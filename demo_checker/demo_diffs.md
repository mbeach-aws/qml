Last update: 2021-12-09  18:04:22 (All times shown in Eastern time)
# List of differences in demonstration outputs

# Table of contents

1. [tutorial_backprop.html](#demo0)
2. [tutorial_general_parshift.html](#demo1)
3. [tutorial_data_reuploading_classifier.html](#demo2)
4. [tutorial_local_cost_functions.html](#demo3)
5. [tutorial_quantum_transfer_learning.html](#demo4)
6. [tutorial_qgrnn.html](#demo5)
7. [tutorial_jax_transformations.html](#demo6)
8. [tutorial_rosalin.html](#demo7)
9. [tutorial_quantum_chemistry.html](#demo8)
10. [tutorial_adaptive_circuits.html](#demo9)
11. [tutorial_error_mitigation.html](#demo10)
12. [tutorial_expressivity_fourier_series.html](#demo11)
13. [tutorial_measurement_optimize.html](#demo12)
14. [tutorial_quanvolution.html](#demo13)
15. [tutorial_quantum_analytic_descent.html](#demo14)
16. [tutorial_vqe_parallel.html](#demo15)
17. [tutorial_qnn_module_tf.html](#demo16)
18. [tutorial_ensemble_multi_qpu.html](#demo17)


Number of demos different/all demos: 18/57

## 1. tutorial_backprop.html <a name="demo0"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_backprop.html):

```
0: ──RX(0.375)──╭C─────────────────╭X──RX(0.599)──╭C──────╭X──╭┤ ⟨Y ⊗ Z⟩
1: ──RY(0.951)──╰X──╭C──RY(0.156)──│──────────────╰X──╭C──│───│┤
2: ──RZ(0.732)──────╰X─────────────╰C──RZ(0.156)──────╰X──╰C──╰┤ ⟨Y ⊗ Z⟩
[-6.51887722e-02 -2.72891905e-02  0.00000000e+00 -9.33934621e-02
 -7.61067572e-01  8.32667268e-17]
-0.06518877224958129
[[-6.51887722e-02 -2.72891905e-02  1.38777878e-17 -9.33934621e-02
  -7.61067572e-01  8.32667268e-17]]
180
0.8947771876917632
Forward pass (best of 3): 0.012585774600029253 sec per loop
Gradient computation (best of 3): 4.671639003399923 sec per loop
4.530878856010531
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_backprop.html):

```
-0.06518877224958129
[-6.51887722e-02 -2.72891905e-02  0.00000000e+00 -9.33934621e-02
 -7.61067572e-01  8.32667268e-17]
[[-6.51887722e-02 -2.72891905e-02  1.38777878e-17 -9.33934621e-02
  -7.61067572e-01  8.32667268e-17]]
180
0.8947771876917632
Forward pass (best of 3): 0.019195340300029784 sec per loop
Gradient computation (best of 3): 5.382104085599986 sec per loop
6.910322508010722
0.9358535378025422
Forward pass (best of 3): 0.05252451469996231 sec per loop
Backward pass (best of 3): 0.10368440069996723 sec per loop
```

---

## 2. tutorial_general_parshift.html <a name="demo1"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_general_parshift.html):

```
For 1 qubits the spectrum is [-1.0, 0, 1.0].
For 2 qubits the spectrum is [-2.0, -1.0, 0, 1.0, 2.0].
For 4 qubits the spectrum is [-4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0].
For 5 qubits the spectrum is [-5.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 5.0].
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_general_parshift.html):

```
For 1 qubits the spectrum is [-1.0, 0.0, 1.0].
For 2 qubits the spectrum is [-2.0, -1.0, 0.0, 1.0, 2.0].
For 4 qubits the spectrum is [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0].
For 5 qubits the spectrum is [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0].
```

---

## 3. tutorial_data_reuploading_classifier.html <a name="demo2"></a>

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

## 4. tutorial_local_cost_functions.html <a name="demo3"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_local_cost_functions.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
--- Global Circuit ---
 0: ──RX(-0.788)──RY(2.83)──╭┤ Probs
 1: ──RX(-0.788)──RY(2.83)──├┤ Probs
 2: ──RX(-0.788)──RY(2.83)──├┤ Probs
 3: ──RX(-0.788)──RY(2.83)──├┤ Probs
 4: ──RX(-0.788)──RY(2.83)──├┤ Probs
 5: ──RX(-0.788)──RY(2.83)──╰┤ Probs
--- Local Circuit
 0: ──RX(-0.788)──RY(2.83)──┤ Probs
 1: ──RX(-0.788)──RY(2.83)──┤ Probs
 2: ──RX(-0.788)──RY(2.83)──┤ Probs
 3: ──RX(-0.788)──RY(2.83)──┤ Probs
 4: ──RX(-0.788)──RY(2.83)──┤ Probs
 5: ──RX(-0.788)──RY(2.83)──┤ Probs
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
 0: ──RX(3)──RY(0)──╭C──────────────────╭┤ Probs
 1: ──RX(3)──RY(0)──╰X──╭C──────────────├┤ Probs
 2: ──RX(3)──RY(0)──────╰X──╭C──────────├┤ Probs
 3: ──RX(3)──RY(0)──────────╰X──╭C──────├┤ Probs
 4: ──RX(3)──RY(0)──────────────╰X──╭C──├┤ Probs
 5: ──RX(3)──RY(0)──────────────────╰X──╰┤ Probs
Cost after step     5:  0.9871000
Cost after step    10:  0.9651000
Cost after step    15:  0.9173000
Cost after step    20:  0.8059000
Cost after step    25:  0.6213000
Cost after step    30:  0.3703000
Cost after step    35:  0.1821000
Cost after step    40:  0.0684000
 0: ──RX(0.44)──RY(-0.00321)──╭C──────────────────┤ Probs
 1: ──RX(3.01)──RY(-4e-05)────╰X──╭C──────────────┤
 2: ──RX(3)─────RY(0)─────────────╰X──╭C──────────┤
 3: ──RX(3)─────RY(0)─────────────────╰X──╭C──────┤
 4: ──RX(3)─────RY(0)─────────────────────╰X──╭C──┤
 5: ──RX(3)─────RY(0)─────────────────────────╰X──┤
tensor(1., requires_grad=True)
Current cost: 0.9999999999972213.
Initial cost: 0.9999999999999843.
Difference: 2.763012041384627e-12
0.9957
 0: ──RX(0.44)──RY(-0.00321)──╭C──────────────────╭┤ Probs
 1: ──RX(3.01)──RY(-4e-05)────╰X──╭C──────────────╰┤ Probs
 2: ──RX(3)─────RY(0)─────────────╰X──╭C───────────┤
 3: ──RX(3)─────RY(0)─────────────────╰X──╭C───────┤
 4: ──RX(3)─────RY(0)─────────────────────╰X──╭C───┤
 5: ──RX(3)─────RY(0)─────────────────────────╰X───┤
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
 0: ──RX(0.00069)──RY(-0.00297)──╭C──────────────────╭┤ Probs
 1: ──RX(0.00331)──RY(-0.0017)───╰X──╭C──────────────├┤ Probs
 2: ──RX(0.017)────RY(-0.00032)──────╰X──╭C──────────├┤ Probs
 3: ──RX(0.0496)───RY(4.5e-05)───────────╰X──╭C──────├┤ Probs
 4: ──RX(0.174)────RY(0.00258)───────────────╰X──╭C──├┤ Probs
 5: ──RX(0.599)────RY(0.0005)────────────────────╰X──╰┤ Probs
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
Cost after step    10:  0.6137000. Locality: 2
---Switching Locality---
Cost after step    20:  0.5159000. Locality: 3
---Switching Locality---
Cost after step    30:  0.7040000. Locality: 4
Cost after step    40:  0.5823000. Locality: 4
Cost after step    50:  0.5055000. Locality: 5
Cost after step    60:  0.6965000. Locality: 6
Cost after step    70:  0.5823000. Locality: 6
---Switching Locality---
Cost after step    80:  0.8792000. Locality: 7
Cost after step    90:  0.7172000. Locality: 7
Cost after step   100:  0.9741000. Locality: 8
Cost after step   110:  0.9329000. Locality: 8
Cost after step   120:  0.8278000. Locality: 8
Cost after step   130:  0.5973000. Locality: 8
Cost after step   140:  0.2649000. Locality: 8
Trained:     1
Plateau'd:     0
--- New run! ---
Cost after step    10:  0.6273000. Locality: 2
---Switching Locality---
Cost after step    20:  0.9218000. Locality: 4
Cost after step    30:  0.8247000. Locality: 4
Cost after step    40:  0.5992000. Locality: 4
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
Cost after step    10:  0.5405000. Locality: 5
---Switching Locality---
Cost after step    20:  0.6024000. Locality: 6
---Switching Locality---
Cost after step    30:  0.6060000. Locality: 7
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
Cost after step    40:  0.5826000. Locality: 4
Cost after step    50:  0.5234000. Locality: 5
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
---Switching Locality---
Cost after step    30:  0.7292000. Locality: 5
Cost after step    40:  0.4696000. Locality: 5
---Switching Locality---
---Switching Locality---
Cost after step    50:  0.5099000. Locality: 7
Cost after step    60:  0.6587000. Locality: 8
Cost after step    70:  0.4912000. Locality: 8
Cost after step    80:  0.2440000. Locality: 8
Trained:     6
Plateau'd:     0
--- New run! ---
---Switching Locality---
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
---Switching Locality---
Cost after step    80:  0.5600000. Locality: 5
---Switching Locality---
Cost after step    90:  0.8307000. Locality: 6
Cost after step   100:  0.6442000. Locality: 6
Cost after step   110:  0.7075000. Locality: 7
Cost after step   120:  0.5936000. Locality: 7
Cost after step   130:  0.7649000. Locality: 8
Cost after step   140:  0.6810000. Locality: 8
Cost after step   150:  0.5936000. Locality: 8
Cost after step   160:  0.4526000. Locality: 8
Cost after step   170:  0.1974000. Locality: 8
Trained:     8
Plateau'd:     0
--- New run! ---
Cost after step    10:  0.5488000. Locality: 1
Cost after step    20:  0.8358000. Locality: 2
Cost after step    30:  0.7614000. Locality: 2
Cost after step    40:  0.6718000. Locality: 2
Cost after step    50:  0.4921000. Locality: 2
---Switching Locality---
---Switching Locality---
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
Cost after step    40:  0.5009000. Locality: 8
Cost after step    50:  0.1676000. Locality: 8
Trained:     3
Plateau'd:     0
--- New run! ---
Cost after step    10:  0.4871000. Locality: 1
---Switching Locality---
Cost after step    20:  0.4948000. Locality: 2
Cost after step    30:  0.6627000. Locality: 3
Cost after step    40:  0.5826000. Locality: 4
---Switching Locality---
Cost after step    50:  0.5234000. Locality: 5
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
---Switching Locality---
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
Cost after step    40:  0.6300000. Locality: 4
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
---Switching Locality---
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
---Switching Locality---
Cost after step    70:  0.7787000. Locality: 5
Cost after step    80:  0.5250000. Locality: 5
---Switching Locality---
Cost after step    90:  0.5270000. Locality: 6
---Switching Locality---
---Switching Locality---
Cost after step   100:  0.9690000. Locality: 8
Cost after step   110:  0.9106000. Locality: 8
Cost after step   120:  0.7561000. Locality: 8
Cost after step   130:  0.4537000. Locality: 8
Cost after step   140:  0.1197000. Locality: 8
Trained:     9
Plateau'd:     0
--- New run! ---
---Switching Locality---
Cost after step    10:  0.6759000. Locality: 2
Cost after step    20:  0.8020000. Locality: 3
Cost after step    30:  0.6412000. Locality: 3
Cost after step    40:  0.6920000. Locality: 4
---Switching Locality---
Cost after step    50:  0.7176000. Locality: 5
Cost after step    60:  0.5973000. Locality: 5
---Switching Locality---
Cost after step    70:  0.5469000. Locality: 6
---Switching Locality---
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
  1%|1         | 616k/44.7M [00:00<00:07, 6.26MB/s]
 18%|#7        | 7.96M/44.7M [00:00<00:00, 46.6MB/s]
 33%|###2      | 14.7M/44.7M [00:00<00:00, 56.4MB/s]
 52%|#####1    | 23.0M/44.7M [00:00<00:00, 68.4MB/s]
 74%|#######4  | 33.1M/44.7M [00:00<00:00, 81.5MB/s]
100%|##########| 44.7M/44.7M [00:00<00:00, 77.7MB/s]
Training started:
Phase: train Epoch: 1/3 Iter: 1/62 Batch time: 0.2356
Phase: train Epoch: 1/3 Iter: 2/62 Batch time: 0.2308
Phase: train Epoch: 1/3 Iter: 3/62 Batch time: 0.2154
Phase: train Epoch: 1/3 Iter: 4/62 Batch time: 0.2154
Phase: train Epoch: 1/3 Iter: 5/62 Batch time: 0.2087
Phase: train Epoch: 1/3 Iter: 6/62 Batch time: 0.2120
Phase: train Epoch: 1/3 Iter: 7/62 Batch time: 0.2093
Phase: train Epoch: 1/3 Iter: 8/62 Batch time: 0.2241
Phase: train Epoch: 1/3 Iter: 9/62 Batch time: 0.2135
Phase: train Epoch: 1/3 Iter: 10/62 Batch time: 0.2129
Phase: train Epoch: 1/3 Iter: 11/62 Batch time: 0.2131
Phase: train Epoch: 1/3 Iter: 12/62 Batch time: 0.2139
Phase: train Epoch: 1/3 Iter: 13/62 Batch time: 0.2091
Phase: train Epoch: 1/3 Iter: 14/62 Batch time: 0.2095
Phase: train Epoch: 1/3 Iter: 15/62 Batch time: 0.2051
Phase: train Epoch: 1/3 Iter: 16/62 Batch time: 0.2112
Phase: train Epoch: 1/3 Iter: 17/62 Batch time: 0.2140
Phase: train Epoch: 1/3 Iter: 18/62 Batch time: 0.2249
Phase: train Epoch: 1/3 Iter: 19/62 Batch time: 0.2137
Phase: train Epoch: 1/3 Iter: 20/62 Batch time: 0.2118
Phase: train Epoch: 1/3 Iter: 21/62 Batch time: 0.2117
Phase: train Epoch: 1/3 Iter: 22/62 Batch time: 0.2109
Phase: train Epoch: 1/3 Iter: 23/62 Batch time: 0.2116
Phase: train Epoch: 1/3 Iter: 24/62 Batch time: 0.2124
Phase: train Epoch: 1/3 Iter: 25/62 Batch time: 0.2097
Phase: train Epoch: 1/3 Iter: 26/62 Batch time: 0.2104
Phase: train Epoch: 1/3 Iter: 27/62 Batch time: 0.2078
Phase: train Epoch: 1/3 Iter: 28/62 Batch time: 0.2137
Phase: train Epoch: 1/3 Iter: 29/62 Batch time: 0.2215
Phase: train Epoch: 1/3 Iter: 30/62 Batch time: 0.2106
Phase: train Epoch: 1/3 Iter: 31/62 Batch time: 0.2074
Phase: train Epoch: 1/3 Iter: 32/62 Batch time: 0.2100
Phase: train Epoch: 1/3 Iter: 33/62 Batch time: 0.2055
Phase: train Epoch: 1/3 Iter: 34/62 Batch time: 0.2028
Phase: train Epoch: 1/3 Iter: 35/62 Batch time: 0.2091
Phase: train Epoch: 1/3 Iter: 36/62 Batch time: 0.2090
Phase: train Epoch: 1/3 Iter: 37/62 Batch time: 0.2138
Phase: train Epoch: 1/3 Iter: 38/62 Batch time: 0.2114
Phase: train Epoch: 1/3 Iter: 39/62 Batch time: 0.2125
Phase: train Epoch: 1/3 Iter: 40/62 Batch time: 0.2062
Phase: train Epoch: 1/3 Iter: 41/62 Batch time: 0.2116
Phase: train Epoch: 1/3 Iter: 42/62 Batch time: 0.2102
Phase: train Epoch: 1/3 Iter: 43/62 Batch time: 0.2039
Phase: train Epoch: 1/3 Iter: 44/62 Batch time: 0.2084
Phase: train Epoch: 1/3 Iter: 45/62 Batch time: 0.2044
Phase: train Epoch: 1/3 Iter: 46/62 Batch time: 0.2138
Phase: train Epoch: 1/3 Iter: 47/62 Batch time: 0.2089
Phase: train Epoch: 1/3 Iter: 48/62 Batch time: 0.2100
Phase: train Epoch: 1/3 Iter: 49/62 Batch time: 0.2156
Phase: train Epoch: 1/3 Iter: 50/62 Batch time: 0.2085
Phase: train Epoch: 1/3 Iter: 51/62 Batch time: 0.2066
Phase: train Epoch: 1/3 Iter: 52/62 Batch time: 0.2149
Phase: train Epoch: 1/3 Iter: 53/62 Batch time: 0.2110
Phase: train Epoch: 1/3 Iter: 54/62 Batch time: 0.2110
Phase: train Epoch: 1/3 Iter: 55/62 Batch time: 0.2130
Phase: train Epoch: 1/3 Iter: 56/62 Batch time: 0.2087
Phase: train Epoch: 1/3 Iter: 57/62 Batch time: 0.2055
Phase: train Epoch: 1/3 Iter: 58/62 Batch time: 0.2049
Phase: train Epoch: 1/3 Iter: 59/62 Batch time: 0.2073
Phase: train Epoch: 1/3 Iter: 60/62 Batch time: 0.2034
Phase: train Epoch: 1/3 Iter: 61/62 Batch time: 0.2087
Phase: train Epoch: 1/3 Loss: 0.6993 Acc: 0.5246
Phase: validation Epoch: 1/3 Iter: 1/39 Batch time: 0.1525
Phase: validation Epoch: 1/3 Iter: 2/39 Batch time: 0.1475
Phase: validation Epoch: 1/3 Iter: 3/39 Batch time: 0.1524
Phase: validation Epoch: 1/3 Iter: 4/39 Batch time: 0.1528
Phase: validation Epoch: 1/3 Iter: 5/39 Batch time: 0.1509
Phase: validation Epoch: 1/3 Iter: 6/39 Batch time: 0.1536
Phase: validation Epoch: 1/3 Iter: 7/39 Batch time: 0.1501
Phase: validation Epoch: 1/3 Iter: 8/39 Batch time: 0.1479
Phase: validation Epoch: 1/3 Iter: 9/39 Batch time: 0.1523
Phase: validation Epoch: 1/3 Iter: 10/39 Batch time: 0.1477
Phase: validation Epoch: 1/3 Iter: 11/39 Batch time: 0.1487
Phase: validation Epoch: 1/3 Iter: 12/39 Batch time: 0.1530
Phase: validation Epoch: 1/3 Iter: 13/39 Batch time: 0.1494
Phase: validation Epoch: 1/3 Iter: 14/39 Batch time: 0.1520
Phase: validation Epoch: 1/3 Iter: 15/39 Batch time: 0.1517
Phase: validation Epoch: 1/3 Iter: 16/39 Batch time: 0.1533
Phase: validation Epoch: 1/3 Iter: 17/39 Batch time: 0.1540
Phase: validation Epoch: 1/3 Iter: 18/39 Batch time: 0.1489
Phase: validation Epoch: 1/3 Iter: 19/39 Batch time: 0.1476
Phase: validation Epoch: 1/3 Iter: 20/39 Batch time: 0.1510
Phase: validation Epoch: 1/3 Iter: 21/39 Batch time: 0.1487
Phase: validation Epoch: 1/3 Iter: 22/39 Batch time: 0.1508
Phase: validation Epoch: 1/3 Iter: 23/39 Batch time: 0.1483
Phase: validation Epoch: 1/3 Iter: 24/39 Batch time: 0.1470
Phase: validation Epoch: 1/3 Iter: 25/39 Batch time: 0.1532
Phase: validation Epoch: 1/3 Iter: 26/39 Batch time: 0.1480
Phase: validation Epoch: 1/3 Iter: 27/39 Batch time: 0.1515
Phase: validation Epoch: 1/3 Iter: 28/39 Batch time: 0.1576
Phase: validation Epoch: 1/3 Iter: 29/39 Batch time: 0.1511
Phase: validation Epoch: 1/3 Iter: 30/39 Batch time: 0.1520
Phase: validation Epoch: 1/3 Iter: 31/39 Batch time: 0.1499
Phase: validation Epoch: 1/3 Iter: 32/39 Batch time: 0.1477
Phase: validation Epoch: 1/3 Iter: 33/39 Batch time: 0.1499
Phase: validation Epoch: 1/3 Iter: 34/39 Batch time: 0.1512
Phase: validation Epoch: 1/3 Iter: 35/39 Batch time: 0.1499
Phase: validation Epoch: 1/3 Iter: 36/39 Batch time: 0.1578
Phase: validation Epoch: 1/3 Iter: 37/39 Batch time: 0.1564
Phase: validation Epoch: 1/3 Iter: 38/39 Batch time: 0.1521
Phase: validation Epoch: 1/3 Iter: 39/39 Batch time: 0.0578
Phase: validation   Epoch: 1/3 Loss: 0.6432 Acc: 0.6536
Phase: train Epoch: 2/3 Iter: 1/62 Batch time: 0.2049
Phase: train Epoch: 2/3 Iter: 2/62 Batch time: 0.2039
Phase: train Epoch: 2/3 Iter: 3/62 Batch time: 0.2104
Phase: train Epoch: 2/3 Iter: 4/62 Batch time: 0.2148
Phase: train Epoch: 2/3 Iter: 5/62 Batch time: 0.2136
Phase: train Epoch: 2/3 Iter: 6/62 Batch time: 0.2105
Phase: train Epoch: 2/3 Iter: 7/62 Batch time: 0.2066
Phase: train Epoch: 2/3 Iter: 8/62 Batch time: 0.2134
Phase: train Epoch: 2/3 Iter: 9/62 Batch time: 0.2119
Phase: train Epoch: 2/3 Iter: 10/62 Batch time: 0.2098
Phase: train Epoch: 2/3 Iter: 11/62 Batch time: 0.2113
Phase: train Epoch: 2/3 Iter: 12/62 Batch time: 0.2109
Phase: train Epoch: 2/3 Iter: 13/62 Batch time: 0.2043
Phase: train Epoch: 2/3 Iter: 14/62 Batch time: 0.2061
Phase: train Epoch: 2/3 Iter: 15/62 Batch time: 0.2093
Phase: train Epoch: 2/3 Iter: 16/62 Batch time: 0.2133
Phase: train Epoch: 2/3 Iter: 17/62 Batch time: 0.2076
Phase: train Epoch: 2/3 Iter: 18/62 Batch time: 0.2071
Phase: train Epoch: 2/3 Iter: 19/62 Batch time: 0.2095
Phase: train Epoch: 2/3 Iter: 20/62 Batch time: 0.2063
Phase: train Epoch: 2/3 Iter: 21/62 Batch time: 0.2113
Phase: train Epoch: 2/3 Iter: 22/62 Batch time: 0.2083
Phase: train Epoch: 2/3 Iter: 23/62 Batch time: 0.2107
Phase: train Epoch: 2/3 Iter: 24/62 Batch time: 0.2112
Phase: train Epoch: 2/3 Iter: 25/62 Batch time: 0.2114
Phase: train Epoch: 2/3 Iter: 26/62 Batch time: 0.2102
Phase: train Epoch: 2/3 Iter: 27/62 Batch time: 0.2087
Phase: train Epoch: 2/3 Iter: 28/62 Batch time: 0.2075
Phase: train Epoch: 2/3 Iter: 29/62 Batch time: 0.2159
Phase: train Epoch: 2/3 Iter: 30/62 Batch time: 0.2110
Phase: train Epoch: 2/3 Iter: 31/62 Batch time: 0.2145
Phase: train Epoch: 2/3 Iter: 32/62 Batch time: 0.2099
Phase: train Epoch: 2/3 Iter: 33/62 Batch time: 0.2169
Phase: train Epoch: 2/3 Iter: 34/62 Batch time: 0.2112
Phase: train Epoch: 2/3 Iter: 35/62 Batch time: 0.2087
Phase: train Epoch: 2/3 Iter: 36/62 Batch time: 0.2202
Phase: train Epoch: 2/3 Iter: 37/62 Batch time: 0.2114
Phase: train Epoch: 2/3 Iter: 38/62 Batch time: 0.2096
Phase: train Epoch: 2/3 Iter: 39/62 Batch time: 0.2117
Phase: train Epoch: 2/3 Iter: 40/62 Batch time: 0.2138
Phase: train Epoch: 2/3 Iter: 41/62 Batch time: 0.2078
Phase: train Epoch: 2/3 Iter: 42/62 Batch time: 0.2168
Phase: train Epoch: 2/3 Iter: 43/62 Batch time: 0.2168
Phase: train Epoch: 2/3 Iter: 44/62 Batch time: 0.2208
Phase: train Epoch: 2/3 Iter: 45/62 Batch time: 0.2198
Phase: train Epoch: 2/3 Iter: 46/62 Batch time: 0.2182
Phase: train Epoch: 2/3 Iter: 47/62 Batch time: 0.2248
Phase: train Epoch: 2/3 Iter: 48/62 Batch time: 0.2155
Phase: train Epoch: 2/3 Iter: 49/62 Batch time: 0.2104
Phase: train Epoch: 2/3 Iter: 50/62 Batch time: 0.2141
Phase: train Epoch: 2/3 Iter: 51/62 Batch time: 0.2174
Phase: train Epoch: 2/3 Iter: 52/62 Batch time: 0.2163
Phase: train Epoch: 2/3 Iter: 53/62 Batch time: 0.2150
Phase: train Epoch: 2/3 Iter: 54/62 Batch time: 0.2140
Phase: train Epoch: 2/3 Iter: 55/62 Batch time: 0.2117
Phase: train Epoch: 2/3 Iter: 56/62 Batch time: 0.2192
Phase: train Epoch: 2/3 Iter: 57/62 Batch time: 0.2140
Phase: train Epoch: 2/3 Iter: 58/62 Batch time: 0.2153
Phase: train Epoch: 2/3 Iter: 59/62 Batch time: 0.2138
Phase: train Epoch: 2/3 Iter: 60/62 Batch time: 0.2180
Phase: train Epoch: 2/3 Iter: 61/62 Batch time: 0.2145
Phase: train Epoch: 2/3 Loss: 0.6141 Acc: 0.7049
Phase: validation Epoch: 2/3 Iter: 1/39 Batch time: 0.1605
Phase: validation Epoch: 2/3 Iter: 2/39 Batch time: 0.1522
Phase: validation Epoch: 2/3 Iter: 3/39 Batch time: 0.1538
Phase: validation Epoch: 2/3 Iter: 4/39 Batch time: 0.1532
Phase: validation Epoch: 2/3 Iter: 5/39 Batch time: 0.1479
Phase: validation Epoch: 2/3 Iter: 6/39 Batch time: 0.1526
Phase: validation Epoch: 2/3 Iter: 7/39 Batch time: 0.1516
Phase: validation Epoch: 2/3 Iter: 8/39 Batch time: 0.1502
Phase: validation Epoch: 2/3 Iter: 9/39 Batch time: 0.1491
Phase: validation Epoch: 2/3 Iter: 10/39 Batch time: 0.1494
Phase: validation Epoch: 2/3 Iter: 11/39 Batch time: 0.1497
Phase: validation Epoch: 2/3 Iter: 12/39 Batch time: 0.1543
Phase: validation Epoch: 2/3 Iter: 13/39 Batch time: 0.1515
Phase: validation Epoch: 2/3 Iter: 14/39 Batch time: 0.1531
Phase: validation Epoch: 2/3 Iter: 15/39 Batch time: 0.1489
Phase: validation Epoch: 2/3 Iter: 16/39 Batch time: 0.1526
Phase: validation Epoch: 2/3 Iter: 17/39 Batch time: 0.1899
Phase: validation Epoch: 2/3 Iter: 18/39 Batch time: 0.1507
Phase: validation Epoch: 2/3 Iter: 19/39 Batch time: 0.1536
Phase: validation Epoch: 2/3 Iter: 20/39 Batch time: 0.1535
Phase: validation Epoch: 2/3 Iter: 21/39 Batch time: 0.1535
Phase: validation Epoch: 2/3 Iter: 22/39 Batch time: 0.1557
Phase: validation Epoch: 2/3 Iter: 23/39 Batch time: 0.1547
Phase: validation Epoch: 2/3 Iter: 24/39 Batch time: 0.1509
Phase: validation Epoch: 2/3 Iter: 25/39 Batch time: 0.1501
Phase: validation Epoch: 2/3 Iter: 26/39 Batch time: 0.1487
Phase: validation Epoch: 2/3 Iter: 27/39 Batch time: 0.1520
Phase: validation Epoch: 2/3 Iter: 28/39 Batch time: 0.1552
Phase: validation Epoch: 2/3 Iter: 29/39 Batch time: 0.1541
Phase: validation Epoch: 2/3 Iter: 30/39 Batch time: 0.1531
Phase: validation Epoch: 2/3 Iter: 31/39 Batch time: 0.1535
Phase: validation Epoch: 2/3 Iter: 32/39 Batch time: 0.1488
Phase: validation Epoch: 2/3 Iter: 33/39 Batch time: 0.1482
Phase: validation Epoch: 2/3 Iter: 34/39 Batch time: 0.1475
Phase: validation Epoch: 2/3 Iter: 35/39 Batch time: 0.1512
Phase: validation Epoch: 2/3 Iter: 36/39 Batch time: 0.1503
Phase: validation Epoch: 2/3 Iter: 37/39 Batch time: 0.1490
Phase: validation Epoch: 2/3 Iter: 38/39 Batch time: 0.1483
Phase: validation Epoch: 2/3 Iter: 39/39 Batch time: 0.0503
Phase: validation   Epoch: 2/3 Loss: 0.5392 Acc: 0.8235
Phase: train Epoch: 3/3 Iter: 1/62 Batch time: 0.2042
Phase: train Epoch: 3/3 Iter: 2/62 Batch time: 0.2037
Phase: train Epoch: 3/3 Iter: 3/62 Batch time: 0.2011
Phase: train Epoch: 3/3 Iter: 4/62 Batch time: 0.2013
Phase: train Epoch: 3/3 Iter: 5/62 Batch time: 0.2005
Phase: train Epoch: 3/3 Iter: 6/62 Batch time: 0.2116
Phase: train Epoch: 3/3 Iter: 7/62 Batch time: 0.2113
Phase: train Epoch: 3/3 Iter: 8/62 Batch time: 0.2113
Phase: train Epoch: 3/3 Iter: 9/62 Batch time: 0.2084
Phase: train Epoch: 3/3 Iter: 10/62 Batch time: 0.2129
Phase: train Epoch: 3/3 Iter: 11/62 Batch time: 0.2123
Phase: train Epoch: 3/3 Iter: 12/62 Batch time: 0.2114
Phase: train Epoch: 3/3 Iter: 13/62 Batch time: 0.2173
Phase: train Epoch: 3/3 Iter: 14/62 Batch time: 0.2172
Phase: train Epoch: 3/3 Iter: 15/62 Batch time: 0.2130
Phase: train Epoch: 3/3 Iter: 16/62 Batch time: 0.2168
Phase: train Epoch: 3/3 Iter: 17/62 Batch time: 0.2100
Phase: train Epoch: 3/3 Iter: 18/62 Batch time: 0.2101
Phase: train Epoch: 3/3 Iter: 19/62 Batch time: 0.2145
Phase: train Epoch: 3/3 Iter: 20/62 Batch time: 0.2152
Phase: train Epoch: 3/3 Iter: 21/62 Batch time: 0.2066
Phase: train Epoch: 3/3 Iter: 22/62 Batch time: 0.2087
Phase: train Epoch: 3/3 Iter: 23/62 Batch time: 0.2080
Phase: train Epoch: 3/3 Iter: 24/62 Batch time: 0.2111
Phase: train Epoch: 3/3 Iter: 25/62 Batch time: 0.2146
Phase: train Epoch: 3/3 Iter: 26/62 Batch time: 0.2122
Phase: train Epoch: 3/3 Iter: 27/62 Batch time: 0.2065
Phase: train Epoch: 3/3 Iter: 28/62 Batch time: 0.2056
Phase: train Epoch: 3/3 Iter: 29/62 Batch time: 0.2104
Phase: train Epoch: 3/3 Iter: 30/62 Batch time: 0.2078
Phase: train Epoch: 3/3 Iter: 31/62 Batch time: 0.2032
Phase: train Epoch: 3/3 Iter: 32/62 Batch time: 0.2141
Phase: train Epoch: 3/3 Iter: 33/62 Batch time: 0.2154
Phase: train Epoch: 3/3 Iter: 34/62 Batch time: 0.2139
Phase: train Epoch: 3/3 Iter: 35/62 Batch time: 0.2129
Phase: train Epoch: 3/3 Iter: 36/62 Batch time: 0.2135
Phase: train Epoch: 3/3 Iter: 37/62 Batch time: 0.2128
Phase: train Epoch: 3/3 Iter: 38/62 Batch time: 0.2097
Phase: train Epoch: 3/3 Iter: 39/62 Batch time: 0.2117
Phase: train Epoch: 3/3 Iter: 40/62 Batch time: 0.2075
Phase: train Epoch: 3/3 Iter: 41/62 Batch time: 0.2150
Phase: train Epoch: 3/3 Iter: 42/62 Batch time: 0.2138
Phase: train Epoch: 3/3 Iter: 43/62 Batch time: 0.2046
Phase: train Epoch: 3/3 Iter: 44/62 Batch time: 0.2173
Phase: train Epoch: 3/3 Iter: 45/62 Batch time: 0.2096
Phase: train Epoch: 3/3 Iter: 46/62 Batch time: 0.2129
Phase: train Epoch: 3/3 Iter: 47/62 Batch time: 0.2031
Phase: train Epoch: 3/3 Iter: 48/62 Batch time: 0.2059
Phase: train Epoch: 3/3 Iter: 49/62 Batch time: 0.2033
Phase: train Epoch: 3/3 Iter: 50/62 Batch time: 0.2120
Phase: train Epoch: 3/3 Iter: 51/62 Batch time: 0.2075
Phase: train Epoch: 3/3 Iter: 52/62 Batch time: 0.2240
Phase: train Epoch: 3/3 Iter: 53/62 Batch time: 0.2117
Phase: train Epoch: 3/3 Iter: 54/62 Batch time: 0.2085
Phase: train Epoch: 3/3 Iter: 55/62 Batch time: 0.2046
Phase: train Epoch: 3/3 Iter: 56/62 Batch time: 0.2070
Phase: train Epoch: 3/3 Iter: 57/62 Batch time: 0.2133
Phase: train Epoch: 3/3 Iter: 58/62 Batch time: 0.2085
Phase: train Epoch: 3/3 Iter: 59/62 Batch time: 0.2109
Phase: train Epoch: 3/3 Iter: 60/62 Batch time: 0.2101
Phase: train Epoch: 3/3 Iter: 61/62 Batch time: 0.2072
Phase: train Epoch: 3/3 Loss: 0.5652 Acc: 0.7336
Phase: validation Epoch: 3/3 Iter: 1/39 Batch time: 0.1581
Phase: validation Epoch: 3/3 Iter: 2/39 Batch time: 0.1524
Phase: validation Epoch: 3/3 Iter: 3/39 Batch time: 0.1557
Phase: validation Epoch: 3/3 Iter: 4/39 Batch time: 0.1538
Phase: validation Epoch: 3/3 Iter: 5/39 Batch time: 0.1501
Phase: validation Epoch: 3/3 Iter: 6/39 Batch time: 0.1502
Phase: validation Epoch: 3/3 Iter: 7/39 Batch time: 0.1542
Phase: validation Epoch: 3/3 Iter: 8/39 Batch time: 0.1494
Phase: validation Epoch: 3/3 Iter: 9/39 Batch time: 0.1522
Phase: validation Epoch: 3/3 Iter: 10/39 Batch time: 0.1514
Phase: validation Epoch: 3/3 Iter: 11/39 Batch time: 0.1486
Phase: validation Epoch: 3/3 Iter: 12/39 Batch time: 0.1515
Phase: validation Epoch: 3/3 Iter: 13/39 Batch time: 0.1527
Phase: validation Epoch: 3/3 Iter: 14/39 Batch time: 0.1532
Phase: validation Epoch: 3/3 Iter: 15/39 Batch time: 0.1507
Phase: validation Epoch: 3/3 Iter: 16/39 Batch time: 0.1525
Phase: validation Epoch: 3/3 Iter: 17/39 Batch time: 0.1535
Phase: validation Epoch: 3/3 Iter: 18/39 Batch time: 0.1507
Phase: validation Epoch: 3/3 Iter: 19/39 Batch time: 0.1512
Phase: validation Epoch: 3/3 Iter: 20/39 Batch time: 0.1531
Phase: validation Epoch: 3/3 Iter: 21/39 Batch time: 0.1546
Phase: validation Epoch: 3/3 Iter: 22/39 Batch time: 0.1586
Phase: validation Epoch: 3/3 Iter: 23/39 Batch time: 0.1579
Phase: validation Epoch: 3/3 Iter: 24/39 Batch time: 0.1580
Phase: validation Epoch: 3/3 Iter: 25/39 Batch time: 0.1566
Phase: validation Epoch: 3/3 Iter: 26/39 Batch time: 0.1556
Phase: validation Epoch: 3/3 Iter: 27/39 Batch time: 0.1534
Phase: validation Epoch: 3/3 Iter: 28/39 Batch time: 0.1570
Phase: validation Epoch: 3/3 Iter: 29/39 Batch time: 0.1589
Phase: validation Epoch: 3/3 Iter: 30/39 Batch time: 0.1559
Phase: validation Epoch: 3/3 Iter: 31/39 Batch time: 0.1570
Phase: validation Epoch: 3/3 Iter: 32/39 Batch time: 0.1534
Phase: validation Epoch: 3/3 Iter: 33/39 Batch time: 0.1502
Phase: validation Epoch: 3/3 Iter: 34/39 Batch time: 0.1489
Phase: validation Epoch: 3/3 Iter: 35/39 Batch time: 0.1542
Phase: validation Epoch: 3/3 Iter: 36/39 Batch time: 0.1481
Phase: validation Epoch: 3/3 Iter: 37/39 Batch time: 0.1538
Phase: validation Epoch: 3/3 Iter: 38/39 Batch time: 0.1509
Phase: validation Epoch: 3/3 Iter: 39/39 Batch time: 0.0525
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
 39%|###9      | 17.6M/44.7M [00:00<00:00, 184MB/s]
 94%|#########4| 42.1M/44.7M [00:00<00:00, 227MB/s]
100%|##########| 44.7M/44.7M [00:00<00:00, 222MB/s]
Training started:
Phase: train Epoch: 1/3 Iter: 1/62 Batch time: 0.3063
Phase: train Epoch: 1/3 Iter: 2/62 Batch time: 0.2914
Phase: train Epoch: 1/3 Iter: 3/62 Batch time: 0.2744
Phase: train Epoch: 1/3 Iter: 4/62 Batch time: 0.2601
Phase: train Epoch: 1/3 Iter: 5/62 Batch time: 0.2994
Phase: train Epoch: 1/3 Iter: 6/62 Batch time: 0.2622
Phase: train Epoch: 1/3 Iter: 7/62 Batch time: 0.2488
Phase: train Epoch: 1/3 Iter: 8/62 Batch time: 0.2540
Phase: train Epoch: 1/3 Iter: 9/62 Batch time: 0.2658
Phase: train Epoch: 1/3 Iter: 10/62 Batch time: 0.2580
Phase: train Epoch: 1/3 Iter: 11/62 Batch time: 0.2830
Phase: train Epoch: 1/3 Iter: 12/62 Batch time: 0.2494
Phase: train Epoch: 1/3 Iter: 13/62 Batch time: 0.2547
Phase: train Epoch: 1/3 Iter: 14/62 Batch time: 0.2767
Phase: train Epoch: 1/3 Iter: 15/62 Batch time: 0.2619
Phase: train Epoch: 1/3 Iter: 16/62 Batch time: 0.2488
Phase: train Epoch: 1/3 Iter: 17/62 Batch time: 0.2630
Phase: train Epoch: 1/3 Iter: 18/62 Batch time: 0.2787
Phase: train Epoch: 1/3 Iter: 19/62 Batch time: 0.2610
Phase: train Epoch: 1/3 Iter: 20/62 Batch time: 0.2603
Phase: train Epoch: 1/3 Iter: 21/62 Batch time: 0.2613
Phase: train Epoch: 1/3 Iter: 22/62 Batch time: 0.2580
Phase: train Epoch: 1/3 Iter: 23/62 Batch time: 0.2488
Phase: train Epoch: 1/3 Iter: 24/62 Batch time: 0.2488
Phase: train Epoch: 1/3 Iter: 25/62 Batch time: 0.2607
Phase: train Epoch: 1/3 Iter: 26/62 Batch time: 0.2701
Phase: train Epoch: 1/3 Iter: 27/62 Batch time: 0.2545
Phase: train Epoch: 1/3 Iter: 28/62 Batch time: 0.2673
Phase: train Epoch: 1/3 Iter: 29/62 Batch time: 0.2740
Phase: train Epoch: 1/3 Iter: 30/62 Batch time: 0.2808
Phase: train Epoch: 1/3 Iter: 31/62 Batch time: 0.2859
Phase: train Epoch: 1/3 Iter: 32/62 Batch time: 0.2807
Phase: train Epoch: 1/3 Iter: 33/62 Batch time: 0.2495
Phase: train Epoch: 1/3 Iter: 34/62 Batch time: 0.2711
Phase: train Epoch: 1/3 Iter: 35/62 Batch time: 0.2611
Phase: train Epoch: 1/3 Iter: 36/62 Batch time: 0.2620
Phase: train Epoch: 1/3 Iter: 37/62 Batch time: 0.2499
Phase: train Epoch: 1/3 Iter: 38/62 Batch time: 0.2743
Phase: train Epoch: 1/3 Iter: 39/62 Batch time: 0.2657
Phase: train Epoch: 1/3 Iter: 40/62 Batch time: 0.2944
Phase: train Epoch: 1/3 Iter: 41/62 Batch time: 0.2901
Phase: train Epoch: 1/3 Iter: 42/62 Batch time: 0.2745
Phase: train Epoch: 1/3 Iter: 43/62 Batch time: 0.2932
Phase: train Epoch: 1/3 Iter: 44/62 Batch time: 0.3024
Phase: train Epoch: 1/3 Iter: 45/62 Batch time: 0.3152
Phase: train Epoch: 1/3 Iter: 46/62 Batch time: 0.3210
Phase: train Epoch: 1/3 Iter: 47/62 Batch time: 0.4990
Phase: train Epoch: 1/3 Iter: 48/62 Batch time: 0.6108
Phase: train Epoch: 1/3 Iter: 49/62 Batch time: 0.3731
Phase: train Epoch: 1/3 Iter: 50/62 Batch time: 0.2473
Phase: train Epoch: 1/3 Iter: 51/62 Batch time: 0.2482
Phase: train Epoch: 1/3 Iter: 52/62 Batch time: 0.2492
Phase: train Epoch: 1/3 Iter: 53/62 Batch time: 0.2872
Phase: train Epoch: 1/3 Iter: 54/62 Batch time: 0.2592
Phase: train Epoch: 1/3 Iter: 55/62 Batch time: 0.2966
Phase: train Epoch: 1/3 Iter: 56/62 Batch time: 0.2663
Phase: train Epoch: 1/3 Iter: 57/62 Batch time: 0.2485
Phase: train Epoch: 1/3 Iter: 58/62 Batch time: 0.2545
Phase: train Epoch: 1/3 Iter: 59/62 Batch time: 0.2764
Phase: train Epoch: 1/3 Iter: 60/62 Batch time: 0.2738
Phase: train Epoch: 1/3 Iter: 61/62 Batch time: 0.2546
Phase: train Epoch: 1/3 Loss: 0.6993 Acc: 0.5246
Phase: validation Epoch: 1/3 Iter: 1/39 Batch time: 0.1978
Phase: validation Epoch: 1/3 Iter: 2/39 Batch time: 0.1933
Phase: validation Epoch: 1/3 Iter: 3/39 Batch time: 0.2190
Phase: validation Epoch: 1/3 Iter: 4/39 Batch time: 0.2312
Phase: validation Epoch: 1/3 Iter: 5/39 Batch time: 0.2019
Phase: validation Epoch: 1/3 Iter: 6/39 Batch time: 0.1882
Phase: validation Epoch: 1/3 Iter: 7/39 Batch time: 0.1891
Phase: validation Epoch: 1/3 Iter: 8/39 Batch time: 0.1870
Phase: validation Epoch: 1/3 Iter: 9/39 Batch time: 0.2019
Phase: validation Epoch: 1/3 Iter: 10/39 Batch time: 0.1975
Phase: validation Epoch: 1/3 Iter: 11/39 Batch time: 0.2222
Phase: validation Epoch: 1/3 Iter: 12/39 Batch time: 0.1994
Phase: validation Epoch: 1/3 Iter: 13/39 Batch time: 0.2052
Phase: validation Epoch: 1/3 Iter: 14/39 Batch time: 0.1960
Phase: validation Epoch: 1/3 Iter: 15/39 Batch time: 0.1984
Phase: validation Epoch: 1/3 Iter: 16/39 Batch time: 0.2034
Phase: validation Epoch: 1/3 Iter: 17/39 Batch time: 0.2065
Phase: validation Epoch: 1/3 Iter: 18/39 Batch time: 0.1983
Phase: validation Epoch: 1/3 Iter: 19/39 Batch time: 0.2189
Phase: validation Epoch: 1/3 Iter: 20/39 Batch time: 0.1960
Phase: validation Epoch: 1/3 Iter: 21/39 Batch time: 0.1891
Phase: validation Epoch: 1/3 Iter: 22/39 Batch time: 0.1885
Phase: validation Epoch: 1/3 Iter: 23/39 Batch time: 0.1982
Phase: validation Epoch: 1/3 Iter: 24/39 Batch time: 0.1974
Phase: validation Epoch: 1/3 Iter: 25/39 Batch time: 0.1940
Phase: validation Epoch: 1/3 Iter: 26/39 Batch time: 0.1977
Phase: validation Epoch: 1/3 Iter: 27/39 Batch time: 0.1936
Phase: validation Epoch: 1/3 Iter: 28/39 Batch time: 0.1917
Phase: validation Epoch: 1/3 Iter: 29/39 Batch time: 0.1988
Phase: validation Epoch: 1/3 Iter: 30/39 Batch time: 0.1993
Phase: validation Epoch: 1/3 Iter: 31/39 Batch time: 0.2039
Phase: validation Epoch: 1/3 Iter: 32/39 Batch time: 0.2265
Phase: validation Epoch: 1/3 Iter: 33/39 Batch time: 0.2004
Phase: validation Epoch: 1/3 Iter: 34/39 Batch time: 0.1904
Phase: validation Epoch: 1/3 Iter: 35/39 Batch time: 0.1914
Phase: validation Epoch: 1/3 Iter: 36/39 Batch time: 0.2035
Phase: validation Epoch: 1/3 Iter: 37/39 Batch time: 0.1992
Phase: validation Epoch: 1/3 Iter: 38/39 Batch time: 0.1946
Phase: validation Epoch: 1/3 Iter: 39/39 Batch time: 0.0641
Phase: validation   Epoch: 1/3 Loss: 0.6432 Acc: 0.6536
Phase: train Epoch: 2/3 Iter: 1/62 Batch time: 0.2459
Phase: train Epoch: 2/3 Iter: 2/62 Batch time: 0.2617
Phase: train Epoch: 2/3 Iter: 3/62 Batch time: 0.2587
Phase: train Epoch: 2/3 Iter: 4/62 Batch time: 0.2508
Phase: train Epoch: 2/3 Iter: 5/62 Batch time: 0.2504
Phase: train Epoch: 2/3 Iter: 6/62 Batch time: 0.2603
Phase: train Epoch: 2/3 Iter: 7/62 Batch time: 0.2636
Phase: train Epoch: 2/3 Iter: 8/62 Batch time: 0.2473
Phase: train Epoch: 2/3 Iter: 9/62 Batch time: 0.2555
Phase: train Epoch: 2/3 Iter: 10/62 Batch time: 0.2721
Phase: train Epoch: 2/3 Iter: 11/62 Batch time: 0.2626
Phase: train Epoch: 2/3 Iter: 12/62 Batch time: 0.2616
Phase: train Epoch: 2/3 Iter: 13/62 Batch time: 0.2718
Phase: train Epoch: 2/3 Iter: 14/62 Batch time: 0.2519
Phase: train Epoch: 2/3 Iter: 15/62 Batch time: 0.2537
Phase: train Epoch: 2/3 Iter: 16/62 Batch time: 0.2579
Phase: train Epoch: 2/3 Iter: 17/62 Batch time: 0.2572
Phase: train Epoch: 2/3 Iter: 18/62 Batch time: 0.2568
Phase: train Epoch: 2/3 Iter: 19/62 Batch time: 0.2719
Phase: train Epoch: 2/3 Iter: 20/62 Batch time: 0.2563
Phase: train Epoch: 2/3 Iter: 21/62 Batch time: 0.2648
Phase: train Epoch: 2/3 Iter: 22/62 Batch time: 0.2548
Phase: train Epoch: 2/3 Iter: 23/62 Batch time: 0.2628
Phase: train Epoch: 2/3 Iter: 24/62 Batch time: 0.2683
Phase: train Epoch: 2/3 Iter: 25/62 Batch time: 0.2496
Phase: train Epoch: 2/3 Iter: 26/62 Batch time: 0.2733
Phase: train Epoch: 2/3 Iter: 27/62 Batch time: 0.2727
Phase: train Epoch: 2/3 Iter: 28/62 Batch time: 0.2634
Phase: train Epoch: 2/3 Iter: 29/62 Batch time: 0.2585
Phase: train Epoch: 2/3 Iter: 30/62 Batch time: 0.2669
Phase: train Epoch: 2/3 Iter: 31/62 Batch time: 0.2789
Phase: train Epoch: 2/3 Iter: 32/62 Batch time: 0.2590
Phase: train Epoch: 2/3 Iter: 33/62 Batch time: 0.2823
Phase: train Epoch: 2/3 Iter: 34/62 Batch time: 0.2697
Phase: train Epoch: 2/3 Iter: 35/62 Batch time: 0.2685
Phase: train Epoch: 2/3 Iter: 36/62 Batch time: 0.2505
Phase: train Epoch: 2/3 Iter: 37/62 Batch time: 0.2751
Phase: train Epoch: 2/3 Iter: 38/62 Batch time: 0.2509
Phase: train Epoch: 2/3 Iter: 39/62 Batch time: 0.2503
Phase: train Epoch: 2/3 Iter: 40/62 Batch time: 0.2639
Phase: train Epoch: 2/3 Iter: 41/62 Batch time: 0.2795
Phase: train Epoch: 2/3 Iter: 42/62 Batch time: 0.2594
Phase: train Epoch: 2/3 Iter: 43/62 Batch time: 0.2584
Phase: train Epoch: 2/3 Iter: 44/62 Batch time: 0.2638
Phase: train Epoch: 2/3 Iter: 45/62 Batch time: 0.2597
Phase: train Epoch: 2/3 Iter: 46/62 Batch time: 0.2619
Phase: train Epoch: 2/3 Iter: 47/62 Batch time: 0.2786
Phase: train Epoch: 2/3 Iter: 48/62 Batch time: 0.2720
Phase: train Epoch: 2/3 Iter: 49/62 Batch time: 0.2567
Phase: train Epoch: 2/3 Iter: 50/62 Batch time: 0.2659
Phase: train Epoch: 2/3 Iter: 51/62 Batch time: 0.2625
Phase: train Epoch: 2/3 Iter: 52/62 Batch time: 0.2546
Phase: train Epoch: 2/3 Iter: 53/62 Batch time: 0.2679
Phase: train Epoch: 2/3 Iter: 54/62 Batch time: 0.2658
Phase: train Epoch: 2/3 Iter: 55/62 Batch time: 0.2687
Phase: train Epoch: 2/3 Iter: 56/62 Batch time: 0.2568
Phase: train Epoch: 2/3 Iter: 57/62 Batch time: 0.2590
Phase: train Epoch: 2/3 Iter: 58/62 Batch time: 0.2672
Phase: train Epoch: 2/3 Iter: 59/62 Batch time: 0.2568
Phase: train Epoch: 2/3 Iter: 60/62 Batch time: 0.2603
Phase: train Epoch: 2/3 Iter: 61/62 Batch time: 0.2650
Phase: train Epoch: 2/3 Loss: 0.6141 Acc: 0.7049
Phase: validation Epoch: 2/3 Iter: 1/39 Batch time: 0.1897
Phase: validation Epoch: 2/3 Iter: 2/39 Batch time: 0.1896
Phase: validation Epoch: 2/3 Iter: 3/39 Batch time: 0.1962
Phase: validation Epoch: 2/3 Iter: 4/39 Batch time: 0.2042
Phase: validation Epoch: 2/3 Iter: 5/39 Batch time: 0.2185
Phase: validation Epoch: 2/3 Iter: 6/39 Batch time: 0.1915
Phase: validation Epoch: 2/3 Iter: 7/39 Batch time: 0.1935
Phase: validation Epoch: 2/3 Iter: 8/39 Batch time: 0.2048
Phase: validation Epoch: 2/3 Iter: 9/39 Batch time: 0.1962
Phase: validation Epoch: 2/3 Iter: 10/39 Batch time: 0.1921
Phase: validation Epoch: 2/3 Iter: 11/39 Batch time: 0.2091
Phase: validation Epoch: 2/3 Iter: 12/39 Batch time: 0.2023
Phase: validation Epoch: 2/3 Iter: 13/39 Batch time: 0.1997
Phase: validation Epoch: 2/3 Iter: 14/39 Batch time: 0.1909
Phase: validation Epoch: 2/3 Iter: 15/39 Batch time: 0.1952
Phase: validation Epoch: 2/3 Iter: 16/39 Batch time: 0.1941
Phase: validation Epoch: 2/3 Iter: 17/39 Batch time: 0.1946
Phase: validation Epoch: 2/3 Iter: 18/39 Batch time: 0.2006
Phase: validation Epoch: 2/3 Iter: 19/39 Batch time: 0.1961
Phase: validation Epoch: 2/3 Iter: 20/39 Batch time: 0.2011
Phase: validation Epoch: 2/3 Iter: 21/39 Batch time: 0.1942
Phase: validation Epoch: 2/3 Iter: 22/39 Batch time: 0.1994
Phase: validation Epoch: 2/3 Iter: 23/39 Batch time: 0.1959
Phase: validation Epoch: 2/3 Iter: 24/39 Batch time: 0.1962
Phase: validation Epoch: 2/3 Iter: 25/39 Batch time: 0.1970
Phase: validation Epoch: 2/3 Iter: 26/39 Batch time: 0.2018
Phase: validation Epoch: 2/3 Iter: 27/39 Batch time: 0.1954
Phase: validation Epoch: 2/3 Iter: 28/39 Batch time: 0.1960
Phase: validation Epoch: 2/3 Iter: 29/39 Batch time: 0.2141
Phase: validation Epoch: 2/3 Iter: 30/39 Batch time: 0.1952
Phase: validation Epoch: 2/3 Iter: 31/39 Batch time: 0.2020
Phase: validation Epoch: 2/3 Iter: 32/39 Batch time: 0.2138
Phase: validation Epoch: 2/3 Iter: 33/39 Batch time: 0.1961
Phase: validation Epoch: 2/3 Iter: 34/39 Batch time: 0.2224
Phase: validation Epoch: 2/3 Iter: 35/39 Batch time: 0.1981
Phase: validation Epoch: 2/3 Iter: 36/39 Batch time: 0.1995
Phase: validation Epoch: 2/3 Iter: 37/39 Batch time: 0.2061
Phase: validation Epoch: 2/3 Iter: 38/39 Batch time: 0.2009
Phase: validation Epoch: 2/3 Iter: 39/39 Batch time: 0.0799
Phase: validation   Epoch: 2/3 Loss: 0.5392 Acc: 0.8235
Phase: train Epoch: 3/3 Iter: 1/62 Batch time: 0.2542
Phase: train Epoch: 3/3 Iter: 2/62 Batch time: 0.2783
Phase: train Epoch: 3/3 Iter: 3/62 Batch time: 0.2685
Phase: train Epoch: 3/3 Iter: 4/62 Batch time: 0.2716
Phase: train Epoch: 3/3 Iter: 5/62 Batch time: 0.2699
Phase: train Epoch: 3/3 Iter: 6/62 Batch time: 0.2715
Phase: train Epoch: 3/3 Iter: 7/62 Batch time: 0.2889
Phase: train Epoch: 3/3 Iter: 8/62 Batch time: 0.2596
Phase: train Epoch: 3/3 Iter: 9/62 Batch time: 0.2728
Phase: train Epoch: 3/3 Iter: 10/62 Batch time: 0.2687
Phase: train Epoch: 3/3 Iter: 11/62 Batch time: 0.2714
Phase: train Epoch: 3/3 Iter: 12/62 Batch time: 0.2733
Phase: train Epoch: 3/3 Iter: 13/62 Batch time: 0.2652
Phase: train Epoch: 3/3 Iter: 14/62 Batch time: 0.2516
Phase: train Epoch: 3/3 Iter: 15/62 Batch time: 0.2493
Phase: train Epoch: 3/3 Iter: 16/62 Batch time: 0.2654
Phase: train Epoch: 3/3 Iter: 17/62 Batch time: 0.2681
Phase: train Epoch: 3/3 Iter: 18/62 Batch time: 0.2589
Phase: train Epoch: 3/3 Iter: 19/62 Batch time: 0.2821
Phase: train Epoch: 3/3 Iter: 20/62 Batch time: 0.2735
Phase: train Epoch: 3/3 Iter: 21/62 Batch time: 0.2660
Phase: train Epoch: 3/3 Iter: 22/62 Batch time: 0.2641
Phase: train Epoch: 3/3 Iter: 23/62 Batch time: 0.2675
Phase: train Epoch: 3/3 Iter: 24/62 Batch time: 0.2568
Phase: train Epoch: 3/3 Iter: 25/62 Batch time: 0.2515
Phase: train Epoch: 3/3 Iter: 26/62 Batch time: 0.2795
Phase: train Epoch: 3/3 Iter: 27/62 Batch time: 0.2814
Phase: train Epoch: 3/3 Iter: 28/62 Batch time: 0.2656
Phase: train Epoch: 3/3 Iter: 29/62 Batch time: 0.3084
Phase: train Epoch: 3/3 Iter: 30/62 Batch time: 0.2929
Phase: train Epoch: 3/3 Iter: 31/62 Batch time: 0.2876
Phase: train Epoch: 3/3 Iter: 32/62 Batch time: 0.2798
Phase: train Epoch: 3/3 Iter: 33/62 Batch time: 0.2796
Phase: train Epoch: 3/3 Iter: 34/62 Batch time: 0.2852
Phase: train Epoch: 3/3 Iter: 35/62 Batch time: 0.2667
Phase: train Epoch: 3/3 Iter: 36/62 Batch time: 0.2897
Phase: train Epoch: 3/3 Iter: 37/62 Batch time: 0.2708
Phase: train Epoch: 3/3 Iter: 38/62 Batch time: 0.2687
Phase: train Epoch: 3/3 Iter: 39/62 Batch time: 0.2854
Phase: train Epoch: 3/3 Iter: 40/62 Batch time: 0.2742
Phase: train Epoch: 3/3 Iter: 41/62 Batch time: 0.2577
Phase: train Epoch: 3/3 Iter: 42/62 Batch time: 0.2827
Phase: train Epoch: 3/3 Iter: 43/62 Batch time: 0.2932
Phase: train Epoch: 3/3 Iter: 44/62 Batch time: 0.2703
Phase: train Epoch: 3/3 Iter: 45/62 Batch time: 0.2881
Phase: train Epoch: 3/3 Iter: 46/62 Batch time: 0.2982
Phase: train Epoch: 3/3 Iter: 47/62 Batch time: 0.2777
Phase: train Epoch: 3/3 Iter: 48/62 Batch time: 0.2767
Phase: train Epoch: 3/3 Iter: 49/62 Batch time: 0.2743
Phase: train Epoch: 3/3 Iter: 50/62 Batch time: 0.2645
Phase: train Epoch: 3/3 Iter: 51/62 Batch time: 0.2789
Phase: train Epoch: 3/3 Iter: 52/62 Batch time: 0.3108
Phase: train Epoch: 3/3 Iter: 53/62 Batch time: 0.3008
Phase: train Epoch: 3/3 Iter: 54/62 Batch time: 0.2603
Phase: train Epoch: 3/3 Iter: 55/62 Batch time: 0.3116
Phase: train Epoch: 3/3 Iter: 56/62 Batch time: 0.3051
Phase: train Epoch: 3/3 Iter: 57/62 Batch time: 0.2905
Phase: train Epoch: 3/3 Iter: 58/62 Batch time: 0.3002
Phase: train Epoch: 3/3 Iter: 59/62 Batch time: 0.2769
Phase: train Epoch: 3/3 Iter: 60/62 Batch time: 0.2866
Phase: train Epoch: 3/3 Iter: 61/62 Batch time: 0.3131
Phase: train Epoch: 3/3 Loss: 0.5652 Acc: 0.7336
Phase: validation Epoch: 3/3 Iter: 1/39 Batch time: 0.2006
Phase: validation Epoch: 3/3 Iter: 2/39 Batch time: 0.2162
Phase: validation Epoch: 3/3 Iter: 3/39 Batch time: 0.1892
Phase: validation Epoch: 3/3 Iter: 4/39 Batch time: 0.2002
Phase: validation Epoch: 3/3 Iter: 5/39 Batch time: 0.1881
Phase: validation Epoch: 3/3 Iter: 6/39 Batch time: 0.1928
Phase: validation Epoch: 3/3 Iter: 7/39 Batch time: 0.1947
Phase: validation Epoch: 3/3 Iter: 8/39 Batch time: 0.2136
Phase: validation Epoch: 3/3 Iter: 9/39 Batch time: 0.1886
Phase: validation Epoch: 3/3 Iter: 10/39 Batch time: 0.2175
Phase: validation Epoch: 3/3 Iter: 11/39 Batch time: 0.1911
Phase: validation Epoch: 3/3 Iter: 12/39 Batch time: 0.2005
Phase: validation Epoch: 3/3 Iter: 13/39 Batch time: 0.2067
Phase: validation Epoch: 3/3 Iter: 14/39 Batch time: 0.1952
Phase: validation Epoch: 3/3 Iter: 15/39 Batch time: 0.2089
Phase: validation Epoch: 3/3 Iter: 16/39 Batch time: 0.1924
Phase: validation Epoch: 3/3 Iter: 17/39 Batch time: 0.2036
Phase: validation Epoch: 3/3 Iter: 18/39 Batch time: 0.1991
Phase: validation Epoch: 3/3 Iter: 19/39 Batch time: 0.1912
Phase: validation Epoch: 3/3 Iter: 20/39 Batch time: 0.1995
Phase: validation Epoch: 3/3 Iter: 21/39 Batch time: 0.1971
Phase: validation Epoch: 3/3 Iter: 22/39 Batch time: 0.2082
Phase: validation Epoch: 3/3 Iter: 23/39 Batch time: 0.1937
Phase: validation Epoch: 3/3 Iter: 24/39 Batch time: 0.1952
Phase: validation Epoch: 3/3 Iter: 25/39 Batch time: 0.2076
Phase: validation Epoch: 3/3 Iter: 26/39 Batch time: 0.2023
Phase: validation Epoch: 3/3 Iter: 27/39 Batch time: 0.1970
Phase: validation Epoch: 3/3 Iter: 28/39 Batch time: 0.2038
Phase: validation Epoch: 3/3 Iter: 29/39 Batch time: 0.1862
Phase: validation Epoch: 3/3 Iter: 30/39 Batch time: 0.1962
Phase: validation Epoch: 3/3 Iter: 31/39 Batch time: 0.2049
Phase: validation Epoch: 3/3 Iter: 32/39 Batch time: 0.1902
Phase: validation Epoch: 3/3 Iter: 33/39 Batch time: 0.2043
Phase: validation Epoch: 3/3 Iter: 34/39 Batch time: 0.1872
Phase: validation Epoch: 3/3 Iter: 35/39 Batch time: 0.1852
Phase: validation Epoch: 3/3 Iter: 36/39 Batch time: 0.1858
Phase: validation Epoch: 3/3 Iter: 37/39 Batch time: 0.1855
Phase: validation Epoch: 3/3 Iter: 38/39 Batch time: 0.1861
Phase: validation Epoch: 3/3 Iter: 39/39 Batch time: 0.0675
Phase: validation   Epoch: 3/3 Loss: 0.4484 Acc: 0.8497
Training completed in 1m 22s
Best test loss: 0.4484 | Best test accuracy: 0.8497
 </code>
 </pre>
 </details>

---

## 6. tutorial_qgrnn.html <a name="demo5"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_qgrnn.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
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
Weights at Step 0: [-0.22603613  0.43887001  0.85859236  0.69735898  0.09417125 -0.02437147]
Bias at Step 0: [-0.23884748 -0.21392016  0.12809368  0.45037793]
Cost at Step 5: -0.9974589524428032
Weights at Step 5: [-0.75106068  1.078707    0.83766935  1.9741555   0.04982793 -0.06747815]
Bias at Step 5: [-0.50836435 -1.32708118  1.57468372  0.11442806]
Cost at Step 10: -0.9971878268304797
Weights at Step 10: [ 0.01577799  0.48771566  0.68379977  1.75747002 -0.21948418 -0.00484698]
Bias at Step 10: [ 0.22007905 -0.90282076  1.58989008 -0.1051542 ]
Cost at Step 15: -0.9981871533122743
Weights at Step 15: [-0.06744249  0.65720464  1.31471457  1.47430241  0.05813038 -0.41315658]
Bias at Step 15: [-0.29045223 -0.67045595  1.19395446  0.24677711]
Cost at Step 20: -0.9995130692146865
Weights at Step 20: [-0.16225009  0.66208813  1.17758061  1.48254064 -0.35468207 -0.01648733]
Bias at Step 20: [-0.56832881 -0.87721581  0.86890622 -0.27734217]
Cost at Step 25: -0.999818156006956
Weights at Step 25: [ 0.030689    0.40363412  1.32430282  1.80692972 -0.14761078 -0.27452275]
Bias at Step 25: [-0.33695681 -1.28646051  1.00509121 -0.19672993]
Cost at Step 30: -0.9997713453918132
Weights at Step 30: [ 0.22014178  0.38063046  1.34934589  1.82854127 -0.201276   -0.35610913]
Bias at Step 30: [-0.40515586 -1.19466624  1.13933672 -0.31753371]
Cost at Step 35: -0.999785863213516
Weights at Step 35: [ 0.22310889  0.50896099  1.36029033  1.73588161 -0.30076632 -0.35201799]
Bias at Step 35: [-0.73605504 -1.06150682  1.07911402 -0.49331792]
Cost at Step 40: -0.9998587245201028
Weights at Step 40: [ 0.32580831  0.34293483  1.38813471  1.76261455 -0.16880133 -0.49831078]
Bias at Step 40: [-0.74534508 -1.20379495  0.91916721 -0.49099848]
Cost at Step 45: -0.9998796449154709
Weights at Step 45: [ 0.37151473  0.26329833  1.29149206  1.84754531 -0.16287055 -0.51793342]
Bias at Step 45: [-0.81082195 -1.35860008  0.88788716 -0.63818331]
Cost at Step 50: -0.9999381279674818
Weights at Step 50: [ 0.36839385  0.36200243  1.3398256   1.82138962 -0.10863003 -0.63499473]
Bias at Step 50: [-1.04909977 -1.27581779  0.93902928 -0.67286639]
Cost at Step 55: -0.9999252881391636
Weights at Step 55: [ 0.52717503  0.22669791  1.27477846  1.75733597 -0.1195999  -0.66147384]
Bias at Step 55: [-1.02458911 -1.19668936  0.89787258 -0.76851797]
Cost at Step 60: -0.9999298586216632
Weights at Step 60: [ 0.46426344  0.2377941   1.30579533  1.86309089 -0.05164041 -0.72495523]
Bias at Step 60: [-1.15189323 -1.3565901   0.92299357 -0.81716179]
Cost at Step 65: -0.9999633163685123
Weights at Step 65: [ 0.54814165  0.15321249  1.29421602  1.79468038 -0.06551864 -0.73442872]
Bias at Step 65: [-1.18093812 -1.2850757   0.8715214  -0.89498456]
Cost at Step 70: -0.9999692974112367
Weights at Step 70: [ 0.54968362  0.15363051  1.32636004  1.81799545 -0.03795287 -0.78176424]
Bias at Step 70: [-1.26004878 -1.29186174  0.93275685 -0.93151559]
Cost at Step 75: -0.999983941803599
Weights at Step 75: [ 0.56926778  0.09899053  1.32784104  1.83800055 -0.02258999 -0.80100631]
Bias at Step 75: [-1.27844754 -1.32707392  0.95121289 -0.97776518]
Cost at Step 80: -0.9999900754964173
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
Cost at Step 100: -0.9999862374755373
Weights at Step 100: [ 0.59133649  0.00671959  1.33149619  1.80654252  0.00739735 -0.85271751]
Bias at Step 100: [-1.3928395  -1.32434117  0.99986843 -1.0746036 ]
---------------------------------------------
Cost at Step 105: -0.9999860417314502
Weights at Step 105: [ 5.98294601e-01 -1.63872117e-03  1.32435903e+00  1.79606830e+00
 -1.50600209e-03 -8.46854780e-01]
Bias at Step 105: [-1.39810406 -1.31716394  1.00632966 -1.08694727]
---------------------------------------------
Cost at Step 110: -0.9999889448376537
Weights at Step 110: [ 0.58790075  0.00252021  1.34070893  1.79969067  0.01353978 -0.86073241]
Bias at Step 110: [-1.41237809 -1.33237829  1.01344356 -1.07306401]
---------------------------------------------
Cost at Step 115: -0.9999907527375035
Weights at Step 115: [ 0.59511523 -0.00485992  1.33269226  1.79443191  0.00548165 -0.85246429]
Bias at Step 115: [-1.40713813 -1.33369881  1.02023634 -1.07811166]
---------------------------------------------
Cost at Step 120: -0.9999894991115471
Weights at Step 120: [ 5.96065192e-01 -3.92737036e-03  1.33288381e+00  1.78858855e+00
  1.75644333e-03 -8.48477853e-01]
Bias at Step 120: [-1.40998319 -1.33422301  1.02362295 -1.07680692]
---------------------------------------------
Cost at Step 125: -0.9999921833903868
Weights at Step 125: [ 5.91985767e-01 -1.57241251e-03  1.33882177e+00  1.79191152e+00
  8.89983556e-03 -8.53306759e-01]
Bias at Step 125: [-1.41164517 -1.34685868  1.0288417  -1.06603474]
Cost at Step 130: -0.9999895629847485
Weights at Step 130: [ 5.93396935e-01 -6.09903975e-04  1.32769014e+00  1.78340627e+00
  3.20989912e-03 -8.45936030e-01]
Bias at Step 130: [-1.41153989 -1.34372306  1.0306921  -1.06641934]
---------------------------------------------
Cost at Step 135: -0.9999893671898541
Weights at Step 135: [ 5.93933082e-01 -1.84879055e-04  1.32433785e+00  1.77929886e+00
  4.38714188e-03 -8.45503125e-01]
Bias at Step 135: [-1.4098866  -1.34497204  1.03415862 -1.06007092]
---------------------------------------------
Cost at Step 140: -0.9999909995915327
Weights at Step 140: [ 0.5908483   0.00239647  1.33016899  1.78168215  0.00438378 -0.84478361]
Bias at Step 140: [-1.41145547 -1.35249113  1.0371159  -1.05630093]
---------------------------------------------
Cost at Step 145: -0.9999882781659925
Weights at Step 145: [ 5.95379262e-01 -3.85481762e-04  1.33209614e+00  1.77901613e+00
 -1.07423798e-03 -8.40357066e-01]
Bias at Step 145: [-1.40745547 -1.35117282  1.03764696 -1.05779208]
---------------------------------------------
Cost at Step 150: -0.9999882605157342
Weights at Step 150: [ 0.59753863 -0.00178702  1.33577445  1.78058366 -0.00414765 -0.83825161]
Bias at Step 150: [-1.40514789 -1.35375725  1.03826782 -1.05953418]
---------------------------------------------
Cost at Step 155: -0.9999873092858326
Weights at Step 155: [ 5.96677467e-01  6.17989620e-04  1.34466485e+00  1.78402267e+00
 -2.65741532e-03 -8.41450495e-01]
Bias at Step 155: [-1.40789298 -1.356471    1.0376034  -1.05877886]
Cost at Step 160: -0.9999872637771746
Weights at Step 160: [ 0.60194839 -0.00191478  1.35123357  1.7884694  -0.00603744 -0.84142449]
Bias at Step 160: [-1.40510396 -1.35668892  1.03451132 -1.06543631]
Cost at Step 165: -0.9999874624869766
Weights at Step 165: [ 0.6030402  -0.00193504  1.34611742  1.78890714 -0.00577061 -0.84238872]
Bias at Step 165: [-1.40613555 -1.35445251  1.02812253 -1.07090041]
Cost at Step 170: -0.9999886747321687
Weights at Step 170: [ 0.5975333   0.00282116  1.32780927  1.78277225  0.00466827 -0.84919749]
Bias at Step 170: [-1.41126245 -1.34857858  1.02684163 -1.0645541 ]
Cost at Step 175: -0.9999875562507218
Weights at Step 175: [ 0.59389571  0.00440434  1.32168971  1.78088367  0.00772899 -0.85019789]
Bias at Step 175: [-1.41279703 -1.34685714  1.02847119 -1.06186318]
Cost at Step 180: -0.9999914836764835
Weights at Step 180: [ 5.95476507e-01  1.04646713e-03  1.33253375e+00  1.78379408e+00
  1.79045580e-03 -8.45539923e-01]
Bias at Step 180: [-1.40924345 -1.34843997  1.03054894 -1.06465758]
---------------------------------------------
Cost at Step 185: -0.9999877384049591
Weights at Step 185: [ 0.60021817 -0.00344807  1.34837992  1.78919267 -0.00485391 -0.84211998]
Bias at Step 185: [-1.40406354 -1.35059434  1.03159602 -1.06906187]
---------------------------------------------
Cost at Step 190: -0.9999943811280624
Weights at Step 190: [ 0.60332647 -0.00502377  1.35580211  1.7937833  -0.00870162 -0.84089377]
Bias at Step 190: [-1.40248031 -1.35189812  1.02840358 -1.07489377]
---------------------------------------------
Cost at Step 195: -0.999989953283451
Weights at Step 195: [ 6.01661213e-01 -1.40258189e-03  1.34845381e+00  1.79177947e+00
 -2.19330435e-03 -8.46882171e-01]
Bias at Step 195: [-1.40624231 -1.34827409  1.02488298 -1.07228167]
---------------------------------------------
Cost at Step 200: -0.9999880990041629
Weights at Step 200: [ 0.59781588  0.00262681  1.33622645  1.7903236   0.00491206 -0.85187394]
Bias at Step 200: [-1.41001483 -1.3468901   1.02276916 -1.06950677]
---------------------------------------------
Cost at Step 205: -0.9999882388738665
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
Weights at Step 220: [ 5.94793970e-01 -8.57403941e-04  1.33609793e+00  1.78453102e+00
 -2.17937735e-03 -8.40640138e-01]
Bias at Step 220: [-1.40459538 -1.34840612  1.03328985 -1.06323676]
Cost at Step 225: -0.9999895237582312
Weights at Step 225: [ 0.59740767 -0.00202435  1.33963888  1.78462403 -0.00566848 -0.83842794]
Bias at Step 225: [-1.40352872 -1.34909283  1.03367034 -1.0646202 ]
Cost at Step 230: -0.999992221715183
Weights at Step 230: [ 0.59830589 -0.00184753  1.33406652  1.78177535 -0.00330205 -0.84015047]
Bias at Step 230: [-1.40427031 -1.34839966  1.03168677 -1.06292291]
Cost at Step 235: -0.999991269286791
Weights at Step 235: [ 5.95836686e-01  8.57240159e-04  1.32786532e+00  1.78035155e+00
  2.56153005e-03 -8.44403818e-01]
Bias at Step 235: [-1.40725255 -1.34955664  1.03260437 -1.05840103]
---------------------------------------------
Cost at Step 240: -0.9999899357699891
Weights at Step 240: [ 0.59351612  0.00209534  1.32485259  1.77824259  0.00444528 -0.8447466 ]
Bias at Step 240: [-1.40903945 -1.35042747  1.03478287 -1.05550792]
---------------------------------------------
Cost at Step 245: -0.9999889869569205
Weights at Step 245: [ 5.95395532e-01  1.23721713e-03  1.33593185e+00  1.78146341e+00
 -1.83303336e-03 -8.40987115e-01]
Bias at Step 245: [-1.40856761 -1.35225776  1.03748333 -1.05978839]
---------------------------------------------
Cost at Step 250: -0.999988129317049
Weights at Step 250: [ 5.95495047e-01 -2.18886873e-04  1.33760142e+00  1.78252108e+00
 -1.04388862e-03 -8.41650674e-01]
Bias at Step 250: [-1.40788422 -1.35473503  1.03617744 -1.05980154]
---------------------------------------------
Cost at Step 255: -0.9999892613809334
Weights at Step 255: [ 5.95095996e-01  1.05345945e-03  1.33120954e+00  1.78110699e+00
  9.09621038e-04 -8.43176791e-01]
Bias at Step 255: [-1.40881827 -1.35271576  1.03732158 -1.05936317]
Cost at Step 260: -0.9999911696516829
Weights at Step 260: [ 0.59215329  0.00249312  1.32162048  1.77469111  0.00411753 -0.84385295]
Bias at Step 260: [-1.41164782 -1.34921326  1.03505213 -1.05565234]
Cost at Step 265: -0.9999897602169294
Weights at Step 265: [ 5.94219509e-01  1.15801805e-03  1.32981272e+00  1.77875998e+00
  2.00797690e-03 -8.43338459e-01]
Bias at Step 265: [-1.40879199 -1.35161558  1.04052305 -1.05548315]
---------------------------------------------
Cost at Step 270: -0.9999877865048056
Weights at Step 270: [ 0.5967199  -0.00228699  1.34015592  1.78207987 -0.00207472 -0.8407702 ]
Bias at Step 270: [-1.40654726 -1.35550984  1.03685756 -1.05892969]
---------------------------------------------
Cost at Step 275: -0.9999913148930895
Weights at Step 275: [ 5.95115179e-01  6.67458889e-04  1.33088691e+00  1.77908492e+00
  2.17950090e-03 -8.43784743e-01]
Bias at Step 275: [-1.40907272 -1.35232601  1.03787329 -1.05577319]
---------------------------------------------
Cost at Step 280: -0.9999912773773976
Weights at Step 280: [ 5.94534323e-01  1.49825127e-03  1.33221200e+00  1.78129607e+00
  1.80930063e-03 -8.43864609e-01]
Bias at Step 280: [-1.41032893 -1.35393432  1.03609549 -1.0579392 ]
---------------------------------------------
Cost at Step 285: -0.9999871922879656
Weights at Step 285: [ 5.96437497e-01  3.12450031e-04  1.33601315e+00  1.78353853e+00
 -7.60576311e-04 -8.42698914e-01]
Bias at Step 285: [-1.40896079 -1.3539148   1.03483345 -1.06189856]
---------------------------------------------
Cost at Step 290: -0.9999918144420524
Weights at Step 290: [ 0.5928268   0.00288728  1.32568915  1.77748739  0.00484885 -0.84553166]
Bias at Step 290: [-1.41115856 -1.34928842  1.03616776 -1.0560555 ]
 </code>
 </pre>
 </details>

---

## 7. tutorial_jax_transformations.html <a name="demo6"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_jax_transformations.html):

```
No jit time: 0.0091 seconds
First run time: 0.0702 seconds
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_jax_transformations.html):

```
No jit time: 0.0079 seconds
First run time: 0.0640 seconds
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
Step 7: cost = -5.983750000000001 shots used = 56000
Step 8: cost = -6.50375 shots used = 64000
Step 9: cost = -6.7749999999999995 shots used = 72000
Step 10: cost = -6.90625 shots used = 80000
Step 11: cost = -7.165 shots used = 88000
Step 12: cost = -7.39375 shots used = 96000
Step 13: cost = -7.52 shots used = 104000
Step 14: cost = -7.547499999999999 shots used = 112000
Step 15: cost = -7.5125 shots used = 120000
Step 16: cost = -7.155 shots used = 128000
Step 17: cost = -7.535 shots used = 136000
Step 18: cost = -7.35 shots used = 144000
Step 19: cost = -7.25625 shots used = 152000
Step 20: cost = -7.39875 shots used = 160000
Step 21: cost = -7.50875 shots used = 168000
Step 22: cost = -7.50375 shots used = 176000
Step 23: cost = -7.55625 shots used = 184000
Step 24: cost = -7.3374999999999995 shots used = 192000
Step 25: cost = -7.7524999999999995 shots used = 200000
Step 26: cost = -7.65 shots used = 208000
Step 27: cost = -7.7524999999999995 shots used = 216000
Step 28: cost = -7.6375 shots used = 224000
Step 29: cost = -7.55125 shots used = 232000
Step 30: cost = -7.7175 shots used = 240000
Step 31: cost = -7.65625 shots used = 248000
Step 32: cost = -7.998749999999999 shots used = 256000
Step 33: cost = -7.67375 shots used = 264000
Step 34: cost = -7.296250000000001 shots used = 272000
Step 35: cost = -7.5874999999999995 shots used = 280000
Step 36: cost = -7.748749999999999 shots used = 288000
Step 37: cost = -7.71375 shots used = 296000
Step 38: cost = -7.735 shots used = 304000
Step 39: cost = -7.893750000000001 shots used = 312000
Step 40: cost = -7.57 shots used = 320000
Step 41: cost = -7.7787500000000005 shots used = 328000
Step 42: cost = -7.83 shots used = 336000
Step 43: cost = -7.8475 shots used = 344000
Step 44: cost = -7.82875 shots used = 352000
Step 45: cost = -7.819999999999999 shots used = 360000
Step 46: cost = -7.836250000000001 shots used = 368000
Step 47: cost = -7.786249999999999 shots used = 376000
Step 48: cost = -7.862500000000001 shots used = 384000
Step 49: cost = -7.951250000000001 shots used = 392000
Step 50: cost = -7.958749999999999 shots used = 400000
Step 51: cost = -8.20375 shots used = 408000
Step 52: cost = -7.692500000000001 shots used = 416000
Step 53: cost = -7.8375 shots used = 424000
Step 54: cost = -7.6312500000000005 shots used = 432000
Step 55: cost = -7.828749999999999 shots used = 440000
Step 56: cost = -7.862500000000001 shots used = 448000
Step 57: cost = -8.09125 shots used = 456000
Step 58: cost = -7.70625 shots used = 464000
Step 59: cost = -7.8237499999999995 shots used = 472000
Step 60: cost = -8.03625 shots used = 480000
Step 61: cost = -7.972499999999999 shots used = 488000
Step 62: cost = -7.81 shots used = 496000
Step 63: cost = -7.86375 shots used = 504000
Step 64: cost = -8.045 shots used = 512000
Step 65: cost = -7.80375 shots used = 520000
Step 66: cost = -7.9037500000000005 shots used = 528000
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
Step 77: cost = -7.720000000000001 shots used = 616000
Step 78: cost = -7.709999999999999 shots used = 624000
Step 79: cost = -7.7125 shots used = 632000
Step 80: cost = -7.7325 shots used = 640000
Step 81: cost = -7.93125 shots used = 648000
Step 82: cost = -7.785 shots used = 656000
Step 83: cost = -7.7625 shots used = 664000
Step 84: cost = -7.6937500000000005 shots used = 672000
Step 85: cost = -8.0525 shots used = 680000
Step 86: cost = -8.06125 shots used = 688000
Step 87: cost = -7.8812500000000005 shots used = 696000
Step 88: cost = -7.97375 shots used = 704000
Step 89: cost = -7.89875 shots used = 712000
Step 90: cost = -7.88 shots used = 720000
Step 91: cost = -7.99875 shots used = 728000
Step 92: cost = -7.86375 shots used = 736000
Step 93: cost = -7.911250000000001 shots used = 744000
Step 94: cost = -7.842499999999999 shots used = 752000
Step 95: cost = -8.00875 shots used = 760000
Step 96: cost = -7.859999999999999 shots used = 768000
Step 97: cost = -7.96375 shots used = 776000
Step 98: cost = -7.772499999999999 shots used = 784000
Step 99: cost = -7.9475 shots used = 792000
Step 0: cost = -4.820380999693457, shots_used = 240
Step 1: cost = -4.937944875992973, shots_used = 336
Step 2: cost = -5.477016391676937, shots_used = 456
Step 3: cost = -5.878302378912975, shots_used = 624
Step 4: cost = -5.740983235298872, shots_used = 768
Step 5: cost = -5.86849950717445, shots_used = 960
Step 6: cost = -5.5725976187549255, shots_used = 1200
Step 7: cost = -6.038511127131682, shots_used = 1512
Step 8: cost = -7.187839850746336, shots_used = 1944
Step 9: cost = -7.244742040043007, shots_used = 2472
Step 10: cost = -6.955119947427081, shots_used = 3144
Step 11: cost = -7.324280331788421, shots_used = 4176
Step 12: cost = -7.305099179560385, shots_used = 5400
Step 13: cost = -7.24133900327795, shots_used = 6528
Step 14: cost = -7.529075219254999, shots_used = 7632
Step 15: cost = -7.095276433317588, shots_used = 9048
Step 16: cost = -7.607336707435147, shots_used = 10584
Step 17: cost = -7.738585612241668, shots_used = 12624
Step 18: cost = -7.792700472104301, shots_used = 15288
Step 19: cost = -7.802640154411867, shots_used = 18048
Step 20: cost = -7.765196284526809, shots_used = 20976
Step 21: cost = -7.77937999521944, shots_used = 24264
Step 22: cost = -7.851415989752514, shots_used = 27576
Step 23: cost = -7.829808564028127, shots_used = 31728
Step 24: cost = -7.813743545091879, shots_used = 36264
Step 25: cost = -7.790635224170294, shots_used = 42024
Step 26: cost = -7.887177094048003, shots_used = 48096
Step 27: cost = -7.877872495361855, shots_used = 54744
Step 28: cost = -7.875679215329842, shots_used = 62448
Step 29: cost = -7.8559020553306125, shots_used = 71352
Step 30: cost = -7.884434864101767, shots_used = 80832
Step 31: cost = -7.891637364212708, shots_used = 90336
Step 32: cost = -7.854723497132758, shots_used = 100680
Step 33: cost = -7.860436984930216, shots_used = 111984
Step 34: cost = -7.885736157831495, shots_used = 123312
Step 35: cost = -7.851714069232078, shots_used = 136632
Step 36: cost = -7.8648485293671975, shots_used = 150744
Step 37: cost = -7.875581235645933, shots_used = 166152
Step 38: cost = -7.809686333605921, shots_used = 183240
Step 39: cost = -7.884245542198442, shots_used = 202752
Step 40: cost = -7.889534749964765, shots_used = 221448
Step 41: cost = -7.894948220964508, shots_used = 240192
Step 42: cost = -7.897425239891586, shots_used = 262368
Step 43: cost = -7.879902900295498, shots_used = 285024
Step 44: cost = -7.873122596846604, shots_used = 307704
Step 45: cost = -7.889285563108286, shots_used = 331272
Step 46: cost = -7.893112373227317, shots_used = 357552
Step 47: cost = -7.878308602523566, shots_used = 385320
Step 48: cost = -7.899236702757056, shots_used = 416208
Step 49: cost = -7.894296334408784, shots_used = 446808
Step 50: cost = -7.89049443519431, shots_used = 479976
Step 51: cost = -7.89229873961093, shots_used = 512928
Step 52: cost = -7.893708744149611, shots_used = 547176
Step 53: cost = -7.898823452831049, shots_used = 582960
Step 54: cost = -7.898889229118191, shots_used = 621072
Step 55: cost = -7.881052733782681, shots_used = 661488
Step 56: cost = -7.8913643791351875, shots_used = 703032
Step 57: cost = -7.89742567457411, shots_used = 747480
Step 58: cost = -7.893221963817919, shots_used = 794808
Step 59: cost = -7.896438792204623, shots_used = 842570
Step 0: cost = -2.12150804866895 shots_used = 2400
Step 1: cost = -3.4462874411421858 shots_used = 4800
Step 2: cost = -4.533723704599176 shots_used = 7200
Step 3: cost = -5.360324618255417 shots_used = 9600
Step 4: cost = -6.010958804727693 shots_used = 12000
Step 5: cost = -6.545008232375085 shots_used = 14400
Step 6: cost = -6.960941130446836 shots_used = 16800
Step 7: cost = -7.248308512586622 shots_used = 19200
Step 8: cost = -7.398432638481056 shots_used = 21600
Step 9: cost = -7.43238804878266 shots_used = 24000
Step 10: cost = -7.374281342889537 shots_used = 26400
Step 11: cost = -7.2878455754894835 shots_used = 28800
Step 12: cost = -7.211212391636272 shots_used = 31200
Step 13: cost = -7.168136331225395 shots_used = 33600
Step 14: cost = -7.17198903781606 shots_used = 36000
Step 15: cost = -7.2153317942728545 shots_used = 38400
Step 16: cost = -7.278024044019375 shots_used = 40800
Step 17: cost = -7.361449931178438 shots_used = 43200
Step 18: cost = -7.442410269187308 shots_used = 45600
Step 19: cost = -7.511315452968971 shots_used = 48000
Step 20: cost = -7.5608730559384805 shots_used = 50400
Step 21: cost = -7.591626968703287 shots_used = 52800
Step 22: cost = -7.608322534779552 shots_used = 55200
Step 23: cost = -7.607043067486307 shots_used = 57600
Step 24: cost = -7.594076978496339 shots_used = 60000
Step 25: cost = -7.579153179578798 shots_used = 62400
Step 26: cost = -7.572266109391965 shots_used = 64800
Step 27: cost = -7.5684397464408555 shots_used = 67200
Step 28: cost = -7.581935968178153 shots_used = 69600
Step 29: cost = -7.610907153836915 shots_used = 72000
Step 30: cost = -7.651198088153217 shots_used = 74400
Step 31: cost = -7.697526604943234 shots_used = 76800
Step 32: cost = -7.746903397102402 shots_used = 79200
Step 33: cost = -7.787293318189747 shots_used = 81600
Step 34: cost = -7.8208748274212425 shots_used = 84000
Step 35: cost = -7.840729913365395 shots_used = 86400
Step 36: cost = -7.858055514435193 shots_used = 88800
Step 37: cost = -7.868752507617257 shots_used = 91200
Step 38: cost = -7.8777750014035055 shots_used = 93600
Step 39: cost = -7.884473822847106 shots_used = 96000
Step 40: cost = -7.887248861002906 shots_used = 98400
Step 41: cost = -7.885930519767967 shots_used = 100800
Step 42: cost = -7.881002890765942 shots_used = 103200
Step 43: cost = -7.875074719805523 shots_used = 105600
Step 44: cost = -7.866129786750201 shots_used = 108000
Step 45: cost = -7.850581153251577 shots_used = 110400
Step 46: cost = -7.843337695237988 shots_used = 112800
Step 47: cost = -7.845453624375395 shots_used = 115200
Step 48: cost = -7.853444576995697 shots_used = 117600
Step 49: cost = -7.858018368114792 shots_used = 120000
Step 50: cost = -7.858043805938921 shots_used = 122400
Step 51: cost = -7.855559046474576 shots_used = 124800
Step 52: cost = -7.850626102015816 shots_used = 127200
Step 53: cost = -7.848969631273946 shots_used = 129600
Step 54: cost = -7.852176020039671 shots_used = 132000
Step 55: cost = -7.86142787416302 shots_used = 134400
Step 56: cost = -7.868403253322003 shots_used = 136800
Step 57: cost = -7.87624175909495 shots_used = 139200
Step 58: cost = -7.88046548748952 shots_used = 141600
Step 59: cost = -7.880026154757118 shots_used = 144000
Step 60: cost = -7.877251725772389 shots_used = 146400
Step 61: cost = -7.870289689150821 shots_used = 148800
Step 62: cost = -7.864543163588924 shots_used = 151200
Step 63: cost = -7.862715331323386 shots_used = 153600
Step 64: cost = -7.861607002909473 shots_used = 156000
Step 65: cost = -7.866735539612959 shots_used = 158400
Step 66: cost = -7.867386735902061 shots_used = 160800
Step 67: cost = -7.867196452121146 shots_used = 163200
Step 68: cost = -7.869567827264065 shots_used = 165600
Step 69: cost = -7.869618725213597 shots_used = 168000
Step 70: cost = -7.866578292195068 shots_used = 170400
Step 71: cost = -7.857706098037315 shots_used = 172800
Step 72: cost = -7.855866345209225 shots_used = 175200
Step 73: cost = -7.858187887358694 shots_used = 177600
Step 74: cost = -7.861466111241278 shots_used = 180000
Step 75: cost = -7.86482587723914 shots_used = 182400
Step 76: cost = -7.863570824947597 shots_used = 184800
Step 77: cost = -7.863497614169012 shots_used = 187200
Step 78: cost = -7.860326845355643 shots_used = 189600
Step 79: cost = -7.8553010865516475 shots_used = 192000
Step 80: cost = -7.85589006991893 shots_used = 194400
Step 81: cost = -7.857406216142363 shots_used = 196800
Step 82: cost = -7.863450868072559 shots_used = 199200
Step 83: cost = -7.8700581426790865 shots_used = 201600
Step 84: cost = -7.8777699578190585 shots_used = 204000
Step 85: cost = -7.883842326350928 shots_used = 206400
Step 86: cost = -7.882633952688104 shots_used = 208800
Step 87: cost = -7.879224942149158 shots_used = 211200
Step 88: cost = -7.872184015334468 shots_used = 213600
Step 89: cost = -7.864992896022883 shots_used = 216000
Step 90: cost = -7.8606069766669036 shots_used = 218400
Step 91: cost = -7.85981012759474 shots_used = 220800
Step 92: cost = -7.8630656606853595 shots_used = 223200
Step 93: cost = -7.868586359942559 shots_used = 225600
Step 94: cost = -7.874757156105922 shots_used = 228000
Step 95: cost = -7.8798938081862495 shots_used = 230400
Step 96: cost = -7.8833173990872165 shots_used = 232800
Step 97: cost = -7.883690480348113 shots_used = 235200
Step 98: cost = -7.882061100381096 shots_used = 237600
Step 99: cost = -7.878907675503163 shots_used = 240000
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
(-46.463906788688874+0j) [] +
(-0.014583648907612752+0j) [X0 X1 Y2 Y3] +
(-3.5707613289482844e-07+0j) [X0 X1 Y2 Z3 Z4 Y5] +
(-0.00565262097801735+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7] +
(-0.008826368514209856+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.7924939575063031e-06+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.5707613289482844e-07+0j) [X0 X1 X3 X4] +
(-0.00565262097801735+0j) [X0 X1 X3 Z4 Z5 X6] +
(-0.008826368514209856+0j) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.7924939575063031e-06+0j) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.0027458364701868137+0j) [X0 X1 Y4 Y5] +
(-2.44732312870767e-07+0j) [X0 X1 Y4 Z5 Z6 Y7] +
(-7.867765103359642e-07+0j) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.003804066171728543+0j) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.44732312870767e-07+0j) [X0 X1 X5 X6] +
(-7.867765103359642e-07+0j) [X0 X1 X5 Z6 Z7 Z8 Z9 X10] +
(-0.003804066171728543+0j) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.00688819435297055+0j) [X0 X1 Y6 Y7] +
(-7.735036880588932e-05+0j) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11] +
(1.7035783553438497e-07+0j) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.735036880588933e-05+0j) [X0 X1 X7 Z8 Z9 X10] +
(1.7035783553438497e-07+0j) [X0 X1 X7 Z8 Z9 Z10 Z11 X12] +
(-0.006509361201177239+0j) [X0 X1 Y8 Y9] +
(-0.00773142525077527+0j) [X0 X1 Y10 Y11] +
(5.627851910864507e-07+0j) [X0 X1 Y10 Z11 Z12 Y13] +
(5.627851910864507e-07+0j) [X0 X1 X11 X12] +
(-0.00528377648840296+0j) [X0 X1 Y12 Y13] +
(0.014583648907612752+0j) [X0 Y1 Y2 X3] +
(3.5707613289482844e-07+0j) [X0 Y1 Y2 Z3 Z4 X5] +
(0.00565262097801735+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7] +
(0.008826368514209856+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-1.7924939575063031e-06+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.5707613289482844e-07+0j) [X0 Y1 Y3 X4] +
(-0.00565262097801735+0j) [X0 Y1 Y3 Z4 Z5 X6] +
(-0.008826368514209856+0j) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.7924939575063031e-06+0j) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0027458364701868137+0j) [X0 Y1 Y4 X5] +
(2.44732312870767e-07+0j) [X0 Y1 Y4 Z5 Z6 X7] +
(7.867765103359642e-07+0j) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.003804066171728543+0j) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.44732312870767e-07+0j) [X0 Y1 Y5 X6] +
(-7.867765103359642e-07+0j) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.003804066171728543+0j) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.00688819435297055+0j) [X0 Y1 Y6 X7] +
(7.735036880588932e-05+0j) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11] +
(-1.7035783553438497e-07+0j) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.735036880588933e-05+0j) [X0 Y1 Y7 Z8 Z9 X10] +
(1.7035783553438497e-07+0j) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12] +
(0.006509361201177239+0j) [X0 Y1 Y8 X9] +
(0.00773142525077527+0j) [X0 Y1 Y10 X11] +
(-5.627851910864507e-07+0j) [X0 Y1 Y10 Z11 Z12 X13] +
(5.627851910864507e-07+0j) [X0 Y1 Y11 X12] +
(0.00528377648840296+0j) [X0 Y1 Y12 X13] +
(0.1250703257977223+0j) [X0 Z1 X2] +
(-1.933241277082068e-07+0j) [X0 Z1 X2 X3 Z4 X5] +
(-0.0022939566113524762+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 X7] +
(-0.001640754855312413+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(3.013471458482464e-07+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.933241277082068e-07+0j) [X0 Z1 X2 Y3 Z4 Y5] +
(-0.0022939566113524762+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7] +
(-0.001640754855312413+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(3.013471458482464e-07+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.003276971931231663+0j) [X0 Z1 X2 Z3] +
(-1.5510539175543743e-07+0j) [X0 Z1 X2 X4 Z5 X6] +
(-1.146837650594576e-06+0j) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.007597464029770625+0j) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.3807781480468706e-07+0j) [X0 Z1 X2 Y4 Z5 Y6] +
(-7.900128985205479e-07+0j) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.005348051582676632+0j) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005530759218631531+0j) [X0 Z1 X2 Z4] +
(-1.3807781480468706e-07+0j) [X0 Z1 X2 X5 Z6 X7] +
(-3.376739308081868e-07+0j) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0018638942824587379+0j) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.3807781480468706e-07+0j) [X0 Z1 X2 Y5 Z6 Y7] +
(-3.376739308081868e-07+0j) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0018638942824587379+0j) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0011059037691896658+0j) [X0 Z1 X2 Z5] +
(0.0057084959859609475+0j) [X0 Z1 X2 X6 Z7 Z8 Z9 X10] +
(-8.352332102023508e-07+0j) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
1.9742253792678284e-08j [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005262642473076851+0j) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10] +
(-8.074305984905768e-07+0j) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.008032520918821387+0j) [X0 Z1 X2 Z6] +
(0.0005940221543005474+0j) [X0 Z1 X2 X7 Z8 Z9 Z10 X11] +
(-8.379773243724439e-08+0j) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005940221543005474+0j) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11] +
(-8.379773243724439e-08+0j) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0033476175306661688+0j) [X0 Z1 X2 Z7] +
(0.011055020596132087+0j) [X0 Z1 X2 Z8] +
(0.002929768674751036+0j) [X0 Z1 X2 Z9] +
(-6.418291573912032e-07+0j) [X0 Z1 X2 X10 Z11 X12] +
(-6.556281913898665e-07+0j) [X0 Z1 X2 Y10 Z11 Y12] +
(0.003555290195504251+0j) [X0 Z1 X2 Z10] +
(-1.1076325598481741e-07+0j) [X0 Z1 X2 X11 Z12 X13] +
(-1.1076325598481741e-07+0j) [X0 Z1 X2 Y11 Z12 Y13] +
(0.001756070701841222+0j) [X0 Z1 X2 Z11] +
(0.006901238249797285+0j) [X0 Z1 X2 Z12] +
(0.0023262306231580697+0j) [X0 Z1 X2 Z13] +
(-3.568247520740279e-07+0j) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.002249412447093993+0j) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.0474716554867549e-08+0j) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.00044585351288409594+0j) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10] +
(-1.9742253792048224e-08+0j) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441865+0j) [X0 Z1 Z2 X3 Y4 Y5] +
(-4.523389677123614e-07+0j) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0034841573002178956+0j) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-8.091637197863892e-07+0j) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10] +
(-0.005733569747311889+0j) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004684903388155217+0j) [X0 Z1 Z2 X3 Y6 Y7] +
(0.004668620318776304+0j) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11] +
(-7.18999097434427e-07+0j) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0051144738316604+0j) [X0 Z1 Z2 X3 X7 Z8 Z9 X10] +
(-7.560692463840117e-07+0j) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381051+0j) [X0 Z1 Z2 X3 Y8 Y9] +
(-0.001799219493663029+0j) [X0 Z1 Z2 X3 Y10 Y11] +
(-5.47164774404475e-07+0j) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13] +
(-5.2876606240696e-07+0j) [X0 Z1 Z2 X3 X11 X12] +
(-0.004575007626639215+0j) [X0 Z1 Z2 X3 Y12 Y13] +
(0.004424855449441865+0j) [X0 Z1 Z2 Y3 Y4 X5] +
(4.523389677123614e-07+0j) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0034841573002178956+0j) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.091637197863892e-07+0j) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.005733569747311889+0j) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.004684903388155217+0j) [X0 Z1 Z2 Y3 Y6 X7] +
(-0.004668620318776304+0j) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11] +
(7.18999097434427e-07+0j) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0051144738316604+0j) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10] +
(-7.560692463840117e-07+0j) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12] +
(0.008125251921381051+0j) [X0 Z1 Z2 Y3 Y8 X9] +
(0.001799219493663029+0j) [X0 Z1 Z2 Y3 Y10 X11] +
(5.47164774404475e-07+0j) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13] +
(-5.2876606240696e-07+0j) [X0 Z1 Z2 Y3 Y11 X12] +
(0.004575007626639215+0j) [X0 Z1 Z2 Y3 Y12 X13] +
(3.2020768802183297e-06+0j) [X0 Z1 Z2 Z3 X4] +
(0.0008533856254125444+0j) [X0 Z1 Z2 Z3 X4 X5 Z6 X7] +
(0.0007870896771024399+0j) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0008533856254125444+0j) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7] +
(0.0007870896771024399+0j) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.2919694861557718e-07+0j) [X0 Z1 Z2 Z3 X4 Z5] +
(4.444597853756031e-07+0j) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10] +
(0.0011726348316441937+0j) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.684915094699634e-07+0j) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10] +
(0.0022009640695004533+0j) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.8394209153981177e-07+0j) [X0 Z1 Z2 Z3 X4 Z6] +
(4.092250615631588e-07+0j) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11] +
(0.0023949726397980197+0j) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.092250615631588e-07+0j) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11] +
(0.0023949726397980197+0j) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.236259961337155e-07+0j) [X0 Z1 Z2 Z3 X4 Z8] +
(8.6493101352855e-08+0j) [X0 Z1 Z2 Z3 X4 Z9] +
(0.001303800478812703+0j) [X0 Z1 Z2 Z3 X4 X10 Z11 X12] +
(0.003989841456619307+0j) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12] +
(-6.733197741273836e-07+0j) [X0 Z1 Z2 Z3 X4 Z10] +
(0.00226196606248234+0j) [X0 Z1 Z2 Z3 X4 X11 Z12 X13] +
(0.00226196606248234+0j) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13] +
(-5.927453081911979e-07+0j) [X0 Z1 Z2 Z3 X4 Z11] +
(1.2393363215922485e-06+0j) [X0 Z1 Z2 Z3 X4 Z12] +
(9.306536650848224e-07+0j) [X0 Z1 Z2 Z3 X4 Z13] +
(-0.0010283292378562596+0j) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.0026860409778066046+0j) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12] +
(-1.839420915398118e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7] +
(-0.00019400857029756686+0j) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0012223378081538266+0j) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-2.371328947808605e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9] +
(8.057446593618566e-08+0j) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11] +
(0.0017278753941369668+0j) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13] +
(-0.0009581655836696368+0j) [X0 Z1 Z2 Z3 Z4 X5 X11 X12] +
(-3.0868265650742613e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13] +
(1.839420915398118e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7] +
(0.00019400857029756686+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0012223378081538266+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(2.371328947808605e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9] +
(-8.057446593618566e-08+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11] +
(-0.0017278753941369668+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13] +
(-0.0009581655836696368+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12] +
(3.0868265650742613e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13] +
(0.0427432770137826+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6] +
(0.0005192743499487633+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(-1.8505641926860465e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005192743499487633+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(-1.8505641926860465e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.002779026799025537+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7] +
(0.004636976661182543+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8] +
(0.001280306097349661+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9] +
(2.3120943049234151e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12] +
(1.071728218205264e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12] +
(0.00537993715583936+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10] +
(7.24697442445612e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13] +
(7.24697442445612e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13] +
(0.0052415353828038445+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11] +
(0.004311038507914305+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12] +
(0.0010435246534907527+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13] +
(1.200428749244804e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12] +
(-0.0033566705638328823+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9] +
(-0.00013840177303551493+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11] +
(-6.175246206250855e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(-4.99701842145304e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12] +
(-0.003267513854423552+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13] +
(0.0033566705638328823+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9] +
(0.00013840177303551493+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11] +
(6.175246206250855e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(-4.99701842145304e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12] +
(0.003267513854423552+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13] +
(0.003876470899336932+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(-7.540341412756515e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.003876470899336932+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(-7.540341412756515e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(0.07165035181002768+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.002141361223101608+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(0.004220813970046446+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(0.0012366478019245185+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(-0.0029841661681219273+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(0.0029841661681219273+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-1.3987009014625403e-05+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(8.94947648634365e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-6.876621657221697e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(-7.661347212047053e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(0.0015324835230730261+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10] +
(-2.9045998838553724e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(0.005408954422409959+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10] +
(-1.0444941296611887e-06+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(0.004767272188278087+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10] +
(-8.105515035980022e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(0.005286546538226852+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10] +
(-9.956079228666066e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(0.001609531381721373+0j) [X0 Z1 Z2 Z3 Z4 X6] +
(-7.141625221157734e-05+0j) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10] +
(-2.6667317544008504e-07+0j) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0024629170071339174+0j) [X0 Z1 Z2 Z3 Z5 X6] +
(0.0007156734248908627+0j) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10] +
(-3.076732531523383e-07+0j) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.60607186793213e-07+0j) [X0 Z1 Z2 X4] +
(0.003961560792496526+0j) [X0 Z1 Z2 Z4 Z5 X6] +
(0.00018787053389549452+0j) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.6569309318261204e-07+0j) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.7379332624059094e-07+0j) [X0 Z1 Z3 X4] +
(0.0016676041811440497+0j) [X0 Z1 Z3 Z4 Z5 X6] +
(-0.0014528843214169185+0j) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(4.6704023903085847e-07+0j) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.10433064780651435+0j) [X0 X2] +
(3.1174479459976944e-06+0j) [X0 Z2 Z3 X4] +
(0.04587947078129789+0j) [X0 Z2 Z3 Z4 Z5 X6] +
(0.05859198873386176+0j) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-1.1463061451698477e-05+0j) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.014583648907612752+0j) [Y0 X1 X2 Y3] +
(3.5707613289482844e-07+0j) [Y0 X1 X2 Z3 Z4 Y5] +
(0.00565262097801735+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7] +
(0.008826368514209856+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.7924939575063031e-06+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.5707613289482844e-07+0j) [Y0 X1 X3 Y4] +
(-0.00565262097801735+0j) [Y0 X1 X3 Z4 Z5 Y6] +
(-0.008826368514209856+0j) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.7924939575063031e-06+0j) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0027458364701868137+0j) [Y0 X1 X4 Y5] +
(2.44732312870767e-07+0j) [Y0 X1 X4 Z5 Z6 Y7] +
(7.867765103359642e-07+0j) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.003804066171728543+0j) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.44732312870767e-07+0j) [Y0 X1 X5 Y6] +
(-7.867765103359642e-07+0j) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.003804066171728543+0j) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.00688819435297055+0j) [Y0 X1 X6 Y7] +
(7.735036880588932e-05+0j) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11] +
(-1.7035783553438497e-07+0j) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.735036880588933e-05+0j) [Y0 X1 X7 Z8 Z9 Y10] +
(1.7035783553438497e-07+0j) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12] +
(0.006509361201177239+0j) [Y0 X1 X8 Y9] +
(0.00773142525077527+0j) [Y0 X1 X10 Y11] +
(-5.627851910864507e-07+0j) [Y0 X1 X10 Z11 Z12 Y13] +
(5.627851910864507e-07+0j) [Y0 X1 X11 Y12] +
(0.00528377648840296+0j) [Y0 X1 X12 Y13] +
(-0.014583648907612752+0j) [Y0 Y1 X2 X3] +
(-3.5707613289482844e-07+0j) [Y0 Y1 X2 Z3 Z4 X5] +
(-0.00565262097801735+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7] +
(-0.008826368514209856+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.7924939575063031e-06+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.5707613289482844e-07+0j) [Y0 Y1 Y3 Y4] +
(-0.00565262097801735+0j) [Y0 Y1 Y3 Z4 Z5 Y6] +
(-0.008826368514209856+0j) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.7924939575063031e-06+0j) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.0027458364701868137+0j) [Y0 Y1 X4 X5] +
(-2.44732312870767e-07+0j) [Y0 Y1 X4 Z5 Z6 X7] +
(-7.867765103359642e-07+0j) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.003804066171728543+0j) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.44732312870767e-07+0j) [Y0 Y1 Y5 Y6] +
(-7.867765103359642e-07+0j) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.003804066171728543+0j) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.00688819435297055+0j) [Y0 Y1 X6 X7] +
(-7.735036880588932e-05+0j) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11] +
(1.7035783553438497e-07+0j) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.735036880588933e-05+0j) [Y0 Y1 Y7 Z8 Z9 Y10] +
(1.7035783553438497e-07+0j) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.006509361201177239+0j) [Y0 Y1 X8 X9] +
(-0.00773142525077527+0j) [Y0 Y1 X10 X11] +
(5.627851910864507e-07+0j) [Y0 Y1 X10 Z11 Z12 X13] +
(5.627851910864507e-07+0j) [Y0 Y1 Y11 Y12] +
(-0.00528377648840296+0j) [Y0 Y1 X12 X13] +
(-3.568247520740279e-07+0j) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.002249412447093993+0j) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.00044585351288409594+0j) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10] +
(-1.9742253792048224e-08+0j) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.0474716554867549e-08+0j) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.1250703257977223+0j) [Y0 Z1 Y2] +
(-1.933241277082068e-07+0j) [Y0 Z1 Y2 X3 Z4 X5] +
(-0.0022939566113524762+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7] +
(-0.001640754855312413+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(3.013471458482464e-07+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.933241277082068e-07+0j) [Y0 Z1 Y2 Y3 Z4 Y5] +
(-0.0022939566113524762+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7] +
(-0.001640754855312413+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(3.013471458482464e-07+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.003276971931231663+0j) [Y0 Z1 Y2 Z3] +
(-1.3807781480468706e-07+0j) [Y0 Z1 Y2 X4 Z5 X6] +
(-7.900128985205479e-07+0j) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.005348051582676632+0j) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.5510539175543743e-07+0j) [Y0 Z1 Y2 Y4 Z5 Y6] +
(-1.146837650594576e-06+0j) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.007597464029770625+0j) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005530759218631531+0j) [Y0 Z1 Y2 Z4] +
(-1.3807781480468706e-07+0j) [Y0 Z1 Y2 X5 Z6 X7] +
(-3.376739308081868e-07+0j) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0018638942824587379+0j) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.3807781480468706e-07+0j) [Y0 Z1 Y2 Y5 Z6 Y7] +
(-3.376739308081868e-07+0j) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0018638942824587379+0j) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0011059037691896658+0j) [Y0 Z1 Y2 Z5] +
(0.005262642473076851+0j) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10] +
(-8.074305984905768e-07+0j) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0057084959859609475+0j) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10] +
-1.9742253792678284e-08j [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-8.352332102023508e-07+0j) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.008032520918821387+0j) [Y0 Z1 Y2 Z6] +
(0.0005940221543005474+0j) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11] +
(-8.379773243724439e-08+0j) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005940221543005474+0j) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11] +
(-8.379773243724439e-08+0j) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0033476175306661688+0j) [Y0 Z1 Y2 Z7] +
(0.011055020596132087+0j) [Y0 Z1 Y2 Z8] +
(0.002929768674751036+0j) [Y0 Z1 Y2 Z9] +
(-6.556281913898665e-07+0j) [Y0 Z1 Y2 X10 Z11 X12] +
(-6.418291573912032e-07+0j) [Y0 Z1 Y2 Y10 Z11 Y12] +
(0.003555290195504251+0j) [Y0 Z1 Y2 Z10] +
(-1.1076325598481741e-07+0j) [Y0 Z1 Y2 X11 Z12 X13] +
(-1.1076325598481741e-07+0j) [Y0 Z1 Y2 Y11 Z12 Y13] +
(0.001756070701841222+0j) [Y0 Z1 Y2 Z11] +
(0.006901238249797285+0j) [Y0 Z1 Y2 Z12] +
(0.0023262306231580697+0j) [Y0 Z1 Y2 Z13] +
(0.004424855449441865+0j) [Y0 Z1 Z2 X3 X4 Y5] +
(4.523389677123614e-07+0j) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.0034841573002178956+0j) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-8.091637197863892e-07+0j) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.005733569747311889+0j) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.004684903388155217+0j) [Y0 Z1 Z2 X3 X6 Y7] +
(-0.004668620318776304+0j) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11] +
(7.18999097434427e-07+0j) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0051144738316604+0j) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10] +
(-7.560692463840117e-07+0j) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12] +
(0.008125251921381051+0j) [Y0 Z1 Z2 X3 X8 Y9] +
(0.001799219493663029+0j) [Y0 Z1 Z2 X3 X10 Y11] +
(5.47164774404475e-07+0j) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13] +
(-5.2876606240696e-07+0j) [Y0 Z1 Z2 X3 X11 Y12] +
(0.004575007626639215+0j) [Y0 Z1 Z2 X3 X12 Y13] +
(-0.004424855449441865+0j) [Y0 Z1 Z2 Y3 X4 X5] +
(-4.523389677123614e-07+0j) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0034841573002178956+0j) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.091637197863892e-07+0j) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.005733569747311889+0j) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004684903388155217+0j) [Y0 Z1 Z2 Y3 X6 X7] +
(0.004668620318776304+0j) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11] +
(-7.18999097434427e-07+0j) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0051144738316604+0j) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10] +
(-7.560692463840117e-07+0j) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381051+0j) [Y0 Z1 Z2 Y3 X8 X9] +
(-0.001799219493663029+0j) [Y0 Z1 Z2 Y3 X10 X11] +
(-5.47164774404475e-07+0j) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13] +
(-5.2876606240696e-07+0j) [Y0 Z1 Z2 Y3 Y11 Y12] +
(-0.004575007626639215+0j) [Y0 Z1 Z2 Y3 X12 X13] +
(-0.0010283292378562596+0j) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.0026860409778066046+0j) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12] +
(3.2020768802183297e-06+0j) [Y0 Z1 Z2 Z3 Y4] +
(0.0008533856254125444+0j) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7] +
(0.0007870896771024399+0j) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0008533856254125444+0j) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7] +
(0.0007870896771024399+0j) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.2919694861557718e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z5] +
(4.684915094699634e-07+0j) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10] +
(0.0022009640695004533+0j) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.444597853756031e-07+0j) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10] +
(0.0011726348316441937+0j) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.8394209153981177e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z6] +
(4.092250615631588e-07+0j) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11] +
(0.0023949726397980197+0j) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.092250615631588e-07+0j) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11] +
(0.0023949726397980197+0j) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.236259961337155e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z8] +
(8.6493101352855e-08+0j) [Y0 Z1 Z2 Z3 Y4 Z9] +
(0.003989841456619307+0j) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12] +
(0.001303800478812703+0j) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12] +
(-6.733197741273836e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z10] +
(0.00226196606248234+0j) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13] +
(0.00226196606248234+0j) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13] +
(-5.927453081911979e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z11] +
(1.2393363215922485e-06+0j) [Y0 Z1 Z2 Z3 Y4 Z12] +
(9.306536650848224e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z13] +
(1.839420915398118e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7] +
(0.00019400857029756686+0j) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0012223378081538266+0j) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(2.371328947808605e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9] +
(-8.057446593618566e-08+0j) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11] +
(-0.0017278753941369668+0j) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13] +
(-0.0009581655836696368+0j) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12] +
(3.0868265650742613e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13] +
(-1.839420915398118e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7] +
(-0.00019400857029756686+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0012223378081538266+0j) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-2.371328947808605e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9] +
(8.057446593618566e-08+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11] +
(0.0017278753941369668+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13] +
(-0.0009581655836696368+0j) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12] +
(-3.0868265650742613e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13] +
(1.200428749244804e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12] +
(0.0427432770137826+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6] +
(0.0005192743499487633+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(-1.8505641926860465e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005192743499487633+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(-1.8505641926860465e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.002779026799025537+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7] +
(0.004636976661182543+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8] +
(0.001280306097349661+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9] +
(1.071728218205264e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12] +
(2.3120943049234151e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12] +
(0.00537993715583936+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10] +
(7.24697442445612e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13] +
(7.24697442445612e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13] +
(0.0052415353828038445+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11] +
(0.004311038507914305+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12] +
(0.0010435246534907527+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13] +
(0.0033566705638328823+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9] +
(0.00013840177303551493+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11] +
(6.175246206250855e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(-4.99701842145304e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12] +
(0.003267513854423552+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13] +
(-0.0033566705638328823+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9] +
(-0.00013840177303551493+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11] +
(-6.175246206250855e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(-4.99701842145304e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12] +
(-0.003267513854423552+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13] +
(0.003876470899336932+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(-7.540341412756515e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.003876470899336932+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(-7.540341412756515e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(0.07165035181002768+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.002141361223101608+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(0.004220813970046446+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(0.0012366478019245185+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(0.0029841661681219273+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-0.0029841661681219273+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-1.3987009014625403e-05+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(8.94947648634365e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-6.876621657221697e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(-7.661347212047053e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(0.0015324835230730261+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10] +
(-2.9045998838553724e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(0.005408954422409959+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10] +
(-1.0444941296611887e-06+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(0.004767272188278087+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10] +
(-8.105515035980022e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(0.005286546538226852+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10] +
(-9.956079228666066e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.001609531381721373+0j) [Y0 Z1 Z2 Z3 Z4 Y6] +
(-7.141625221157734e-05+0j) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10] +
(-2.6667317544008504e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0024629170071339174+0j) [Y0 Z1 Z2 Z3 Z5 Y6] +
(0.0007156734248908627+0j) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10] +
(-3.076732531523383e-07+0j) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(3.60607186793213e-07+0j) [Y0 Z1 Z2 Y4] +
(0.003961560792496526+0j) [Y0 Z1 Z2 Z4 Z5 Y6] +
(0.00018787053389549452+0j) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.6569309318261204e-07+0j) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.7379332624059094e-07+0j) [Y0 Z1 Z3 Y4] +
(0.0016676041811440497+0j) [Y0 Z1 Z3 Z4 Z5 Y6] +
(-0.0014528843214169185+0j) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(4.6704023903085847e-07+0j) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.10433064780651435+0j) [Y0 Y2] +
(3.1174479459976944e-06+0j) [Y0 Z2 Z3 Y4] +
(0.04587947078129789+0j) [Y0 Z2 Z3 Z4 Z5 Y6] +
(0.05859198873386176+0j) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-1.1463061451698477e-05+0j) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(12.41263074211178+0j) [Z0] +
(0.10433064780651435+0j) [Z0 X1 Z2 X3] +
(3.1174479459976944e-06+0j) [Z0 X1 Z2 Z3 Z4 X5] +
(0.04587947078129789+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7] +
(0.05859198873386176+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-1.1463061451698477e-05+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.10433064780651435+0j) [Z0 Y1 Z2 Y3] +
(3.1174479459976944e-06+0j) [Z0 Y1 Z2 Z3 Z4 Y5] +
(0.04587947078129789+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7] +
(0.05859198873386176+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.1463061451698477e-05+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.186176373486049+0j) [Z0 Z1] +
(-8.337746753270062e-07+0j) [Z0 X2 Z3 X4] +
(-0.027115036845273124+0j) [Z0 X2 Z3 Z4 Z5 X6] +
(-0.06752385099214002+0j) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.401710973362013e-05+0j) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-8.337746753270062e-07+0j) [Z0 Y2 Z3 Y4] +
(-0.027115036845273124+0j) [Z0 Y2 Z3 Z4 Z5 Y6] +
(-0.06752385099214002+0j) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.401710973362013e-05+0j) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.23671080783830456+0j) [Z0 Z2] +
(-1.1908508082218347e-06+0j) [Z0 X3 Z4 X5] +
(-0.032767657823290476+0j) [Z0 X3 Z4 Z5 Z6 X7] +
(-0.07635021950634988+0j) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.580960369112643e-05+0j) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.1908508082218347e-06+0j) [Z0 Y3 Z4 Y5] +
(-0.032767657823290476+0j) [Z0 Y3 Z4 Z5 Z6 Y7] +
(-0.07635021950634988+0j) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.580960369112643e-05+0j) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.2512944567459173+0j) [Z0 Z3] +
(-3.099349243490603e-06+0j) [Z0 X4 Z5 X6] +
(-1.531680879353339e-05+0j) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.08684737589863617+0j) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-3.099349243490603e-06+0j) [Z0 Y4 Z5 Y6] +
(-1.531680879353339e-05+0j) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.08684737589863617+0j) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.19661770890342137+0j) [Z0 Z4] +
(-3.344081556361369e-06+0j) [Z0 X5 Z6 X7] +
(-1.6103585303869358e-05+0j) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.09065144207036471+0j) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.344081556361369e-06+0j) [Z0 Y5 Z6 Y7] +
(-1.6103585303869358e-05+0j) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.09065144207036471+0j) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.19936354537360818+0j) [Z0 Z5] +
(0.056084681246613644+0j) [Z0 X6 Z7 Z8 Z9 X10] +
(-6.652209668479086e-06+0j) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.056084681246613644+0j) [Z0 Y6 Z7 Z8 Z9 Y10] +
(-6.652209668479086e-06+0j) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.2416466393601717+0j) [Z0 Z6] +
(0.05600733087780776+0j) [Z0 X7 Z8 Z9 Z10 X11] +
(-6.4818518329447e-06+0j) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.05600733087780776+0j) [Z0 Y7 Z8 Z9 Z10 Y11] +
(-6.4818518329447e-06+0j) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.24853483371314225+0j) [Z0 Z7] +
(0.27232518306605663+0j) [Z0 Z8] +
(-2.177664604670446e-06+0j) [Z0 X10 Z11 X12] +
(-2.177664604670446e-06+0j) [Z0 Y10 Z11 Y12] +
(0.19299723935364205+0j) [Z0 Z10] +
(-1.6148794135839955e-06+0j) [Z0 X11 Z12 X13] +
(-1.6148794135839955e-06+0j) [Z0 Y11 Z12 Y13] +
(0.2007286646044173+0j) [Z0 Z11] +
(0.2110265984979151+0j) [Z0 Z12] +
(0.21631037498631805+0j) [Z0 Z13] +
(1.933241277082068e-07+0j) [X1 X2 Y3 Y4] +
(0.001640754855312413+0j) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-3.0134714584824643e-07+0j) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004424855449441865+0j) [X1 X2 X4 X5] +
(-8.091637197863892e-07+0j) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.005733569747311889+0j) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-4.523389677123614e-07+0j) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.0034841573002178956+0j) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004684903388155217+0j) [X1 X2 X6 X7] +
(0.0051144738316604+0j) [X1 X2 X6 Z7 Z8 Z9 Z10 X11] +
(-7.560692463840117e-07+0j) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.004668620318776304+0j) [X1 X2 Y7 Z8 Z9 Y10] +
(-7.18999097434427e-07+0j) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381051+0j) [X1 X2 X8 X9] +
(-0.0017992194936630292+0j) [X1 X2 X10 X11] +
(-5.2876606240696e-07+0j) [X1 X2 X10 Z11 Z12 X13] +
(-5.47164774404475e-07+0j) [X1 X2 Y11 Y12] +
(-0.004575007626639216+0j) [X1 X2 X12 X13] +
(-1.933241277082068e-07+0j) [X1 Y2 Y3 X4] +
(-0.001640754855312413+0j) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(3.0134714584824643e-07+0j) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441865+0j) [X1 Y2 Y4 X5] +
(-8.091637197863892e-07+0j) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.005733569747311889+0j) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.523389677123614e-07+0j) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10] +
(0.0034841573002178956+0j) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004684903388155217+0j) [X1 Y2 Y6 X7] +
(0.0051144738316604+0j) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11] +
(-7.560692463840117e-07+0j) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.004668620318776304+0j) [X1 Y2 Y7 Z8 Z9 X10] +
(7.18999097434427e-07+0j) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381051+0j) [X1 Y2 Y8 X9] +
(-0.0017992194936630292+0j) [X1 Y2 Y10 X11] +
(-5.2876606240696e-07+0j) [X1 Y2 Y10 Z11 Z12 X13] +
(5.47164774404475e-07+0j) [X1 Y2 Y11 X12] +
(-0.004575007626639216+0j) [X1 Y2 Y12 X13] +
(0.12507032579772223+0j) [X1 Z2 X3] +
(-1.3807781480468706e-07+0j) [X1 Z2 X3 X4 Z5 X6] +
(-3.376739308081868e-07+0j) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0018638942824587379+0j) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.3807781480468706e-07+0j) [X1 Z2 X3 Y4 Z5 Y6] +
(-3.376739308081868e-07+0j) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0018638942824587379+0j) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0011059037691896658+0j) [X1 Z2 X3 Z4] +
(-1.5510539175543743e-07+0j) [X1 Z2 X3 X5 Z6 X7] +
(-1.146837650594576e-06+0j) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.007597464029770625+0j) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.3807781480468706e-07+0j) [X1 Z2 X3 Y5 Z6 Y7] +
(-7.900128985205479e-07+0j) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.005348051582676632+0j) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005530759218631531+0j) [X1 Z2 X3 Z5] +
(0.0005940221543005474+0j) [X1 Z2 X3 X6 Z7 Z8 Z9 X10] +
(-8.379773243724439e-08+0j) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0005940221543005474+0j) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10] +
(-8.379773243724439e-08+0j) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0033476175306661688+0j) [X1 Z2 X3 Z6] +
(0.0057084959859609475+0j) [X1 Z2 X3 X7 Z8 Z9 Z10 X11] +
(-8.352332102023508e-07+0j) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
1.9742253792678284e-08j [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005262642473076851+0j) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11] +
(-8.074305984905768e-07+0j) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.008032520918821387+0j) [X1 Z2 X3 Z7] +
(0.002929768674751036+0j) [X1 Z2 X3 Z8] +
(0.011055020596132087+0j) [X1 Z2 X3 Z9] +
(-1.1076325598481741e-07+0j) [X1 Z2 X3 X10 Z11 X12] +
(-1.1076325598481741e-07+0j) [X1 Z2 X3 Y10 Z11 Y12] +
(0.001756070701841222+0j) [X1 Z2 X3 Z10] +
(-6.418291573912032e-07+0j) [X1 Z2 X3 X11 Z12 X13] +
(-6.556281913898665e-07+0j) [X1 Z2 X3 Y11 Z12 Y13] +
(0.003555290195504251+0j) [X1 Z2 X3 Z11] +
(0.0023262306231580697+0j) [X1 Z2 X3 Z12] +
(0.006901238249797285+0j) [X1 Z2 X3 Z13] +
(-3.568247520740279e-07+0j) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.002249412447093993+0j) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.0474716554867549e-08+0j) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.00044585351288409594+0j) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11] +
(-1.9742253792048224e-08+0j) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0008533856254125444+0j) [X1 Z2 Z3 X4 Y5 Y6] +
(-0.0007870896771024399+0j) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10] +
(-1.839420915398118e-07+0j) [X1 Z2 Z3 X4 X6 X7] +
(-0.0012223378081538266+0j) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.00019400857029756686+0j) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12] +
(-2.371328947808605e-07+0j) [X1 Z2 Z3 X4 X8 X9] +
(8.057446593618566e-08+0j) [X1 Z2 Z3 X4 X10 X11] +
(-0.0009581655836696368+0j) [X1 Z2 Z3 X4 X10 Z11 Z12 X13] +
(0.0017278753941369668+0j) [X1 Z2 Z3 X4 Y11 Y12] +
(-3.086826565074262e-07+0j) [X1 Z2 Z3 X4 X12 X13] +
(0.0008533856254125444+0j) [X1 Z2 Z3 Y4 Y5 X6] +
(0.0007870896771024399+0j) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10] +
(-1.839420915398118e-07+0j) [X1 Z2 Z3 Y4 Y6 X7] +
(-0.0012223378081538266+0j) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.00019400857029756686+0j) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12] +
(-2.371328947808605e-07+0j) [X1 Z2 Z3 Y4 Y8 X9] +
(8.057446593618566e-08+0j) [X1 Z2 Z3 Y4 Y10 X11] +
(-0.0009581655836696368+0j) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13] +
(-0.0017278753941369668+0j) [X1 Z2 Z3 Y4 Y11 X12] +
(-3.086826565074262e-07+0j) [X1 Z2 Z3 Y4 Y12 X13] +
(3.202076880218329e-06+0j) [X1 Z2 Z3 Z4 X5] +
(4.092250615631588e-07+0j) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10] +
(0.0023949726397980197+0j) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.092250615631588e-07+0j) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10] +
(0.0023949726397980197+0j) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.444597853756031e-07+0j) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11] +
(0.0011726348316441937+0j) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.684915094699634e-07+0j) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11] +
(0.0022009640695004533+0j) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.8394209153981177e-07+0j) [X1 Z2 Z3 Z4 X5 Z7] +
(8.6493101352855e-08+0j) [X1 Z2 Z3 Z4 X5 Z8] +
(3.236259961337155e-07+0j) [X1 Z2 Z3 Z4 X5 Z9] +
(0.00226196606248234+0j) [X1 Z2 Z3 Z4 X5 X10 Z11 X12] +
(0.00226196606248234+0j) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12] +
(-5.927453081911979e-07+0j) [X1 Z2 Z3 Z4 X5 Z10] +
(0.001303800478812703+0j) [X1 Z2 Z3 Z4 X5 X11 Z12 X13] +
(0.003989841456619307+0j) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13] +
(-6.733197741273836e-07+0j) [X1 Z2 Z3 Z4 X5 Z11] +
(9.306536650848224e-07+0j) [X1 Z2 Z3 Z4 X5 Z12] +
(1.2393363215922485e-06+0j) [X1 Z2 Z3 Z4 X5 Z13] +
(-0.0010283292378562596+0j) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0026860409778066046+0j) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13] +
(-0.0005192743499487633+0j) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10] +
(1.8505641926860465e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.0033566705638328823+0j) [X1 Z2 Z3 Z4 Z5 X6 X8 X9] +
(-0.00013840177303551493+0j) [X1 Z2 Z3 Z4 Z5 X6 X10 X11] +
(-4.99701842145304e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13] +
(-6.175246206250855e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12] +
(-0.003267513854423552+0j) [X1 Z2 Z3 Z4 Z5 X6 X12 X13] +
(0.0005192743499487633+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10] +
(-1.8505641926860465e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.0033566705638328823+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9] +
(-0.00013840177303551493+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11] +
(-4.99701842145304e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13] +
(6.175246206250855e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12] +
(-0.003267513854423552+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13] +
(0.0427432770137826+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7] +
(0.001280306097349661+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8] +
(0.004636976661182543+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9] +
(7.24697442445612e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12] +
(7.24697442445612e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12] +
(0.0052415353828038445+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10] +
(2.3120943049234151e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13] +
(1.071728218205264e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13] +
(0.00537993715583936+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11] +
(0.0010435246534907527+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12] +
(0.004311038507914305+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13] +
(1.200428749244804e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13] +
(-0.0038764708993369325+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10] +
(7.540341412756516e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(0.0038764708993369325+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10] +
(-7.540341412756516e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(-0.0029841661681219277+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-0.0029841661681219277+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(0.07165035181002766+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0012366478019245185+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(0.004220813970046446+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(-1.3987009014625403e-05+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(8.949476486343652e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(-7.661347212047053e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(-0.002141361223101608+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11] +
(-6.876621657221697e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(0.005408954422409959+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11] +
(-1.0444941296611887e-06+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(0.0015324835230730261+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11] +
(-2.9045998838553724e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(0.005286546538226852+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11] +
(-9.956079228666066e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.002779026799025537+0j) [X1 Z2 Z3 Z4 Z5 X7] +
(0.004767272188278087+0j) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11] +
(-8.105515035980022e-07+0j) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0024629170071339174+0j) [X1 Z2 Z3 Z4 Z6 X7] +
(0.0007156734248908627+0j) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11] +
(-3.076732531523383e-07+0j) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2919694861557718e-07+0j) [X1 Z2 Z3 X5] +
(0.001609531381721373+0j) [X1 Z2 Z3 Z5 Z6 X7] +
(-7.141625221157734e-05+0j) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-2.6667317544008504e-07+0j) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.7379332624059094e-07+0j) [X1 Z2 Z4 X5] +
(0.0016676041811440497+0j) [X1 Z2 Z4 Z5 Z6 X7] +
(-0.0014528843214169185+0j) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(4.6704023903085847e-07+0j) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0032769719312316634+0j) [X1 X3] +
(3.60607186793213e-07+0j) [X1 Z3 Z4 X5] +
(0.003961560792496526+0j) [X1 Z3 Z4 Z5 Z6 X7] +
(0.00018787053389549452+0j) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.6569309318261204e-07+0j) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.933241277082068e-07+0j) [Y1 X2 X3 Y4] +
(-0.001640754855312413+0j) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(3.0134714584824643e-07+0j) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004424855449441865+0j) [Y1 X2 X4 Y5] +
(-8.091637197863892e-07+0j) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.005733569747311889+0j) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(4.523389677123614e-07+0j) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10] +
(0.0034841573002178956+0j) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004684903388155217+0j) [Y1 X2 X6 Y7] +
(0.0051144738316604+0j) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11] +
(-7.560692463840117e-07+0j) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.004668620318776304+0j) [Y1 X2 X7 Z8 Z9 Y10] +
(7.18999097434427e-07+0j) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381051+0j) [Y1 X2 X8 Y9] +
(-0.0017992194936630292+0j) [Y1 X2 X10 Y11] +
(-5.2876606240696e-07+0j) [Y1 X2 X10 Z11 Z12 Y13] +
(5.47164774404475e-07+0j) [Y1 X2 X11 Y12] +
(-0.004575007626639216+0j) [Y1 X2 X12 Y13] +
(1.933241277082068e-07+0j) [Y1 Y2 X3 X4] +
(0.001640754855312413+0j) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-3.0134714584824643e-07+0j) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441865+0j) [Y1 Y2 Y4 Y5] +
(-8.091637197863892e-07+0j) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.005733569747311889+0j) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-4.523389677123614e-07+0j) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10] +
(-0.0034841573002178956+0j) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004684903388155217+0j) [Y1 Y2 Y6 Y7] +
(0.0051144738316604+0j) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11] +
(-7.560692463840117e-07+0j) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.004668620318776304+0j) [Y1 Y2 X7 Z8 Z9 X10] +
(-7.18999097434427e-07+0j) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381051+0j) [Y1 Y2 Y8 Y9] +
(-0.0017992194936630292+0j) [Y1 Y2 Y10 Y11] +
(-5.2876606240696e-07+0j) [Y1 Y2 Y10 Z11 Z12 Y13] +
(-5.47164774404475e-07+0j) [Y1 Y2 X11 X12] +
(-0.004575007626639216+0j) [Y1 Y2 Y12 Y13] +
(-3.568247520740279e-07+0j) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.002249412447093993+0j) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.00044585351288409594+0j) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11] +
(-1.9742253792048224e-08+0j) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.0474716554867549e-08+0j) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.12507032579772223+0j) [Y1 Z2 Y3] +
(-1.3807781480468706e-07+0j) [Y1 Z2 Y3 X4 Z5 X6] +
(-3.376739308081868e-07+0j) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0018638942824587379+0j) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.3807781480468706e-07+0j) [Y1 Z2 Y3 Y4 Z5 Y6] +
(-3.376739308081868e-07+0j) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0018638942824587379+0j) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0011059037691896658+0j) [Y1 Z2 Y3 Z4] +
(-1.3807781480468706e-07+0j) [Y1 Z2 Y3 X5 Z6 X7] +
(-7.900128985205479e-07+0j) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.005348051582676632+0j) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.5510539175543743e-07+0j) [Y1 Z2 Y3 Y5 Z6 Y7] +
(-1.146837650594576e-06+0j) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.007597464029770625+0j) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005530759218631531+0j) [Y1 Z2 Y3 Z5] +
(0.0005940221543005474+0j) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10] +
(-8.379773243724439e-08+0j) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0005940221543005474+0j) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10] +
(-8.379773243724439e-08+0j) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0033476175306661688+0j) [Y1 Z2 Y3 Z6] +
(0.005262642473076851+0j) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11] +
(-8.074305984905768e-07+0j) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0057084959859609475+0j) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11] +
-1.9742253792678284e-08j [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.352332102023508e-07+0j) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.008032520918821387+0j) [Y1 Z2 Y3 Z7] +
(0.002929768674751036+0j) [Y1 Z2 Y3 Z8] +
(0.011055020596132087+0j) [Y1 Z2 Y3 Z9] +
(-1.1076325598481741e-07+0j) [Y1 Z2 Y3 X10 Z11 X12] +
(-1.1076325598481741e-07+0j) [Y1 Z2 Y3 Y10 Z11 Y12] +
(0.001756070701841222+0j) [Y1 Z2 Y3 Z10] +
(-6.556281913898665e-07+0j) [Y1 Z2 Y3 X11 Z12 X13] +
(-6.418291573912032e-07+0j) [Y1 Z2 Y3 Y11 Z12 Y13] +
(0.003555290195504251+0j) [Y1 Z2 Y3 Z11] +
(0.0023262306231580697+0j) [Y1 Z2 Y3 Z12] +
(0.006901238249797285+0j) [Y1 Z2 Y3 Z13] +
(0.0008533856254125444+0j) [Y1 Z2 Z3 X4 X5 Y6] +
(0.0007870896771024399+0j) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10] +
(-1.839420915398118e-07+0j) [Y1 Z2 Z3 X4 X6 Y7] +
(-0.0012223378081538266+0j) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.00019400857029756686+0j) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12] +
(-2.371328947808605e-07+0j) [Y1 Z2 Z3 X4 X8 Y9] +
(8.057446593618566e-08+0j) [Y1 Z2 Z3 X4 X10 Y11] +
(-0.0009581655836696368+0j) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13] +
(-0.0017278753941369668+0j) [Y1 Z2 Z3 X4 X11 Y12] +
(-3.086826565074262e-07+0j) [Y1 Z2 Z3 X4 X12 Y13] +
(-0.0008533856254125444+0j) [Y1 Z2 Z3 Y4 X5 X6] +
(-0.0007870896771024399+0j) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10] +
(-1.839420915398118e-07+0j) [Y1 Z2 Z3 Y4 Y6 Y7] +
(-0.0012223378081538266+0j) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.00019400857029756686+0j) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12] +
(-2.371328947808605e-07+0j) [Y1 Z2 Z3 Y4 Y8 Y9] +
(8.057446593618566e-08+0j) [Y1 Z2 Z3 Y4 Y10 Y11] +
(-0.0009581655836696368+0j) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13] +
(0.0017278753941369668+0j) [Y1 Z2 Z3 Y4 X11 X12] +
(-3.086826565074262e-07+0j) [Y1 Z2 Z3 Y4 Y12 Y13] +
(-0.0010283292378562596+0j) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0026860409778066046+0j) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13] +
(3.202076880218329e-06+0j) [Y1 Z2 Z3 Z4 Y5] +
(4.092250615631588e-07+0j) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10] +
(0.0023949726397980197+0j) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.092250615631588e-07+0j) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10] +
(0.0023949726397980197+0j) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.684915094699634e-07+0j) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11] +
(0.0022009640695004533+0j) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.444597853756031e-07+0j) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11] +
(0.0011726348316441937+0j) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.8394209153981177e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z7] +
(8.6493101352855e-08+0j) [Y1 Z2 Z3 Z4 Y5 Z8] +
(3.236259961337155e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z9] +
(0.00226196606248234+0j) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12] +
(0.00226196606248234+0j) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12] +
(-5.927453081911979e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z10] +
(0.003989841456619307+0j) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13] +
(0.001303800478812703+0j) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13] +
(-6.733197741273836e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z11] +
(9.306536650848224e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z12] +
(1.2393363215922485e-06+0j) [Y1 Z2 Z3 Z4 Y5 Z13] +
(0.0005192743499487633+0j) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10] +
(-1.8505641926860465e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.0033566705638328823+0j) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9] +
(-0.00013840177303551493+0j) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11] +
(-4.99701842145304e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13] +
(6.175246206250855e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12] +
(-0.003267513854423552+0j) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13] +
(-0.0005192743499487633+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10] +
(1.8505641926860465e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-0.0033566705638328823+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9] +
(-0.00013840177303551493+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11] +
(-4.99701842145304e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13] +
(-6.175246206250855e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12] +
(-0.003267513854423552+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13] +
(1.200428749244804e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13] +
(0.0427432770137826+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7] +
(0.001280306097349661+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8] +
(0.004636976661182543+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9] +
(7.24697442445612e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12] +
(7.24697442445612e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12] +
(0.0052415353828038445+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10] +
(1.071728218205264e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13] +
(2.3120943049234151e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13] +
(0.00537993715583936+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11] +
(0.0010435246534907527+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12] +
(0.004311038507914305+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13] +
(0.0038764708993369325+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10] +
(-7.540341412756516e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-0.0038764708993369325+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10] +
(7.540341412756516e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-0.0029841661681219277+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(-0.0029841661681219277+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(0.07165035181002766+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.0012366478019245185+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(0.004220813970046446+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(-1.3987009014625403e-05+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(8.949476486343652e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(-7.661347212047053e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(-0.002141361223101608+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11] +
(-6.876621657221697e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(0.005408954422409959+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11] +
(-1.0444941296611887e-06+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(0.0015324835230730261+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11] +
(-2.9045998838553724e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(0.005286546538226852+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11] +
(-9.956079228666066e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.002779026799025537+0j) [Y1 Z2 Z3 Z4 Z5 Y7] +
(0.004767272188278087+0j) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11] +
(-8.105515035980022e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0024629170071339174+0j) [Y1 Z2 Z3 Z4 Z6 Y7] +
(0.0007156734248908627+0j) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11] +
(-3.076732531523383e-07+0j) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.2919694861557718e-07+0j) [Y1 Z2 Z3 Y5] +
(0.001609531381721373+0j) [Y1 Z2 Z3 Z5 Z6 Y7] +
(-7.141625221157734e-05+0j) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-2.6667317544008504e-07+0j) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.7379332624059094e-07+0j) [Y1 Z2 Z4 Y5] +
(0.0016676041811440497+0j) [Y1 Z2 Z4 Z5 Z6 Y7] +
(-0.0014528843214169185+0j) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(4.6704023903085847e-07+0j) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0032769719312316634+0j) [Y1 Y3] +
(3.60607186793213e-07+0j) [Y1 Z3 Z4 Y5] +
(0.003961560792496526+0j) [Y1 Z3 Z4 Z5 Z6 Y7] +
(0.00018787053389549452+0j) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.6569309318261204e-07+0j) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(12.41263074211178+0j) [Z1] +
(-1.1908508082218347e-06+0j) [Z1 X2 Z3 X4] +
(-0.032767657823290476+0j) [Z1 X2 Z3 Z4 Z5 X6] +
(-0.07635021950634988+0j) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.580960369112643e-05+0j) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.1908508082218347e-06+0j) [Z1 Y2 Z3 Y4] +
(-0.032767657823290476+0j) [Z1 Y2 Z3 Z4 Z5 Y6] +
(-0.07635021950634988+0j) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.580960369112643e-05+0j) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.2512944567459173+0j) [Z1 Z2] +
(-8.337746753270062e-07+0j) [Z1 X3 Z4 X5] +
(-0.027115036845273124+0j) [Z1 X3 Z4 Z5 Z6 X7] +
(-0.06752385099214002+0j) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.401710973362013e-05+0j) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.337746753270062e-07+0j) [Z1 Y3 Z4 Y5] +
(-0.027115036845273124+0j) [Z1 Y3 Z4 Z5 Z6 Y7] +
(-0.06752385099214002+0j) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.401710973362013e-05+0j) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.23671080783830456+0j) [Z1 Z3] +
(-3.344081556361369e-06+0j) [Z1 X4 Z5 X6] +
(-1.6103585303869358e-05+0j) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.09065144207036471+0j) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-3.344081556361369e-06+0j) [Z1 Y4 Z5 Y6] +
(-1.6103585303869358e-05+0j) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.09065144207036471+0j) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.19936354537360818+0j) [Z1 Z4] +
(-3.099349243490603e-06+0j) [Z1 X5 Z6 X7] +
(-1.531680879353339e-05+0j) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.08684737589863617+0j) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.099349243490603e-06+0j) [Z1 Y5 Z6 Y7] +
(-1.531680879353339e-05+0j) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.08684737589863617+0j) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.19661770890342137+0j) [Z1 Z5] +
(0.05600733087780776+0j) [Z1 X6 Z7 Z8 Z9 X10] +
(-6.4818518329447e-06+0j) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.05600733087780776+0j) [Z1 Y6 Z7 Z8 Z9 Y10] +
(-6.4818518329447e-06+0j) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.24853483371314225+0j) [Z1 Z6] +
(0.056084681246613644+0j) [Z1 X7 Z8 Z9 Z10 X11] +
(-6.652209668479086e-06+0j) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.056084681246613644+0j) [Z1 Y7 Z8 Z9 Z10 Y11] +
(-6.652209668479086e-06+0j) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.2416466393601717+0j) [Z1 Z7] +
(0.27232518306605663+0j) [Z1 Z9] +
(-1.6148794135839955e-06+0j) [Z1 X10 Z11 X12] +
(-1.6148794135839955e-06+0j) [Z1 Y10 Z11 Y12] +
(0.2007286646044173+0j) [Z1 Z10] +
(-2.177664604670446e-06+0j) [Z1 X11 Z12 X13] +
(-2.177664604670446e-06+0j) [Z1 Y11 Z12 Y13] +
(0.19299723935364205+0j) [Z1 Z11] +
(0.21631037498631805+0j) [Z1 Z12] +
(0.2110265984979151+0j) [Z1 Z13] +
(-0.03583956795335344+0j) [X2 X3 Y4 Y5] +
(-2.199051618018829e-07+0j) [X2 X3 Y4 Z5 Z6 Y7] +
(-2.360956320043574e-06+0j) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.010311482489831894+0j) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.199051618018829e-07+0j) [X2 X3 X5 X6] +
(-2.360956320043574e-06+0j) [X2 X3 X5 Z6 Z7 Z8 Z9 X10] +
(-0.010311482489831894+0j) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.031143817988967173+0j) [X2 X3 Y6 Y7] +
(0.005368659358109611+0j) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11] +
(9.209350652291171e-08+0j) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005368659358109611+0j) [X2 X3 X7 Z8 Z9 X10] +
(9.209350652291171e-08+0j) [X2 X3 X7 Z8 Z9 Z10 Z11 X12] +
(-0.0361941235590427+0j) [X2 X3 Y8 Y9] +
(-0.025384657508457316+0j) [X2 X3 Y10 Y11] +
(2.1726691012502567e-06+0j) [X2 X3 Y10 Z11 Z12 Y13] +
(2.172669101250257e-06+0j) [X2 X3 X11 X12] +
(-0.015577208063976469+0j) [X2 X3 Y12 Y13] +
(0.03583956795335344+0j) [X2 Y3 Y4 X5] +
(2.199051618018829e-07+0j) [X2 Y3 Y4 Z5 Z6 X7] +
(2.360956320043574e-06+0j) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.010311482489831894+0j) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.199051618018829e-07+0j) [X2 Y3 Y5 X6] +
(-2.360956320043574e-06+0j) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.010311482489831894+0j) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.031143817988967173+0j) [X2 Y3 Y6 X7] +
(-0.005368659358109611+0j) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11] +
(-9.209350652291171e-08+0j) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005368659358109611+0j) [X2 Y3 Y7 Z8 Z9 X10] +
(9.209350652291171e-08+0j) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12] +
(0.0361941235590427+0j) [X2 Y3 Y8 X9] +
(0.025384657508457316+0j) [X2 Y3 Y10 X11] +
(-2.1726691012502567e-06+0j) [X2 Y3 Y10 Z11 Z12 X13] +
(2.172669101250257e-06+0j) [X2 Y3 Y11 X12] +
(0.015577208063976469+0j) [X2 Y3 Y12 X13] +
(-3.887051672420326e-06+0j) [X2 Z3 X4] +
(-0.005143391768825138+0j) [X2 Z3 X4 X5 Z6 X7] +
(-0.009841749246962623+0j) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.988511706101805e-06+0j) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.005143391768825138+0j) [X2 Z3 X4 Y5 Z6 Y7] +
(-0.009841749246962623+0j) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.988511706101805e-06+0j) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.764994118334913e-07+0j) [X2 Z3 X4 Z5] +
(1.6893489512604343e-06+0j) [X2 Z3 X4 X6 Z7 Z8 Z9 X10] +
(0.01075756395390896+0j) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.537178095352438e-08+0j) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10] +
(4.205548411217964e-05+0j) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.59353439138071e-07+0j) [X2 Z3 X4 Z6] +
(3.2118420187238283e-06+0j) [X2 Z3 X4 X7 Z8 Z9 Z10 X11] +
(0.019299560579363776+0j) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.2118420187238283e-06+0j) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11] +
(0.019299560579363776+0j) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.1954890098621335e-06+0j) [X2 Z3 X4 Z7] +
(2.1868423777396504e-07+0j) [X2 Z3 X4 Z8] +
(-5.770052994438331e-07+0j) [X2 Z3 X4 Z9] +
(0.0155882501023802+0j) [X2 Z3 X4 X10 Z11 X12] +
(0.005324835234221699+0j) [X2 Z3 X4 Y10 Z11 Y12] +
(-3.158656431607071e-06+0j) [X2 Z3 X4 Z10] +
(0.024353077678068904+0j) [X2 Z3 X4 X11 Z12 X13] +
(0.024353077678068904+0j) [X2 Z3 X4 Y11 Z12 Y13] +
(-7.801707499546218e-06+0j) [X2 Z3 X4 Z11] +
(3.5390541841113903e-06+0j) [X2 Z3 X4 Z12] +
(8.814937305600543e-06+0j) [X2 Z3 X4 Z13] +
(1.6288532433231053e-06+0j) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10] +
(0.010715508469796783+0j) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.010263414868158497+0j) [X2 Z3 Y4 Y10 Z11 X12] +
(-1.4548424490002047e-06+0j) [X2 Z3 Z4 X5 Y6 Y7] +
(-3.151346310786498e-06+0j) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11] +
(-0.0192575050952516+0j) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.522493067463393e-06+0j) [X2 Z3 Z4 X5 X7 Z8 Z9 X10] +
(-0.008541996625454816+0j) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-7.956895372177981e-07+0j) [X2 Z3 Z4 X5 Y8 Y9] +
(-4.643051067939146e-06+0j) [X2 Z3 Z4 X5 Y10 Y11] +
(-0.019028242443847203+0j) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13] +
(-0.008764827575688706+0j) [X2 Z3 Z4 X5 X11 X12] +
(5.275883121489154e-06+0j) [X2 Z3 Z4 X5 Y12 Y13] +
(1.4548424490002047e-06+0j) [X2 Z3 Z4 Y5 Y6 X7] +
(3.151346310786498e-06+0j) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11] +
(0.0192575050952516+0j) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.522493067463393e-06+0j) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10] +
(-0.008541996625454816+0j) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(7.956895372177981e-07+0j) [X2 Z3 Z4 Y5 Y8 X9] +
(4.643051067939146e-06+0j) [X2 Z3 Z4 Y5 Y10 X11] +
(0.019028242443847203+0j) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13] +
(-0.008764827575688706+0j) [X2 Z3 Z4 Y5 Y11 X12] +
(-5.275883121489154e-06+0j) [X2 Z3 Z4 Y5 Y12 X13] +
(-0.121332769110423+0j) [X2 Z3 Z4 Z5 X6] +
(-0.008469978791023992+0j) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(2.6863815444635537e-07+0j) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.008469978791023992+0j) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(2.6863815444635537e-07+0j) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.02599617759802111+0j) [X2 Z3 Z4 Z5 X6 Z7] +
(-0.005805188989826928+0j) [X2 Z3 Z4 Z5 X6 Z8] +
(-0.01756120240964616+0j) [X2 Z3 Z4 Z5 X6 Z9] +
(-7.988770288204933e-07+0j) [X2 Z3 Z4 Z5 X6 X10 Z11 X12] +
(-3.427323108624411e-07+0j) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12] +
(-0.0008145313270956668+0j) [X2 Z3 Z4 Z5 X6 Z10] +
(2.7455184000411633e-06+0j) [X2 Z3 Z4 Z5 X6 X11 Z12 X13] +
(2.7455184000411633e-06+0j) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13] +
(0.014411099430130874+0j) [X2 Z3 Z4 Z5 X6 Z11] +
(0.0006650070219499397+0j) [X2 Z3 Z4 Z5 X6 Z12] +
(-0.0034937903598901325+0j) [X2 Z3 Z4 Z5 X6 Z13] +
(-4.5614471795805214e-07+0j) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12] +
(-0.01175601341981923+0j) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9] +
(0.01522563075722654+0j) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11] +
(-3.0882507109036053e-06+0j) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(-3.544395428861657e-06+0j) [X2 Z3 Z4 Z5 Z6 X7 X11 X12] +
(-0.004158797381840071+0j) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13] +
(0.01175601341981923+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9] +
(-0.01522563075722654+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11] +
(3.0882507109036053e-06+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(-3.544395428861657e-06+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12] +
(0.004158797381840071+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13] +
(0.014603704729162056+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(-2.8742990710205646e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.014603704729162056+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(-2.8742990710205646e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(-0.2816425776702271+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-1.3002946560810144e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(-1.3002946560810144e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(-0.024282117354692864+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(-0.019538050311314628+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(-0.01709155315589878+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(0.0024464971554158505+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(-0.0024464971554158505+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(5.7759505264483675e-05+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(2.883676575650813e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(5.1464963268510794e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(3.846201670770064e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(-0.03935916802205294+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10] +
(7.979825792401739e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(-0.024755463292890884+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10] +
(5.1055267213811745e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(-0.021433810721600756+0j) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10] +
(5.159350501322191e-06+0j) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(-0.02990378951262474+0j) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10] +
(5.427988655768547e-06+0j) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0016638798784907787+0j) [X2 Z3 Z4 X6] +
(-0.01888903030494285+0j) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10] +
(2.947356011239427e-06+0j) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.00347951189033436+0j) [X2 Z3 Z5 X6] +
(-0.02873077955190547+0j) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10] +
(5.935867717341232e-06+0j) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.6021167406246093e-06+0j) [X2 X4] +
(0.0004956762314915653+0j) [X2 Z4 Z5 X6] +
(-0.03560837898831257+0j) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(7.253273347240699e-06+0j) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.03583956795335344+0j) [Y2 X3 X4 Y5] +
(2.199051618018829e-07+0j) [Y2 X3 X4 Z5 Z6 Y7] +
(2.360956320043574e-06+0j) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.010311482489831894+0j) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.199051618018829e-07+0j) [Y2 X3 X5 Y6] +
(-2.360956320043574e-06+0j) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.010311482489831894+0j) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.031143817988967173+0j) [Y2 X3 X6 Y7] +
(-0.005368659358109611+0j) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11] +
(-9.209350652291171e-08+0j) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005368659358109611+0j) [Y2 X3 X7 Z8 Z9 Y10] +
(9.209350652291171e-08+0j) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12] +
(0.0361941235590427+0j) [Y2 X3 X8 Y9] +
(0.025384657508457316+0j) [Y2 X3 X10 Y11] +
(-2.1726691012502567e-06+0j) [Y2 X3 X10 Z11 Z12 Y13] +
(2.172669101250257e-06+0j) [Y2 X3 X11 Y12] +
(0.015577208063976469+0j) [Y2 X3 X12 Y13] +
(-0.03583956795335344+0j) [Y2 Y3 X4 X5] +
(-2.199051618018829e-07+0j) [Y2 Y3 X4 Z5 Z6 X7] +
(-2.360956320043574e-06+0j) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.010311482489831894+0j) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.199051618018829e-07+0j) [Y2 Y3 Y5 Y6] +
(-2.360956320043574e-06+0j) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.010311482489831894+0j) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.031143817988967173+0j) [Y2 Y3 X6 X7] +
(0.005368659358109611+0j) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11] +
(9.209350652291171e-08+0j) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005368659358109611+0j) [Y2 Y3 Y7 Z8 Z9 Y10] +
(9.209350652291171e-08+0j) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.0361941235590427+0j) [Y2 Y3 X8 X9] +
(-0.025384657508457316+0j) [Y2 Y3 X10 X11] +
(2.1726691012502567e-06+0j) [Y2 Y3 X10 Z11 Z12 X13] +
(2.172669101250257e-06+0j) [Y2 Y3 Y11 Y12] +
(-0.015577208063976469+0j) [Y2 Y3 X12 X13] +
(1.6288532433231053e-06+0j) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10] +
(0.010715508469796783+0j) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.010263414868158497+0j) [Y2 Z3 X4 X10 Z11 Y12] +
(-3.887051672420326e-06+0j) [Y2 Z3 Y4] +
(-0.005143391768825138+0j) [Y2 Z3 Y4 X5 Z6 X7] +
(-0.009841749246962623+0j) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.988511706101805e-06+0j) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.005143391768825138+0j) [Y2 Z3 Y4 Y5 Z6 Y7] +
(-0.009841749246962623+0j) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.988511706101805e-06+0j) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.764994118334913e-07+0j) [Y2 Z3 Y4 Z5] +
(4.537178095352438e-08+0j) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10] +
(4.205548411217964e-05+0j) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.6893489512604343e-06+0j) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10] +
(0.01075756395390896+0j) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.59353439138071e-07+0j) [Y2 Z3 Y4 Z6] +
(3.2118420187238283e-06+0j) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11] +
(0.019299560579363776+0j) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.2118420187238283e-06+0j) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11] +
(0.019299560579363776+0j) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.1954890098621335e-06+0j) [Y2 Z3 Y4 Z7] +
(2.1868423777396504e-07+0j) [Y2 Z3 Y4 Z8] +
(-5.770052994438331e-07+0j) [Y2 Z3 Y4 Z9] +
(0.005324835234221699+0j) [Y2 Z3 Y4 X10 Z11 X12] +
(0.0155882501023802+0j) [Y2 Z3 Y4 Y10 Z11 Y12] +
(-3.158656431607071e-06+0j) [Y2 Z3 Y4 Z10] +
(0.024353077678068904+0j) [Y2 Z3 Y4 X11 Z12 X13] +
(0.024353077678068904+0j) [Y2 Z3 Y4 Y11 Z12 Y13] +
(-7.801707499546218e-06+0j) [Y2 Z3 Y4 Z11] +
(3.5390541841113903e-06+0j) [Y2 Z3 Y4 Z12] +
(8.814937305600543e-06+0j) [Y2 Z3 Y4 Z13] +
(1.4548424490002047e-06+0j) [Y2 Z3 Z4 X5 X6 Y7] +
(3.151346310786498e-06+0j) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11] +
(0.0192575050952516+0j) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.522493067463393e-06+0j) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10] +
(-0.008541996625454816+0j) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(7.956895372177981e-07+0j) [Y2 Z3 Z4 X5 X8 Y9] +
(4.643051067939146e-06+0j) [Y2 Z3 Z4 X5 X10 Y11] +
(0.019028242443847203+0j) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13] +
(-0.008764827575688706+0j) [Y2 Z3 Z4 X5 X11 Y12] +
(-5.275883121489154e-06+0j) [Y2 Z3 Z4 X5 X12 Y13] +
(-1.4548424490002047e-06+0j) [Y2 Z3 Z4 Y5 X6 X7] +
(-3.151346310786498e-06+0j) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11] +
(-0.0192575050952516+0j) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.522493067463393e-06+0j) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10] +
(-0.008541996625454816+0j) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895372177981e-07+0j) [Y2 Z3 Z4 Y5 X8 X9] +
(-4.643051067939146e-06+0j) [Y2 Z3 Z4 Y5 X10 X11] +
(-0.019028242443847203+0j) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13] +
(-0.008764827575688706+0j) [Y2 Z3 Z4 Y5 Y11 Y12] +
(5.275883121489154e-06+0j) [Y2 Z3 Z4 Y5 X12 X13] +
(-4.5614471795805214e-07+0j) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12] +
(-0.121332769110423+0j) [Y2 Z3 Z4 Z5 Y6] +
(-0.008469978791023992+0j) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(2.6863815444635537e-07+0j) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.008469978791023992+0j) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(2.6863815444635537e-07+0j) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.02599617759802111+0j) [Y2 Z3 Z4 Z5 Y6 Z7] +
(-0.005805188989826928+0j) [Y2 Z3 Z4 Z5 Y6 Z8] +
(-0.01756120240964616+0j) [Y2 Z3 Z4 Z5 Y6 Z9] +
(-3.427323108624411e-07+0j) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12] +
(-7.988770288204933e-07+0j) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12] +
(-0.0008145313270956668+0j) [Y2 Z3 Z4 Z5 Y6 Z10] +
(2.7455184000411633e-06+0j) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13] +
(2.7455184000411633e-06+0j) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13] +
(0.014411099430130874+0j) [Y2 Z3 Z4 Z5 Y6 Z11] +
(0.0006650070219499397+0j) [Y2 Z3 Z4 Z5 Y6 Z12] +
(-0.0034937903598901325+0j) [Y2 Z3 Z4 Z5 Y6 Z13] +
(0.01175601341981923+0j) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9] +
(-0.01522563075722654+0j) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11] +
(3.0882507109036053e-06+0j) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(-3.544395428861657e-06+0j) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12] +
(0.004158797381840071+0j) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13] +
(-0.01175601341981923+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9] +
(0.01522563075722654+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11] +
(-3.0882507109036053e-06+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(-3.544395428861657e-06+0j) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12] +
(-0.004158797381840071+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13] +
(0.014603704729162056+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(-2.8742990710205646e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.014603704729162056+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(-2.8742990710205646e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(-0.2816425776702271+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-1.3002946560810144e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(-1.3002946560810144e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(-0.024282117354692864+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(-0.019538050311314628+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(-0.01709155315589878+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(-0.0024464971554158505+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(0.0024464971554158505+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(5.7759505264483675e-05+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.883676575650813e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(5.1464963268510794e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(3.846201670770064e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(-0.03935916802205294+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10] +
(7.979825792401739e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(-0.024755463292890884+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10] +
(5.1055267213811745e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(-0.021433810721600756+0j) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10] +
(5.159350501322191e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(-0.02990378951262474+0j) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10] +
(5.427988655768547e-06+0j) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0016638798784907787+0j) [Y2 Z3 Z4 Y6] +
(-0.01888903030494285+0j) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10] +
(2.947356011239427e-06+0j) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.00347951189033436+0j) [Y2 Z3 Z5 Y6] +
(-0.02873077955190547+0j) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10] +
(5.935867717341232e-06+0j) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.6021167406246093e-06+0j) [Y2 Y4] +
(0.0004956762314915653+0j) [Y2 Z4 Z5 Y6] +
(-0.03560837898831257+0j) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(7.253273347240699e-06+0j) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.6538942226831732+0j) [Z2] +
(1.6021167406246093e-06+0j) [Z2 X3 Z4 X5] +
(0.0004956762314915653+0j) [Z2 X3 Z4 Z5 Z6 X7] +
(-0.03560837898831257+0j) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(7.253273347240699e-06+0j) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.6021167406246093e-06+0j) [Z2 Y3 Z4 Y5] +
(0.0004956762314915653+0j) [Z2 Y3 Z4 Z5 Z6 Y7] +
(-0.03560837898831257+0j) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(7.253273347240699e-06+0j) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.18189085790751408+0j) [Z2 Z3] +
(-9.509249751672286e-07+0j) [Z2 X4 Z5 X6] +
(-4.72884314673879e-06+0j) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.024591860883830013+0j) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-9.509249751672286e-07+0j) [Z2 Y4 Z5 Y6] +
(-4.72884314673879e-06+0j) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.024591860883830013+0j) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.12495807739503235+0j) [Z2 Z4] +
(-1.1708301369691115e-06+0j) [Z2 X5 Z6 X7] +
(-7.089799466782364e-06+0j) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0349033433736619+0j) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.1708301369691115e-06+0j) [Z2 Y5 Z6 Y7] +
(-7.089799466782364e-06+0j) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0349033433736619+0j) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.1607976453483858+0j) [Z2 Z5] +
(0.01902042317303998+0j) [Z2 X6 Z7 Z8 Z9 X10] +
(-2.1032156043281722e-06+0j) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.01902042317303998+0j) [Z2 Y6 Z7 Z8 Z9 Y10] +
(-2.1032156043281722e-06+0j) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.13739104762683238+0j) [Z2 Z6] +
(0.02438908253114959+0j) [Z2 X7 Z8 Z9 Z10 X11] +
(-2.0111220978052604e-06+0j) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.02438908253114959+0j) [Z2 Y7 Z8 Z9 Z10 Y11] +
(-2.0111220978052604e-06+0j) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.16853486561579956+0j) [Z2 Z7] +
(0.150714081210083+0j) [Z2 Z8] +
(0.18690820476912573+0j) [Z2 Z9] +
(-1.0632283420986983e-06+0j) [Z2 X10 Z11 X12] +
(-1.0632283420986983e-06+0j) [Z2 Y10 Z11 Y12] +
(0.12799502492468418+0j) [Z2 Z10] +
(1.1094407591515586e-06+0j) [Z2 X11 Z12 X13] +
(1.1094407591515586e-06+0j) [Z2 Y11 Z12 Y13] +
(0.15337968243314148+0j) [Z2 Z11] +
(0.14011289865354828+0j) [Z2 Z12] +
(0.15569010671752476+0j) [Z2 Z13] +
(0.005143391768825139+0j) [X3 X4 Y5 Y6] +
(0.009841749246962623+0j) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10] +
(-2.988511706101805e-06+0j) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.454842449000205e-06+0j) [X3 X4 X6 X7] +
(-1.522493067463393e-06+0j) [X3 X4 X6 Z7 Z8 Z9 Z10 X11] +
(-0.008541996625454816+0j) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.151346310786498e-06+0j) [X3 X4 Y7 Z8 Z9 Y10] +
(-0.0192575050952516+0j) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895372177981e-07+0j) [X3 X4 X8 X9] +
(-4.643051067939147e-06+0j) [X3 X4 X10 X11] +
(-0.008764827575688706+0j) [X3 X4 X10 Z11 Z12 X13] +
(-0.019028242443847203+0j) [X3 X4 Y11 Y12] +
(5.275883121489154e-06+0j) [X3 X4 X12 X13] +
(-0.005143391768825139+0j) [X3 Y4 Y5 X6] +
(-0.009841749246962623+0j) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10] +
(2.988511706101805e-06+0j) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.454842449000205e-06+0j) [X3 Y4 Y6 X7] +
(-1.522493067463393e-06+0j) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11] +
(-0.008541996625454816+0j) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.151346310786498e-06+0j) [X3 Y4 Y7 Z8 Z9 X10] +
(0.0192575050952516+0j) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12] +
(-7.956895372177981e-07+0j) [X3 Y4 Y8 X9] +
(-4.643051067939147e-06+0j) [X3 Y4 Y10 X11] +
(-0.008764827575688706+0j) [X3 Y4 Y10 Z11 Z12 X13] +
(0.019028242443847203+0j) [X3 Y4 Y11 X12] +
(5.275883121489154e-06+0j) [X3 Y4 Y12 X13] +
(-3.887051672420328e-06+0j) [X3 Z4 X5] +
(3.2118420187238283e-06+0j) [X3 Z4 X5 X6 Z7 Z8 Z9 X10] +
(0.019299560579363776+0j) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.2118420187238283e-06+0j) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10] +
(0.019299560579363776+0j) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.1954890098621335e-06+0j) [X3 Z4 X5 Z6] +
(1.6893489512604343e-06+0j) [X3 Z4 X5 X7 Z8 Z9 Z10 X11] +
(0.01075756395390896+0j) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.537178095352438e-08+0j) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11] +
(4.205548411217964e-05+0j) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.59353439138071e-07+0j) [X3 Z4 X5 Z7] +
(-5.770052994438331e-07+0j) [X3 Z4 X5 Z8] +
(2.1868423777396504e-07+0j) [X3 Z4 X5 Z9] +
(0.024353077678068904+0j) [X3 Z4 X5 X10 Z11 X12] +
(0.024353077678068904+0j) [X3 Z4 X5 Y10 Z11 Y12] +
(-7.801707499546218e-06+0j) [X3 Z4 X5 Z10] +
(0.0155882501023802+0j) [X3 Z4 X5 X11 Z12 X13] +
(0.005324835234221699+0j) [X3 Z4 X5 Y11 Z12 Y13] +
(-3.158656431607071e-06+0j) [X3 Z4 X5 Z11] +
(8.814937305600543e-06+0j) [X3 Z4 X5 Z12] +
(3.5390541841113903e-06+0j) [X3 Z4 X5 Z13] +
(1.6288532433231053e-06+0j) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11] +
(0.010715508469796783+0j) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.010263414868158497+0j) [X3 Z4 Y5 Y11 Z12 X13] +
(0.008469978791023992+0j) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10] +
(-2.6863815444635537e-07+0j) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.01175601341981923+0j) [X3 Z4 Z5 X6 X8 X9] +
(0.01522563075722654+0j) [X3 Z4 Z5 X6 X10 X11] +
(-3.544395428861657e-06+0j) [X3 Z4 Z5 X6 X10 Z11 Z12 X13] +
(-3.0882507109036053e-06+0j) [X3 Z4 Z5 X6 Y11 Y12] +
(-0.004158797381840071+0j) [X3 Z4 Z5 X6 X12 X13] +
(-0.008469978791023992+0j) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10] +
(2.6863815444635537e-07+0j) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.01175601341981923+0j) [X3 Z4 Z5 Y6 Y8 X9] +
(0.01522563075722654+0j) [X3 Z4 Z5 Y6 Y10 X11] +
(-3.544395428861657e-06+0j) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13] +
(3.0882507109036053e-06+0j) [X3 Z4 Z5 Y6 Y11 X12] +
(-0.004158797381840071+0j) [X3 Z4 Z5 Y6 Y12 X13] +
(-0.12133276911042304+0j) [X3 Z4 Z5 Z6 X7] +
(-0.01756120240964616+0j) [X3 Z4 Z5 Z6 X7 Z8] +
(-0.005805188989826928+0j) [X3 Z4 Z5 Z6 X7 Z9] +
(2.7455184000411633e-06+0j) [X3 Z4 Z5 Z6 X7 X10 Z11 X12] +
(2.7455184000411633e-06+0j) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12] +
(0.014411099430130874+0j) [X3 Z4 Z5 Z6 X7 Z10] +
(-7.988770288204933e-07+0j) [X3 Z4 Z5 Z6 X7 X11 Z12 X13] +
(-3.427323108624411e-07+0j) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13] +
(-0.0008145313270956668+0j) [X3 Z4 Z5 Z6 X7 Z11] +
(-0.0034937903598901325+0j) [X3 Z4 Z5 Z6 X7 Z12] +
(0.0006650070219499397+0j) [X3 Z4 Z5 Z6 X7 Z13] +
(-4.5614471795805214e-07+0j) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13] +
(-0.014603704729162056+0j) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10] +
(2.8742990710205646e-06+0j) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(0.014603704729162056+0j) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10] +
(-2.8742990710205646e-06+0j) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(1.3002946560810144e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12] +
(0.00244649715541585+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-1.3002946560810144e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12] +
(0.00244649715541585+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(-0.281642577670227+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.01709155315589878+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(-0.019538050311314628+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(5.7759505264483654e-05+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(2.8836765756508134e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(3.846201670770064e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(-0.02428211735469286+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11] +
(5.1464963268510794e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(-0.024755463292890884+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11] +
(5.1055267213811745e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(-0.03935916802205294+0j) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11] +
(7.979825792401739e-06+0j) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(-0.02990378951262474+0j) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11] +
(5.427988655768547e-06+0j) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.025996177598021107+0j) [X3 Z4 Z5 X7] +
(-0.021433810721600756+0j) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11] +
(5.159350501322191e-06+0j) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.00347951189033436+0j) [X3 Z4 Z6 X7] +
(-0.02873077955190547+0j) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11] +
(5.935867717341232e-06+0j) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.764994118334912e-07+0j) [X3 X5] +
(0.0016638798784907787+0j) [X3 Z5 Z6 X7] +
(-0.01888903030494285+0j) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.947356011239427e-06+0j) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.005143391768825139+0j) [Y3 X4 X5 Y6] +
(-0.009841749246962623+0j) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10] +
(2.988511706101805e-06+0j) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.454842449000205e-06+0j) [Y3 X4 X6 Y7] +
(-1.522493067463393e-06+0j) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11] +
(-0.008541996625454816+0j) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.151346310786498e-06+0j) [Y3 X4 X7 Z8 Z9 Y10] +
(0.0192575050952516+0j) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895372177981e-07+0j) [Y3 X4 X8 Y9] +
(-4.643051067939147e-06+0j) [Y3 X4 X10 Y11] +
(-0.008764827575688706+0j) [Y3 X4 X10 Z11 Z12 Y13] +
(0.019028242443847203+0j) [Y3 X4 X11 Y12] +
(5.275883121489154e-06+0j) [Y3 X4 X12 Y13] +
(0.005143391768825139+0j) [Y3 Y4 X5 X6] +
(0.009841749246962623+0j) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10] +
(-2.988511706101805e-06+0j) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.454842449000205e-06+0j) [Y3 Y4 Y6 Y7] +
(-1.522493067463393e-06+0j) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11] +
(-0.008541996625454816+0j) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.151346310786498e-06+0j) [Y3 Y4 X7 Z8 Z9 X10] +
(-0.0192575050952516+0j) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12] +
(-7.956895372177981e-07+0j) [Y3 Y4 Y8 Y9] +
(-4.643051067939147e-06+0j) [Y3 Y4 Y10 Y11] +
(-0.008764827575688706+0j) [Y3 Y4 Y10 Z11 Z12 Y13] +
(-0.019028242443847203+0j) [Y3 Y4 X11 X12] +
(5.275883121489154e-06+0j) [Y3 Y4 Y12 Y13] +
(1.6288532433231053e-06+0j) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11] +
(0.010715508469796783+0j) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.010263414868158497+0j) [Y3 Z4 X5 X11 Z12 Y13] +
(-3.887051672420328e-06+0j) [Y3 Z4 Y5] +
(3.2118420187238283e-06+0j) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10] +
(0.019299560579363776+0j) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.2118420187238283e-06+0j) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10] +
(0.019299560579363776+0j) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.1954890098621335e-06+0j) [Y3 Z4 Y5 Z6] +
(4.537178095352438e-08+0j) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11] +
(4.205548411217964e-05+0j) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.6893489512604343e-06+0j) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11] +
(0.01075756395390896+0j) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.59353439138071e-07+0j) [Y3 Z4 Y5 Z7] +
(-5.770052994438331e-07+0j) [Y3 Z4 Y5 Z8] +
(2.1868423777396504e-07+0j) [Y3 Z4 Y5 Z9] +
(0.024353077678068904+0j) [Y3 Z4 Y5 X10 Z11 X12] +
(0.024353077678068904+0j) [Y3 Z4 Y5 Y10 Z11 Y12] +
(-7.801707499546218e-06+0j) [Y3 Z4 Y5 Z10] +
(0.005324835234221699+0j) [Y3 Z4 Y5 X11 Z12 X13] +
(0.0155882501023802+0j) [Y3 Z4 Y5 Y11 Z12 Y13] +
(-3.158656431607071e-06+0j) [Y3 Z4 Y5 Z11] +
(8.814937305600543e-06+0j) [Y3 Z4 Y5 Z12] +
(3.5390541841113903e-06+0j) [Y3 Z4 Y5 Z13] +
(-0.008469978791023992+0j) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10] +
(2.6863815444635537e-07+0j) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.01175601341981923+0j) [Y3 Z4 Z5 X6 X8 Y9] +
(0.01522563075722654+0j) [Y3 Z4 Z5 X6 X10 Y11] +
(-3.544395428861657e-06+0j) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13] +
(3.0882507109036053e-06+0j) [Y3 Z4 Z5 X6 X11 Y12] +
(-0.004158797381840071+0j) [Y3 Z4 Z5 X6 X12 Y13] +
(0.008469978791023992+0j) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10] +
(-2.6863815444635537e-07+0j) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-0.01175601341981923+0j) [Y3 Z4 Z5 Y6 Y8 Y9] +
(0.01522563075722654+0j) [Y3 Z4 Z5 Y6 Y10 Y11] +
(-3.544395428861657e-06+0j) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13] +
(-3.0882507109036053e-06+0j) [Y3 Z4 Z5 Y6 X11 X12] +
(-0.004158797381840071+0j) [Y3 Z4 Z5 Y6 Y12 Y13] +
(-4.5614471795805214e-07+0j) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13] +
(-0.12133276911042304+0j) [Y3 Z4 Z5 Z6 Y7] +
(-0.01756120240964616+0j) [Y3 Z4 Z5 Z6 Y7 Z8] +
(-0.005805188989826928+0j) [Y3 Z4 Z5 Z6 Y7 Z9] +
(2.7455184000411633e-06+0j) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12] +
(2.7455184000411633e-06+0j) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12] +
(0.014411099430130874+0j) [Y3 Z4 Z5 Z6 Y7 Z10] +
(-3.427323108624411e-07+0j) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13] +
(-7.988770288204933e-07+0j) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13] +
(-0.0008145313270956668+0j) [Y3 Z4 Z5 Z6 Y7 Z11] +
(-0.0034937903598901325+0j) [Y3 Z4 Z5 Z6 Y7 Z12] +
(0.0006650070219499397+0j) [Y3 Z4 Z5 Z6 Y7 Z13] +
(0.014603704729162056+0j) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10] +
(-2.8742990710205646e-06+0j) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-0.014603704729162056+0j) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10] +
(2.8742990710205646e-06+0j) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-1.3002946560810144e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12] +
(0.00244649715541585+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(1.3002946560810144e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12] +
(0.00244649715541585+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(-0.281642577670227+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.01709155315589878+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(-0.019538050311314628+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(5.7759505264483654e-05+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.8836765756508134e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(3.846201670770064e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(-0.02428211735469286+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11] +
(5.1464963268510794e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(-0.024755463292890884+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11] +
(5.1055267213811745e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(-0.03935916802205294+0j) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11] +
(7.979825792401739e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(-0.02990378951262474+0j) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11] +
(5.427988655768547e-06+0j) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.025996177598021107+0j) [Y3 Z4 Z5 Y7] +
(-0.021433810721600756+0j) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11] +
(5.159350501322191e-06+0j) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.00347951189033436+0j) [Y3 Z4 Z6 Y7] +
(-0.02873077955190547+0j) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11] +
(5.935867717341232e-06+0j) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.764994118334912e-07+0j) [Y3 Y5] +
(0.0016638798784907787+0j) [Y3 Z5 Z6 Y7] +
(-0.01888903030494285+0j) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.947356011239427e-06+0j) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.653894222683173+0j) [Z3] +
(-1.1708301369691115e-06+0j) [Z3 X4 Z5 X6] +
(-7.089799466782364e-06+0j) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0349033433736619+0j) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.1708301369691115e-06+0j) [Z3 Y4 Z5 Y6] +
(-7.089799466782364e-06+0j) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0349033433736619+0j) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.1607976453483858+0j) [Z3 Z4] +
(-9.509249751672286e-07+0j) [Z3 X5 Z6 X7] +
(-4.72884314673879e-06+0j) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.024591860883830013+0j) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-9.509249751672286e-07+0j) [Z3 Y5 Z6 Y7] +
(-4.72884314673879e-06+0j) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.024591860883830013+0j) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.12495807739503235+0j) [Z3 Z5] +
(0.02438908253114959+0j) [Z3 X6 Z7 Z8 Z9 X10] +
(-2.0111220978052604e-06+0j) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.02438908253114959+0j) [Z3 Y6 Z7 Z8 Z9 Y10] +
(-2.0111220978052604e-06+0j) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.16853486561579956+0j) [Z3 Z6] +
(0.01902042317303998+0j) [Z3 X7 Z8 Z9 Z10 X11] +
(-2.1032156043281722e-06+0j) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.01902042317303998+0j) [Z3 Y7 Z8 Z9 Z10 Y11] +
(-2.1032156043281722e-06+0j) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.13739104762683238+0j) [Z3 Z7] +
(0.18690820476912573+0j) [Z3 Z8] +
(0.150714081210083+0j) [Z3 Z9] +
(1.1094407591515586e-06+0j) [Z3 X10 Z11 X12] +
(1.1094407591515586e-06+0j) [Z3 Y10 Z11 Y12] +
(0.15337968243314148+0j) [Z3 Z10] +
(-1.0632283420986983e-06+0j) [Z3 X11 Z12 X13] +
(-1.0632283420986983e-06+0j) [Z3 Y11 Z12 Y13] +
(0.12799502492468418+0j) [Z3 Z11] +
(0.15569010671752476+0j) [Z3 Z12] +
(0.14011289865354828+0j) [Z3 Z13] +
(-0.011982389010247977+0j) [X4 X5 Y6 Y7] +
(-0.007306759928832988+0j) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11] +
(-2.888293596241997e-07+0j) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.007306759928832987+0j) [X4 X5 X7 Z8 Z9 X10] +
(-2.888293596241997e-07+0j) [X4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-0.007156934919856944+0j) [X4 X5 Y8 Y9] +
(-0.01768006795248153+0j) [X4 X5 Y10 Y11] +
(-3.6945132940652373e-06+0j) [X4 X5 Y10 Z11 Z12 Y13] +
(-3.694513294065237e-06+0j) [X4 X5 X11 X12] +
(0.011982389010247977+0j) [X4 Y5 Y6 X7] +
(0.007306759928832988+0j) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11] +
(2.888293596241997e-07+0j) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.007306759928832987+0j) [X4 Y5 Y7 Z8 Z9 X10] +
(-2.888293596241997e-07+0j) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(0.007156934919856944+0j) [X4 Y5 Y8 X9] +
(0.01768006795248153+0j) [X4 Y5 Y10 X11] +
(3.6945132940652373e-06+0j) [X4 Y5 Y10 Z11 Z12 X13] +
(-3.694513294065237e-06+0j) [X4 Y5 Y11 X12] +
(-1.226048498846592e-05+0j) [X4 Z5 X6] +
(-1.228333782439862e-06+0j) [X4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(0.00024636437569574036+0j) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.228333782439862e-06+0j) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(0.00024636437569574036+0j) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.854060857906315e-06+0j) [X4 Z5 X6 Z7] +
(-1.398044908078433e-06+0j) [X4 Z5 X6 Z8] +
(-1.8818501831499925e-06+0j) [X4 Z5 X6 Z9] +
(0.00796088072592158+0j) [X4 Z5 X6 X10 Z11 X12] +
(-0.0009298507967730376+0j) [X4 Z5 X6 Y10 Z11 Y12] +
(-1.6923978284327481e-06+0j) [X4 Z5 X6 Z10] +
(-0.012215040997614003+0j) [X4 Z5 X6 X11 Z12 X13] +
(-0.012215040997614003+0j) [X4 Z5 X6 Y11 Z12 Y13] +
(4.281913884343135e-06+0j) [X4 Z5 X6 Z11] +
(-4.588855155208364e-06+0j) [X4 Z5 X6 Z13] +
(0.008890731522694616+0j) [X4 Z5 Y6 Y10 Z11 X12] +
(-4.838052750715597e-07+0j) [X4 Z5 Z6 X7 Y8 Y9] +
(5.974311712775884e-06+0j) [X4 Z5 Z6 X7 Y10 Y11] +
(0.011285190200840966+0j) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(0.02017592172353558+0j) [X4 Z5 Z6 X7 X11 X12] +
(-4.556569217606453e-06+0j) [X4 Z5 Z6 X7 Y12 Y13] +
(4.838052750715597e-07+0j) [X4 Z5 Z6 Y7 Y8 X9] +
(-5.974311712775884e-06+0j) [X4 Z5 Z6 Y7 Y10 X11] +
(-0.011285190200840966+0j) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(0.02017592172353558+0j) [X4 Z5 Z6 Y7 Y11 X12] +
(4.556569217606453e-06+0j) [X4 Z5 Z6 Y7 Y12 X13] +
(1.3304731885404095e-06+0j) [X4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(0.005923798336561342+0j) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(1.3304731885404095e-06+0j) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(0.005923798336561342+0j) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(-6.631277927681174e-05+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.016024603689179587+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(-0.016024603689179587+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(3.3343312890108305e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(-4.734622038207982e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(-9.806102774070867e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(-5.071480735862885e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(5.071480735862885e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-0.36937089366156106+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.02314513092952902+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-0.009612634606847232+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(-0.02563723829602682+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(-8.77481786361013e-06+0j) [X4 Z5 Z6 Z7 Z8 X10] +
(-0.04764261217638307+0j) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(-7.4443446750697205e-06+0j) [X4 Z5 Z6 Z7 Z9 X10] +
(-0.041718813839821726+0j) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(-6.290028432436617e-06+0j) [X4 Z5 Z6 Z8 Z9 X10] +
(-0.039564416322893245+0j) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(-7.518362214876479e-06+0j) [X4 Z5 Z7 Z8 Z9 X10] +
(-0.0393180519471975+0j) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(-5.929765815576792e-07+0j) [X4 X6] +
(-4.253224225104777e-06+0j) [X4 Z6 Z7 Z8 Z9 X10] +
(-0.022528440196012994+0j) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.011982389010247977+0j) [Y4 X5 X6 Y7] +
(0.007306759928832988+0j) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11] +
(2.888293596241997e-07+0j) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.007306759928832987+0j) [Y4 X5 X7 Z8 Z9 Y10] +
(-2.888293596241997e-07+0j) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(0.007156934919856944+0j) [Y4 X5 X8 Y9] +
(0.01768006795248153+0j) [Y4 X5 X10 Y11] +
(3.6945132940652373e-06+0j) [Y4 X5 X10 Z11 Z12 Y13] +
(-3.694513294065237e-06+0j) [Y4 X5 X11 Y12] +
(-0.011982389010247977+0j) [Y4 Y5 X6 X7] +
(-0.007306759928832988+0j) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11] +
(-2.888293596241997e-07+0j) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.007306759928832987+0j) [Y4 Y5 Y7 Z8 Z9 Y10] +
(-2.888293596241997e-07+0j) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.007156934919856944+0j) [Y4 Y5 X8 X9] +
(-0.01768006795248153+0j) [Y4 Y5 X10 X11] +
(-3.6945132940652373e-06+0j) [Y4 Y5 X10 Z11 Z12 X13] +
(-3.694513294065237e-06+0j) [Y4 Y5 Y11 Y12] +
(0.008890731522694616+0j) [Y4 Z5 X6 X10 Z11 Y12] +
(-1.226048498846592e-05+0j) [Y4 Z5 Y6] +
(-1.228333782439862e-06+0j) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(0.00024636437569574036+0j) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.228333782439862e-06+0j) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(0.00024636437569574036+0j) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.854060857906315e-06+0j) [Y4 Z5 Y6 Z7] +
(-1.398044908078433e-06+0j) [Y4 Z5 Y6 Z8] +
(-1.8818501831499925e-06+0j) [Y4 Z5 Y6 Z9] +
(-0.0009298507967730376+0j) [Y4 Z5 Y6 X10 Z11 X12] +
(0.00796088072592158+0j) [Y4 Z5 Y6 Y10 Z11 Y12] +
(-1.6923978284327481e-06+0j) [Y4 Z5 Y6 Z10] +
(-0.012215040997614003+0j) [Y4 Z5 Y6 X11 Z12 X13] +
(-0.012215040997614003+0j) [Y4 Z5 Y6 Y11 Z12 Y13] +
(4.281913884343135e-06+0j) [Y4 Z5 Y6 Z11] +
(-4.588855155208364e-06+0j) [Y4 Z5 Y6 Z13] +
(4.838052750715597e-07+0j) [Y4 Z5 Z6 X7 X8 Y9] +
(-5.974311712775884e-06+0j) [Y4 Z5 Z6 X7 X10 Y11] +
(-0.011285190200840966+0j) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(0.02017592172353558+0j) [Y4 Z5 Z6 X7 X11 Y12] +
(4.556569217606453e-06+0j) [Y4 Z5 Z6 X7 X12 Y13] +
(-4.838052750715597e-07+0j) [Y4 Z5 Z6 Y7 X8 X9] +
(5.974311712775884e-06+0j) [Y4 Z5 Z6 Y7 X10 X11] +
(0.011285190200840966+0j) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(0.02017592172353558+0j) [Y4 Z5 Z6 Y7 Y11 Y12] +
(-4.556569217606453e-06+0j) [Y4 Z5 Z6 Y7 X12 X13] +
(1.3304731885404095e-06+0j) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(0.005923798336561342+0j) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(1.3304731885404095e-06+0j) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(0.005923798336561342+0j) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(-6.631277927681174e-05+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.016024603689179587+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(-0.016024603689179587+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(3.3343312890108305e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(-4.734622038207982e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(-9.806102774070867e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(5.071480735862885e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-5.071480735862885e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-0.36937089366156106+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.02314513092952902+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-0.009612634606847232+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(-0.02563723829602682+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(-8.77481786361013e-06+0j) [Y4 Z5 Z6 Z7 Z8 Y10] +
(-0.04764261217638307+0j) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(-7.4443446750697205e-06+0j) [Y4 Z5 Z6 Z7 Z9 Y10] +
(-0.041718813839821726+0j) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(-6.290028432436617e-06+0j) [Y4 Z5 Z6 Z8 Z9 Y10] +
(-0.039564416322893245+0j) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(-7.518362214876479e-06+0j) [Y4 Z5 Z7 Z8 Z9 Y10] +
(-0.0393180519471975+0j) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(-5.929765815576792e-07+0j) [Y4 Y6] +
(-4.253224225104777e-06+0j) [Y4 Z6 Z7 Z8 Z9 Y10] +
(-0.022528440196012994+0j) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.2034402289145636+0j) [Z4] +
(-5.929765815576792e-07+0j) [Z4 X5 Z6 X7] +
(-4.253224225104777e-06+0j) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.022528440196012994+0j) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-5.929765815576792e-07+0j) [Z4 Y5 Z6 Y7] +
(-4.253224225104777e-06+0j) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.022528440196012994+0j) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.15755314797985667+0j) [Z4 Z5] +
(0.018266834869375585+0j) [Z4 X6 Z7 Z8 Z9 X10] +
(-1.6541174767943236e-06+0j) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.018266834869375585+0j) [Z4 Y6 Z7 Z8 Z9 Y10] +
(-1.6541174767943236e-06+0j) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.13701191674040747+0j) [Z4 Z6] +
(0.010960074940542599+0j) [Z4 X7 Z8 Z9 Z10 X11] +
(-1.942946836418523e-06+0j) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.010960074940542599+0j) [Z4 Y7 Z8 Z9 Z10 Y11] +
(-1.942946836418523e-06+0j) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.14899430575065542+0j) [Z4 Z7] +
(0.14960702684445293+0j) [Z4 Z8] +
(0.1567639617643099+0j) [Z4 Z9] +
(1.8782101245281466e-06+0j) [Z4 X10 Z11 X12] +
(1.8782101245281466e-06+0j) [Z4 Y10 Z11 Y12] +
(0.12489990917237588+0j) [Z4 Z10] +
(-1.8163031695370907e-06+0j) [Z4 X11 Z12 X13] +
(-1.8163031695370907e-06+0j) [Z4 Y11 Z12 Y13] +
(0.14257997712485743+0j) [Z4 Z11] +
(0.11383573679388664+0j) [Z4 Z12] +
(0.15215040708869054+0j) [Z4 Z13] +
(1.228333782439862e-06+0j) [X5 X6 Y7 Z8 Z9 Y10] +
(-0.00024636437569574036+0j) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-4.838052750715597e-07+0j) [X5 X6 X8 X9] +
(5.974311712775884e-06+0j) [X5 X6 X10 X11] +
(0.02017592172353558+0j) [X5 X6 X10 Z11 Z12 X13] +
(0.011285190200840966+0j) [X5 X6 Y11 Y12] +
(-4.556569217606453e-06+0j) [X5 X6 X12 X13] +
(-1.228333782439862e-06+0j) [X5 Y6 Y7 Z8 Z9 X10] +
(0.00024636437569574036+0j) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-4.838052750715597e-07+0j) [X5 Y6 Y8 X9] +
(5.974311712775884e-06+0j) [X5 Y6 Y10 X11] +
(0.02017592172353558+0j) [X5 Y6 Y10 Z11 Z12 X13] +
(-0.011285190200840966+0j) [X5 Y6 Y11 X12] +
(-4.556569217606453e-06+0j) [X5 Y6 Y12 X13] +
(-1.2260484988465916e-05+0j) [X5 Z6 X7] +
(-1.8818501831499925e-06+0j) [X5 Z6 X7 Z8] +
(-1.398044908078433e-06+0j) [X5 Z6 X7 Z9] +
(-0.012215040997614003+0j) [X5 Z6 X7 X10 Z11 X12] +
(-0.012215040997614003+0j) [X5 Z6 X7 Y10 Z11 Y12] +
(4.281913884343135e-06+0j) [X5 Z6 X7 Z10] +
(0.00796088072592158+0j) [X5 Z6 X7 X11 Z12 X13] +
(-0.0009298507967730376+0j) [X5 Z6 X7 Y11 Z12 Y13] +
(-1.6923978284327481e-06+0j) [X5 Z6 X7 Z11] +
(-4.588855155208364e-06+0j) [X5 Z6 X7 Z12] +
(0.008890731522694616+0j) [X5 Z6 Y7 Y11 Z12 X13] +
(-1.3304731885404095e-06+0j) [X5 Z6 Z7 X8 Y9 Y10] +
(-0.005923798336561342+0j) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(1.3304731885404095e-06+0j) [X5 Z6 Z7 Y8 Y9 X10] +
(0.005923798336561342+0j) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(0.016024603689179587+0j) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12] +
(-5.071480735862883e-06+0j) [X5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-0.016024603689179587+0j) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12] +
(-5.071480735862883e-06+0j) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(-6.631277927681177e-05+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-9.806102774070867e-06+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(-4.734622038207982e-06+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(-0.36937089366156106+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.02314513092952902+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(-0.02563723829602682+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(3.33433128901083e-06+0j) [X5 Z6 Z7 Z8 Z9 X11] +
(-0.009612634606847232+0j) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(-7.4443446750697205e-06+0j) [X5 Z6 Z7 Z8 Z10 X11] +
(-0.041718813839821726+0j) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(-8.77481786361013e-06+0j) [X5 Z6 Z7 Z9 Z10 X11] +
(-0.04764261217638307+0j) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(-7.518362214876479e-06+0j) [X5 Z6 Z8 Z9 Z10 X11] +
(-0.0393180519471975+0j) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.8540608579063153e-06+0j) [X5 X7] +
(-6.290028432436617e-06+0j) [X5 Z7 Z8 Z9 Z10 X11] +
(-0.039564416322893245+0j) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.228333782439862e-06+0j) [Y5 X6 X7 Z8 Z9 Y10] +
(0.00024636437569574036+0j) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-4.838052750715597e-07+0j) [Y5 X6 X8 Y9] +
(5.974311712775884e-06+0j) [Y5 X6 X10 Y11] +
(0.02017592172353558+0j) [Y5 X6 X10 Z11 Z12 Y13] +
(-0.011285190200840966+0j) [Y5 X6 X11 Y12] +
(-4.556569217606453e-06+0j) [Y5 X6 X12 Y13] +
(1.228333782439862e-06+0j) [Y5 Y6 X7 Z8 Z9 X10] +
(-0.00024636437569574036+0j) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-4.838052750715597e-07+0j) [Y5 Y6 Y8 Y9] +
(5.974311712775884e-06+0j) [Y5 Y6 Y10 Y11] +
(0.02017592172353558+0j) [Y5 Y6 Y10 Z11 Z12 Y13] +
(0.011285190200840966+0j) [Y5 Y6 X11 X12] +
(-4.556569217606453e-06+0j) [Y5 Y6 Y12 Y13] +
(0.008890731522694616+0j) [Y5 Z6 X7 X11 Z12 Y13] +
(-1.2260484988465916e-05+0j) [Y5 Z6 Y7] +
(-1.8818501831499925e-06+0j) [Y5 Z6 Y7 Z8] +
(-1.398044908078433e-06+0j) [Y5 Z6 Y7 Z9] +
(-0.012215040997614003+0j) [Y5 Z6 Y7 X10 Z11 X12] +
(-0.012215040997614003+0j) [Y5 Z6 Y7 Y10 Z11 Y12] +
(4.281913884343135e-06+0j) [Y5 Z6 Y7 Z10] +
(-0.0009298507967730376+0j) [Y5 Z6 Y7 X11 Z12 X13] +
(0.00796088072592158+0j) [Y5 Z6 Y7 Y11 Z12 Y13] +
(-1.6923978284327481e-06+0j) [Y5 Z6 Y7 Z11] +
(-4.588855155208364e-06+0j) [Y5 Z6 Y7 Z12] +
(1.3304731885404095e-06+0j) [Y5 Z6 Z7 X8 X9 Y10] +
(0.005923798336561342+0j) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-1.3304731885404095e-06+0j) [Y5 Z6 Z7 Y8 X9 X10] +
(-0.005923798336561342+0j) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-0.016024603689179587+0j) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12] +
(-5.071480735862883e-06+0j) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(0.016024603689179587+0j) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12] +
(-5.071480735862883e-06+0j) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(-6.631277927681177e-05+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-9.806102774070867e-06+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(-4.734622038207982e-06+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(-0.36937089366156106+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.02314513092952902+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(-0.02563723829602682+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(3.33433128901083e-06+0j) [Y5 Z6 Z7 Z8 Z9 Y11] +
(-0.009612634606847232+0j) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(-7.4443446750697205e-06+0j) [Y5 Z6 Z7 Z8 Z10 Y11] +
(-0.041718813839821726+0j) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(-8.77481786361013e-06+0j) [Y5 Z6 Z7 Z9 Z10 Y11] +
(-0.04764261217638307+0j) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(-7.518362214876479e-06+0j) [Y5 Z6 Z8 Z9 Z10 Y11] +
(-0.0393180519471975+0j) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.8540608579063153e-06+0j) [Y5 Y7] +
(-6.290028432436617e-06+0j) [Y5 Z7 Z8 Z9 Z10 Y11] +
(-0.039564416322893245+0j) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.2034402289145634+0j) [Z5] +
(0.010960074940542599+0j) [Z5 X6 Z7 Z8 Z9 X10] +
(-1.942946836418523e-06+0j) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.010960074940542599+0j) [Z5 Y6 Z7 Z8 Z9 Y10] +
(-1.942946836418523e-06+0j) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.14899430575065542+0j) [Z5 Z6] +
(0.018266834869375585+0j) [Z5 X7 Z8 Z9 Z10 X11] +
(-1.6541174767943236e-06+0j) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.018266834869375585+0j) [Z5 Y7 Z8 Z9 Z10 Y11] +
(-1.6541174767943236e-06+0j) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.13701191674040747+0j) [Z5 Z7] +
(0.1567639617643099+0j) [Z5 Z8] +
(0.14960702684445293+0j) [Z5 Z9] +
(-1.8163031695370907e-06+0j) [Z5 X10 Z11 X12] +
(-1.8163031695370907e-06+0j) [Z5 Y10 Z11 Y12] +
(0.14257997712485743+0j) [Z5 Z10] +
(1.8782101245281466e-06+0j) [Z5 X11 Z12 X13] +
(1.8782101245281466e-06+0j) [Z5 Y11 Z12 Y13] +
(0.12489990917237588+0j) [Z5 Z11] +
(0.15215040708869054+0j) [Z5 Z12] +
(0.11383573679388664+0j) [Z5 Z13] +
(-0.013873381748426063+0j) [X6 X7 Y8 Y9] +
(-0.017825140995786543+0j) [X6 X7 Y10 Y11] +
(-1.0358477600843729e-06+0j) [X6 X7 Y10 Z11 Z12 Y13] +
(-1.0358477600843729e-06+0j) [X6 X7 X11 X12] +
(0.013873381748426063+0j) [X6 Y7 Y8 X9] +
(0.017825140995786543+0j) [X6 Y7 Y10 X11] +
(1.0358477600843729e-06+0j) [X6 Y7 Y10 Z11 Z12 X13] +
(-1.0358477600843729e-06+0j) [X6 Y7 Y11 X12] +
(0.00029219862611101576+0j) [X6 Z7 X8 X9 Z10 X11] +
(-3.3281393502328436e-07+0j) [X6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.00029219862611101576+0j) [X6 Z7 X8 Y9 Z10 Y11] +
(-3.3281393502328436e-07+0j) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(0.22848106564918894+0j) [X6 Z7 Z8 Z9 X10] +
(3.313145499809178e-06+0j) [X6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(3.313145499809178e-06+0j) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(0.011307274008848118+0j) [X6 Z7 Z8 Z9 X10 Z11] +
(0.02510495713884455+0j) [X6 Z7 Z8 Z9 X10 Z12] +
(0.01054042590767153+0j) [X6 Z7 Z8 Z9 X10 Z13] +
(-0.014564531231173022+0j) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(0.014564531231173022+0j) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-2.595086006618798e-05+0j) [X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.183932558964794e-06+0j) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-6.524373847745341e-06+0j) [X6 Z7 Z8 Z9 Z10 X12] +
(-3.211228347936163e-06+0j) [X6 Z7 Z8 Z9 Z11 X12] +
(0.02981242451734583+0j) [X6 Z7 Z8 X10] +
(-3.277483194998421e-06+0j) [X6 Z7 Z8 Z10 Z11 X12] +
(0.030104623143456848+0j) [X6 Z7 Z9 X10] +
(-3.610297130021705e-06+0j) [X6 Z7 Z9 Z10 Z11 X12] +
(0.030787505389143956+0j) [X6 Z8 Z9 X10] +
(-3.7696594514428103e-06+0j) [X6 Z8 Z9 Z10 Z11 X12] +
(0.013873381748426063+0j) [Y6 X7 X8 Y9] +
(0.017825140995786543+0j) [Y6 X7 X10 Y11] +
(1.0358477600843729e-06+0j) [Y6 X7 X10 Z11 Z12 Y13] +
(-1.0358477600843729e-06+0j) [Y6 X7 X11 Y12] +
(-0.013873381748426063+0j) [Y6 Y7 X8 X9] +
(-0.017825140995786543+0j) [Y6 Y7 X10 X11] +
(-1.0358477600843729e-06+0j) [Y6 Y7 X10 Z11 Z12 X13] +
(-1.0358477600843729e-06+0j) [Y6 Y7 Y11 Y12] +
(0.00029219862611101576+0j) [Y6 Z7 Y8 X9 Z10 X11] +
(-3.3281393502328436e-07+0j) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.00029219862611101576+0j) [Y6 Z7 Y8 Y9 Z10 Y11] +
(-3.3281393502328436e-07+0j) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(0.22848106564918894+0j) [Y6 Z7 Z8 Z9 Y10] +
(3.313145499809178e-06+0j) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(3.313145499809178e-06+0j) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(0.011307274008848118+0j) [Y6 Z7 Z8 Z9 Y10 Z11] +
(0.02510495713884455+0j) [Y6 Z7 Z8 Z9 Y10 Z12] +
(0.01054042590767153+0j) [Y6 Z7 Z8 Z9 Y10 Z13] +
(0.014564531231173022+0j) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-0.014564531231173022+0j) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-2.595086006618798e-05+0j) [Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.183932558964794e-06+0j) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-6.524373847745341e-06+0j) [Y6 Z7 Z8 Z9 Z10 Y12] +
(-3.211228347936163e-06+0j) [Y6 Z7 Z8 Z9 Z11 Y12] +
(0.02981242451734583+0j) [Y6 Z7 Z8 Y10] +
(-3.277483194998421e-06+0j) [Y6 Z7 Z8 Z10 Z11 Y12] +
(0.030104623143456848+0j) [Y6 Z7 Z9 Y10] +
(-3.610297130021705e-06+0j) [Y6 Z7 Z9 Z10 Z11 Y12] +
(0.030787505389143956+0j) [Y6 Z8 Z9 Y10] +
(-3.7696594514428103e-06+0j) [Y6 Z8 Z9 Z10 Z11 Y12] +
(1.3096862988615423+0j) [Z6] +
(0.030787505389143956+0j) [Z6 X7 Z8 Z9 Z10 X11] +
(-3.7696594514428103e-06+0j) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.030787505389143956+0j) [Z6 Y7 Z8 Z9 Z10 Y11] +
(-3.7696594514428103e-06+0j) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.19392534613270165+0j) [Z6 Z7] +
(0.16756653265461252+0j) [Z6 Z8] +
(0.18143991440303858+0j) [Z6 Z9] +
(-1.8551201212447217e-06+0j) [Z6 X10 Z11 X12] +
(-1.8551201212447217e-06+0j) [Z6 Y10 Z11 Y12] +
(0.11952438964682653+0j) [Z6 Z10] +
(-2.890967881329095e-06+0j) [Z6 X11 Z12 X13] +
(-2.890967881329095e-06+0j) [Z6 Y11 Z12 Y13] +
(0.13734953064261307+0j) [Z6 Z11] +
(0.134017152619637+0j) [Z6 Z12] +
(0.15138327161428838+0j) [Z6 Z13] +
(-0.0002921986261110157+0j) [X7 X8 Y9 Y10] +
(3.3281393502328436e-07+0j) [X7 X8 Y9 Z10 Z11 Y12] +
(0.0002921986261110157+0j) [X7 Y8 Y9 X10] +
(-3.3281393502328436e-07+0j) [X7 Y8 Y9 Z10 Z11 X12] +
(-3.3131454998091783e-06+0j) [X7 Z8 Z9 X10 Y11 Y12] +
(-0.014564531231173022+0j) [X7 Z8 Z9 X10 X12 X13] +
(3.3131454998091783e-06+0j) [X7 Z8 Z9 Y10 Y11 X12] +
(-0.014564531231173022+0j) [X7 Z8 Z9 Y10 Y12 X13] +
(0.228481065649189+0j) [X7 Z8 Z9 Z10 X11] +
(0.01054042590767153+0j) [X7 Z8 Z9 Z10 X11 Z12] +
(0.02510495713884455+0j) [X7 Z8 Z9 Z10 X11 Z13] +
(-2.5950860066187984e-05+0j) [X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.183932558964794e-06+0j) [X7 Z8 Z9 Z10 Z11 X13] +
(-3.211228347936163e-06+0j) [X7 Z8 Z9 Z10 Z12 X13] +
(0.011307274008848116+0j) [X7 Z8 Z9 X11] +
(-6.524373847745341e-06+0j) [X7 Z8 Z9 Z11 Z12 X13] +
(0.030104623143456848+0j) [X7 Z8 Z10 X11] +
(-3.610297130021705e-06+0j) [X7 Z8 Z10 Z11 Z12 X13] +
(0.02981242451734583+0j) [X7 Z9 Z10 X11] +
(-3.277483194998421e-06+0j) [X7 Z9 Z10 Z11 Z12 X13] +
(0.0002921986261110157+0j) [Y7 X8 X9 Y10] +
(-3.3281393502328436e-07+0j) [Y7 X8 X9 Z10 Z11 Y12] +
(-0.0002921986261110157+0j) [Y7 Y8 X9 X10] +
(3.3281393502328436e-07+0j) [Y7 Y8 X9 Z10 Z11 X12] +
(3.3131454998091783e-06+0j) [Y7 Z8 Z9 X10 X11 Y12] +
(-0.014564531231173022+0j) [Y7 Z8 Z9 X10 X12 Y13] +
(-3.3131454998091783e-06+0j) [Y7 Z8 Z9 Y10 X11 X12] +
(-0.014564531231173022+0j) [Y7 Z8 Z9 Y10 Y12 Y13] +
(0.228481065649189+0j) [Y7 Z8 Z9 Z10 Y11] +
(0.01054042590767153+0j) [Y7 Z8 Z9 Z10 Y11 Z12] +
(0.02510495713884455+0j) [Y7 Z8 Z9 Z10 Y11 Z13] +
(-2.5950860066187984e-05+0j) [Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(4.183932558964794e-06+0j) [Y7 Z8 Z9 Z10 Z11 Y13] +
(-3.211228347936163e-06+0j) [Y7 Z8 Z9 Z10 Z12 Y13] +
(0.011307274008848116+0j) [Y7 Z8 Z9 Y11] +
(-6.524373847745341e-06+0j) [Y7 Z8 Z9 Z11 Z12 Y13] +
(0.030104623143456848+0j) [Y7 Z8 Z10 Y11] +
(-3.610297130021705e-06+0j) [Y7 Z8 Z10 Z11 Z12 Y13] +
(0.02981242451734583+0j) [Y7 Z9 Z10 Y11] +
(-3.277483194998421e-06+0j) [Y7 Z9 Z10 Z11 Z12 Y13] +
(1.3096862988615423+0j) [Z7] +
(0.18143991440303858+0j) [Z7 Z8] +
(0.16756653265461252+0j) [Z7 Z9] +
(-2.890967881329095e-06+0j) [Z7 X10 Z11 X12] +
(-2.890967881329095e-06+0j) [Z7 Y10 Z11 Y12] +
(0.13734953064261307+0j) [Z7 Z10] +
(-1.8551201212447217e-06+0j) [Z7 X11 Z12 X13] +
(-1.8551201212447217e-06+0j) [Z7 Y11 Z12 Y13] +
(0.11952438964682653+0j) [Z7 Z11] +
(0.15138327161428838+0j) [Z7 Z12] +
(0.134017152619637+0j) [Z7 Z13] +
(-0.009560705729135885+0j) [X8 X9 Y10 Y11] +
(6.628614200767354e-07+0j) [X8 X9 Y10 Z11 Z12 Y13] +
(6.628614200767354e-07+0j) [X8 X9 X11 X12] +
(-0.006087822480561855+0j) [X8 X9 Y12 Y13] +
(0.009560705729135885+0j) [X8 Y9 Y10 X11] +
(-6.628614200767354e-07+0j) [X8 Y9 Y10 Z11 Z12 X13] +
(6.628614200767354e-07+0j) [X8 Y9 Y11 X12] +
(0.006087822480561855+0j) [X8 Y9 Y12 X13] +
(0.009560705729135885+0j) [Y8 X9 X10 Y11] +
(-6.628614200767354e-07+0j) [Y8 X9 X10 Z11 Z12 Y13] +
(6.628614200767354e-07+0j) [Y8 X9 X11 Y12] +
(0.006087822480561855+0j) [Y8 X9 X12 Y13] +
(-0.009560705729135885+0j) [Y8 Y9 X10 X11] +
(6.628614200767354e-07+0j) [Y8 Y9 X10 Z11 Z12 X13] +
(6.628614200767354e-07+0j) [Y8 Y9 Y11 Y12] +
(-0.006087822480561855+0j) [Y8 Y9 X12 X13] +
(1.3693525634718178+0j) [Z8] +
(0.2200397733437608+0j) [Z8 Z9] +
(-1.597317197535129e-06+0j) [Z8 X10 Z11 X12] +
(-1.597317197535129e-06+0j) [Z8 Y10 Z11 Y12] +
(0.1376687264585256+0j) [Z8 Z10] +
(-9.344557774583934e-07+0j) [Z8 X11 Z12 X13] +
(-9.344557774583934e-07+0j) [Z8 Y11 Z12 Y13] +
(0.1472294321876615+0j) [Z8 Z11] +
(0.14973486803496927+0j) [Z8 Z12] +
(0.1558226905155311+0j) [Z8 Z13] +
(1.369352563471818+0j) [Z9] +
(-9.344557774583934e-07+0j) [Z9 X10 Z11 X12] +
(-9.344557774583934e-07+0j) [Z9 Y10 Z11 Y12] +
(0.1472294321876615+0j) [Z9 Z10] +
(-1.597317197535129e-06+0j) [Z9 X11 Z12 X13] +
(-1.597317197535129e-06+0j) [Z9 Y11 Z12 Y13] +
(0.1376687264585256+0j) [Z9 Z11] +
(0.1558226905155311+0j) [Z9 Z12] +
(0.14973486803496927+0j) [Z9 Z13] +
(-0.02868518371610592+0j) [X10 X11 Y12 Y13] +
(0.02868518371610592+0j) [X10 Y11 Y12 X13] +
(-1.0722312157091923e-05+0j) [X10 Z11 X12] +
(7.954413175365363e-06+0j) [X10 Z11 X12 Z13] +
(-8.194261371229622e-06+0j) [X10 X12] +
(0.02868518371610592+0j) [Y10 X11 X12 Y13] +
(-0.02868518371610592+0j) [Y10 Y11 X12 X13] +
(-1.0722312157091923e-05+0j) [Y10 Z11 Y12] +
(7.954413175365363e-06+0j) [Y10 Z11 Y12 Z13] +
(-8.194261371229622e-06+0j) [Y10 Y12] +
(0.7829661725950186+0j) [Z10] +
(-8.194261371229622e-06+0j) [Z10 X11 Z12 X13] +
(-8.194261371229622e-06+0j) [Z10 Y11 Z12 Y13] +
(0.1492635514738887+0j) [Z10 Z11] +
(0.11270386920332211+0j) [Z10 Z12] +
(0.14138905291942805+0j) [Z10 Z13] +
(-1.0722312157091927e-05+0j) [X11 Z12 X13] +
(7.954413175365363e-06+0j) [X11 X13] +
(-1.0722312157091927e-05+0j) [Y11 Z12 Y13] +
(7.954413175365363e-06+0j) [Y11 Y13] +
(0.7829661725950187+0j) [Z11] +
(0.14138905291942805+0j) [Z11 Z12] +
(0.11270386920332211+0j) [Z11 Z13] +
(0.8084581961720478+0j) [Z12] +
(0.15435748657223647+0j) [Z12 Z13] +
(0.8084581961720478+0j) [Z13]
  (-46.46390678868899) [I0]
+ (0.7829661725950198) [Z10]
+ (0.7829661725950199) [Z11]
+ (0.8084581961720468) [Z12]
+ (0.8084581961720468) [Z13]
+ (1.2034402289145651) [Z5]
+ (1.2034402289145656) [Z4]
+ (1.3096862988615414) [Z6]
+ (1.3096862988615414) [Z7]
+ (1.3693525634718169) [Z8]
+ (1.3693525634718173) [Z9]
+ (1.6538942226831688) [Z2]
+ (1.653894222683169) [Z3]
+ (12.412630742111787) [Z0]
+ (12.412630742111787) [Z1]
+ (-8.194261372910286e-06) [Y10 Y12]
+ (-8.194261372910286e-06) [X10 X12]
+ (-1.8540608578852472e-06) [Y5 Y7]
+ (-1.8540608578852472e-06) [X5 X7]
+ (-7.76499411844001e-07) [Y3 Y5]
+ (-7.76499411844001e-07) [X3 X5]
+ (-5.929765815456163e-07) [Y4 Y6]
+ (-5.929765815456163e-07) [X4 X6]
+ (1.6021167405529254e-06) [Y2 Y4]
+ (1.6021167405529254e-06) [X2 X4]
+ (7.95441317701788e-06) [Y11 Y13]
+ (7.95441317701788e-06) [X11 X13]
+ (0.0032769719312316656) [Y1 Y3]
+ (0.0032769719312316656) [X1 X3]
+ (0.10433064780651422) [Y0 Y2]
+ (0.10433064780651422) [X0 X2]
+ (0.11270386920332202) [Z10 Z12]
+ (0.11270386920332202) [Z11 Z13]
+ (0.11383573679388642) [Z4 Z12]
+ (0.11383573679388642) [Z5 Z13]
+ (0.1195243896468267) [Z6 Z10]
+ (0.1195243896468267) [Z7 Z11]
+ (0.12489990917237595) [Z4 Z10]
+ (0.12489990917237595) [Z5 Z11]
+ (0.12495807739503195) [Z2 Z4]
+ (0.12495807739503195) [Z3 Z5]
+ (0.12799502492468395) [Z2 Z10]
+ (0.12799502492468395) [Z3 Z11]
+ (0.13401715261963687) [Z6 Z12]
+ (0.13401715261963687) [Z7 Z13]
+ (0.1370119167404074) [Z4 Z6]
+ (0.1370119167404074) [Z5 Z7]
+ (0.13734953064261318) [Z6 Z11]
+ (0.13734953064261318) [Z7 Z10]
+ (0.13739104762683207) [Z2 Z6]
+ (0.13739104762683207) [Z3 Z7]
+ (0.1376687264585257) [Z8 Z10]
+ (0.1376687264585257) [Z9 Z11]
+ (0.1401128986535478) [Z2 Z12]
+ (0.1401128986535478) [Z3 Z13]
+ (0.14138905291942788) [Z10 Z13]
+ (0.14138905291942788) [Z11 Z12]
+ (0.14257997712485748) [Z4 Z11]
+ (0.14257997712485748) [Z5 Z10]
+ (0.14722943218766166) [Z8 Z11]
+ (0.14722943218766166) [Z9 Z10]
+ (0.14899430575065536) [Z4 Z7]
+ (0.14899430575065536) [Z5 Z6]
+ (0.1492635514738889) [Z10 Z11]
+ (0.1496070268444529) [Z4 Z8]
+ (0.1496070268444529) [Z5 Z9]
+ (0.14973486803496908) [Z8 Z12]
+ (0.14973486803496908) [Z9 Z13]
+ (0.1507140812100827) [Z2 Z8]
+ (0.1507140812100827) [Z3 Z9]
+ (0.15138327161428822) [Z6 Z13]
+ (0.15138327161428822) [Z7 Z12]
+ (0.15215040708869026) [Z4 Z13]
+ (0.15215040708869026) [Z5 Z12]
+ (0.15337968243314126) [Z2 Z11]
+ (0.15337968243314126) [Z3 Z10]
+ (0.15435748657223597) [Z12 Z13]
+ (0.15569010671752426) [Z2 Z13]
+ (0.15569010671752426) [Z3 Z12]
+ (0.15582269051553094) [Z8 Z13]
+ (0.15582269051553094) [Z9 Z12]
+ (0.15676396176430984) [Z4 Z9]
+ (0.15676396176430984) [Z5 Z8]
+ (0.15755314797985656) [Z4 Z5]
+ (0.16079764534838537) [Z2 Z5]
+ (0.16079764534838537) [Z3 Z4]
+ (0.1675665326546126) [Z6 Z8]
+ (0.1675665326546126) [Z7 Z9]
+ (0.16853486561579922) [Z2 Z7]
+ (0.16853486561579922) [Z3 Z6]
+ (0.18143991440303867) [Z6 Z9]
+ (0.18143991440303867) [Z7 Z8]
+ (0.18189085790751316) [Z2 Z3]
+ (0.18690820476912529) [Z2 Z9]
+ (0.18690820476912529) [Z3 Z8]
+ (0.1929972393536426) [Z0 Z10]
+ (0.1929972393536426) [Z1 Z11]
+ (0.1939253461327018) [Z6 Z7]
+ (0.19661770890342165) [Z0 Z4]
+ (0.19661770890342165) [Z1 Z5]
+ (0.19936354537360845) [Z0 Z5]
+ (0.19936354537360845) [Z1 Z4]
+ (0.20072866460441788) [Z0 Z11]
+ (0.20072866460441788) [Z1 Z10]
+ (0.21102659849791527) [Z0 Z12]
+ (0.21102659849791527) [Z1 Z13]
+ (0.21631037498631822) [Z0 Z13]
+ (0.21631037498631822) [Z1 Z12]
+ (0.2200397733437608) [Z8 Z9]
+ (0.2367108078383043) [Z0 Z2]
+ (0.2367108078383043) [Z1 Z3]
+ (0.24164663936017225) [Z0 Z6]
+ (0.24164663936017225) [Z1 Z7]
+ (0.2485348337131428) [Z0 Z7]
+ (0.2485348337131428) [Z1 Z6]
+ (0.251294456745917) [Z0 Z3]
+ (0.251294456745917) [Z1 Z2]
+ (0.2723251830660571) [Z0 Z8]
+ (0.2723251830660571) [Z1 Z9]
+ (0.2788345442672343) [Z0 Z9]
+ (0.2788345442672343) [Z1 Z8]
+ (1.186176373486053) [Z0 Z1]
+ (-1.2260484988573093e-05) [Y4 Z5 Y6]
+ (-1.2260484988573093e-05) [X4 Z5 X6]
+ (-1.2260484988573093e-05) [Y5 Z6 Y7]
+ (-1.2260484988573093e-05) [X5 Z6 X7]
+ (-1.0722312157764274e-05) [Y10 Z11 Y12]
+ (-1.0722312157764274e-05) [X10 Z11 X12]
+ (-1.0722312157764274e-05) [Y11 Z12 Y13]
+ (-1.0722312157764274e-05) [X11 Z12 X13]
+ (-3.887051672571808e-06) [Y3 Z4 Y5]
+ (-3.887051672571808e-06) [X3 Z4 X5]
+ (-3.887051672571807e-06) [Y2 Z3 Y4]
+ (-3.887051672571807e-06) [X2 Z3 X4]
+ (0.12507032579771993) [Y1 Z2 Y3]
+ (0.12507032579771993) [X1 Z2 X3]
+ (0.12507032579772) [Y0 Z1 Y2]
+ (0.12507032579772) [X0 Z1 X2]
+ (-0.03831467029480384) [Y4 Y5 X12 X13]
+ (-0.03831467029480384) [X4 X5 Y12 Y13]
+ (-0.036194123559042564) [Y2 Y3 X8 X9]
+ (-0.036194123559042564) [X2 X3 Y8 Y9]
+ (-0.03583956795335342) [Y2 Y3 X4 X5]
+ (-0.03583956795335342) [X2 X3 Y4 Y5]
+ (-0.031143817988967145) [Y2 Y3 X6 X7]
+ (-0.031143817988967145) [X2 X3 Y6 Y7]
+ (-0.028685183716105876) [Y10 Y11 X12 X13]
+ (-0.028685183716105876) [X10 X11 Y12 Y13]
+ (-0.02599617759802112) [Y3 Z4 Z5 Y7]
+ (-0.02599617759802112) [X3 Z4 Z5 X7]
+ (-0.025384657508457316) [Y2 Y3 X10 X11]
+ (-0.025384657508457316) [X2 X3 Y10 Y11]
+ (-0.01902824244384725) [Y3 Y4 X11 X12]
+ (-0.01902824244384725) [X3 X4 Y11 Y12]
+ (-0.017825140995786495) [Y6 Y7 X10 X11]
+ (-0.017825140995786495) [X6 X7 Y10 Y11]
+ (-0.017680067952481508) [Y4 Y5 X10 X11]
+ (-0.017680067952481508) [X4 X5 Y10 Y11]
+ (-0.01736611899465136) [Y6 Y7 X12 X13]
+ (-0.01736611899465136) [X6 X7 Y12 Y13]
+ (-0.015577208063976443) [Y2 Y3 X12 X13]
+ (-0.015577208063976443) [X2 X3 Y12 Y13]
+ (-0.01458364890761267) [Y0 Y1 X2 X3]
+ (-0.01458364890761267) [X0 X1 Y2 Y3]
+ (-0.013873381748426072) [Y6 Y7 X8 X9]
+ (-0.013873381748426072) [X6 X7 Y8 Y9]
+ (-0.011982389010247955) [Y4 Y5 X6 X7]
+ (-0.011982389010247955) [X4 X5 Y6 Y7]
+ (-0.011285190200840895) [Y5 X6 X11 Y12]
+ (-0.011285190200840895) [X5 Y6 Y11 X12]
+ (-0.009560705729135947) [Y8 Y9 X10 X11]
+ (-0.009560705729135947) [X8 X9 Y10 Y11]
+ (-0.008125251921381025) [Y1 X2 X8 Y9]
+ (-0.008125251921381025) [Y1 Y2 Y8 Y9]
+ (-0.008125251921381025) [X1 X2 X8 X9]
+ (-0.008125251921381025) [X1 Y2 Y8 X9]
+ (-0.0077314252507752765) [Y0 Y1 X10 X11]
+ (-0.0077314252507752765) [X0 X1 Y10 Y11]
+ (-0.0068881943529705576) [Y0 Y1 X6 X7]
+ (-0.0068881943529705576) [X0 X1 Y6 Y7]
+ (-0.006087822480561852) [Y8 Y9 X12 X13]
+ (-0.006087822480561852) [X8 X9 Y12 Y13]
+ (-0.005143391768825144) [Y3 X4 X5 Y6]
+ (-0.005143391768825144) [X3 Y4 Y5 X6]
+ (-0.004684903388155211) [Y1 X2 X6 Y7]
+ (-0.004684903388155211) [Y1 Y2 Y6 Y7]
+ (-0.004684903388155211) [X1 X2 X6 X7]
+ (-0.004684903388155211) [X1 Y2 Y6 X7]
+ (-0.004575007626639203) [Y1 X2 X12 Y13]
+ (-0.004575007626639203) [Y1 Y2 Y12 Y13]
+ (-0.004575007626639203) [X1 X2 X12 X13]
+ (-0.004575007626639203) [X1 Y2 Y12 X13]
+ (-0.004424855449441849) [Y1 X2 X4 Y5]
+ (-0.004424855449441849) [Y1 Y2 Y4 Y5]
+ (-0.004424855449441849) [X1 X2 X4 X5]
+ (-0.004424855449441849) [X1 Y2 Y4 X5]
+ (-0.003479511890334331) [Y2 Z3 Z5 Y6]
+ (-0.003479511890334331) [X2 Z3 Z5 X6]
+ (-0.003479511890334331) [Y3 Z4 Z6 Y7]
+ (-0.003479511890334331) [X3 Z4 Z6 X7]
+ (-0.0027458364701868116) [Y0 Y1 X4 X5]
+ (-0.0027458364701868116) [X0 X1 Y4 Y5]
+ (-0.0017992194936630253) [Y1 X2 X10 Y11]
+ (-0.0017992194936630253) [Y1 Y2 Y10 Y11]
+ (-0.0017992194936630253) [X1 X2 X10 X11]
+ (-0.0017992194936630253) [X1 Y2 Y10 X11]
+ (-0.0002921986261110464) [Y7 Y8 X9 X10]
+ (-0.0002921986261110464) [X7 X8 Y9 Y10]
+ (-8.194261372910286e-06) [Z10 Y11 Z12 Y13]
+ (-8.194261372910286e-06) [Z10 X11 Z12 X13]
+ (-7.80170750129276e-06) [Y2 Z3 Y4 Z11]
+ (-7.80170750129276e-06) [X2 Z3 X4 Z11]
+ (-7.80170750129276e-06) [Y3 Z4 Y5 Z10]
+ (-7.80170750129276e-06) [X3 Z4 X5 Z10]
+ (-4.643051068943363e-06) [Y3 X4 X10 Y11]
+ (-4.643051068943363e-06) [Y3 Y4 Y10 Y11]
+ (-4.643051068943363e-06) [X3 X4 X10 X11]
+ (-4.643051068943363e-06) [X3 Y4 Y10 X11]
+ (-4.588855156064921e-06) [Y4 Z5 Y6 Z13]
+ (-4.588855156064921e-06) [X4 Z5 X6 Z13]
+ (-4.588855156064921e-06) [Y5 Z6 Y7 Z12]
+ (-4.588855156064921e-06) [X5 Z6 X7 Z12]
+ (-4.55656921869332e-06) [Y5 X6 X12 Y13]
+ (-4.55656921869332e-06) [Y5 Y6 Y12 Y13]
+ (-4.55656921869332e-06) [X5 X6 X12 X13]
+ (-4.55656921869332e-06) [X5 Y6 Y12 X13]
+ (-3.6945132948614733e-06) [Y4 X5 X11 Y12]
+ (-3.6945132948614733e-06) [Y4 Y5 Y11 Y12]
+ (-3.6945132948614733e-06) [X4 X5 X11 X12]
+ (-3.6945132948614733e-06) [X4 Y5 Y11 X12]
+ (-3.344081556325843e-06) [Z0 Y5 Z6 Y7]
+ (-3.344081556325843e-06) [Z0 X5 Z6 X7]
+ (-3.344081556325843e-06) [Z1 Y4 Z5 Y6]
+ (-3.344081556325843e-06) [Z1 X4 Z5 X6]
+ (-3.158656432349397e-06) [Y2 Z3 Y4 Z10]
+ (-3.158656432349397e-06) [X2 Z3 X4 Z10]
+ (-3.158656432349397e-06) [Y3 Z4 Y5 Z11]
+ (-3.158656432349397e-06) [X3 Z4 X5 Z11]
+ (-3.099349243464496e-06) [Z0 Y4 Z5 Y6]
+ (-3.099349243464496e-06) [Z0 X4 Z5 X6]
+ (-3.099349243464496e-06) [Z1 Y5 Z6 Y7]
+ (-3.099349243464496e-06) [Z1 X5 Z6 X7]
+ (-2.8909678817889047e-06) [Z6 Y11 Z12 Y13]
+ (-2.8909678817889047e-06) [Z6 X11 Z12 X13]
+ (-2.8909678817889047e-06) [Z7 Y10 Z11 Y12]
+ (-2.8909678817889047e-06) [Z7 X10 Z11 X12]
+ (-2.177664605243604e-06) [Z0 Y10 Z11 Y12]
+ (-2.177664605243604e-06) [Z0 X10 Z11 X12]
+ (-2.177664605243604e-06) [Z1 Y11 Z12 Y13]
+ (-2.177664605243604e-06) [Z1 X11 Z12 X13]
+ (-1.881850183130704e-06) [Y4 Z5 Y6 Z9]
+ (-1.881850183130704e-06) [X4 Z5 X6 Z9]
+ (-1.881850183130704e-06) [Y5 Z6 Y7 Z8]
+ (-1.881850183130704e-06) [X5 Z6 X7 Z8]
+ (-1.8551201217451443e-06) [Z6 Y10 Z11 Y12]
+ (-1.8551201217451443e-06) [Z6 X10 Z11 X12]
+ (-1.8551201217451443e-06) [Z7 Y11 Z12 Y13]
+ (-1.8551201217451443e-06) [Z7 X11 Z12 X13]
+ (-1.8540608578852472e-06) [Y4 Z5 Y6 Z7]
+ (-1.8540608578852472e-06) [X4 Z5 X6 Z7]
+ (-1.8163031698680062e-06) [Z4 Y11 Z12 Y13]
+ (-1.8163031698680062e-06) [Z4 X11 Z12 X13]
+ (-1.8163031698680062e-06) [Z5 Y10 Z11 Y12]
+ (-1.8163031698680062e-06) [Z5 X10 Z11 X12]
+ (-1.6923978287043082e-06) [Y4 Z5 Y6 Z10]
+ (-1.6923978287043082e-06) [X4 Z5 X6 Z10]
+ (-1.6923978287043082e-06) [Y5 Z6 Y7 Z11]
+ (-1.6923978287043082e-06) [X5 Z6 X7 Z11]
+ (-1.6148794140644442e-06) [Z0 Y11 Z12 Y13]
+ (-1.6148794140644442e-06) [Z0 X11 Z12 X13]
+ (-1.6148794140644442e-06) [Z1 Y10 Z11 Y12]
+ (-1.6148794140644442e-06) [Z1 X10 Z11 X12]
+ (-1.5973171979398326e-06) [Z8 Y10 Z11 Y12]
+ (-1.5973171979398326e-06) [Z8 X10 Z11 X12]
+ (-1.5973171979398326e-06) [Z9 Y11 Z12 Y13]
+ (-1.5973171979398326e-06) [Z9 X11 Z12 X13]
+ (-1.4548424489428526e-06) [Y3 X4 X6 Y7]
+ (-1.4548424489428526e-06) [Y3 Y4 Y6 Y7]
+ (-1.4548424489428526e-06) [X3 X4 X6 X7]
+ (-1.4548424489428526e-06) [X3 Y4 Y6 X7]
+ (-1.3980449080716089e-06) [Y4 Z5 Y6 Z8]
+ (-1.3980449080716089e-06) [X4 Z5 X6 Z8]
+ (-1.3980449080716089e-06) [Y5 Z6 Y7 Z9]
+ (-1.3980449080716089e-06) [X5 Z6 X7 Z9]
+ (-1.1954890098505404e-06) [Y2 Z3 Y4 Z7]
+ (-1.1954890098505404e-06) [X2 Z3 X4 Z7]
+ (-1.1954890098505404e-06) [Y3 Z4 Y5 Z6]
+ (-1.1954890098505404e-06) [X3 Z4 X5 Z6]
+ (-1.1908508082447188e-06) [Z0 Y3 Z4 Y5]
+ (-1.1908508082447188e-06) [Z0 X3 Z4 X5]
+ (-1.1908508082447188e-06) [Z1 Y2 Z3 Y4]
+ (-1.1908508082447188e-06) [Z1 X2 Z3 X4]
+ (-1.1708301370076388e-06) [Z2 Y5 Z6 Y7]
+ (-1.1708301370076388e-06) [Z2 X5 Z6 X7]
+ (-1.1708301370076388e-06) [Z3 Y4 Z5 Y6]
+ (-1.1708301370076388e-06) [Z3 X4 Z5 X6]
+ (-1.0632283424992835e-06) [Z2 Y10 Z11 Y12]
+ (-1.0632283424992835e-06) [Z2 X10 Z11 X12]
+ (-1.0632283424992835e-06) [Z3 Y11 Z12 Y13]
+ (-1.0632283424992835e-06) [Z3 X11 Z12 X13]
+ (-1.0358477600437604e-06) [Y6 X7 X11 Y12]
+ (-1.0358477600437604e-06) [Y6 Y7 Y11 Y12]
+ (-1.0358477600437604e-06) [X6 X7 X11 X12]
+ (-1.0358477600437604e-06) [X6 Y7 Y11 X12]
+ (-9.509249751278073e-07) [Z2 Y4 Z5 Y6]
+ (-9.509249751278073e-07) [Z2 X4 Z5 X6]
+ (-9.509249751278073e-07) [Z3 Y5 Z6 Y7]
+ (-9.509249751278073e-07) [Z3 X5 Z6 X7]
+ (-9.344557777144457e-07) [Z8 Y11 Z12 Y13]
+ (-9.344557777144457e-07) [Z8 X11 Z12 X13]
+ (-9.344557777144457e-07) [Z9 Y10 Z11 Y12]
+ (-9.344557777144457e-07) [Z9 X10 Z11 X12]
+ (-8.33774675365343e-07) [Z0 Y2 Z3 Y4]
+ (-8.33774675365343e-07) [Z0 X2 Z3 X4]
+ (-8.33774675365343e-07) [Z1 Y3 Z4 Y5]
+ (-8.33774675365343e-07) [Z1 X3 Z4 X5]
+ (-7.956895371934957e-07) [Y3 X4 X8 Y9]
+ (-7.956895371934957e-07) [Y3 Y4 Y8 Y9]
+ (-7.956895371934957e-07) [X3 X4 X8 X9]
+ (-7.956895371934957e-07) [X3 Y4 Y8 X9]
+ (-7.764994118440009e-07) [Y2 Z3 Y4 Z5]
+ (-7.764994118440009e-07) [X2 Z3 X4 Z5]
+ (-5.929765815456163e-07) [Z4 Y5 Z6 Y7]
+ (-5.929765815456163e-07) [Z4 X5 Z6 X7]
+ (-5.7700529944331e-07) [Y2 Z3 Y4 Z9]
+ (-5.7700529944331e-07) [X2 Z3 X4 Z9]
+ (-5.7700529944331e-07) [Y3 Z4 Y5 Z8]
+ (-5.7700529944331e-07) [X3 Z4 X5 Z8]
+ (-5.471647745038769e-07) [Y1 Y2 X11 X12]
+ (-5.471647745038769e-07) [X1 X2 Y11 Y12]
+ (-4.838052750590951e-07) [Y5 X6 X8 Y9]
+ (-4.838052750590951e-07) [Y5 Y6 Y8 Y9]
+ (-4.838052750590951e-07) [X5 X6 X8 X9]
+ (-4.838052750590951e-07) [X5 Y6 Y8 X9]
+ (-3.57076132879376e-07) [Y0 X1 X3 Y4]
+ (-3.57076132879376e-07) [Y0 Y1 Y3 Y4]
+ (-3.57076132879376e-07) [X0 X1 X3 X4]
+ (-3.57076132879376e-07) [X0 Y1 Y3 X4]
+ (-2.447323128613471e-07) [Y0 X1 X5 Y6]
+ (-2.447323128613471e-07) [Y0 Y1 Y5 Y6]
+ (-2.447323128613471e-07) [X0 X1 X5 X6]
+ (-2.447323128613471e-07) [X0 Y1 Y5 X6]
+ (-2.1990516187983153e-07) [Y2 X3 X5 Y6]
+ (-2.1990516187983153e-07) [Y2 Y3 Y5 Y6]
+ (-2.1990516187983153e-07) [X2 X3 X5 X6]
+ (-2.1990516187983153e-07) [X2 Y3 Y5 X6]
+ (-1.9332412769139038e-07) [Y1 X2 X3 Y4]
+ (-1.9332412769139038e-07) [X1 Y2 Y3 X4]
+ (-1.2919694863285088e-07) [Y1 Z2 Z3 Y5]
+ (-1.2919694863285088e-07) [X1 Z2 Z3 X5]
+ (1.7379332621433922e-07) [Y0 Z1 Z3 Y4]
+ (1.7379332621433922e-07) [X0 Z1 Z3 X4]
+ (1.7379332621433922e-07) [Y1 Z2 Z4 Y5]
+ (1.7379332621433922e-07) [X1 Z2 Z4 X5]
+ (1.9332412769139038e-07) [Y1 Y2 X3 X4]
+ (1.9332412769139038e-07) [X1 X2 Y3 Y4]
+ (2.1868423775018557e-07) [Y2 Z3 Y4 Z8]
+ (2.1868423775018557e-07) [X2 Z3 X4 Z8]
+ (2.1868423775018557e-07) [Y3 Z4 Y5 Z9]
+ (2.1868423775018557e-07) [X3 Z4 X5 Z9]
+ (2.5935343909231196e-07) [Y2 Z3 Y4 Z6]
+ (2.5935343909231196e-07) [X2 Z3 X4 Z6]
+ (2.5935343909231196e-07) [Y3 Z4 Y5 Z7]
+ (2.5935343909231196e-07) [X3 Z4 X5 Z7]
+ (3.606071867465883e-07) [Y0 Z1 Z2 Y4]
+ (3.606071867465883e-07) [X0 Z1 Z2 X4]
+ (3.606071867465883e-07) [Y1 Z3 Z4 Y5]
+ (3.606071867465883e-07) [X1 Z3 Z4 X5]
+ (5.471647745038769e-07) [Y1 X2 X11 Y12]
+ (5.471647745038769e-07) [X1 Y2 Y11 X12]
+ (5.627851911791599e-07) [Y0 X1 X11 Y12]
+ (5.627851911791599e-07) [Y0 Y1 Y11 Y12]
+ (5.627851911791599e-07) [X0 X1 X11 X12]
+ (5.627851911791599e-07) [X0 Y1 Y11 X12]
+ (6.62861420225387e-07) [Y8 X9 X11 Y12]
+ (6.62861420225387e-07) [Y8 Y9 Y11 Y12]
+ (6.62861420225387e-07) [X8 X9 X11 X12]
+ (6.62861420225387e-07) [X8 Y9 Y11 X12]
+ (1.1094407591371508e-06) [Z2 Y11 Z12 Y13]
+ (1.1094407591371508e-06) [Z2 X11 Z12 X13]
+ (1.1094407591371508e-06) [Z3 Y10 Z11 Y12]
+ (1.1094407591371508e-06) [Z3 X10 Z11 X12]
+ (1.6021167405529254e-06) [Z2 Y3 Z4 Y5]
+ (1.6021167405529254e-06) [Z2 X3 Z4 X5]
+ (1.878210124993467e-06) [Z4 Y10 Z11 Y12]
+ (1.878210124993467e-06) [Z4 X10 Z11 X12]
+ (1.878210124993467e-06) [Z5 Y11 Z12 Y13]
+ (1.878210124993467e-06) [Z5 X11 Z12 X13]
+ (2.1726691016364343e-06) [Y2 X3 X11 Y12]
+ (2.1726691016364343e-06) [Y2 Y3 Y11 Y12]
+ (2.1726691016364343e-06) [X2 X3 X11 X12]
+ (2.1726691016364343e-06) [X2 Y3 Y11 X12]
+ (3.117447945777388e-06) [Y0 Z2 Z3 Y4]
+ (3.117447945777388e-06) [X0 Z2 Z3 X4]
+ (3.5390541848576204e-06) [Y2 Z3 Y4 Z12]
+ (3.5390541848576204e-06) [X2 Z3 X4 Z12]
+ (3.5390541848576204e-06) [Y3 Z4 Y5 Z13]
+ (3.5390541848576204e-06) [X3 Z4 X5 Z13]
+ (4.281913885189189e-06) [Y4 Z5 Y6 Z11]
+ (4.281913885189189e-06) [X4 Z5 X6 Z11]
+ (4.281913885189189e-06) [Y5 Z6 Y7 Z10]
+ (4.281913885189189e-06) [X5 Z6 X7 Z10]
+ (5.275883122552817e-06) [Y3 X4 X12 Y13]
+ (5.275883122552817e-06) [Y3 Y4 Y12 Y13]
+ (5.275883122552817e-06) [X3 X4 X12 X13]
+ (5.275883122552817e-06) [X3 Y4 Y12 X13]
+ (5.974311713893498e-06) [Y5 X6 X10 Y11]
+ (5.974311713893498e-06) [Y5 Y6 Y10 Y11]
+ (5.974311713893498e-06) [X5 X6 X10 X11]
+ (5.974311713893498e-06) [X5 Y6 Y10 X11]
+ (7.95441317701788e-06) [Y10 Z11 Y12 Z13]
+ (7.95441317701788e-06) [X10 Z11 X12 Z13]
+ (8.814937307410437e-06) [Y2 Z3 Y4 Z13]
+ (8.814937307410437e-06) [X2 Z3 X4 Z13]
+ (8.814937307410437e-06) [Y3 Z4 Y5 Z12]
+ (8.814937307410437e-06) [X3 Z4 X5 Z12]
+ (0.0002921986261110464) [Y7 X8 X9 Y10]
+ (0.0002921986261110464) [X7 Y8 Y9 X10]
+ (0.0004956762314916009) [Y2 Z4 Z5 Y6]
+ (0.0004956762314916009) [X2 Z4 Z5 X6]
+ (0.0011059037691896947) [Y0 Z1 Y2 Z5]
+ (0.0011059037691896947) [X0 Z1 X2 Z5]
+ (0.0011059037691896947) [Y1 Z2 Y3 Z4]
+ (0.0011059037691896947) [X1 Z2 X3 Z4]
+ (0.001663879878490813) [Y2 Z3 Z4 Y6]
+ (0.001663879878490813) [X2 Z3 Z4 X6]
+ (0.001663879878490813) [Y3 Z5 Z6 Y7]
+ (0.001663879878490813) [X3 Z5 Z6 X7]
+ (0.0017560707018412507) [Y0 Z1 Y2 Z11]
+ (0.0017560707018412507) [X0 Z1 X2 Z11]
+ (0.0017560707018412507) [Y1 Z2 Y3 Z10]
+ (0.0017560707018412507) [X1 Z2 X3 Z10]
+ (0.0023262306231580905) [Y0 Z1 Y2 Z13]
+ (0.0023262306231580905) [X0 Z1 X2 Z13]
+ (0.0023262306231580905) [Y1 Z2 Y3 Z12]
+ (0.0023262306231580905) [X1 Z2 X3 Z12]
+ (0.0027458364701868116) [Y0 X1 X4 Y5]
+ (0.0027458364701868116) [X0 Y1 Y4 X5]
+ (0.0029297686747510585) [Y0 Z1 Y2 Z9]
+ (0.0029297686747510585) [X0 Z1 X2 Z9]
+ (0.0029297686747510585) [Y1 Z2 Y3 Z8]
+ (0.0029297686747510585) [X1 Z2 X3 Z8]
+ (0.003276971931231665) [Y0 Z1 Y2 Z3]
+ (0.003276971931231665) [X0 Z1 X2 Z3]
+ (0.0033476175306661805) [Y0 Z1 Y2 Z7]
+ (0.0033476175306661805) [X0 Z1 X2 Z7]
+ (0.0033476175306661805) [Y1 Z2 Y3 Z6]
+ (0.0033476175306661805) [X1 Z2 X3 Z6]
+ (0.003555290195504276) [Y0 Z1 Y2 Z10]
+ (0.003555290195504276) [X0 Z1 X2 Z10]
+ (0.003555290195504276) [Y1 Z2 Y3 Z11]
+ (0.003555290195504276) [X1 Z2 X3 Z11]
+ (0.005143391768825144) [Y3 Y4 X5 X6]
+ (0.005143391768825144) [X3 X4 Y5 Y6]
+ (0.005530759218631544) [Y0 Z1 Y2 Z4]
+ (0.005530759218631544) [X0 Z1 X2 Z4]
+ (0.005530759218631544) [Y1 Z2 Y3 Z5]
+ (0.005530759218631544) [X1 Z2 X3 Z5]
+ (0.006087822480561852) [Y8 X9 X12 Y13]
+ (0.006087822480561852) [X8 Y9 Y12 X13]
+ (0.0068881943529705576) [Y0 X1 X6 Y7]
+ (0.0068881943529705576) [X0 Y1 Y6 X7]
+ (0.006901238249797294) [Y0 Z1 Y2 Z12]
+ (0.006901238249797294) [X0 Z1 X2 Z12]
+ (0.006901238249797294) [Y1 Z2 Y3 Z13]
+ (0.006901238249797294) [X1 Z2 X3 Z13]
+ (0.0077314252507752765) [Y0 X1 X10 Y11]
+ (0.0077314252507752765) [X0 Y1 Y10 X11]
+ (0.00803252091882139) [Y0 Z1 Y2 Z6]
+ (0.00803252091882139) [X0 Z1 X2 Z6]
+ (0.00803252091882139) [Y1 Z2 Y3 Z7]
+ (0.00803252091882139) [X1 Z2 X3 Z7]
+ (0.009560705729135947) [Y8 X9 X10 Y11]
+ (0.009560705729135947) [X8 Y9 Y10 X11]
+ (0.011055020596132083) [Y0 Z1 Y2 Z8]
+ (0.011055020596132083) [X0 Z1 X2 Z8]
+ (0.011055020596132083) [Y1 Z2 Y3 Z9]
+ (0.011055020596132083) [X1 Z2 X3 Z9]
+ (0.011285190200840895) [Y5 Y6 X11 X12]
+ (0.011285190200840895) [X5 X6 Y11 Y12]
+ (0.011307274008848227) [Y7 Z8 Z9 Y11]
+ (0.011307274008848227) [X7 Z8 Z9 X11]
+ (0.011982389010247955) [Y4 X5 X6 Y7]
+ (0.011982389010247955) [X4 Y5 Y6 X7]
+ (0.013873381748426072) [Y6 X7 X8 Y9]
+ (0.013873381748426072) [X6 Y7 Y8 X9]
+ (0.01458364890761267) [Y0 X1 X2 Y3]
+ (0.01458364890761267) [X0 Y1 Y2 X3]
+ (0.015577208063976443) [Y2 X3 X12 Y13]
+ (0.015577208063976443) [X2 Y3 Y12 X13]
+ (0.01736611899465136) [Y6 X7 X12 Y13]
+ (0.01736611899465136) [X6 Y7 Y12 X13]
+ (0.017680067952481508) [Y4 X5 X10 Y11]
+ (0.017680067952481508) [X4 Y5 Y10 X11]
+ (0.017825140995786495) [Y6 X7 X10 Y11]
+ (0.017825140995786495) [X6 Y7 Y10 X11]
+ (0.01902824244384725) [Y3 X4 X11 Y12]
+ (0.01902824244384725) [X3 Y4 Y11 X12]
+ (0.025384657508457316) [Y2 X3 X10 Y11]
+ (0.025384657508457316) [X2 Y3 Y10 X11]
+ (0.028685183716105876) [Y10 X11 X12 Y13]
+ (0.028685183716105876) [X10 Y11 Y12 X13]
+ (0.02981242451734579) [Y6 Z7 Z8 Y10]
+ (0.02981242451734579) [X6 Z7 Z8 X10]
+ (0.02981242451734579) [Y7 Z9 Z10 Y11]
+ (0.02981242451734579) [X7 Z9 Z10 X11]
+ (0.03010462314345684) [Y6 Z7 Z9 Y10]
+ (0.03010462314345684) [X6 Z7 Z9 X10]
+ (0.03010462314345684) [Y7 Z8 Z10 Y11]
+ (0.03010462314345684) [X7 Z8 Z10 X11]
+ (0.030787505389143918) [Y6 Z8 Z9 Y10]
+ (0.030787505389143918) [X6 Z8 Z9 X10]
+ (0.031143817988967145) [Y2 X3 X6 Y7]
+ (0.031143817988967145) [X2 Y3 Y6 X7]
+ (0.03583956795335342) [Y2 X3 X4 Y5]
+ (0.03583956795335342) [X2 Y3 Y4 X5]
+ (0.036194123559042564) [Y2 X3 X8 Y9]
+ (0.036194123559042564) [X2 Y3 Y8 X9]
+ (0.03831467029480384) [Y4 X5 X12 Y13]
+ (0.03831467029480384) [X4 Y5 Y12 X13]
+ (0.10433064780651422) [Z0 Y1 Z2 Y3]
+ (0.10433064780651422) [Z0 X1 Z2 X3]
+ (-0.12133276911042289) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133276911042289) [X3 Z4 Z5 Z6 X7]
+ (-0.12133276911042286) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042286) [X2 Z3 Z4 Z5 X6]
+ (3.202076879098679e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.202076879098679e-06) [X0 Z1 Z2 Z3 X4]
+ (3.202076879098681e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.202076879098681e-06) [X1 Z2 Z3 Z4 X5]
+ (0.22848106564918863) [Y6 Z7 Z8 Z9 Y10]
+ (0.22848106564918863) [X6 Z7 Z8 Z9 X10]
+ (0.22848106564918866) [Y7 Z8 Z9 Z10 Y11]
+ (0.22848106564918866) [X7 Z8 Z9 Z10 X11]
+ (-0.03276765782329051) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.03276765782329051) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.03276765782329051) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.03276765782329051) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.02711503684527317) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.02711503684527317) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.02711503684527317) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.02711503684527317) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.02599617759802112) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.02599617759802112) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.017561202409646124) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.017561202409646124) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.017561202409646124) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.017561202409646124) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.01456453123117298) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.01456453123117298) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.01456453123117298) [X7 Z8 Z9 X10 X12 X13]
+ (-0.01456453123117298) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.012215040997613957) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.012215040997613957) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.012215040997613957) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.012215040997613957) [X4 Z5 X6 X11 Z12 X13]
+ (-0.012215040997613957) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.012215040997613957) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.012215040997613957) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.012215040997613957) [X5 Z6 X7 X10 Z11 X12]
+ (-0.011756013419819215) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.011756013419819215) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.011756013419819215) [X3 Z4 Z5 X6 X8 X9]
+ (-0.011756013419819215) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.00876482757568875) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.00876482757568875) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.00876482757568875) [X2 Z3 Z4 X5 X11 X12]
+ (-0.00876482757568875) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.00876482757568875) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.00876482757568875) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.00876482757568875) [X3 X4 X10 Z11 Z12 X13]
+ (-0.00876482757568875) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.008125251921381025) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125251921381025) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.0073067599288329736) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.0073067599288329736) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.0073067599288329736) [X4 X5 X7 Z8 Z9 X10]
+ (-0.0073067599288329736) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005805188989826909) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.005805188989826909) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.005805188989826909) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.005805188989826909) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.005652620978017343) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.005652620978017343) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.005652620978017343) [X0 X1 X3 Z4 Z5 X6]
+ (-0.005652620978017343) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.005143391768825144) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.005143391768825144) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.005143391768825144) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.005143391768825144) [X2 Z3 X4 X5 Z6 X7]
+ (-0.004684903388155211) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.004684903388155211) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668620318776291) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668620318776291) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.004575007626639203) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.004575007626639203) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424855449441849) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424855449441849) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.00415879738184005) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.00415879738184005) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.00415879738184005) [X3 Z4 Z5 X6 X12 X13]
+ (-0.00415879738184005) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.003493790359890137) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.003493790359890137) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.003493790359890137) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.003493790359890137) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.0027790267990255462) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.0027790267990255462) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.0022939566113524563) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.0022939566113524563) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.0017992194936630253) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.0017992194936630253) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278753941369486) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.0017278753941369486) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.000929850796773061) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.000929850796773061) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.000929850796773061) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.000929850796773061) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533856254125437) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533856254125437) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008145313270956404) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008145313270956404) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008145313270956404) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008145313270956404) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.735036880588997e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735036880588997e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735036880588997e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735036880588997e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.774817865393397e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.774817865393397e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.774817865393397e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.774817865393397e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.51836221633334e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.51836221633334e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.51836221633334e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.51836221633334e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.444344676646171e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.444344676646171e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.444344676646171e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.444344676646171e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.524373849192588e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.524373849192588e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.524373849192588e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.524373849192588e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.2900284339380365e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.2900284339380365e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.2900284339380365e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.2900284339380365e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.974311713893498e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.974311713893498e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.275883122552816e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.275883122552816e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.643051068943363e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.643051068943363e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.556569218693319e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.556569218693319e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.2532242259371355e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.2532242259371355e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.7696594525468487e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.7696594525468487e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.6945132948614733e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.6945132948614733e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.6102971311418465e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.6102971311418465e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.6102971311418465e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.6102971311418465e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.3131455003432597e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.3131455003432597e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.2774831960907486e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.2774831960907486e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.2774831960907486e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.2774831960907486e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.211228348849329e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.211228348849329e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.211228348849329e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.211228348849329e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.1513463114604274e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.1513463114604274e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.0882507116503267e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.0882507116503267e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.1726691016364343e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.1726691016364343e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.4548424489428526e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.4548424489428526e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.330473188747227e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.330473188747227e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2283337823953025e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2283337823953025e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.0358477600437604e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.0358477600437604e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956895371934957e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956895371934957e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733197743483748e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733197743483748e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733197743483748e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733197743483748e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.62861420225387e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.62861420225387e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.556281915028602e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.556281915028602e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.556281915028602e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.556281915028602e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.418291575060646e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.418291575060646e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.418291575060646e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.418291575060646e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.927453083803202e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927453083803202e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927453083803202e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927453083803202e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627851911791599e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627851911791599e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660625087007e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660625087007e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660625087007e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660625087007e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660625087007e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660625087007e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660625087007e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660625087007e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.838052750590951e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.838052750590951e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.57076132879376e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.57076132879376e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.3281393505109753e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.3281393505109753e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.08682656509559e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.08682656509559e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.08682656509559e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.08682656509559e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.447323128613471e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.447323128613471e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.3713289477318758e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.3713289477318758e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.3713289477318758e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.3713289477318758e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.199051618798315e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.199051618798315e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.9332412769139038e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.9332412769139038e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.9332412769139038e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.9332412769139038e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.8394209153233028e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.8394209153233028e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.8394209153233028e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.8394209153233028e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.5510539175483892e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.5510539175483892e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.5510539175483892e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.5510539175483892e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.380778147960358e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.380778147960358e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.380778147960358e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.380778147960358e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.380778147960358e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.380778147960358e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.380778147960358e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.380778147960358e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.380778147960358e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.380778147960358e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.380778147960358e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.380778147960358e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.2919694863285088e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.2919694863285088e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.107632559981736e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.107632559981736e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.107632559981736e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.107632559981736e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.107632559981736e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.107632559981736e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.107632559981736e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.107632559981736e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.057446596805467e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057446596805467e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057446596805467e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057446596805467e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649310131910353e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649310131910353e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649310131910353e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649310131910353e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.839420915323303e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.839420915323303e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.839420915323303e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.839420915323303e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.199051618798315e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.199051618798315e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.447323128613471e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.447323128613471e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.236259960922911e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.236259960922911e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.236259960922911e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.236259960922911e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.3281393505109753e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.3281393505109753e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.57076132879376e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.57076132879376e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.838052750590951e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.838052750590951e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.627851911791599e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627851911791599e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.62861420225387e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.62861420225387e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956895371934957e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956895371934957e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.306536652406885e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.306536652406885e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.306536652406885e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.306536652406885e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.0358477600437604e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.0358477600437604e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2283337823953025e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2283337823953025e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.2393363217502477e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393363217502477e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393363217502477e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393363217502477e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.330473188747227e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.330473188747227e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.4548424489428526e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.4548424489428526e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.1726691016364343e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.1726691016364343e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.0882507116503267e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.0882507116503267e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.117447945777388e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.117447945777388e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.1513463114604274e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.1513463114604274e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.3131455003432597e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.3131455003432597e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.334331289740288e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.334331289740288e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.6945132948614733e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.6945132948614733e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.183932559649843e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.183932559649843e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.556569218693319e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.556569218693319e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.643051068943363e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.643051068943363e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.275883122552816e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.275883122552816e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.974311713893498e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.974311713893498e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.0002921986261110464) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.0002921986261110464) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.0002921986261110464) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.0002921986261110464) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004956762314916009) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004956762314916009) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650070219499129) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650070219499129) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650070219499129) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650070219499129) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533856254125437) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533856254125437) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.0016095313817213685) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.0016095313817213685) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.0016095313817213685) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.0016095313817213685) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0016676041811440395) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.0016676041811440395) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.0016676041811440395) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.0016676041811440395) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.0017278753941369486) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.0017278753941369486) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.0017992194936630253) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.0017992194936630253) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.0022939566113524563) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.0022939566113524563) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.0024629170071339117) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.0024629170071339117) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.0024629170071339117) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.0024629170071339117) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.003961560792496497) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.003961560792496497) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.003961560792496497) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.003961560792496497) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.004424855449441849) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424855449441849) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004575007626639203) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.004575007626639203) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668620318776291) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668620318776291) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.004684903388155211) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.004684903388155211) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005324835234221689) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.005324835234221689) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.005324835234221689) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.005324835234221689) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.005368659358109543) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368659358109543) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368659358109543) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368659358109543) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.007960880725921549) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.007960880725921549) [X4 Z5 X6 X10 Z11 X12]
+ (0.007960880725921549) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.007960880725921549) [X5 Z6 X7 X11 Z12 X13]
+ (0.008125251921381025) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125251921381025) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.008890731522694612) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.008890731522694612) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.008890731522694612) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.008890731522694612) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.010263414868158498) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.010263414868158498) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.010263414868158498) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.010263414868158498) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.01054042590767155) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.01054042590767155) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.01054042590767155) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.01054042590767155) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010960074940542611) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010960074940542611) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010960074940542611) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010960074940542611) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307274008848227) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307274008848227) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.014411099430130941) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411099430130941) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411099430130941) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411099430130941) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.015225630757226584) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.015225630757226584) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.015225630757226584) [X3 Z4 Z5 X6 X10 X11]
+ (0.015225630757226584) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.015588250102380189) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.015588250102380189) [X2 Z3 X4 X10 Z11 X12]
+ (0.015588250102380189) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.015588250102380189) [X3 Z4 X5 X11 Z12 X13]
+ (0.01826683486937558) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.01826683486937558) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.01826683486937558) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.01826683486937558) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.019020423173039983) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.019020423173039983) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.019020423173039983) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.019020423173039983) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.020175921723535505) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.020175921723535505) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.020175921723535505) [X4 Z5 Z6 X7 X11 X12]
+ (0.020175921723535505) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.020175921723535505) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.020175921723535505) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.020175921723535505) [X5 X6 X10 Z11 Z12 X13]
+ (0.020175921723535505) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.02435307767806894) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.02435307767806894) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.02435307767806894) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.02435307767806894) [X2 Z3 X4 X11 Z12 X13]
+ (0.02435307767806894) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.02435307767806894) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.02435307767806894) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.02435307767806894) [X3 Z4 X5 X10 Z11 X12]
+ (0.024389082531149527) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.024389082531149527) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.024389082531149527) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.024389082531149527) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.025104957138844527) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.025104957138844527) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.025104957138844527) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.025104957138844527) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.030787505389143918) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.030787505389143918) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.04587947078129795) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.04587947078129795) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.05600733087780779) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.05600733087780779) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.05600733087780779) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.05600733087780779) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661368) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.05608468124661368) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661368) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.05608468124661368) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.631277929048022e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631277929048022e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.631277929048022e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631277929048022e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.5950860074384065e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.5950860074384065e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.5950860074384058e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.5950860074384058e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.04274327701378231) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04274327701378231) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (0.042743277013782326) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.042743277013782326) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (-0.04764261217638309) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.04764261217638309) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.04764261217638309) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.04764261217638309) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.04171881383982175) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.04171881383982175) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.04171881383982175) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.04171881383982175) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.0395644163228933) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.0395644163228933) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.0395644163228933) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0395644163228933) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.039359168022052984) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.039359168022052984) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.039359168022052984) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.039359168022052984) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.039318051947197535) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.039318051947197535) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.039318051947197535) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.039318051947197535) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0356083789883124) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0356083789883124) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.029903789512624762) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.029903789512624762) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.029903789512624762) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.029903789512624762) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.028730779551905436) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.028730779551905436) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.028730779551905436) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.028730779551905436) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.02563723829602683) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.02563723829602683) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.02563723829602683) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.02563723829602683) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.024755463292890884) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.024755463292890884) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.024755463292890884) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.024755463292890884) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.024282117354692913) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.024282117354692913) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.023145130929529065) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.023145130929529065) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.022528440196013005) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.022528440196013005) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.021433810721600846) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.021433810721600846) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.021433810721600846) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.021433810721600846) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251575) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.019257505095251575) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.01902824244384725) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.01902824244384725) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.018889030304942836) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.018889030304942836) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.018889030304942836) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.018889030304942836) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.01602460368917952) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.01602460368917952) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.015225630757226584) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225630757226584) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.0146037047291621) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.0146037047291621) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.01456453123117298) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.01456453123117298) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.011756013419819215) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.011756013419819215) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.011285190200840895) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.011285190200840895) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.0098417492469626) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0098417492469626) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.009612634606847307) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612634606847307) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612634606847307) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612634606847307) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.008469978791023913) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.008469978791023913) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.0073067599288329736) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.0073067599288329736) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0059237983365613405) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.0059237983365613405) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.005652620978017342) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.005652620978017342) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368659358109543) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368659358109543) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.00415879738184005) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.00415879738184005) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0033566705638328784) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.0033566705638328784) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.0033566705638328784) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.0033566705638328784) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.0032675138544235446) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.0032675138544235446) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.0032675138544235446) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.0032675138544235446) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.0027790267990255462) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.0027790267990255462) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.0026860409778066007) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.0026860409778066007) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.0026860409778066007) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.0026860409778066007) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.0022939566113524563) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524563) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.0022939566113524563) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524563) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.0009581655836696522) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.0009581655836696522) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0009581655836696522) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.0009581655836696522) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.0009581655836696522) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.0009581655836696522) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0009581655836696522) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.0009581655836696522) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.0002463643756957689) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0002463643756957689) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303549336) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.00013840177303549336) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.00013840177303549336) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.00013840177303549336) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.735036880588997e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735036880588997e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.61035853072231e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.61035853072231e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.61035853072231e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.61035853072231e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808796757236e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.5316808796757236e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808796757236e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5316808796757236e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.806102776061835e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.806102776061835e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.806102776061835e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.806102776061835e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.089799468111037e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.089799468111037e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.089799468111037e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.089799468111037e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.652209670545842e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.652209670545842e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.652209670545842e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.652209670545842e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.4818518350134064e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.4818518350134064e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.4818518350134064e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.4818518350134064e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.0714807368580936e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.0714807368580936e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.0714807368580936e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.0714807368580936e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.7346220392037415e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.7346220392037415e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.7346220392037415e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.7346220392037415e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.728843147626325e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.728843147626325e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.728843147626325e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.728843147626325e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.2532242259371355e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.2532242259371355e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.7696594525468487e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.7696594525468487e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.5443954295946626e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.5443954295946626e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.5443954295946626e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.5443954295946626e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.5443954295946626e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.5443954295946626e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.5443954295946626e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.5443954295946626e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.360956320484712e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.360956320484712e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.360956320484712e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.360956320484712e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.1032156050376762e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.1032156050376762e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.1032156050376762e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.1032156050376762e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.011122098725354e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.011122098725354e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.011122098725354e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.011122098725354e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9429468368305244e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.9429468368305244e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9429468368305244e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9429468368305244e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174774883958e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.6541174774883958e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174774883958e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.6541174774883958e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.5224930677582867e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224930677582867e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224930677582867e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224930677582867e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224930677582867e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930677582867e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930677582867e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224930677582867e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.2283337823953025e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337823953025e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.2283337823953025e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337823953025e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.988770288278514e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988770288278514e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988770288278514e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988770288278514e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.86776510465863e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.86776510465863e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.86776510465863e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.86776510465863e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.189990976130453e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.189990976130453e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246207486192e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246207486192e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.471647745038769e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.471647745038769e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.5614471794433593e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.5614471794433593e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.5614471794433593e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.5614471794433593e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.523389678610142e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.523389678610142e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.4273231088351525e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.4273231088351525e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.4273231088351525e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.4273231088351525e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.328139350510975e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350510975e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.328139350510975e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350510975e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.08682656509559e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.08682656509559e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.8882935934212865e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935934212865e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935934212865e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.8882935934212865e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.3713289477318758e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.3713289477318758e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.8394209153233033e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.8394209153233033e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.057446596805467e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057446596805467e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.537178095452364e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.537178095452364e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.537178095452364e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.537178095452364e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.057446596805467e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057446596805467e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.209350631232183e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350631232183e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350631232183e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.209350631232183e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783553243488e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783553243488e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783553243488e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783553243488e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.8394209153233033e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.8394209153233033e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.3713289477318758e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.3713289477318758e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.08682656509559e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.08682656509559e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.523389678610142e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.523389678610142e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.471647745038769e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.471647745038769e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246207486192e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246207486192e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189990976130453e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.189990976130453e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.330473188747227e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.330473188747227e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.330473188747227e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.330473188747227e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288532437021404e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288532437021404e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288532437021404e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288532437021404e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.6893489516408625e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.6893489516408625e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.6893489516408625e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.6893489516408625e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.7455184007668114e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.7455184007668114e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.7455184007668114e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.7455184007668114e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.7455184007668114e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.7455184007668114e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.7455184007668114e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.7455184007668114e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.21184201939915e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.21184201939915e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.21184201939915e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.21184201939915e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.21184201939915e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.21184201939915e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.21184201939915e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.21184201939915e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.3131455003432592e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.3131455003432592e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.3131455003432592e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.3131455003432592e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.334331289740288e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.334331289740288e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.183932559649843e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.183932559649843e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.735036880588997e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735036880588997e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.0002463643756957689) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.0002463643756957689) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00044585351288409486) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.00044585351288409486) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.00044585351288409486) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.00044585351288409486) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940221543005475) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005475) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005475) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005475) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005475) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005475) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940221543005475) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005475) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533856254125438) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533856254125438) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533856254125438) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533856254125438) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0010435246534907497) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435246534907497) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435246534907497) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435246534907497) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.0012803060973496608) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.0012803060973496608) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.0012803060973496608) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.0012803060973496608) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.0013038004788126938) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.0013038004788126938) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.0013038004788126938) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.0013038004788126938) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.002261966062482346) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.002261966062482346) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.002261966062482346) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.002261966062482346) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.002261966062482346) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.002261966062482346) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.002261966062482346) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.002261966062482346) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.0039898414566192945) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.0039898414566192945) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.0039898414566192945) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.0039898414566192945) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.00415879738184005) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.00415879738184005) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.0043110385079142945) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.0043110385079142945) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.0043110385079142945) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.0043110385079142945) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.004636976661182539) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.004636976661182539) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.004636976661182539) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.004636976661182539) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.005114473831660386) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.005114473831660386) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.005114473831660386) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.005114473831660386) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.005114473831660386) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660386) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660386) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005114473831660386) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005241535382803857) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.005241535382803857) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.005241535382803857) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.005241535382803857) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.005262642473076838) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262642473076838) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262642473076838) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262642473076838) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368659358109543) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368659358109543) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.00537993715583935) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.00537993715583935) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.00537993715583935) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.00537993715583935) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.005652620978017342) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.005652620978017342) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.005708495985960933) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708495985960933) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708495985960933) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708495985960933) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.0059237983365613405) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.0059237983365613405) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.0073067599288329736) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.0073067599288329736) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469978791023913) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.008469978791023913) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.0098417492469626) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.0098417492469626) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.011285190200840895) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.011285190200840895) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.011756013419819215) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.011756013419819215) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.01456453123117298) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.01456453123117298) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.0146037047291621) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.0146037047291621) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.015225630757226584) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225630757226584) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.01602460368917952) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.01602460368917952) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.01902824244384725) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.01902824244384725) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.019257505095251575) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.019257505095251575) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.04587947078129794) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04587947078129794) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.3693708936615614) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.3693708936615614) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.3693708936615614) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.3693708936615614) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.281642577670228) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.281642577670228) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.2816425776702278) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.2816425776702278) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.09065144207036485) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.09065144207036485) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.09065144207036485) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.09065144207036485) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863632) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684737589863632) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863632) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684737589863632) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635021950635004) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635021950635004) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635021950635004) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635021950635004) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0356083789883124) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0356083789883124) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03490334337366177) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03490334337366177) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03490334337366177) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03490334337366177) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02459186088382997) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.02459186088382997) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02459186088382997) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.02459186088382997) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.024282117354692913) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.024282117354692913) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.02314513092952906) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.02314513092952906) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.022528440196013005) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.022528440196013005) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.01953805031131464) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.01953805031131464) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.01953805031131464) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.01953805031131464) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.017091553155898796) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.017091553155898796) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.017091553155898796) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.017091553155898796) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.01602460368917952) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.01602460368917952) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.01602460368917952) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.01602460368917952) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.0103114824898318) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0103114824898318) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0103114824898318) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0103114824898318) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0098417492469626) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0098417492469626) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0098417492469626) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0098417492469626) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209836) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209836) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209836) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008826368514209836) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008541996625454816) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454816) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454816) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454816) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454816) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454816) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454816) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008541996625454816) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008469978791023913) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023913) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.008469978791023913) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023913) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.004668620318776291) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668620318776291) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.0038764708993369473) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.0038764708993369473) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.0038040661717285425) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0038040661717285425) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0038040661717285425) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0038040661717285425) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178817) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178817) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003356670563832879) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.003356670563832879) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.0032675138544235446) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.0032675138544235446) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0021413612231016167) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.0021413612231016167) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.0017278753941369486) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.0017278753941369486) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.001640754855312416) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.001640754855312416) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214169302) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0014528843214169302) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214169302) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0014528843214169302) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.000787089677102447) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.000787089677102447) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.000519274349948777) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.000519274349948777) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.00019400857029756982) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00019400857029756982) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303549336) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.00013840177303549336) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.141625221158184e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.141625221158184e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.141625221158184e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.141625221158184e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.071480736858093e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.071480736858093e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.1513463114604274e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.1513463114604274e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0882507116503267e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.0882507116503267e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.988511706358717e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.988511706358717e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.874299071606893e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.874299071606893e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.360956320484712e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.360956320484712e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.3002946564159884e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3002946564159884e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.1468376508779301e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.1468376508779301e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.1468376508779301e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.1468376508779301e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.352332104111667e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.352332104111667e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.352332104111667e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.352332104111667e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091637200059449e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637200059449e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637200059449e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637200059449e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637200059449e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637200059449e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637200059449e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091637200059449e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.074305986851858e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.074305986851858e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.074305986851858e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.074305986851858e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128987329992e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128987329992e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128987329992e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128987329992e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.86776510465863e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.86776510465863e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560692465812873e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692465812873e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692465812873e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692465812873e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692465812873e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692465812873e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692465812873e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560692465812873e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.997018422648679e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.997018422648679e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.997018422648679e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.997018422648679e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.997018422648679e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.997018422648679e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.997018422648679e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.997018422648679e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.5682475214493075e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.5682475214493075e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.5682475214493075e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.5682475214493075e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.37673930871985e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.37673930871985e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.37673930871985e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.37673930871985e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.37673930871985e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.37673930871985e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.37673930871985e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.37673930871985e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.888293593421286e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.888293593421286e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.686381547806208e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.686381547806208e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035783553243488e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7035783553243488e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.209350631232183e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.209350631232183e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773245101009e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773245101009e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773245101009e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773245101009e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773245101009e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773245101009e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773245101009e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773245101009e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253798405286e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253798405286e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9742253798405286e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9742253798405286e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.047471655647805e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.047471655647805e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.047471655647805e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.047471655647805e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.209350631232183e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.209350631232183e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0717282186606975e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0717282186606975e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0717282186606975e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0717282186606975e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.200428749466787e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.200428749466787e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.200428749466787e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.200428749466787e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.7035783553243488e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7035783553243488e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.3120943054448153e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.3120943054448153e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.3120943054448153e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.3120943054448153e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.686381547806208e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.686381547806208e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.888293593421286e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.888293593421286e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.0922506164310997e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506164310997e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.0922506164310997e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506164310997e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.0922506164310997e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506164310997e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.0922506164310997e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506164310997e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.444597854166314e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.444597854166314e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.444597854166314e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.444597854166314e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.6849150954493563e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.6849150954493563e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.6849150954493563e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.6849150954493563e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246974426146891e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246974426146891e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246974426146891e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246974426146891e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246974426146891e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246974426146891e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246974426146891e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246974426146891e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.86776510465863e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.86776510465863e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3002946564159884e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3002946564159884e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.360956320484712e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.360956320484712e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.874299071606893e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.874299071606893e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.88367657617592e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.88367657617592e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.9473560119838607e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9473560119838607e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.9473560119838607e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9473560119838607e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.988511706358717e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.988511706358717e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.0882507116503267e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.0882507116503267e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.1513463114604274e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.1513463114604274e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.846201671499729e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846201671499729e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846201671499729e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846201671499729e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.071480736858093e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.071480736858093e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.105526722287584e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.105526722287584e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.105526722287584e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.105526722287584e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146496327915717e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146496327915717e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146496327915717e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146496327915717e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159350502117206e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159350502117206e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159350502117206e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159350502117206e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427988656897827e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427988656897827e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427988656897827e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427988656897827e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.9358677183425784e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.9358677183425784e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.9358677183425784e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.9358677183425784e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348521488e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.253273348521488e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979825793894477e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979825793894477e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979825793894477e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979825793894477e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.205548411216316e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.205548411216316e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.205548411216316e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.205548411216316e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00013840177303549336) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.00013840177303549336) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.00018787053389548615) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.00018787053389548615) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.00018787053389548615) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00018787053389548615) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00019400857029756982) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00019400857029756982) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0002463643756957689) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0002463643756957689) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0002463643756957689) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0002463643756957689) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.000519274349948777) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.000519274349948777) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156734248908652) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156734248908652) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156734248908652) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156734248908652) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.000787089677102447) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.000787089677102447) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.0015324835230730105) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.0015324835230730105) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.0015324835230730105) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.0015324835230730105) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.001640754855312416) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.001640754855312416) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0017278753941369486) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.0017278753941369486) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.0024464971554158453) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.0024464971554158453) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.0024464971554158453) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.0024464971554158453) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.0032675138544235446) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.0032675138544235446) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.003356670563832879) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.003356670563832879) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.0034841573002178817) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0034841573002178817) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0038764708993369473) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.0038764708993369473) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.004668620318776291) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668620318776291) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278071) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.004767272188278071) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.004767272188278071) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278071) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.005286546538226848) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.005286546538226848) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.005286546538226848) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.005286546538226848) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.005408954422409959) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.005408954422409959) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.005408954422409959) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.005408954422409959) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.0059237983365613405) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.0059237983365613405) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.0059237983365613405) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.0059237983365613405) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.010715508469796763) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010715508469796763) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010715508469796763) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010715508469796763) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.010757563953908927) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010757563953908927) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010757563953908927) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010757563953908927) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0146037047291621) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.0146037047291621) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.0146037047291621) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.0146037047291621) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.01929956057936374) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01929956057936374) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01929956057936374) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01929956057936374) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01929956057936374) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01929956057936374) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.01929956057936374) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01929956057936374) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.05859198873386186) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.05859198873386186) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.775950527605424e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.775950527605424e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.775950527605429e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.775950527605429e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.07165035181002458) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07165035181002458) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.07165035181002458) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.07165035181002458) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251575) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019257505095251575) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0103114824898318) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0103114824898318) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008826368514209836) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209836) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.007597464029770597) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.007597464029770597) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.007597464029770597) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.007597464029770597) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311868) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311868) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311868) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311868) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311868) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311868) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311868) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311868) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676612) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005348051582676612) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005348051582676612) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676612) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0038040661717285425) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0038040661717285425) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984166168121932) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.002984166168121932) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.002984166168121932) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.002984166168121932) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.0024464971554158453) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.0024464971554158453) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.002249412447093987) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.002249412447093987) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.002249412447093987) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002249412447093987) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0021413612231016167) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.0021413612231016167) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.001863894282458731) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.001863894282458731) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.001863894282458731) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.001863894282458731) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.001863894282458731) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.001863894282458731) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.001863894282458731) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.001863894282458731) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.001640754855312416) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.001640754855312416) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.001640754855312416) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.001640754855312416) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.001222337808153827) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.001222337808153827) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.001222337808153827) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.001222337808153827) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.001222337808153827) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.001222337808153827) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.001222337808153827) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.001222337808153827) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0010283292378562572) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0010283292378562572) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0010283292378562572) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0010283292378562572) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1463061453880335e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1463061453880335e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.874299071606893e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.874299071606893e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.874299071606893e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.874299071606893e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3002946564159884e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3002946564159884e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3002946564159884e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3002946564159884e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.044494129870546e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.044494129870546e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.044494129870546e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.044494129870546e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.956079230661803e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.956079230661803e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.956079230661803e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.956079230661803e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.10551503779804e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.10551503779804e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.10551503779804e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.10551503779804e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661347213609852e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661347213609852e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661347213609852e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661347213609852e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341414306419e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341414306419e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189990976130453e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.189990976130453e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876621658673942e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876621658673942e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876621658673942e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876621658673942e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246207486192e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246207486192e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.523389678610142e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.523389678610142e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0767325317710633e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.0767325317710633e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0767325317710633e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.0767325317710633e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.013471459186943e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.013471459186943e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.9045998843990437e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.9045998843990437e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.9045998843990437e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.9045998843990437e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.666731754367204e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.666731754367204e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.666731754367204e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.666731754367204e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.850564192863764e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.850564192863764e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309315108433e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6569309315108433e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309315108433e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6569309315108433e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.850564192863764e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.850564192863764e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.686381547806208e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.686381547806208e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.686381547806208e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.686381547806208e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471459186943e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.013471459186943e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.523389678610142e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.523389678610142e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.670402390697786e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.670402390697786e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.670402390697786e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.670402390697786e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246207486192e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246207486192e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189990976130453e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.189990976130453e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341414306419e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341414306419e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949476488201989e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949476488201989e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.7924939578246941e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939578246941e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939578246941e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.7924939578246941e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.88367657617592e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.88367657617592e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.988511706358717e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.988511706358717e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.988511706358717e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.988511706358717e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348521488e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253273348521488e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4017109736113185e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4017109736113185e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4017109736113185e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4017109736113185e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.580960369393788e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.580960369393788e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.580960369393788e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.580960369393788e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.000519274349948777) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.000519274349948777) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.000519274349948777) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.000519274349948777) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.000787089677102447) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.000787089677102447) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.000787089677102447) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.000787089677102447) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0011726348316441874) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0011726348316441874) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0011726348316441874) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0011726348316441874) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.001236647801924513) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.001236647801924513) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.001236647801924513) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.001236647801924513) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.0022009640695004446) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0022009640695004446) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0022009640695004446) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0022009640695004446) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980145) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980145) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0023949726397980145) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980145) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0023949726397980145) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0023949726397980145) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0023949726397980145) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0023949726397980145) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0024464971554158453) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.0024464971554158453) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.0038040661717285425) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0038040661717285425) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0038764708993369473) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.0038764708993369473) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.0038764708993369473) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.0038764708993369473) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.0042208139700464446) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.0042208139700464446) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.0042208139700464446) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.0042208139700464446) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.008826368514209836) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.008826368514209836) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0103114824898318) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0103114824898318) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257505095251575) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019257505095251575) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.05859198873386186) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.05859198873386186) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3987009015512934e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.3987009015512934e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.3987009015512934e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3987009015512934e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178817) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178817) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984166168121931) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.002984166168121931) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.00019400857029756982) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00019400857029756982) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453880335e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453880335e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7924939578246941e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7924939578246941e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540341414306419e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341414306419e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540341414306419e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341414306419e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.850564192863764e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.850564192863764e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.850564192863764e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.850564192863764e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471459186943e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471459186943e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471459186943e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471459186943e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949476488201988e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949476488201988e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.7924939578246941e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7924939578246941e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756982) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756982) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002984166168121931) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.002984166168121931) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.0034841573002178817) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0034841573002178817) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
  (-73.13873231352538) [I0]
+ (-0.1806679265658352) [Z6]
+ (-0.1806679265658352) [Z7]
+ (-0.15961432501810105) [Z4]
+ (-0.15961432501810105) [Z5]
+ (0.17419956155055527) [Z3]
+ (0.1741995615505553) [Z2]
+ (0.2275726900545334) [Z0]
+ (0.22757269005453357) [Z1]
+ (-8.194261372114766e-06) [Y4 Y6]
+ (-8.194261372114766e-06) [X4 X6]
+ (7.954413176230292e-06) [Y5 Y7]
+ (7.954413176230292e-06) [X5 X7]
+ (0.11270386920332207) [Z4 Z6]
+ (0.11270386920332207) [Z5 Z7]
+ (0.11952438964682666) [Z0 Z4]
+ (0.11952438964682666) [Z1 Z5]
+ (0.13401715261963698) [Z0 Z6]
+ (0.13401715261963698) [Z1 Z7]
+ (0.13734953064261313) [Z0 Z5]
+ (0.13734953064261313) [Z1 Z4]
+ (0.141389052919428) [Z4 Z7]
+ (0.141389052919428) [Z5 Z6]
+ (0.14722943218766166) [Z2 Z5]
+ (0.14722943218766166) [Z3 Z4]
+ (0.1492635514738889) [Z4 Z5]
+ (0.14973486803496922) [Z2 Z6]
+ (0.14973486803496922) [Z3 Z7]
+ (0.15138327161428833) [Z0 Z7]
+ (0.15138327161428833) [Z1 Z6]
+ (0.15435748657223625) [Z6 Z7]
+ (0.15582269051553105) [Z2 Z7]
+ (0.15582269051553105) [Z3 Z6]
+ (0.1675665326546127) [Z0 Z2]
+ (0.1675665326546127) [Z1 Z3]
+ (0.18143991440303875) [Z0 Z3]
+ (0.18143991440303875) [Z1 Z2]
+ (0.19392534613270188) [Z0 Z1]
+ (-7.037887510925774e-06) [Y5 Z6 Y7]
+ (-7.037887510925774e-06) [X5 Z6 X7]
+ (-7.0378875109257695e-06) [Y4 Z5 Y6]
+ (-7.0378875109257695e-06) [X4 Z5 X6]
+ (-0.02868518371610595) [Y4 Y5 X6 X7]
+ (-0.02868518371610595) [X4 X5 Y6 Y7]
+ (-0.017366118994651365) [Y0 Y1 X6 X7]
+ (-0.017366118994651365) [X0 X1 Y6 Y7]
+ (-0.013873381748426089) [Y0 Y1 X2 X3]
+ (-0.013873381748426089) [X0 X1 Y2 Y3]
+ (-0.009560705729135926) [Y2 Y3 X4 X5]
+ (-0.009560705729135926) [X2 X3 Y4 Y5]
+ (-0.006087822480561852) [Y2 Y3 X6 X7]
+ (-0.006087822480561852) [X2 X3 Y6 Y7]
+ (-0.0002921986261110458) [Y1 Y2 X3 X4]
+ (-0.0002921986261110458) [X1 X2 Y3 Y4]
+ (-8.194261372114766e-06) [Z4 Y5 Z6 Y7]
+ (-8.194261372114766e-06) [Z4 X5 Z6 X7]
+ (-2.8909678816874377e-06) [Z0 Y5 Z6 Y7]
+ (-2.8909678816874377e-06) [Z0 X5 Z6 X7]
+ (-2.8909678816874377e-06) [Z1 Y4 Z5 Y6]
+ (-2.8909678816874377e-06) [Z1 X4 Z5 X6]
+ (-1.8551201214416778e-06) [Z0 Y4 Z5 Y6]
+ (-1.8551201214416778e-06) [Z0 X4 Z5 X6]
+ (-1.8551201214416778e-06) [Z1 Y5 Z6 Y7]
+ (-1.8551201214416778e-06) [Z1 X5 Z6 X7]
+ (-1.5973171977483997e-06) [Z2 Y4 Z5 Y6]
+ (-1.5973171977483997e-06) [Z2 X4 Z5 X6]
+ (-1.5973171977483997e-06) [Z3 Y5 Z6 Y7]
+ (-1.5973171977483997e-06) [Z3 X5 Z6 X7]
+ (-1.03584776024576e-06) [Y0 X1 X5 Y6]
+ (-1.03584776024576e-06) [Y0 Y1 Y5 Y6]
+ (-1.03584776024576e-06) [X0 X1 X5 X6]
+ (-1.03584776024576e-06) [X0 Y1 Y5 X6]
+ (-9.344557775811811e-07) [Z2 Y5 Z6 Y7]
+ (-9.344557775811811e-07) [Z2 X5 Z6 X7]
+ (-9.344557775811811e-07) [Z3 Y4 Z5 Y6]
+ (-9.344557775811811e-07) [Z3 X4 Z5 X6]
+ (6.628614201672186e-07) [Y2 X3 X5 Y6]
+ (6.628614201672186e-07) [Y2 Y3 Y5 Y6]
+ (6.628614201672186e-07) [X2 X3 X5 X6]
+ (6.628614201672186e-07) [X2 Y3 Y5 X6]
+ (7.954413176230292e-06) [Y4 Z5 Y6 Z7]
+ (7.954413176230292e-06) [X4 Z5 X6 Z7]
+ (0.0002921986261110458) [Y1 X2 X3 Y4]
+ (0.0002921986261110458) [X1 Y2 Y3 X4]
+ (0.006087822480561852) [Y2 X3 X6 Y7]
+ (0.006087822480561852) [X2 Y3 Y6 X7]
+ (0.009560705729135926) [Y2 X3 X4 Y5]
+ (0.009560705729135926) [X2 Y3 Y4 X5]
+ (0.011307274008848199) [Y1 Z2 Z3 Y5]
+ (0.011307274008848199) [X1 Z2 Z3 X5]
+ (0.013873381748426089) [Y0 X1 X2 Y3]
+ (0.013873381748426089) [X0 Y1 Y2 X3]
+ (0.017366118994651365) [Y0 X1 X6 Y7]
+ (0.017366118994651365) [X0 Y1 Y6 X7]
+ (0.02868518371610595) [Y4 X5 X6 Y7]
+ (0.02868518371610595) [X4 Y5 Y6 X7]
+ (0.029812424517345823) [Y0 Z1 Z2 Y4]
+ (0.029812424517345823) [X0 Z1 Z2 X4]
+ (0.029812424517345823) [Y1 Z3 Z4 Y5]
+ (0.029812424517345823) [X1 Z3 Z4 X5]
+ (0.03010462314345687) [Y0 Z1 Z3 Y4]
+ (0.03010462314345687) [X0 Z1 Z3 X4]
+ (0.03010462314345687) [Y1 Z2 Z4 Y5]
+ (0.03010462314345687) [X1 Z2 Z4 X5]
+ (0.030787505389143963) [Y0 Z2 Z3 Y4]
+ (0.030787505389143963) [X0 Z2 Z3 X4]
+ (0.04375263801065963) [Y1 Z2 Z3 Z4 Y5]
+ (0.04375263801065963) [X1 Z2 Z3 Z4 X5]
+ (0.04375263801065964) [Y0 Z1 Z2 Z3 Y4]
+ (0.04375263801065964) [X0 Z1 Z2 Z3 X4]
+ (-0.014564531231172989) [Y1 Z2 Z3 X4 X6 Y7]
+ (-0.014564531231172989) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-0.014564531231172989) [X1 Z2 Z3 X4 X6 X7]
+ (-0.014564531231172989) [X1 Z2 Z3 Y4 Y6 X7]
+ (-6.524373848441078e-06) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (-6.524373848441078e-06) [X0 Z1 Z2 Z3 Z4 X6]
+ (-6.524373848441078e-06) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (-6.524373848441078e-06) [X1 Z2 Z3 Z5 Z6 X7]
+ (-3.769659451758355e-06) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (-3.769659451758355e-06) [X0 Z2 Z3 Z4 Z5 X6]
+ (-3.6102971303535964e-06) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (-3.6102971303535964e-06) [X0 Z1 Z3 Z4 Z5 X6]
+ (-3.6102971303535964e-06) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (-3.6102971303535964e-06) [X1 Z2 Z4 Z5 Z6 X7]
+ (-3.313145500194655e-06) [Y1 Z2 Z3 Y4 X5 X6]
+ (-3.313145500194655e-06) [X1 Z2 Z3 X4 Y5 Y6]
+ (-3.2774831952720154e-06) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (-3.2774831952720154e-06) [X0 Z1 Z2 Z4 Z5 X6]
+ (-3.2774831952720154e-06) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (-3.2774831952720154e-06) [X1 Z3 Z4 Z5 Z6 X7]
+ (-3.211228348246422e-06) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (-3.211228348246422e-06) [X0 Z1 Z2 Z3 Z5 X6]
+ (-3.211228348246422e-06) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (-3.211228348246422e-06) [X1 Z2 Z3 Z4 Z6 X7]
+ (-1.03584776024576e-06) [Y0 Y1 X4 Z5 Z6 X7]
+ (-1.03584776024576e-06) [X0 X1 Y4 Z5 Z6 Y7]
+ (-6.628614201672188e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (-6.628614201672188e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (-3.3281393508158056e-07) [Y1 X2 X3 Z4 Z5 Y6]
+ (-3.3281393508158056e-07) [X1 Y2 Y3 Z4 Z5 X6]
+ (3.3281393508158056e-07) [Y1 Y2 X3 Z4 Z5 X6]
+ (3.3281393508158056e-07) [X1 X2 Y3 Z4 Z5 Y6]
+ (6.628614201672188e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (6.628614201672188e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (1.03584776024576e-06) [Y0 X1 X4 Z5 Z6 Y7]
+ (1.03584776024576e-06) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.313145500194655e-06) [Y1 Z2 Z3 X4 X5 Y6]
+ (3.313145500194655e-06) [X1 Z2 Z3 Y4 Y5 X6]
+ (4.183932559487133e-06) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (4.183932559487133e-06) [X1 Z2 Z3 Z4 Z5 X7]
+ (0.00029219862611104574) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (0.00029219862611104574) [Y0 Z1 Y2 X3 Z4 X5]
+ (0.00029219862611104574) [X0 Z1 X2 Y3 Z4 Y5]
+ (0.00029219862611104574) [X0 Z1 X2 X3 Z4 X5]
+ (0.010540425907671529) [Y0 Z1 Z2 Z3 Y4 Z7]
+ (0.010540425907671529) [X0 Z1 Z2 Z3 X4 Z7]
+ (0.010540425907671529) [Y1 Z2 Z3 Z4 Y5 Z6]
+ (0.010540425907671529) [X1 Z2 Z3 Z4 X5 Z6]
+ (0.011307274008848199) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (0.011307274008848199) [X0 Z1 Z2 Z3 X4 Z5]
+ (0.025104957138844516) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (0.025104957138844516) [X0 Z1 Z2 Z3 X4 Z6]
+ (0.025104957138844516) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (0.025104957138844516) [X1 Z2 Z3 Z4 X5 Z7]
+ (0.030787505389143963) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (0.030787505389143963) [Z0 X1 Z2 Z3 Z4 X5]
+ (-5.1053965494129645e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-5.1053965494129645e-06) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-5.105396549412964e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (-5.105396549412964e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (-0.014564531231172987) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-0.014564531231172987) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-3.769659451758355e-06) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-3.769659451758355e-06) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-3.328139350815805e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-3.328139350815805e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-3.328139350815805e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-3.328139350815805e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (3.3131455001946553e-06) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (3.3131455001946553e-06) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (3.3131455001946553e-06) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (3.3131455001946553e-06) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (4.183932559487133e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (4.183932559487133e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (0.014564531231172987) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (0.014564531231172987) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
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
(-46.46390678868891+0j) [] +
(-0.014583648907612698+0j) [X0 X1 Y2 Y3] +
(-3.5707613292425203e-07+0j) [X0 X1 Y2 Z3 Z4 Y5] +
(-0.005652620978017369+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7] +
(-0.008826368514209874+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.7924939576044865e-06+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.5707613292425203e-07+0j) [X0 X1 X3 X4] +
(-0.005652620978017369+0j) [X0 X1 X3 Z4 Z5 X6] +
(-0.008826368514209874+0j) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.7924939576044865e-06+0j) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.0027458364701868003+0j) [X0 X1 Y4 Y5] +
(-2.447323128832056e-07+0j) [X0 X1 Y4 Z5 Z6 Y7] +
(-7.867765103819956e-07+0j) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.003804066171728536+0j) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.447323128832056e-07+0j) [X0 X1 X5 X6] +
(-7.867765103819956e-07+0j) [X0 X1 X5 Z6 Z7 Z8 Z9 X10] +
(-0.003804066171728536+0j) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.006888194352970576+0j) [X0 X1 Y6 Y7] +
(-7.735036880592475e-05+0j) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11] +
(1.7035783554644396e-07+0j) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.735036880592475e-05+0j) [X0 X1 X7 Z8 Z9 X10] +
(1.7035783554644402e-07+0j) [X0 X1 X7 Z8 Z9 Z10 Z11 X12] +
(-0.006509361201177232+0j) [X0 X1 Y8 Y9] +
(-0.007731425250775303+0j) [X0 X1 Y10 Y11] +
(5.627851911239184e-07+0j) [X0 X1 Y10 Z11 Z12 Y13] +
(5.627851911239184e-07+0j) [X0 X1 X11 X12] +
(-0.005283776488402967+0j) [X0 X1 Y12 Y13] +
(0.014583648907612698+0j) [X0 Y1 Y2 X3] +
(3.5707613292425203e-07+0j) [X0 Y1 Y2 Z3 Z4 X5] +
(0.005652620978017369+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7] +
(0.008826368514209874+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-1.7924939576044865e-06+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.5707613292425203e-07+0j) [X0 Y1 Y3 X4] +
(-0.005652620978017369+0j) [X0 Y1 Y3 Z4 Z5 X6] +
(-0.008826368514209874+0j) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.7924939576044865e-06+0j) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0027458364701868003+0j) [X0 Y1 Y4 X5] +
(2.447323128832056e-07+0j) [X0 Y1 Y4 Z5 Z6 X7] +
(7.867765103819956e-07+0j) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.003804066171728536+0j) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.447323128832056e-07+0j) [X0 Y1 Y5 X6] +
(-7.867765103819956e-07+0j) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.003804066171728536+0j) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.006888194352970576+0j) [X0 Y1 Y6 X7] +
(7.735036880592475e-05+0j) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11] +
(-1.7035783554644396e-07+0j) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.735036880592475e-05+0j) [X0 Y1 Y7 Z8 Z9 X10] +
(1.7035783554644402e-07+0j) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12] +
(0.006509361201177232+0j) [X0 Y1 Y8 X9] +
(0.007731425250775303+0j) [X0 Y1 Y10 X11] +
(-5.627851911239184e-07+0j) [X0 Y1 Y10 Z11 Z12 X13] +
(5.627851911239184e-07+0j) [X0 Y1 Y11 X12] +
(0.005283776488402967+0j) [X0 Y1 Y12 X13] +
(0.12507032579772331+0j) [X0 Z1 X2] +
(-1.9332412771118596e-07+0j) [X0 Z1 X2 X3 Z4 X5] +
(-0.0022939566113524767+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 X7] +
(-0.0016407548553124436+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(3.013471458736137e-07+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.9332412771118596e-07+0j) [X0 Z1 X2 Y3 Z4 Y5] +
(-0.0022939566113524767+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7] +
(-0.0016407548553124436+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(3.013471458736137e-07+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0032769719312317307+0j) [X0 Z1 X2 Z3] +
(-1.5510539176126766e-07+0j) [X0 Z1 X2 X4 Z5 X6] +
(-1.1468376506598585e-06+0j) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.007597464029770627+0j) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.3807781480675324e-07+0j) [X0 Z1 X2 Y4 Z5 Y6] +
(-7.900128985725649e-07+0j) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.005348051582676637+0j) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005530759218631571+0j) [X0 Z1 X2 Z4] +
(-1.3807781480675324e-07+0j) [X0 Z1 X2 X5 Z6 X7] +
(-3.3767393082598746e-07+0j) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0018638942824587587+0j) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.3807781480675324e-07+0j) [X0 Z1 X2 Y5 Z6 Y7] +
(-3.3767393082598746e-07+0j) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0018638942824587587+0j) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0011059037691897291+0j) [X0 Z1 X2 Z5] +
(0.005708495985960935+0j) [X0 Z1 X2 X6 Z7 Z8 Z9 X10] +
(-8.35233210242304e-07+0j) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
1.9742253793246246e-08j [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005262642473076855+0j) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10] +
(-8.07430598527014e-07+0j) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.008032520918821454+0j) [X0 Z1 X2 Z6] +
(0.0005940221543005544+0j) [X0 Z1 X2 X7 Z8 Z9 Z10 X11] +
(-8.379773243198647e-08+0j) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005940221543005544+0j) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11] +
(-8.379773243198647e-08+0j) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0033476175306662572+0j) [X0 Z1 X2 Z7] +
(0.01105502059613215+0j) [X0 Z1 X2 Z8] +
(0.0029297686747511284+0j) [X0 Z1 X2 Z9] +
(-6.418291574217039e-07+0j) [X0 Z1 X2 X10 Z11 X12] +
(-6.556281914112998e-07+0j) [X0 Z1 X2 Y10 Z11 Y12] +
(0.003555290195504306+0j) [X0 Z1 X2 Z10] +
(-1.1076325598317528e-07+0j) [X0 Z1 X2 X11 Z12 X13] +
(-1.1076325598317528e-07+0j) [X0 Z1 X2 Y11 Z12 Y13] +
(0.001756070701841304+0j) [X0 Z1 X2 Z11] +
(0.006901238249797359+0j) [X0 Z1 X2 Z12] +
(0.002326230623158148+0j) [X0 Z1 X2 Z13] +
(-3.5682475208729344e-07+0j) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0022494124470939904+0j) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.0474716554332859e-08+0j) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0004458535128840802+0j) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10] +
(-1.9742253793283145e-08+0j) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441841+0j) [X0 Z1 Z2 X3 Y4 Y5] +
(-4.523389677465775e-07+0j) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.003484157300217878+0j) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-8.091637198338709e-07+0j) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10] +
(-0.0057335697473118695+0j) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004684903388155197+0j) [X0 Z1 Z2 X3 Y6 Y7] +
(0.004668620318776302+0j) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11] +
(-7.189990974755708e-07+0j) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005114473831660382+0j) [X0 Z1 Z2 X3 X7 Z8 Z9 X10] +
(-7.560692464297741e-07+0j) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12] +
(-0.00812525192138102+0j) [X0 Z1 Z2 X3 Y8 Y9] +
(-0.0017992194936630014+0j) [X0 Z1 Z2 X3 Y10 Y11] +
(-5.47164774426425e-07+0j) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13] +
(-5.287660624402284e-07+0j) [X0 Z1 Z2 X3 X11 X12] +
(-0.004575007626639211+0j) [X0 Z1 Z2 X3 Y12 Y13] +
(0.004424855449441841+0j) [X0 Z1 Z2 Y3 Y4 X5] +
(4.523389677465775e-07+0j) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.003484157300217878+0j) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.091637198338709e-07+0j) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.0057335697473118695+0j) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.004684903388155197+0j) [X0 Z1 Z2 Y3 Y6 X7] +
(-0.004668620318776302+0j) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11] +
(7.189990974755708e-07+0j) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005114473831660382+0j) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10] +
(-7.560692464297741e-07+0j) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12] +
(0.00812525192138102+0j) [X0 Z1 Z2 Y3 Y8 X9] +
(0.0017992194936630014+0j) [X0 Z1 Z2 Y3 Y10 X11] +
(5.47164774426425e-07+0j) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13] +
(-5.287660624402284e-07+0j) [X0 Z1 Z2 Y3 Y11 X12] +
(0.004575007626639211+0j) [X0 Z1 Z2 Y3 Y12 X13] +
(3.2020768797081216e-06+0j) [X0 Z1 Z2 Z3 X4] +
(0.0008533856254125473+0j) [X0 Z1 Z2 Z3 X4 X5 Z6 X7] +
(0.0007870896771024431+0j) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0008533856254125473+0j) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7] +
(0.0007870896771024431+0j) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.2919694865350102e-07+0j) [X0 Z1 Z2 Z3 X4 Z5] +
(4.4445978539252117e-07+0j) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10] +
(0.001172634831644184+0j) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.6849150949283104e-07+0j) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10] +
(0.0022009640695004563+0j) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.8394209155175266e-07+0j) [X0 Z1 Z2 Z3 X4 Z6] +
(4.092250615741314e-07+0j) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11] +
(0.0023949726397980214+0j) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.092250615741314e-07+0j) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11] +
(0.0023949726397980214+0j) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.236259961263184e-07+0j) [X0 Z1 Z2 Z3 X4 Z8] +
(8.649310132521708e-08+0j) [X0 Z1 Z2 Z3 X4 Z9] +
(0.0013038004788126943+0j) [X0 Z1 Z2 Z3 X4 X10 Z11 X12] +
(0.003989841456619317+0j) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12] +
(-6.733197741843018e-07+0j) [X0 Z1 Z2 Z3 X4 Z10] +
(0.00226196606248235+0j) [X0 Z1 Z2 Z3 X4 X11 Z12 X13] +
(0.00226196606248235+0j) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13] +
(-5.927453082391738e-07+0j) [X0 Z1 Z2 Z3 X4 Z11] +
(1.2393363216420002e-06+0j) [X0 Z1 Z2 Z3 X4 Z12] +
(9.306536651221527e-07+0j) [X0 Z1 Z2 Z3 X4 Z13] +
(-0.0010283292378562726+0j) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.002686040977806622+0j) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12] +
(-1.8394209155175266e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7] +
(-0.0001940085702975649+0j) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0012223378081538377+0j) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-2.3713289480110134e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9] +
(8.057446594512792e-08+0j) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11] +
(0.0017278753941369677+0j) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13] +
(-0.0009581655836696554+0j) [X0 Z1 Z2 Z3 Z4 X5 X11 X12] +
(-3.086826565198475e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13] +
(1.8394209155175266e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7] +
(0.0001940085702975649+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0012223378081538377+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(2.3713289480110134e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9] +
(-8.057446594512792e-08+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11] +
(-0.0017278753941369677+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13] +
(-0.0009581655836696554+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12] +
(3.086826565198475e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13] +
(0.04274327701378319+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6] +
(0.0005192743499487691+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(-1.8505641927803864e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005192743499487691+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(-1.8505641927803864e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0027790267990255545+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7] +
(0.004636976661182564+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8] +
(0.0012803060973496701+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9] +
(2.3120943051420206e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12] +
(1.0717282182567258e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12] +
(0.005379937155839379+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10] +
(7.246974424931244e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13] +
(7.246974424931244e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13] +
(0.005241535382803876+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11] +
(0.004311038507914315+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12] +
(0.0010435246534907584+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13] +
(1.2004287493541528e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12] +
(-0.003356670563832894+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9] +
(-0.00013840177303550328+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11] +
(-6.175246206674518e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(-4.997018421675544e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12] +
(-0.003267513854423557+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13] +
(0.003356670563832894+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9] +
(0.00013840177303550328+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11] +
(6.175246206674518e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(-4.997018421675544e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12] +
(0.003267513854423557+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13] +
(0.0038764708993369507+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(-7.540341413240216e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.0038764708993369507+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(-7.540341413240216e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(0.07165035181002737+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0021413612231015585+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(0.004220813970046511+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(0.0012366478019245745+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(-0.0029841661681219373+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(0.0029841661681219373+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-1.3987009015439525e-05+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(8.949476486467516e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-6.87662165802578e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(-7.661347212839333e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(0.001532483523073075+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10] +
(-2.9045998844893215e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(0.005408954422410026+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10] +
(-1.0444941297729537e-06+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(0.004767272188278155+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10] +
(-8.105515036898032e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(0.005286546538226924+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10] +
(-9.956079229678416e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(0.001609531381721372+0j) [X0 Z1 Z2 Z3 Z4 X6] +
(-7.141625221153381e-05+0j) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10] +
(-2.666731755093001e-07+0j) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.002462917007133919+0j) [X0 Z1 Z2 Z3 Z5 X6] +
(0.0007156734248909091+0j) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10] +
(-3.0767325321580573e-07+0j) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.6060718677803603e-07+0j) [X0 Z1 Z2 X4] +
(0.003961560792496535+0j) [X0 Z1 Z2 Z4 Z5 X6] +
(0.0001878705338955407+0j) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.6569309313458506e-07+0j) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.7379332622789548e-07+0j) [X0 Z1 Z3 X4] +
(0.001667604181144058+0j) [X0 Z1 Z3 Z4 Z5 X6] +
(-0.001452884321416903+0j) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(4.6704023900819875e-07+0j) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.10433064780651422+0j) [X0 X2] +
(3.117447946137541e-06+0j) [X0 Z2 Z3 X4] +
(0.04587947078129805+0j) [X0 Z2 Z3 Z4 Z5 X6] +
(0.058591988733862024+0j) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-1.1463061452457945e-05+0j) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.014583648907612698+0j) [Y0 X1 X2 Y3] +
(3.5707613292425203e-07+0j) [Y0 X1 X2 Z3 Z4 Y5] +
(0.005652620978017369+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7] +
(0.008826368514209874+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.7924939576044865e-06+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.5707613292425203e-07+0j) [Y0 X1 X3 Y4] +
(-0.005652620978017369+0j) [Y0 X1 X3 Z4 Z5 Y6] +
(-0.008826368514209874+0j) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.7924939576044865e-06+0j) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0027458364701868003+0j) [Y0 X1 X4 Y5] +
(2.447323128832056e-07+0j) [Y0 X1 X4 Z5 Z6 Y7] +
(7.867765103819956e-07+0j) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.003804066171728536+0j) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.447323128832056e-07+0j) [Y0 X1 X5 Y6] +
(-7.867765103819956e-07+0j) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.003804066171728536+0j) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.006888194352970576+0j) [Y0 X1 X6 Y7] +
(7.735036880592475e-05+0j) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11] +
(-1.7035783554644396e-07+0j) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.735036880592475e-05+0j) [Y0 X1 X7 Z8 Z9 Y10] +
(1.7035783554644402e-07+0j) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12] +
(0.006509361201177232+0j) [Y0 X1 X8 Y9] +
(0.007731425250775303+0j) [Y0 X1 X10 Y11] +
(-5.627851911239184e-07+0j) [Y0 X1 X10 Z11 Z12 Y13] +
(5.627851911239184e-07+0j) [Y0 X1 X11 Y12] +
(0.005283776488402967+0j) [Y0 X1 X12 Y13] +
(-0.014583648907612698+0j) [Y0 Y1 X2 X3] +
(-3.5707613292425203e-07+0j) [Y0 Y1 X2 Z3 Z4 X5] +
(-0.005652620978017369+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7] +
(-0.008826368514209874+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.7924939576044865e-06+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.5707613292425203e-07+0j) [Y0 Y1 Y3 Y4] +
(-0.005652620978017369+0j) [Y0 Y1 Y3 Z4 Z5 Y6] +
(-0.008826368514209874+0j) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.7924939576044865e-06+0j) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.0027458364701868003+0j) [Y0 Y1 X4 X5] +
(-2.447323128832056e-07+0j) [Y0 Y1 X4 Z5 Z6 X7] +
(-7.867765103819956e-07+0j) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.003804066171728536+0j) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.447323128832056e-07+0j) [Y0 Y1 Y5 Y6] +
(-7.867765103819956e-07+0j) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.003804066171728536+0j) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.006888194352970576+0j) [Y0 Y1 X6 X7] +
(-7.735036880592475e-05+0j) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11] +
(1.7035783554644396e-07+0j) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.735036880592475e-05+0j) [Y0 Y1 Y7 Z8 Z9 Y10] +
(1.7035783554644402e-07+0j) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.006509361201177232+0j) [Y0 Y1 X8 X9] +
(-0.007731425250775303+0j) [Y0 Y1 X10 X11] +
(5.627851911239184e-07+0j) [Y0 Y1 X10 Z11 Z12 X13] +
(5.627851911239184e-07+0j) [Y0 Y1 Y11 Y12] +
(-0.005283776488402967+0j) [Y0 Y1 X12 X13] +
(-3.5682475208729344e-07+0j) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0022494124470939904+0j) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0004458535128840802+0j) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10] +
(-1.9742253793283145e-08+0j) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.0474716554332859e-08+0j) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.12507032579772331+0j) [Y0 Z1 Y2] +
(-1.9332412771118596e-07+0j) [Y0 Z1 Y2 X3 Z4 X5] +
(-0.0022939566113524767+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7] +
(-0.0016407548553124436+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(3.013471458736137e-07+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.9332412771118596e-07+0j) [Y0 Z1 Y2 Y3 Z4 Y5] +
(-0.0022939566113524767+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7] +
(-0.0016407548553124436+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(3.013471458736137e-07+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0032769719312317307+0j) [Y0 Z1 Y2 Z3] +
(-1.3807781480675324e-07+0j) [Y0 Z1 Y2 X4 Z5 X6] +
(-7.900128985725649e-07+0j) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.005348051582676637+0j) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.5510539176126766e-07+0j) [Y0 Z1 Y2 Y4 Z5 Y6] +
(-1.1468376506598585e-06+0j) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.007597464029770627+0j) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005530759218631571+0j) [Y0 Z1 Y2 Z4] +
(-1.3807781480675324e-07+0j) [Y0 Z1 Y2 X5 Z6 X7] +
(-3.3767393082598746e-07+0j) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0018638942824587587+0j) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.3807781480675324e-07+0j) [Y0 Z1 Y2 Y5 Z6 Y7] +
(-3.3767393082598746e-07+0j) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0018638942824587587+0j) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0011059037691897291+0j) [Y0 Z1 Y2 Z5] +
(0.005262642473076855+0j) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10] +
(-8.07430598527014e-07+0j) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.005708495985960935+0j) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10] +
-1.9742253793246246e-08j [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-8.35233210242304e-07+0j) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.008032520918821454+0j) [Y0 Z1 Y2 Z6] +
(0.0005940221543005544+0j) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11] +
(-8.379773243198647e-08+0j) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005940221543005544+0j) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11] +
(-8.379773243198647e-08+0j) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0033476175306662572+0j) [Y0 Z1 Y2 Z7] +
(0.01105502059613215+0j) [Y0 Z1 Y2 Z8] +
(0.0029297686747511284+0j) [Y0 Z1 Y2 Z9] +
(-6.556281914112998e-07+0j) [Y0 Z1 Y2 X10 Z11 X12] +
(-6.418291574217039e-07+0j) [Y0 Z1 Y2 Y10 Z11 Y12] +
(0.003555290195504306+0j) [Y0 Z1 Y2 Z10] +
(-1.1076325598317528e-07+0j) [Y0 Z1 Y2 X11 Z12 X13] +
(-1.1076325598317528e-07+0j) [Y0 Z1 Y2 Y11 Z12 Y13] +
(0.001756070701841304+0j) [Y0 Z1 Y2 Z11] +
(0.006901238249797359+0j) [Y0 Z1 Y2 Z12] +
(0.002326230623158148+0j) [Y0 Z1 Y2 Z13] +
(0.004424855449441841+0j) [Y0 Z1 Z2 X3 X4 Y5] +
(4.523389677465775e-07+0j) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.003484157300217878+0j) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-8.091637198338709e-07+0j) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.0057335697473118695+0j) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.004684903388155197+0j) [Y0 Z1 Z2 X3 X6 Y7] +
(-0.004668620318776302+0j) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11] +
(7.189990974755708e-07+0j) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005114473831660382+0j) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10] +
(-7.560692464297741e-07+0j) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12] +
(0.00812525192138102+0j) [Y0 Z1 Z2 X3 X8 Y9] +
(0.0017992194936630014+0j) [Y0 Z1 Z2 X3 X10 Y11] +
(5.47164774426425e-07+0j) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13] +
(-5.287660624402284e-07+0j) [Y0 Z1 Z2 X3 X11 Y12] +
(0.004575007626639211+0j) [Y0 Z1 Z2 X3 X12 Y13] +
(-0.004424855449441841+0j) [Y0 Z1 Z2 Y3 X4 X5] +
(-4.523389677465775e-07+0j) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.003484157300217878+0j) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.091637198338709e-07+0j) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.0057335697473118695+0j) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004684903388155197+0j) [Y0 Z1 Z2 Y3 X6 X7] +
(0.004668620318776302+0j) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11] +
(-7.189990974755708e-07+0j) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005114473831660382+0j) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10] +
(-7.560692464297741e-07+0j) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.00812525192138102+0j) [Y0 Z1 Z2 Y3 X8 X9] +
(-0.0017992194936630014+0j) [Y0 Z1 Z2 Y3 X10 X11] +
(-5.47164774426425e-07+0j) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13] +
(-5.287660624402284e-07+0j) [Y0 Z1 Z2 Y3 Y11 Y12] +
(-0.004575007626639211+0j) [Y0 Z1 Z2 Y3 X12 X13] +
(-0.0010283292378562726+0j) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.002686040977806622+0j) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12] +
(3.2020768797081216e-06+0j) [Y0 Z1 Z2 Z3 Y4] +
(0.0008533856254125473+0j) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7] +
(0.0007870896771024431+0j) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0008533856254125473+0j) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7] +
(0.0007870896771024431+0j) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.2919694865350102e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z5] +
(4.6849150949283104e-07+0j) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10] +
(0.0022009640695004563+0j) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.4445978539252117e-07+0j) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10] +
(0.001172634831644184+0j) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.8394209155175266e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z6] +
(4.092250615741314e-07+0j) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11] +
(0.0023949726397980214+0j) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.092250615741314e-07+0j) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11] +
(0.0023949726397980214+0j) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.236259961263184e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z8] +
(8.649310132521708e-08+0j) [Y0 Z1 Z2 Z3 Y4 Z9] +
(0.003989841456619317+0j) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12] +
(0.0013038004788126943+0j) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12] +
(-6.733197741843018e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z10] +
(0.00226196606248235+0j) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13] +
(0.00226196606248235+0j) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13] +
(-5.927453082391738e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z11] +
(1.2393363216420002e-06+0j) [Y0 Z1 Z2 Z3 Y4 Z12] +
(9.306536651221527e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z13] +
(1.8394209155175266e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7] +
(0.0001940085702975649+0j) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0012223378081538377+0j) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(2.3713289480110134e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9] +
(-8.057446594512792e-08+0j) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11] +
(-0.0017278753941369677+0j) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13] +
(-0.0009581655836696554+0j) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12] +
(3.086826565198475e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13] +
(-1.8394209155175266e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7] +
(-0.0001940085702975649+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0012223378081538377+0j) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-2.3713289480110134e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9] +
(8.057446594512792e-08+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11] +
(0.0017278753941369677+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13] +
(-0.0009581655836696554+0j) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12] +
(-3.086826565198475e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13] +
(1.2004287493541528e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12] +
(0.04274327701378319+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6] +
(0.0005192743499487691+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(-1.8505641927803864e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005192743499487691+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(-1.8505641927803864e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0027790267990255545+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7] +
(0.004636976661182564+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8] +
(0.0012803060973496701+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9] +
(1.0717282182567258e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12] +
(2.3120943051420206e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12] +
(0.005379937155839379+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10] +
(7.246974424931244e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13] +
(7.246974424931244e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13] +
(0.005241535382803876+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11] +
(0.004311038507914315+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12] +
(0.0010435246534907584+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13] +
(0.003356670563832894+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9] +
(0.00013840177303550328+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11] +
(6.175246206674518e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(-4.997018421675544e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12] +
(0.003267513854423557+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13] +
(-0.003356670563832894+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9] +
(-0.00013840177303550328+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11] +
(-6.175246206674518e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(-4.997018421675544e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12] +
(-0.003267513854423557+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13] +
(0.0038764708993369507+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(-7.540341413240216e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.0038764708993369507+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(-7.540341413240216e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(0.07165035181002737+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0021413612231015585+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(0.004220813970046511+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(0.0012366478019245745+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(0.0029841661681219373+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-0.0029841661681219373+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-1.3987009015439525e-05+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(8.949476486467516e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-6.87662165802578e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(-7.661347212839333e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(0.001532483523073075+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10] +
(-2.9045998844893215e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(0.005408954422410026+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10] +
(-1.0444941297729537e-06+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(0.004767272188278155+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10] +
(-8.105515036898032e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(0.005286546538226924+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10] +
(-9.956079229678416e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.001609531381721372+0j) [Y0 Z1 Z2 Z3 Z4 Y6] +
(-7.141625221153381e-05+0j) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10] +
(-2.666731755093001e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.002462917007133919+0j) [Y0 Z1 Z2 Z3 Z5 Y6] +
(0.0007156734248909091+0j) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10] +
(-3.0767325321580573e-07+0j) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(3.6060718677803603e-07+0j) [Y0 Z1 Z2 Y4] +
(0.003961560792496535+0j) [Y0 Z1 Z2 Z4 Z5 Y6] +
(0.0001878705338955407+0j) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.6569309313458506e-07+0j) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.7379332622789548e-07+0j) [Y0 Z1 Z3 Y4] +
(0.001667604181144058+0j) [Y0 Z1 Z3 Z4 Z5 Y6] +
(-0.001452884321416903+0j) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(4.6704023900819875e-07+0j) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.10433064780651422+0j) [Y0 Y2] +
(3.117447946137541e-06+0j) [Y0 Z2 Z3 Y4] +
(0.04587947078129805+0j) [Y0 Z2 Z3 Z4 Z5 Y6] +
(0.058591988733862024+0j) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-1.1463061452457945e-05+0j) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(12.412630742111762+0j) [Z0] +
(0.10433064780651422+0j) [Z0 X1 Z2 X3] +
(3.117447946137541e-06+0j) [Z0 X1 Z2 Z3 Z4 X5] +
(0.04587947078129805+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7] +
(0.05859198873386202+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-1.1463061452457945e-05+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.10433064780651422+0j) [Z0 Y1 Z2 Y3] +
(3.117447946137541e-06+0j) [Z0 Y1 Z2 Z3 Z4 Y5] +
(0.04587947078129805+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7] +
(0.05859198873386202+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.1463061452457945e-05+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.1861763734860487+0j) [Z0 Z1] +
(-8.337746755398009e-07+0j) [Z0 X2 Z3 X4] +
(-0.027115036845273183+0j) [Z0 X2 Z3 Z4 Z5 X6] +
(-0.06752385099214042+0j) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.4017109734481867e-05+0j) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-8.337746755398009e-07+0j) [Z0 Y2 Z3 Y4] +
(-0.027115036845273183+0j) [Z0 Y2 Z3 Z4 Z5 Y6] +
(-0.06752385099214042+0j) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.4017109734481867e-05+0j) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.23671080783830417+0j) [Z0 Z2] +
(-1.1908508084640527e-06+0j) [Z0 X3 Z4 X5] +
(-0.03276765782329055+0j) [Z0 X3 Z4 Z5 Z6 X7] +
(-0.07635021950635029+0j) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.5809603692086352e-05+0j) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.1908508084640527e-06+0j) [Z0 Y3 Z4 Y5] +
(-0.03276765782329055+0j) [Z0 Y3 Z4 Z5 Z6 Y7] +
(-0.07635021950635029+0j) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.5809603692086352e-05+0j) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.2512944567459169+0j) [Z0 Z3] +
(-3.099349243575037e-06+0j) [Z0 X4 Z5 X6] +
(-1.5316808794326665e-05+0j) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.08684737589863618+0j) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-3.099349243575037e-06+0j) [Z0 Y4 Z5 Y6] +
(-1.5316808794326665e-05+0j) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.08684737589863618+0j) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.19661770890342117+0j) [Z0 Z4] +
(-3.344081556458242e-06+0j) [Z0 X5 Z6 X7] +
(-1.6103585304708656e-05+0j) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.09065144207036469+0j) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.344081556458242e-06+0j) [Z0 Y5 Z6 Y7] +
(-1.6103585304708656e-05+0j) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.09065144207036469+0j) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.19936354537360795+0j) [Z0 Z5] +
(0.056084681246613394+0j) [Z0 X6 Z7 Z8 Z9 X10] +
(-6.652209668791919e-06+0j) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.056084681246613394+0j) [Z0 Y6 Z7 Z8 Z9 Y10] +
(-6.652209668791919e-06+0j) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.24164663936017197+0j) [Z0 Z6] +
(0.05600733087780746+0j) [Z0 X7 Z8 Z9 Z10 X11] +
(-6.481851833245475e-06+0j) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.05600733087780746+0j) [Z0 Y7 Z8 Z9 Z10 Y11] +
(-6.481851833245475e-06+0j) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.24853483371314253+0j) [Z0 Z7] +
(0.2723251830660567+0j) [Z0 Z8] +
(-2.177664604637626e-06+0j) [Z0 X10 Z11 X12] +
(-2.177664604637626e-06+0j) [Z0 Y10 Z11 Y12] +
(0.19299723935364246+0j) [Z0 Z10] +
(-1.6148794135137072e-06+0j) [Z0 X11 Z12 X13] +
(-1.6148794135137072e-06+0j) [Z0 Y11 Z12 Y13] +
(0.20072866460441777+0j) [Z0 Z11] +
(0.21102659849791544+0j) [Z0 Z12] +
(0.2163103749863184+0j) [Z0 Z13] +
(1.9332412771118596e-07+0j) [X1 X2 Y3 Y4] +
(0.0016407548553124436+0j) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-3.013471458736137e-07+0j) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004424855449441841+0j) [X1 X2 X4 X5] +
(-8.091637198338709e-07+0j) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0057335697473118695+0j) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-4.523389677465775e-07+0j) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.003484157300217878+0j) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004684903388155197+0j) [X1 X2 X6 X7] +
(0.005114473831660382+0j) [X1 X2 X6 Z7 Z8 Z9 Z10 X11] +
(-7.560692464297741e-07+0j) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.004668620318776302+0j) [X1 X2 Y7 Z8 Z9 Y10] +
(-7.189990974755708e-07+0j) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.00812525192138102+0j) [X1 X2 X8 X9] +
(-0.0017992194936630017+0j) [X1 X2 X10 X11] +
(-5.287660624402284e-07+0j) [X1 X2 X10 Z11 Z12 X13] +
(-5.47164774426425e-07+0j) [X1 X2 Y11 Y12] +
(-0.004575007626639211+0j) [X1 X2 X12 X13] +
(-1.9332412771118596e-07+0j) [X1 Y2 Y3 X4] +
(-0.0016407548553124436+0j) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(3.013471458736137e-07+0j) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441841+0j) [X1 Y2 Y4 X5] +
(-8.091637198338709e-07+0j) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0057335697473118695+0j) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.523389677465775e-07+0j) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10] +
(0.003484157300217878+0j) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004684903388155197+0j) [X1 Y2 Y6 X7] +
(0.005114473831660382+0j) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11] +
(-7.560692464297741e-07+0j) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.004668620318776302+0j) [X1 Y2 Y7 Z8 Z9 X10] +
(7.189990974755708e-07+0j) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.00812525192138102+0j) [X1 Y2 Y8 X9] +
(-0.0017992194936630017+0j) [X1 Y2 Y10 X11] +
(-5.287660624402284e-07+0j) [X1 Y2 Y10 Z11 Z12 X13] +
(5.47164774426425e-07+0j) [X1 Y2 Y11 X12] +
(-0.004575007626639211+0j) [X1 Y2 Y12 X13] +
(0.1250703257977234+0j) [X1 Z2 X3] +
(-1.3807781480675324e-07+0j) [X1 Z2 X3 X4 Z5 X6] +
(-3.3767393082598746e-07+0j) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0018638942824587587+0j) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.3807781480675324e-07+0j) [X1 Z2 X3 Y4 Z5 Y6] +
(-3.3767393082598746e-07+0j) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0018638942824587587+0j) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0011059037691897291+0j) [X1 Z2 X3 Z4] +
(-1.5510539176126766e-07+0j) [X1 Z2 X3 X5 Z6 X7] +
(-1.1468376506598585e-06+0j) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.007597464029770627+0j) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.3807781480675324e-07+0j) [X1 Z2 X3 Y5 Z6 Y7] +
(-7.900128985725649e-07+0j) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.005348051582676637+0j) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005530759218631571+0j) [X1 Z2 X3 Z5] +
(0.0005940221543005544+0j) [X1 Z2 X3 X6 Z7 Z8 Z9 X10] +
(-8.379773243198647e-08+0j) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0005940221543005544+0j) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10] +
(-8.379773243198647e-08+0j) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0033476175306662572+0j) [X1 Z2 X3 Z6] +
(0.005708495985960935+0j) [X1 Z2 X3 X7 Z8 Z9 Z10 X11] +
(-8.35233210242304e-07+0j) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
1.9742253793246246e-08j [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005262642473076855+0j) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11] +
(-8.07430598527014e-07+0j) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.008032520918821454+0j) [X1 Z2 X3 Z7] +
(0.0029297686747511284+0j) [X1 Z2 X3 Z8] +
(0.01105502059613215+0j) [X1 Z2 X3 Z9] +
(-1.1076325598317528e-07+0j) [X1 Z2 X3 X10 Z11 X12] +
(-1.1076325598317528e-07+0j) [X1 Z2 X3 Y10 Z11 Y12] +
(0.001756070701841304+0j) [X1 Z2 X3 Z10] +
(-6.418291574217039e-07+0j) [X1 Z2 X3 X11 Z12 X13] +
(-6.556281914112998e-07+0j) [X1 Z2 X3 Y11 Z12 Y13] +
(0.003555290195504306+0j) [X1 Z2 X3 Z11] +
(0.002326230623158148+0j) [X1 Z2 X3 Z12] +
(0.006901238249797359+0j) [X1 Z2 X3 Z13] +
(-3.5682475208729344e-07+0j) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0022494124470939904+0j) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.0474716554332859e-08+0j) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0004458535128840802+0j) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11] +
(-1.9742253793283145e-08+0j) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0008533856254125472+0j) [X1 Z2 Z3 X4 Y5 Y6] +
(-0.0007870896771024431+0j) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10] +
(-1.8394209155175266e-07+0j) [X1 Z2 Z3 X4 X6 X7] +
(-0.0012223378081538377+0j) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0001940085702975649+0j) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12] +
(-2.3713289480110134e-07+0j) [X1 Z2 Z3 X4 X8 X9] +
(8.057446594512792e-08+0j) [X1 Z2 Z3 X4 X10 X11] +
(-0.0009581655836696554+0j) [X1 Z2 Z3 X4 X10 Z11 Z12 X13] +
(0.0017278753941369677+0j) [X1 Z2 Z3 X4 Y11 Y12] +
(-3.086826565198475e-07+0j) [X1 Z2 Z3 X4 X12 X13] +
(0.0008533856254125472+0j) [X1 Z2 Z3 Y4 Y5 X6] +
(0.0007870896771024431+0j) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10] +
(-1.8394209155175266e-07+0j) [X1 Z2 Z3 Y4 Y6 X7] +
(-0.0012223378081538377+0j) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0001940085702975649+0j) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12] +
(-2.3713289480110134e-07+0j) [X1 Z2 Z3 Y4 Y8 X9] +
(8.057446594512792e-08+0j) [X1 Z2 Z3 Y4 Y10 X11] +
(-0.0009581655836696554+0j) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13] +
(-0.0017278753941369677+0j) [X1 Z2 Z3 Y4 Y11 X12] +
(-3.086826565198475e-07+0j) [X1 Z2 Z3 Y4 Y12 X13] +
(3.2020768797081232e-06+0j) [X1 Z2 Z3 Z4 X5] +
(4.092250615741314e-07+0j) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10] +
(0.0023949726397980214+0j) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.092250615741314e-07+0j) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10] +
(0.0023949726397980214+0j) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.4445978539252117e-07+0j) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11] +
(0.001172634831644184+0j) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.6849150949283104e-07+0j) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11] +
(0.0022009640695004563+0j) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.8394209155175266e-07+0j) [X1 Z2 Z3 Z4 X5 Z7] +
(8.649310132521708e-08+0j) [X1 Z2 Z3 Z4 X5 Z8] +
(3.236259961263184e-07+0j) [X1 Z2 Z3 Z4 X5 Z9] +
(0.00226196606248235+0j) [X1 Z2 Z3 Z4 X5 X10 Z11 X12] +
(0.00226196606248235+0j) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12] +
(-5.927453082391738e-07+0j) [X1 Z2 Z3 Z4 X5 Z10] +
(0.0013038004788126943+0j) [X1 Z2 Z3 Z4 X5 X11 Z12 X13] +
(0.003989841456619317+0j) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13] +
(-6.733197741843018e-07+0j) [X1 Z2 Z3 Z4 X5 Z11] +
(9.306536651221527e-07+0j) [X1 Z2 Z3 Z4 X5 Z12] +
(1.2393363216420002e-06+0j) [X1 Z2 Z3 Z4 X5 Z13] +
(-0.0010283292378562726+0j) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.002686040977806622+0j) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13] +
(-0.0005192743499487691+0j) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10] +
(1.8505641927803864e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.0033566705638328944+0j) [X1 Z2 Z3 Z4 Z5 X6 X8 X9] +
(-0.00013840177303550328+0j) [X1 Z2 Z3 Z4 Z5 X6 X10 X11] +
(-4.997018421675544e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13] +
(-6.175246206674518e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12] +
(-0.003267513854423557+0j) [X1 Z2 Z3 Z4 Z5 X6 X12 X13] +
(0.0005192743499487691+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10] +
(-1.8505641927803864e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.0033566705638328944+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9] +
(-0.00013840177303550328+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11] +
(-4.997018421675544e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13] +
(6.175246206674518e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12] +
(-0.003267513854423557+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13] +
(0.042743277013783186+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7] +
(0.0012803060973496701+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8] +
(0.004636976661182564+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9] +
(7.246974424931244e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12] +
(7.246974424931244e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12] +
(0.005241535382803876+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10] +
(2.3120943051420206e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13] +
(1.0717282182567258e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13] +
(0.005379937155839379+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11] +
(0.0010435246534907584+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12] +
(0.004311038507914315+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13] +
(1.2004287493541528e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13] +
(-0.003876470899336951+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10] +
(7.540341413240216e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(0.003876470899336951+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10] +
(-7.540341413240216e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(-0.0029841661681219373+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-0.0029841661681219373+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(0.07165035181002738+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0012366478019245745+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(0.004220813970046511+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(-1.3987009015439521e-05+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(8.949476486467515e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(-7.661347212839333e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(-0.0021413612231015585+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11] +
(-6.87662165802578e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(0.005408954422410026+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11] +
(-1.0444941297729537e-06+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(0.001532483523073075+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11] +
(-2.9045998844893215e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(0.005286546538226924+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11] +
(-9.956079229678416e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.002779026799025555+0j) [X1 Z2 Z3 Z4 Z5 X7] +
(0.004767272188278155+0j) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11] +
(-8.105515036898032e-07+0j) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.002462917007133919+0j) [X1 Z2 Z3 Z4 Z6 X7] +
(0.0007156734248909091+0j) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11] +
(-3.0767325321580573e-07+0j) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2919694865350102e-07+0j) [X1 Z2 Z3 X5] +
(0.001609531381721372+0j) [X1 Z2 Z3 Z5 Z6 X7] +
(-7.141625221153381e-05+0j) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-2.666731755093001e-07+0j) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.7379332622789548e-07+0j) [X1 Z2 Z4 X5] +
(0.001667604181144058+0j) [X1 Z2 Z4 Z5 Z6 X7] +
(-0.001452884321416903+0j) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(4.6704023900819875e-07+0j) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0032769719312317307+0j) [X1 X3] +
(3.6060718677803603e-07+0j) [X1 Z3 Z4 X5] +
(0.003961560792496535+0j) [X1 Z3 Z4 Z5 Z6 X7] +
(0.0001878705338955407+0j) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.6569309313458506e-07+0j) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.9332412771118596e-07+0j) [Y1 X2 X3 Y4] +
(-0.0016407548553124436+0j) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(3.013471458736137e-07+0j) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004424855449441841+0j) [Y1 X2 X4 Y5] +
(-8.091637198338709e-07+0j) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0057335697473118695+0j) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(4.523389677465775e-07+0j) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10] +
(0.003484157300217878+0j) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004684903388155197+0j) [Y1 X2 X6 Y7] +
(0.005114473831660382+0j) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11] +
(-7.560692464297741e-07+0j) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.004668620318776302+0j) [Y1 X2 X7 Z8 Z9 Y10] +
(7.189990974755708e-07+0j) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.00812525192138102+0j) [Y1 X2 X8 Y9] +
(-0.0017992194936630017+0j) [Y1 X2 X10 Y11] +
(-5.287660624402284e-07+0j) [Y1 X2 X10 Z11 Z12 Y13] +
(5.47164774426425e-07+0j) [Y1 X2 X11 Y12] +
(-0.004575007626639211+0j) [Y1 X2 X12 Y13] +
(1.9332412771118596e-07+0j) [Y1 Y2 X3 X4] +
(0.0016407548553124436+0j) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-3.013471458736137e-07+0j) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441841+0j) [Y1 Y2 Y4 Y5] +
(-8.091637198338709e-07+0j) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0057335697473118695+0j) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-4.523389677465775e-07+0j) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10] +
(-0.003484157300217878+0j) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004684903388155197+0j) [Y1 Y2 Y6 Y7] +
(0.005114473831660382+0j) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11] +
(-7.560692464297741e-07+0j) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.004668620318776302+0j) [Y1 Y2 X7 Z8 Z9 X10] +
(-7.189990974755708e-07+0j) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12] +
(-0.00812525192138102+0j) [Y1 Y2 Y8 Y9] +
(-0.0017992194936630017+0j) [Y1 Y2 Y10 Y11] +
(-5.287660624402284e-07+0j) [Y1 Y2 Y10 Z11 Z12 Y13] +
(-5.47164774426425e-07+0j) [Y1 Y2 X11 X12] +
(-0.004575007626639211+0j) [Y1 Y2 Y12 Y13] +
(-3.5682475208729344e-07+0j) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0022494124470939904+0j) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0004458535128840802+0j) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11] +
(-1.9742253793283145e-08+0j) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.0474716554332859e-08+0j) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.1250703257977234+0j) [Y1 Z2 Y3] +
(-1.3807781480675324e-07+0j) [Y1 Z2 Y3 X4 Z5 X6] +
(-3.3767393082598746e-07+0j) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0018638942824587587+0j) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.3807781480675324e-07+0j) [Y1 Z2 Y3 Y4 Z5 Y6] +
(-3.3767393082598746e-07+0j) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0018638942824587587+0j) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0011059037691897291+0j) [Y1 Z2 Y3 Z4] +
(-1.3807781480675324e-07+0j) [Y1 Z2 Y3 X5 Z6 X7] +
(-7.900128985725649e-07+0j) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.005348051582676637+0j) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.5510539176126766e-07+0j) [Y1 Z2 Y3 Y5 Z6 Y7] +
(-1.1468376506598585e-06+0j) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.007597464029770627+0j) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005530759218631571+0j) [Y1 Z2 Y3 Z5] +
(0.0005940221543005544+0j) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10] +
(-8.379773243198647e-08+0j) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0005940221543005544+0j) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10] +
(-8.379773243198647e-08+0j) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0033476175306662572+0j) [Y1 Z2 Y3 Z6] +
(0.005262642473076855+0j) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11] +
(-8.07430598527014e-07+0j) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005708495985960935+0j) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11] +
-1.9742253793246246e-08j [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.35233210242304e-07+0j) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.008032520918821454+0j) [Y1 Z2 Y3 Z7] +
(0.0029297686747511284+0j) [Y1 Z2 Y3 Z8] +
(0.01105502059613215+0j) [Y1 Z2 Y3 Z9] +
(-1.1076325598317528e-07+0j) [Y1 Z2 Y3 X10 Z11 X12] +
(-1.1076325598317528e-07+0j) [Y1 Z2 Y3 Y10 Z11 Y12] +
(0.001756070701841304+0j) [Y1 Z2 Y3 Z10] +
(-6.556281914112998e-07+0j) [Y1 Z2 Y3 X11 Z12 X13] +
(-6.418291574217039e-07+0j) [Y1 Z2 Y3 Y11 Z12 Y13] +
(0.003555290195504306+0j) [Y1 Z2 Y3 Z11] +
(0.002326230623158148+0j) [Y1 Z2 Y3 Z12] +
(0.006901238249797359+0j) [Y1 Z2 Y3 Z13] +
(0.0008533856254125472+0j) [Y1 Z2 Z3 X4 X5 Y6] +
(0.0007870896771024431+0j) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10] +
(-1.8394209155175266e-07+0j) [Y1 Z2 Z3 X4 X6 Y7] +
(-0.0012223378081538377+0j) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0001940085702975649+0j) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12] +
(-2.3713289480110134e-07+0j) [Y1 Z2 Z3 X4 X8 Y9] +
(8.057446594512792e-08+0j) [Y1 Z2 Z3 X4 X10 Y11] +
(-0.0009581655836696554+0j) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13] +
(-0.0017278753941369677+0j) [Y1 Z2 Z3 X4 X11 Y12] +
(-3.086826565198475e-07+0j) [Y1 Z2 Z3 X4 X12 Y13] +
(-0.0008533856254125472+0j) [Y1 Z2 Z3 Y4 X5 X6] +
(-0.0007870896771024431+0j) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10] +
(-1.8394209155175266e-07+0j) [Y1 Z2 Z3 Y4 Y6 Y7] +
(-0.0012223378081538377+0j) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0001940085702975649+0j) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12] +
(-2.3713289480110134e-07+0j) [Y1 Z2 Z3 Y4 Y8 Y9] +
(8.057446594512792e-08+0j) [Y1 Z2 Z3 Y4 Y10 Y11] +
(-0.0009581655836696554+0j) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13] +
(0.0017278753941369677+0j) [Y1 Z2 Z3 Y4 X11 X12] +
(-3.086826565198475e-07+0j) [Y1 Z2 Z3 Y4 Y12 Y13] +
(-0.0010283292378562726+0j) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.002686040977806622+0j) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13] +
(3.2020768797081232e-06+0j) [Y1 Z2 Z3 Z4 Y5] +
(4.092250615741314e-07+0j) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10] +
(0.0023949726397980214+0j) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.092250615741314e-07+0j) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10] +
(0.0023949726397980214+0j) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.6849150949283104e-07+0j) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11] +
(0.0022009640695004563+0j) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.4445978539252117e-07+0j) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11] +
(0.001172634831644184+0j) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.8394209155175266e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z7] +
(8.649310132521708e-08+0j) [Y1 Z2 Z3 Z4 Y5 Z8] +
(3.236259961263184e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z9] +
(0.00226196606248235+0j) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12] +
(0.00226196606248235+0j) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12] +
(-5.927453082391738e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z10] +
(0.003989841456619317+0j) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13] +
(0.0013038004788126943+0j) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13] +
(-6.733197741843018e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z11] +
(9.306536651221527e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z12] +
(1.2393363216420002e-06+0j) [Y1 Z2 Z3 Z4 Y5 Z13] +
(0.0005192743499487691+0j) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10] +
(-1.8505641927803864e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.0033566705638328944+0j) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9] +
(-0.00013840177303550328+0j) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11] +
(-4.997018421675544e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13] +
(6.175246206674518e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12] +
(-0.003267513854423557+0j) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13] +
(-0.0005192743499487691+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10] +
(1.8505641927803864e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-0.0033566705638328944+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9] +
(-0.00013840177303550328+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11] +
(-4.997018421675544e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13] +
(-6.175246206674518e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12] +
(-0.003267513854423557+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13] +
(1.2004287493541528e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13] +
(0.042743277013783186+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7] +
(0.0012803060973496701+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8] +
(0.004636976661182564+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9] +
(7.246974424931244e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12] +
(7.246974424931244e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12] +
(0.005241535382803876+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10] +
(1.0717282182567258e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13] +
(2.3120943051420206e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13] +
(0.005379937155839379+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11] +
(0.0010435246534907584+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12] +
(0.004311038507914315+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13] +
(0.003876470899336951+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10] +
(-7.540341413240216e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-0.003876470899336951+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10] +
(7.540341413240216e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-0.0029841661681219373+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(-0.0029841661681219373+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(0.07165035181002738+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.0012366478019245745+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(0.004220813970046511+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(-1.3987009015439521e-05+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(8.949476486467515e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(-7.661347212839333e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(-0.0021413612231015585+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11] +
(-6.87662165802578e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(0.005408954422410026+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11] +
(-1.0444941297729537e-06+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(0.001532483523073075+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11] +
(-2.9045998844893215e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(0.005286546538226924+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11] +
(-9.956079229678416e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.002779026799025555+0j) [Y1 Z2 Z3 Z4 Z5 Y7] +
(0.004767272188278155+0j) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11] +
(-8.105515036898032e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.002462917007133919+0j) [Y1 Z2 Z3 Z4 Z6 Y7] +
(0.0007156734248909091+0j) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11] +
(-3.0767325321580573e-07+0j) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.2919694865350102e-07+0j) [Y1 Z2 Z3 Y5] +
(0.001609531381721372+0j) [Y1 Z2 Z3 Z5 Z6 Y7] +
(-7.141625221153381e-05+0j) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-2.666731755093001e-07+0j) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.7379332622789548e-07+0j) [Y1 Z2 Z4 Y5] +
(0.001667604181144058+0j) [Y1 Z2 Z4 Z5 Z6 Y7] +
(-0.001452884321416903+0j) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(4.6704023900819875e-07+0j) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0032769719312317307+0j) [Y1 Y3] +
(3.6060718677803603e-07+0j) [Y1 Z3 Z4 Y5] +
(0.003961560792496535+0j) [Y1 Z3 Z4 Z5 Z6 Y7] +
(0.0001878705338955407+0j) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.6569309313458506e-07+0j) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(12.412630742111762+0j) [Z1] +
(-1.1908508084640527e-06+0j) [Z1 X2 Z3 X4] +
(-0.03276765782329055+0j) [Z1 X2 Z3 Z4 Z5 X6] +
(-0.07635021950635029+0j) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.5809603692086352e-05+0j) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.1908508084640527e-06+0j) [Z1 Y2 Z3 Y4] +
(-0.03276765782329055+0j) [Z1 Y2 Z3 Z4 Z5 Y6] +
(-0.07635021950635029+0j) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.5809603692086352e-05+0j) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.2512944567459169+0j) [Z1 Z2] +
(-8.337746755398009e-07+0j) [Z1 X3 Z4 X5] +
(-0.027115036845273183+0j) [Z1 X3 Z4 Z5 Z6 X7] +
(-0.06752385099214042+0j) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.4017109734481867e-05+0j) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.337746755398009e-07+0j) [Z1 Y3 Z4 Y5] +
(-0.027115036845273183+0j) [Z1 Y3 Z4 Z5 Z6 Y7] +
(-0.06752385099214042+0j) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.4017109734481867e-05+0j) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.23671080783830417+0j) [Z1 Z3] +
(-3.344081556458242e-06+0j) [Z1 X4 Z5 X6] +
(-1.6103585304708656e-05+0j) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.09065144207036469+0j) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-3.344081556458242e-06+0j) [Z1 Y4 Z5 Y6] +
(-1.6103585304708656e-05+0j) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.09065144207036469+0j) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.19936354537360795+0j) [Z1 Z4] +
(-3.099349243575037e-06+0j) [Z1 X5 Z6 X7] +
(-1.5316808794326665e-05+0j) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.08684737589863618+0j) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.099349243575037e-06+0j) [Z1 Y5 Z6 Y7] +
(-1.5316808794326665e-05+0j) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.08684737589863618+0j) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.19661770890342117+0j) [Z1 Z5] +
(0.05600733087780746+0j) [Z1 X6 Z7 Z8 Z9 X10] +
(-6.481851833245475e-06+0j) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.05600733087780746+0j) [Z1 Y6 Z7 Z8 Z9 Y10] +
(-6.481851833245475e-06+0j) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.24853483371314253+0j) [Z1 Z6] +
(0.056084681246613394+0j) [Z1 X7 Z8 Z9 Z10 X11] +
(-6.652209668791919e-06+0j) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.056084681246613394+0j) [Z1 Y7 Z8 Z9 Z10 Y11] +
(-6.652209668791919e-06+0j) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.24164663936017197+0j) [Z1 Z7] +
(0.2723251830660567+0j) [Z1 Z9] +
(-1.6148794135137072e-06+0j) [Z1 X10 Z11 X12] +
(-1.6148794135137072e-06+0j) [Z1 Y10 Z11 Y12] +
(0.20072866460441777+0j) [Z1 Z10] +
(-2.177664604637626e-06+0j) [Z1 X11 Z12 X13] +
(-2.177664604637626e-06+0j) [Z1 Y11 Z12 Y13] +
(0.19299723935364246+0j) [Z1 Z11] +
(0.2163103749863184+0j) [Z1 Z12] +
(0.21102659849791544+0j) [Z1 Z13] +
(-0.035839567953353475+0j) [X2 X3 Y4 Y5] +
(-2.199051618834916e-07+0j) [X2 X3 Y4 Z5 Z6 Y7] +
(-2.3609563202366283e-06+0j) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.010311482489831778+0j) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.199051618834916e-07+0j) [X2 X3 X5 X6] +
(-2.360956320236628e-06+0j) [X2 X3 X5 Z6 Z7 Z8 Z9 X10] +
(-0.010311482489831778+0j) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.031143817988967096+0j) [X2 X3 Y6 Y7] +
(0.005368659358109495+0j) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11] +
(9.209350654361721e-08+0j) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005368659358109495+0j) [X2 X3 X7 Z8 Z9 X10] +
(9.209350654361721e-08+0j) [X2 X3 X7 Z8 Z9 Z10 Z11 X12] +
(-0.036194123559042606+0j) [X2 X3 Y8 Y9] +
(-0.025384657508457482+0j) [X2 X3 Y10 Y11] +
(2.172669101434983e-06+0j) [X2 X3 Y10 Z11 Z12 Y13] +
(2.172669101434983e-06+0j) [X2 X3 X11 X12] +
(-0.015577208063976484+0j) [X2 X3 Y12 Y13] +
(0.035839567953353475+0j) [X2 Y3 Y4 X5] +
(2.199051618834916e-07+0j) [X2 Y3 Y4 Z5 Z6 X7] +
(2.3609563202366283e-06+0j) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.010311482489831778+0j) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.199051618834916e-07+0j) [X2 Y3 Y5 X6] +
(-2.360956320236628e-06+0j) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.010311482489831778+0j) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.031143817988967096+0j) [X2 Y3 Y6 X7] +
(-0.005368659358109495+0j) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11] +
(-9.209350654361721e-08+0j) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005368659358109495+0j) [X2 Y3 Y7 Z8 Z9 X10] +
(9.209350654361721e-08+0j) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12] +
(0.036194123559042606+0j) [X2 Y3 Y8 X9] +
(0.025384657508457482+0j) [X2 Y3 Y10 X11] +
(-2.172669101434983e-06+0j) [X2 Y3 Y10 Z11 Z12 X13] +
(2.172669101434983e-06+0j) [X2 Y3 Y11 X12] +
(0.015577208063976484+0j) [X2 Y3 Y12 X13] +
(-3.88705167387878e-06+0j) [X2 Z3 X4] +
(-0.005143391768825129+0j) [X2 Z3 X4 X5 Z6 X7] +
(-0.00984174924696265+0j) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.9885117062387915e-06+0j) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.005143391768825129+0j) [X2 Z3 X4 Y5 Z6 Y7] +
(-0.00984174924696265+0j) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.9885117062387915e-06+0j) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.764994119620165e-07+0j) [X2 Z3 X4 Z5] +
(1.6893489513418401e-06+0j) [X2 Z3 X4 X6 Z7 Z8 Z9 X10] +
(0.01075756395390895+0j) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.537178095426712e-08+0j) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10] +
(4.2055484112183544e-05+0j) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.5935343908545746e-07+0j) [X2 Z3 X4 Z6] +
(3.2118420188330066e-06+0j) [X2 Z3 X4 X7 Z8 Z9 Z10 X11] +
(0.019299560579363793+0j) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.2118420188330066e-06+0j) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11] +
(0.019299560579363793+0j) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.1954890099718503e-06+0j) [X2 Z3 X4 Z7] +
(2.1868423773069423e-07+0j) [X2 Z3 X4 Z8] +
(-5.77005299559363e-07+0j) [X2 Z3 X4 Z9] +
(0.015588250102380219+0j) [X2 Z3 X4 X10 Z11 X12] +
(0.005324835234221673+0j) [X2 Z3 X4 Y10 Z11 Y12] +
(-3.1586564317899996e-06+0j) [X2 Z3 X4 Z10] +
(0.02435307767806902+0j) [X2 Z3 X4 X11 Z12 X13] +
(0.02435307767806902+0j) [X2 Z3 X4 Y11 Z12 Y13] +
(-7.80170749993665e-06+0j) [X2 Z3 X4 Z11] +
(3.5390541842202535e-06+0j) [X2 Z3 X4 Z12] +
(8.8149373060548e-06+0j) [X2 Z3 X4 Z13] +
(1.6288532434023657e-06+0j) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10] +
(0.010715508469796766+0j) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.010263414868158544+0j) [X2 Z3 Y4 Y10 Z11 X12] +
(-1.4548424490573079e-06+0j) [X2 Z3 Z4 X5 Y6 Y7] +
(-3.151346310893532e-06+0j) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11] +
(-0.019257505095251606+0j) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.5224930674911664e-06+0j) [X2 Z3 Z4 X5 X7 Z8 Z9 X10] +
(-0.008541996625454844+0j) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-7.956895372900573e-07+0j) [X2 Z3 Z4 X5 Y8 Y9] +
(-4.64305106814665e-06+0j) [X2 Z3 Z4 X5 Y10 Y11] +
(-0.01902824244384734+0j) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13] +
(-0.008764827575688798+0j) [X2 Z3 Z4 X5 X11 X12] +
(5.2758831218345475e-06+0j) [X2 Z3 Z4 X5 Y12 Y13] +
(1.4548424490573079e-06+0j) [X2 Z3 Z4 Y5 Y6 X7] +
(3.151346310893532e-06+0j) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11] +
(0.019257505095251606+0j) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.5224930674911664e-06+0j) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10] +
(-0.008541996625454844+0j) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(7.956895372900573e-07+0j) [X2 Z3 Z4 Y5 Y8 X9] +
(4.64305106814665e-06+0j) [X2 Z3 Z4 Y5 Y10 X11] +
(0.01902824244384734+0j) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13] +
(-0.008764827575688798+0j) [X2 Z3 Z4 Y5 Y11 X12] +
(-5.2758831218345475e-06+0j) [X2 Z3 Z4 Y5 Y12 X13] +
(-0.12133276911042408+0j) [X2 Z3 Z4 Z5 X6] +
(-0.008469978791023906+0j) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(2.6863815444737853e-07+0j) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.008469978791023906+0j) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(2.6863815444737853e-07+0j) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.025996177598021253+0j) [X2 Z3 Z4 Z5 X6 Z7] +
(-0.005805188989826926+0j) [X2 Z3 Z4 Z5 X6 Z8] +
(-0.0175612024096462+0j) [X2 Z3 Z4 Z5 X6 Z9] +
(-7.988770288602579e-07+0j) [X2 Z3 Z4 Z5 X6 X10 Z11 X12] +
(-3.427323108624932e-07+0j) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12] +
(-0.0008145313270956889+0j) [X2 Z3 Z4 Z5 X6 Z10] +
(2.7455184002282526e-06+0j) [X2 Z3 Z4 Z5 X6 X11 Z12 X13] +
(2.7455184002282526e-06+0j) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13] +
(0.014411099430130907+0j) [X2 Z3 Z4 Z5 X6 Z11] +
(0.0006650070219498614+0j) [X2 Z3 Z4 Z5 X6 Z12] +
(-0.0034937903598902115+0j) [X2 Z3 Z4 Z5 X6 Z13] +
(-4.561447179977648e-07+0j) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12] +
(-0.011756013419819274+0j) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9] +
(0.015225630757226596+0j) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11] +
(-3.088250711090746e-06+0j) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(-3.5443954290885104e-06+0j) [X2 Z3 Z4 Z5 Z6 X7 X11 X12] +
(-0.004158797381840073+0j) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13] +
(0.011756013419819274+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9] +
(-0.015225630757226596+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11] +
(3.088250711090746e-06+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(-3.5443954290885104e-06+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12] +
(0.004158797381840073+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13] +
(0.014603704729162146+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(-2.874299071224402e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.014603704729162146+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(-2.874299071224402e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(-0.28164257767023054+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-1.300294656131589e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(-1.300294656131589e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(-0.02428211735469306+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(-0.019538050311314805+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(-0.017091553155898945+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(0.0024464971554158583+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(-0.0024464971554158583+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(5.7759505269753304e-05+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(2.883676575926599e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(5.146496327152698e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(3.846201671021108e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(-0.03935916802205318+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10] +
(7.979825792943652e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(-0.02475546329289104+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10] +
(5.10552672171925e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(-0.021433810721601+0j) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10] +
(5.159350501692646e-06+0j) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(-0.0299037895126249+0j) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10] +
(5.427988656140024e-06+0j) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(0.001663879878490798+0j) [X2 Z3 Z4 X6] +
(-0.018889030304942905+0j) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10] +
(2.9473560114557034e-06+0j) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.003479511890334331+0j) [X2 Z3 Z5 X6] +
(-0.028730779551905554+0j) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10] +
(5.935867717694495e-06+0j) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.6021167406578304e-06+0j) [X2 X4] +
(0.0004956762314916488+0j) [X2 Z4 Z5 X6] +
(-0.035608378988312664+0j) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(7.253273347712183e-06+0j) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.035839567953353475+0j) [Y2 X3 X4 Y5] +
(2.199051618834916e-07+0j) [Y2 X3 X4 Z5 Z6 Y7] +
(2.3609563202366283e-06+0j) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.010311482489831778+0j) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.199051618834916e-07+0j) [Y2 X3 X5 Y6] +
(-2.360956320236628e-06+0j) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.010311482489831778+0j) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.031143817988967096+0j) [Y2 X3 X6 Y7] +
(-0.005368659358109495+0j) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11] +
(-9.209350654361721e-08+0j) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005368659358109495+0j) [Y2 X3 X7 Z8 Z9 Y10] +
(9.209350654361721e-08+0j) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12] +
(0.036194123559042606+0j) [Y2 X3 X8 Y9] +
(0.025384657508457482+0j) [Y2 X3 X10 Y11] +
(-2.172669101434983e-06+0j) [Y2 X3 X10 Z11 Z12 Y13] +
(2.172669101434983e-06+0j) [Y2 X3 X11 Y12] +
(0.015577208063976484+0j) [Y2 X3 X12 Y13] +
(-0.035839567953353475+0j) [Y2 Y3 X4 X5] +
(-2.199051618834916e-07+0j) [Y2 Y3 X4 Z5 Z6 X7] +
(-2.3609563202366283e-06+0j) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.010311482489831778+0j) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.199051618834916e-07+0j) [Y2 Y3 Y5 Y6] +
(-2.360956320236628e-06+0j) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.010311482489831778+0j) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.031143817988967096+0j) [Y2 Y3 X6 X7] +
(0.005368659358109495+0j) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11] +
(9.209350654361721e-08+0j) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005368659358109495+0j) [Y2 Y3 Y7 Z8 Z9 Y10] +
(9.209350654361721e-08+0j) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.036194123559042606+0j) [Y2 Y3 X8 X9] +
(-0.025384657508457482+0j) [Y2 Y3 X10 X11] +
(2.172669101434983e-06+0j) [Y2 Y3 X10 Z11 Z12 X13] +
(2.172669101434983e-06+0j) [Y2 Y3 Y11 Y12] +
(-0.015577208063976484+0j) [Y2 Y3 X12 X13] +
(1.6288532434023657e-06+0j) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10] +
(0.010715508469796766+0j) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.010263414868158544+0j) [Y2 Z3 X4 X10 Z11 Y12] +
(-3.88705167387878e-06+0j) [Y2 Z3 Y4] +
(-0.005143391768825129+0j) [Y2 Z3 Y4 X5 Z6 X7] +
(-0.00984174924696265+0j) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.9885117062387915e-06+0j) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.005143391768825129+0j) [Y2 Z3 Y4 Y5 Z6 Y7] +
(-0.00984174924696265+0j) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.9885117062387915e-06+0j) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.764994119620165e-07+0j) [Y2 Z3 Y4 Z5] +
(4.537178095426712e-08+0j) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10] +
(4.2055484112183544e-05+0j) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.6893489513418401e-06+0j) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10] +
(0.01075756395390895+0j) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.5935343908545746e-07+0j) [Y2 Z3 Y4 Z6] +
(3.2118420188330066e-06+0j) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11] +
(0.019299560579363793+0j) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.2118420188330066e-06+0j) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11] +
(0.019299560579363793+0j) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.1954890099718503e-06+0j) [Y2 Z3 Y4 Z7] +
(2.1868423773069423e-07+0j) [Y2 Z3 Y4 Z8] +
(-5.77005299559363e-07+0j) [Y2 Z3 Y4 Z9] +
(0.005324835234221673+0j) [Y2 Z3 Y4 X10 Z11 X12] +
(0.015588250102380219+0j) [Y2 Z3 Y4 Y10 Z11 Y12] +
(-3.1586564317899996e-06+0j) [Y2 Z3 Y4 Z10] +
(0.02435307767806902+0j) [Y2 Z3 Y4 X11 Z12 X13] +
(0.02435307767806902+0j) [Y2 Z3 Y4 Y11 Z12 Y13] +
(-7.80170749993665e-06+0j) [Y2 Z3 Y4 Z11] +
(3.5390541842202535e-06+0j) [Y2 Z3 Y4 Z12] +
(8.8149373060548e-06+0j) [Y2 Z3 Y4 Z13] +
(1.4548424490573079e-06+0j) [Y2 Z3 Z4 X5 X6 Y7] +
(3.151346310893532e-06+0j) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11] +
(0.019257505095251606+0j) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.5224930674911664e-06+0j) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10] +
(-0.008541996625454844+0j) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(7.956895372900573e-07+0j) [Y2 Z3 Z4 X5 X8 Y9] +
(4.64305106814665e-06+0j) [Y2 Z3 Z4 X5 X10 Y11] +
(0.01902824244384734+0j) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13] +
(-0.008764827575688798+0j) [Y2 Z3 Z4 X5 X11 Y12] +
(-5.2758831218345475e-06+0j) [Y2 Z3 Z4 X5 X12 Y13] +
(-1.4548424490573079e-06+0j) [Y2 Z3 Z4 Y5 X6 X7] +
(-3.151346310893532e-06+0j) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11] +
(-0.019257505095251606+0j) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.5224930674911664e-06+0j) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10] +
(-0.008541996625454844+0j) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895372900573e-07+0j) [Y2 Z3 Z4 Y5 X8 X9] +
(-4.64305106814665e-06+0j) [Y2 Z3 Z4 Y5 X10 X11] +
(-0.01902824244384734+0j) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13] +
(-0.008764827575688798+0j) [Y2 Z3 Z4 Y5 Y11 Y12] +
(5.2758831218345475e-06+0j) [Y2 Z3 Z4 Y5 X12 X13] +
(-4.561447179977648e-07+0j) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12] +
(-0.12133276911042408+0j) [Y2 Z3 Z4 Z5 Y6] +
(-0.008469978791023906+0j) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(2.6863815444737853e-07+0j) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.008469978791023906+0j) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(2.6863815444737853e-07+0j) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.025996177598021253+0j) [Y2 Z3 Z4 Z5 Y6 Z7] +
(-0.005805188989826926+0j) [Y2 Z3 Z4 Z5 Y6 Z8] +
(-0.0175612024096462+0j) [Y2 Z3 Z4 Z5 Y6 Z9] +
(-3.427323108624932e-07+0j) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12] +
(-7.988770288602579e-07+0j) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12] +
(-0.0008145313270956889+0j) [Y2 Z3 Z4 Z5 Y6 Z10] +
(2.7455184002282526e-06+0j) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13] +
(2.7455184002282526e-06+0j) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13] +
(0.014411099430130907+0j) [Y2 Z3 Z4 Z5 Y6 Z11] +
(0.0006650070219498614+0j) [Y2 Z3 Z4 Z5 Y6 Z12] +
(-0.0034937903598902115+0j) [Y2 Z3 Z4 Z5 Y6 Z13] +
(0.011756013419819274+0j) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9] +
(-0.015225630757226596+0j) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11] +
(3.088250711090746e-06+0j) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(-3.5443954290885104e-06+0j) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12] +
(0.004158797381840073+0j) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13] +
(-0.011756013419819274+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9] +
(0.015225630757226596+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11] +
(-3.088250711090746e-06+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(-3.5443954290885104e-06+0j) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12] +
(-0.004158797381840073+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13] +
(0.014603704729162146+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(-2.874299071224402e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.014603704729162146+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(-2.874299071224402e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(-0.28164257767023054+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-1.300294656131589e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(-1.300294656131589e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(-0.02428211735469306+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(-0.019538050311314805+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(-0.017091553155898945+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(-0.0024464971554158583+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(0.0024464971554158583+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(5.7759505269753304e-05+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.883676575926599e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(5.146496327152698e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(3.846201671021108e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(-0.03935916802205318+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10] +
(7.979825792943652e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(-0.02475546329289104+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10] +
(5.10552672171925e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(-0.021433810721601+0j) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10] +
(5.159350501692646e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(-0.0299037895126249+0j) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10] +
(5.427988656140024e-06+0j) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.001663879878490798+0j) [Y2 Z3 Z4 Y6] +
(-0.018889030304942905+0j) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10] +
(2.9473560114557034e-06+0j) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.003479511890334331+0j) [Y2 Z3 Z5 Y6] +
(-0.028730779551905554+0j) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10] +
(5.935867717694495e-06+0j) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.6021167406578304e-06+0j) [Y2 Y4] +
(0.0004956762314916488+0j) [Y2 Z4 Z5 Y6] +
(-0.035608378988312664+0j) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(7.253273347712183e-06+0j) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.6538942226831694+0j) [Z2] +
(1.6021167406578306e-06+0j) [Z2 X3 Z4 X5] +
(0.0004956762314916487+0j) [Z2 X3 Z4 Z5 Z6 X7] +
(-0.03560837898831266+0j) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(7.253273347712183e-06+0j) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.6021167406578306e-06+0j) [Z2 Y3 Z4 Y5] +
(0.0004956762314916487+0j) [Z2 Y3 Z4 Z5 Z6 Y7] +
(-0.03560837898831266+0j) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(7.253273347712183e-06+0j) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.1818908579075138+0j) [Z2 Z3] +
(-9.50924975139167e-07+0j) [Z2 X4 Z5 X6] +
(-4.728843146939049e-06+0j) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.024591860883829964+0j) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-9.50924975139167e-07+0j) [Z2 Y4 Z5 Y6] +
(-4.728843146939049e-06+0j) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.024591860883829964+0j) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.12495807739503215+0j) [Z2 Z4] +
(-1.1708301370226586e-06+0j) [Z2 X5 Z6 X7] +
(-7.0897994671756765e-06+0j) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.034903343373661744+0j) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.1708301370226586e-06+0j) [Z2 Y5 Z6 Y7] +
(-7.0897994671756765e-06+0j) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.034903343373661744+0j) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.16079764534838564+0j) [Z2 Z5] +
(0.019020423173039903+0j) [Z2 X6 Z7 Z8 Z9 X10] +
(-2.1032156044294735e-06+0j) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.019020423173039903+0j) [Z2 Y6 Z7 Z8 Z9 Y10] +
(-2.1032156044294735e-06+0j) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.13739104762683232+0j) [Z2 Z6] +
(0.024389082531149398+0j) [Z2 X7 Z8 Z9 Z10 X11] +
(-2.0111220978858564e-06+0j) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.024389082531149398+0j) [Z2 Y7 Z8 Z9 Z10 Y11] +
(-2.0111220978858564e-06+0j) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.16853486561579942+0j) [Z2 Z7] +
(0.15071408121008295+0j) [Z2 Z8] +
(0.18690820476912556+0j) [Z2 Z9] +
(-1.0632283420830098e-06+0j) [Z2 X10 Z11 X12] +
(-1.0632283420830098e-06+0j) [Z2 Y10 Z11 Y12] +
(0.1279950249246842+0j) [Z2 Z10] +
(1.1094407593519738e-06+0j) [Z2 X11 Z12 X13] +
(1.1094407593519738e-06+0j) [Z2 Y11 Z12 Y13] +
(0.1533796824331417+0j) [Z2 Z11] +
(0.14011289865354834+0j) [Z2 Z12] +
(0.15569010671752484+0j) [Z2 Z13] +
(0.005143391768825129+0j) [X3 X4 Y5 Y6] +
(0.00984174924696265+0j) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10] +
(-2.9885117062387915e-06+0j) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.4548424490573079e-06+0j) [X3 X4 X6 X7] +
(-1.5224930674911664e-06+0j) [X3 X4 X6 Z7 Z8 Z9 Z10 X11] +
(-0.008541996625454844+0j) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.151346310893532e-06+0j) [X3 X4 Y7 Z8 Z9 Y10] +
(-0.019257505095251606+0j) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895372900573e-07+0j) [X3 X4 X8 X9] +
(-4.643051068146651e-06+0j) [X3 X4 X10 X11] +
(-0.008764827575688798+0j) [X3 X4 X10 Z11 Z12 X13] +
(-0.01902824244384734+0j) [X3 X4 Y11 Y12] +
(5.2758831218345475e-06+0j) [X3 X4 X12 X13] +
(-0.005143391768825129+0j) [X3 Y4 Y5 X6] +
(-0.00984174924696265+0j) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10] +
(2.9885117062387915e-06+0j) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.4548424490573079e-06+0j) [X3 Y4 Y6 X7] +
(-1.5224930674911664e-06+0j) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11] +
(-0.008541996625454844+0j) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.151346310893532e-06+0j) [X3 Y4 Y7 Z8 Z9 X10] +
(0.019257505095251606+0j) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12] +
(-7.956895372900573e-07+0j) [X3 Y4 Y8 X9] +
(-4.643051068146651e-06+0j) [X3 Y4 Y10 X11] +
(-0.008764827575688798+0j) [X3 Y4 Y10 Z11 Z12 X13] +
(0.01902824244384734+0j) [X3 Y4 Y11 X12] +
(5.2758831218345475e-06+0j) [X3 Y4 Y12 X13] +
(-3.887051673878778e-06+0j) [X3 Z4 X5] +
(3.2118420188330066e-06+0j) [X3 Z4 X5 X6 Z7 Z8 Z9 X10] +
(0.019299560579363793+0j) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.2118420188330066e-06+0j) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10] +
(0.019299560579363793+0j) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.1954890099718503e-06+0j) [X3 Z4 X5 Z6] +
(1.6893489513418401e-06+0j) [X3 Z4 X5 X7 Z8 Z9 Z10 X11] +
(0.01075756395390895+0j) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.537178095426712e-08+0j) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11] +
(4.2055484112183544e-05+0j) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.5935343908545746e-07+0j) [X3 Z4 X5 Z7] +
(-5.77005299559363e-07+0j) [X3 Z4 X5 Z8] +
(2.1868423773069423e-07+0j) [X3 Z4 X5 Z9] +
(0.02435307767806902+0j) [X3 Z4 X5 X10 Z11 X12] +
(0.02435307767806902+0j) [X3 Z4 X5 Y10 Z11 Y12] +
(-7.80170749993665e-06+0j) [X3 Z4 X5 Z10] +
(0.015588250102380219+0j) [X3 Z4 X5 X11 Z12 X13] +
(0.005324835234221673+0j) [X3 Z4 X5 Y11 Z12 Y13] +
(-3.1586564317899996e-06+0j) [X3 Z4 X5 Z11] +
(8.8149373060548e-06+0j) [X3 Z4 X5 Z12] +
(3.5390541842202535e-06+0j) [X3 Z4 X5 Z13] +
(1.6288532434023657e-06+0j) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11] +
(0.010715508469796766+0j) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.010263414868158544+0j) [X3 Z4 Y5 Y11 Z12 X13] +
(0.008469978791023906+0j) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10] +
(-2.6863815444737853e-07+0j) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.011756013419819272+0j) [X3 Z4 Z5 X6 X8 X9] +
(0.015225630757226596+0j) [X3 Z4 Z5 X6 X10 X11] +
(-3.5443954290885104e-06+0j) [X3 Z4 Z5 X6 X10 Z11 Z12 X13] +
(-3.088250711090746e-06+0j) [X3 Z4 Z5 X6 Y11 Y12] +
(-0.004158797381840073+0j) [X3 Z4 Z5 X6 X12 X13] +
(-0.008469978791023906+0j) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10] +
(2.6863815444737853e-07+0j) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.011756013419819272+0j) [X3 Z4 Z5 Y6 Y8 X9] +
(0.015225630757226596+0j) [X3 Z4 Z5 Y6 Y10 X11] +
(-3.5443954290885104e-06+0j) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13] +
(3.088250711090746e-06+0j) [X3 Z4 Z5 Y6 Y11 X12] +
(-0.004158797381840073+0j) [X3 Z4 Z5 Y6 Y12 X13] +
(-0.12133276911042408+0j) [X3 Z4 Z5 Z6 X7] +
(-0.0175612024096462+0j) [X3 Z4 Z5 Z6 X7 Z8] +
(-0.005805188989826926+0j) [X3 Z4 Z5 Z6 X7 Z9] +
(2.7455184002282526e-06+0j) [X3 Z4 Z5 Z6 X7 X10 Z11 X12] +
(2.7455184002282526e-06+0j) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12] +
(0.014411099430130907+0j) [X3 Z4 Z5 Z6 X7 Z10] +
(-7.988770288602579e-07+0j) [X3 Z4 Z5 Z6 X7 X11 Z12 X13] +
(-3.427323108624932e-07+0j) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13] +
(-0.0008145313270956889+0j) [X3 Z4 Z5 Z6 X7 Z11] +
(-0.0034937903598902115+0j) [X3 Z4 Z5 Z6 X7 Z12] +
(0.0006650070219498614+0j) [X3 Z4 Z5 Z6 X7 Z13] +
(-4.561447179977648e-07+0j) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13] +
(-0.014603704729162146+0j) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10] +
(2.874299071224402e-06+0j) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(0.014603704729162146+0j) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10] +
(-2.874299071224402e-06+0j) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(1.300294656131589e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12] +
(0.0024464971554158583+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-1.300294656131589e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12] +
(0.0024464971554158583+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(-0.28164257767023065+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.017091553155898945+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(-0.019538050311314805+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(5.775950526975332e-05+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(2.8836765759265984e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(3.846201671021108e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(-0.02428211735469306+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11] +
(5.146496327152698e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(-0.02475546329289104+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11] +
(5.10552672171925e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(-0.03935916802205318+0j) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11] +
(7.979825792943652e-06+0j) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(-0.0299037895126249+0j) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11] +
(5.427988656140024e-06+0j) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.025996177598021253+0j) [X3 Z4 Z5 X7] +
(-0.021433810721601+0j) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11] +
(5.159350501692646e-06+0j) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.003479511890334331+0j) [X3 Z4 Z6 X7] +
(-0.028730779551905554+0j) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11] +
(5.935867717694495e-06+0j) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.764994119620164e-07+0j) [X3 X5] +
(0.001663879878490798+0j) [X3 Z5 Z6 X7] +
(-0.018889030304942905+0j) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.9473560114557034e-06+0j) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.005143391768825129+0j) [Y3 X4 X5 Y6] +
(-0.00984174924696265+0j) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10] +
(2.9885117062387915e-06+0j) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.4548424490573079e-06+0j) [Y3 X4 X6 Y7] +
(-1.5224930674911664e-06+0j) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11] +
(-0.008541996625454844+0j) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.151346310893532e-06+0j) [Y3 X4 X7 Z8 Z9 Y10] +
(0.019257505095251606+0j) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895372900573e-07+0j) [Y3 X4 X8 Y9] +
(-4.643051068146651e-06+0j) [Y3 X4 X10 Y11] +
(-0.008764827575688798+0j) [Y3 X4 X10 Z11 Z12 Y13] +
(0.01902824244384734+0j) [Y3 X4 X11 Y12] +
(5.2758831218345475e-06+0j) [Y3 X4 X12 Y13] +
(0.005143391768825129+0j) [Y3 Y4 X5 X6] +
(0.00984174924696265+0j) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10] +
(-2.9885117062387915e-06+0j) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.4548424490573079e-06+0j) [Y3 Y4 Y6 Y7] +
(-1.5224930674911664e-06+0j) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11] +
(-0.008541996625454844+0j) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.151346310893532e-06+0j) [Y3 Y4 X7 Z8 Z9 X10] +
(-0.019257505095251606+0j) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12] +
(-7.956895372900573e-07+0j) [Y3 Y4 Y8 Y9] +
(-4.643051068146651e-06+0j) [Y3 Y4 Y10 Y11] +
(-0.008764827575688798+0j) [Y3 Y4 Y10 Z11 Z12 Y13] +
(-0.01902824244384734+0j) [Y3 Y4 X11 X12] +
(5.2758831218345475e-06+0j) [Y3 Y4 Y12 Y13] +
(1.6288532434023657e-06+0j) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11] +
(0.010715508469796766+0j) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.010263414868158544+0j) [Y3 Z4 X5 X11 Z12 Y13] +
(-3.887051673878778e-06+0j) [Y3 Z4 Y5] +
(3.2118420188330066e-06+0j) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10] +
(0.019299560579363793+0j) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.2118420188330066e-06+0j) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10] +
(0.019299560579363793+0j) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.1954890099718503e-06+0j) [Y3 Z4 Y5 Z6] +
(4.537178095426712e-08+0j) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11] +
(4.2055484112183544e-05+0j) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.6893489513418401e-06+0j) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11] +
(0.01075756395390895+0j) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.5935343908545746e-07+0j) [Y3 Z4 Y5 Z7] +
(-5.77005299559363e-07+0j) [Y3 Z4 Y5 Z8] +
(2.1868423773069423e-07+0j) [Y3 Z4 Y5 Z9] +
(0.02435307767806902+0j) [Y3 Z4 Y5 X10 Z11 X12] +
(0.02435307767806902+0j) [Y3 Z4 Y5 Y10 Z11 Y12] +
(-7.80170749993665e-06+0j) [Y3 Z4 Y5 Z10] +
(0.005324835234221673+0j) [Y3 Z4 Y5 X11 Z12 X13] +
(0.015588250102380219+0j) [Y3 Z4 Y5 Y11 Z12 Y13] +
(-3.1586564317899996e-06+0j) [Y3 Z4 Y5 Z11] +
(8.8149373060548e-06+0j) [Y3 Z4 Y5 Z12] +
(3.5390541842202535e-06+0j) [Y3 Z4 Y5 Z13] +
(-0.008469978791023906+0j) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10] +
(2.6863815444737853e-07+0j) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.011756013419819272+0j) [Y3 Z4 Z5 X6 X8 Y9] +
(0.015225630757226596+0j) [Y3 Z4 Z5 X6 X10 Y11] +
(-3.5443954290885104e-06+0j) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13] +
(3.088250711090746e-06+0j) [Y3 Z4 Z5 X6 X11 Y12] +
(-0.004158797381840073+0j) [Y3 Z4 Z5 X6 X12 Y13] +
(0.008469978791023906+0j) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10] +
(-2.6863815444737853e-07+0j) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-0.011756013419819272+0j) [Y3 Z4 Z5 Y6 Y8 Y9] +
(0.015225630757226596+0j) [Y3 Z4 Z5 Y6 Y10 Y11] +
(-3.5443954290885104e-06+0j) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13] +
(-3.088250711090746e-06+0j) [Y3 Z4 Z5 Y6 X11 X12] +
(-0.004158797381840073+0j) [Y3 Z4 Z5 Y6 Y12 Y13] +
(-4.561447179977648e-07+0j) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13] +
(-0.12133276911042408+0j) [Y3 Z4 Z5 Z6 Y7] +
(-0.0175612024096462+0j) [Y3 Z4 Z5 Z6 Y7 Z8] +
(-0.005805188989826926+0j) [Y3 Z4 Z5 Z6 Y7 Z9] +
(2.7455184002282526e-06+0j) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12] +
(2.7455184002282526e-06+0j) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12] +
(0.014411099430130907+0j) [Y3 Z4 Z5 Z6 Y7 Z10] +
(-3.427323108624932e-07+0j) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13] +
(-7.988770288602579e-07+0j) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13] +
(-0.0008145313270956889+0j) [Y3 Z4 Z5 Z6 Y7 Z11] +
(-0.0034937903598902115+0j) [Y3 Z4 Z5 Z6 Y7 Z12] +
(0.0006650070219498614+0j) [Y3 Z4 Z5 Z6 Y7 Z13] +
(0.014603704729162146+0j) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10] +
(-2.874299071224402e-06+0j) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-0.014603704729162146+0j) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10] +
(2.874299071224402e-06+0j) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-1.300294656131589e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12] +
(0.0024464971554158583+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(1.300294656131589e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12] +
(0.0024464971554158583+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(-0.28164257767023065+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.017091553155898945+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(-0.019538050311314805+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(5.775950526975332e-05+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.8836765759265984e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(3.846201671021108e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(-0.02428211735469306+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11] +
(5.146496327152698e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(-0.02475546329289104+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11] +
(5.10552672171925e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(-0.03935916802205318+0j) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11] +
(7.979825792943652e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(-0.0299037895126249+0j) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11] +
(5.427988656140024e-06+0j) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.025996177598021253+0j) [Y3 Z4 Z5 Y7] +
(-0.021433810721601+0j) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11] +
(5.159350501692646e-06+0j) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.003479511890334331+0j) [Y3 Z4 Z6 Y7] +
(-0.028730779551905554+0j) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11] +
(5.935867717694495e-06+0j) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.764994119620164e-07+0j) [Y3 Y5] +
(0.001663879878490798+0j) [Y3 Z5 Z6 Y7] +
(-0.018889030304942905+0j) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.9473560114557034e-06+0j) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.65389422268317+0j) [Z3] +
(-1.1708301370226586e-06+0j) [Z3 X4 Z5 X6] +
(-7.0897994671756765e-06+0j) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.034903343373661744+0j) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.1708301370226586e-06+0j) [Z3 Y4 Z5 Y6] +
(-7.0897994671756765e-06+0j) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.034903343373661744+0j) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.16079764534838564+0j) [Z3 Z4] +
(-9.50924975139167e-07+0j) [Z3 X5 Z6 X7] +
(-4.728843146939049e-06+0j) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.024591860883829964+0j) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-9.50924975139167e-07+0j) [Z3 Y5 Z6 Y7] +
(-4.728843146939049e-06+0j) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.024591860883829964+0j) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.12495807739503215+0j) [Z3 Z5] +
(0.024389082531149398+0j) [Z3 X6 Z7 Z8 Z9 X10] +
(-2.0111220978858564e-06+0j) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.024389082531149398+0j) [Z3 Y6 Z7 Z8 Z9 Y10] +
(-2.0111220978858564e-06+0j) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.16853486561579942+0j) [Z3 Z6] +
(0.019020423173039903+0j) [Z3 X7 Z8 Z9 Z10 X11] +
(-2.1032156044294735e-06+0j) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.019020423173039903+0j) [Z3 Y7 Z8 Z9 Z10 Y11] +
(-2.1032156044294735e-06+0j) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.13739104762683232+0j) [Z3 Z7] +
(0.18690820476912556+0j) [Z3 Z8] +
(0.15071408121008295+0j) [Z3 Z9] +
(1.1094407593519738e-06+0j) [Z3 X10 Z11 X12] +
(1.1094407593519738e-06+0j) [Z3 Y10 Z11 Y12] +
(0.1533796824331417+0j) [Z3 Z10] +
(-1.0632283420830098e-06+0j) [Z3 X11 Z12 X13] +
(-1.0632283420830098e-06+0j) [Z3 Y11 Z12 Y13] +
(0.1279950249246842+0j) [Z3 Z11] +
(0.15569010671752484+0j) [Z3 Z12] +
(0.14011289865354834+0j) [Z3 Z13] +
(-0.01198238901024794+0j) [X4 X5 Y6 Y7] +
(-0.007306759928832983+0j) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11] +
(-2.888293596288883e-07+0j) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.007306759928832983+0j) [X4 X5 X7 Z8 Z9 X10] +
(-2.888293596288883e-07+0j) [X4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-0.0071569349198569296+0j) [X4 X5 Y8 Y9] +
(-0.01768006795248155+0j) [X4 X5 Y10 Y11] +
(-3.6945132942591346e-06+0j) [X4 X5 Y10 Z11 Z12 Y13] +
(-3.6945132942591346e-06+0j) [X4 X5 X11 X12] +
(0.01198238901024794+0j) [X4 Y5 Y6 X7] +
(0.007306759928832983+0j) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11] +
(2.888293596288883e-07+0j) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.007306759928832983+0j) [X4 Y5 Y7 Z8 Z9 X10] +
(-2.888293596288883e-07+0j) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(0.0071569349198569296+0j) [X4 Y5 Y8 X9] +
(0.01768006795248155+0j) [X4 Y5 Y10 X11] +
(3.6945132942591346e-06+0j) [X4 Y5 Y10 Z11 Z12 X13] +
(-3.6945132942591346e-06+0j) [X4 Y5 Y11 X12] +
(-1.2260484988755717e-05+0j) [X4 Z5 X6] +
(-1.2283337824815328e-06+0j) [X4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(0.000246364375695814+0j) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2283337824815328e-06+0j) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(0.000246364375695814+0j) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.8540608579572954e-06+0j) [X4 Z5 X6 Z7] +
(-1.398044908091774e-06+0j) [X4 Z5 X6 Z8] +
(-1.881850183190604e-06+0j) [X4 Z5 X6 Z9] +
(0.007960880725921542+0j) [X4 Z5 X6 X10 Z11 X12] +
(-0.0009298507967730285+0j) [X4 Z5 X6 Y10 Z11 Y12] +
(-1.6923978284660816e-06+0j) [X4 Z5 X6 Z10] +
(-0.012215040997613925+0j) [X4 Z5 X6 X11 Z12 X13] +
(-0.012215040997613925+0j) [X4 Z5 X6 Y11 Z12 Y13] +
(4.281913884591407e-06+0j) [X4 Z5 X6 Z11] +
(-4.588855155360685e-06+0j) [X4 Z5 X6 Z13] +
(0.00889073152269457+0j) [X4 Z5 Y6 Y10 Z11 X12] +
(-4.838052750988301e-07+0j) [X4 Z5 Z6 X7 Y8 Y9] +
(5.974311713057489e-06+0j) [X4 Z5 Z6 X7 Y10 Y11] +
(0.011285190200840893+0j) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(0.020175921723535464+0j) [X4 Z5 Z6 X7 X11 X12] +
(-4.55656921776986e-06+0j) [X4 Z5 Z6 X7 Y12 Y13] +
(4.838052750988301e-07+0j) [X4 Z5 Z6 Y7 Y8 X9] +
(-5.974311713057489e-06+0j) [X4 Z5 Z6 Y7 Y10 X11] +
(-0.011285190200840893+0j) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(0.020175921723535464+0j) [X4 Z5 Z6 Y7 Y11 X12] +
(4.55656921776986e-06+0j) [X4 Z5 Z6 Y7 Y12 X13] +
(1.3304731886137888e-06+0j) [X4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(0.005923798336561347+0j) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(1.3304731886137888e-06+0j) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(0.005923798336561347+0j) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(-6.631277927964914e-05+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.016024603689179545+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(-0.016024603689179545+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(3.334331289141671e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(-4.7346220384615475e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(-9.806102774489722e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(-5.071480736028172e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(5.071480736028172e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-0.3693708936615624+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.023145130929529113+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-0.009612634606847283+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(-0.025637238296026835+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(-8.774817864051353e-06+0j) [X4 Z5 Z6 Z7 Z8 X10] +
(-0.04764261217638309+0j) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(-7.444344675437563e-06+0j) [X4 Z5 Z6 Z7 Z9 X10] +
(-0.04171881383982175+0j) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(-6.290028432753968e-06+0j) [X4 Z5 Z6 Z8 Z9 X10] +
(-0.03956441632289337+0j) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(-7.518362215235501e-06+0j) [X4 Z5 Z7 Z8 Z9 X10] +
(-0.039318051947197556+0j) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(-5.929765815528812e-07+0j) [X4 X6] +
(-4.253224225302965e-06+0j) [X4 Z6 Z7 Z8 Z9 X10] +
(-0.022528440196012897+0j) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.01198238901024794+0j) [Y4 X5 X6 Y7] +
(0.007306759928832983+0j) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11] +
(2.888293596288883e-07+0j) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.007306759928832983+0j) [Y4 X5 X7 Z8 Z9 Y10] +
(-2.888293596288883e-07+0j) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(0.0071569349198569296+0j) [Y4 X5 X8 Y9] +
(0.01768006795248155+0j) [Y4 X5 X10 Y11] +
(3.6945132942591346e-06+0j) [Y4 X5 X10 Z11 Z12 Y13] +
(-3.6945132942591346e-06+0j) [Y4 X5 X11 Y12] +
(-0.01198238901024794+0j) [Y4 Y5 X6 X7] +
(-0.007306759928832983+0j) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11] +
(-2.888293596288883e-07+0j) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.007306759928832983+0j) [Y4 Y5 Y7 Z8 Z9 Y10] +
(-2.888293596288883e-07+0j) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.0071569349198569296+0j) [Y4 Y5 X8 X9] +
(-0.01768006795248155+0j) [Y4 Y5 X10 X11] +
(-3.6945132942591346e-06+0j) [Y4 Y5 X10 Z11 Z12 X13] +
(-3.6945132942591346e-06+0j) [Y4 Y5 Y11 Y12] +
(0.00889073152269457+0j) [Y4 Z5 X6 X10 Z11 Y12] +
(-1.2260484988755717e-05+0j) [Y4 Z5 Y6] +
(-1.2283337824815328e-06+0j) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(0.000246364375695814+0j) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2283337824815328e-06+0j) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(0.000246364375695814+0j) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.8540608579572954e-06+0j) [Y4 Z5 Y6 Z7] +
(-1.398044908091774e-06+0j) [Y4 Z5 Y6 Z8] +
(-1.881850183190604e-06+0j) [Y4 Z5 Y6 Z9] +
(-0.0009298507967730285+0j) [Y4 Z5 Y6 X10 Z11 X12] +
(0.007960880725921542+0j) [Y4 Z5 Y6 Y10 Z11 Y12] +
(-1.6923978284660816e-06+0j) [Y4 Z5 Y6 Z10] +
(-0.012215040997613925+0j) [Y4 Z5 Y6 X11 Z12 X13] +
(-0.012215040997613925+0j) [Y4 Z5 Y6 Y11 Z12 Y13] +
(4.281913884591407e-06+0j) [Y4 Z5 Y6 Z11] +
(-4.588855155360685e-06+0j) [Y4 Z5 Y6 Z13] +
(4.838052750988301e-07+0j) [Y4 Z5 Z6 X7 X8 Y9] +
(-5.974311713057489e-06+0j) [Y4 Z5 Z6 X7 X10 Y11] +
(-0.011285190200840893+0j) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(0.020175921723535464+0j) [Y4 Z5 Z6 X7 X11 Y12] +
(4.55656921776986e-06+0j) [Y4 Z5 Z6 X7 X12 Y13] +
(-4.838052750988301e-07+0j) [Y4 Z5 Z6 Y7 X8 X9] +
(5.974311713057489e-06+0j) [Y4 Z5 Z6 Y7 X10 X11] +
(0.011285190200840893+0j) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(0.020175921723535464+0j) [Y4 Z5 Z6 Y7 Y11 Y12] +
(-4.55656921776986e-06+0j) [Y4 Z5 Z6 Y7 X12 X13] +
(1.3304731886137888e-06+0j) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(0.005923798336561347+0j) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(1.3304731886137888e-06+0j) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(0.005923798336561347+0j) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(-6.631277927964914e-05+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.016024603689179545+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(-0.016024603689179545+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(3.334331289141671e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(-4.7346220384615475e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(-9.806102774489722e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(5.071480736028172e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-5.071480736028172e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-0.3693708936615624+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.023145130929529113+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-0.009612634606847283+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(-0.025637238296026835+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(-8.774817864051353e-06+0j) [Y4 Z5 Z6 Z7 Z8 Y10] +
(-0.04764261217638309+0j) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(-7.444344675437563e-06+0j) [Y4 Z5 Z6 Z7 Z9 Y10] +
(-0.04171881383982175+0j) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(-6.290028432753968e-06+0j) [Y4 Z5 Z6 Z8 Z9 Y10] +
(-0.03956441632289337+0j) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(-7.518362215235501e-06+0j) [Y4 Z5 Z7 Z8 Z9 Y10] +
(-0.039318051947197556+0j) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(-5.929765815528812e-07+0j) [Y4 Y6] +
(-4.253224225302965e-06+0j) [Y4 Z6 Z7 Z8 Z9 Y10] +
(-0.022528440196012897+0j) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.203440228914563+0j) [Z4] +
(-5.929765815528812e-07+0j) [Z4 X5 Z6 X7] +
(-4.253224225302965e-06+0j) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.022528440196012897+0j) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-5.929765815528812e-07+0j) [Z4 Y5 Z6 Y7] +
(-4.253224225302965e-06+0j) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.022528440196012897+0j) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.1575531479798565+0j) [Z4 Z5] +
(0.018266834869375456+0j) [Z4 X6 Z7 Z8 Z9 X10] +
(-1.654117476927522e-06+0j) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.018266834869375456+0j) [Z4 Y6 Z7 Z8 Z9 Y10] +
(-1.654117476927522e-06+0j) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.13701191674040736+0j) [Z4 Z6] +
(0.010960074940542472+0j) [Z4 X7 Z8 Z9 Z10 X11] +
(-1.9429468365564106e-06+0j) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.010960074940542472+0j) [Z4 Y7 Z8 Z9 Z10 Y11] +
(-1.9429468365564106e-06+0j) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.1489943057506553+0j) [Z4 Z7] +
(0.14960702684445287+0j) [Z4 Z8] +
(0.1567639617643098+0j) [Z4 Z9] +
(1.8782101246652918e-06+0j) [Z4 X10 Z11 X12] +
(1.8782101246652918e-06+0j) [Z4 Y10 Z11 Y12] +
(0.124899909172376+0j) [Z4 Z10] +
(-1.8163031695938426e-06+0j) [Z4 X11 Z12 X13] +
(-1.8163031695938426e-06+0j) [Z4 Y11 Z12 Y13] +
(0.14257997712485754+0j) [Z4 Z11] +
(0.11383573679388667+0j) [Z4 Z12] +
(0.15215040708869057+0j) [Z4 Z13] +
(1.2283337824815328e-06+0j) [X5 X6 Y7 Z8 Z9 Y10] +
(-0.000246364375695814+0j) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-4.838052750988301e-07+0j) [X5 X6 X8 X9] +
(5.974311713057489e-06+0j) [X5 X6 X10 X11] +
(0.020175921723535464+0j) [X5 X6 X10 Z11 Z12 X13] +
(0.011285190200840893+0j) [X5 X6 Y11 Y12] +
(-4.55656921776986e-06+0j) [X5 X6 X12 X13] +
(-1.2283337824815328e-06+0j) [X5 Y6 Y7 Z8 Z9 X10] +
(0.000246364375695814+0j) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-4.838052750988301e-07+0j) [X5 Y6 Y8 X9] +
(5.974311713057489e-06+0j) [X5 Y6 Y10 X11] +
(0.020175921723535464+0j) [X5 Y6 Y10 Z11 Z12 X13] +
(-0.011285190200840893+0j) [X5 Y6 Y11 X12] +
(-4.55656921776986e-06+0j) [X5 Y6 Y12 X13] +
(-1.2260484988755717e-05+0j) [X5 Z6 X7] +
(-1.881850183190604e-06+0j) [X5 Z6 X7 Z8] +
(-1.398044908091774e-06+0j) [X5 Z6 X7 Z9] +
(-0.012215040997613925+0j) [X5 Z6 X7 X10 Z11 X12] +
(-0.012215040997613925+0j) [X5 Z6 X7 Y10 Z11 Y12] +
(4.281913884591407e-06+0j) [X5 Z6 X7 Z10] +
(0.007960880725921542+0j) [X5 Z6 X7 X11 Z12 X13] +
(-0.0009298507967730285+0j) [X5 Z6 X7 Y11 Z12 Y13] +
(-1.6923978284660816e-06+0j) [X5 Z6 X7 Z11] +
(-4.588855155360685e-06+0j) [X5 Z6 X7 Z12] +
(0.00889073152269457+0j) [X5 Z6 Y7 Y11 Z12 X13] +
(-1.3304731886137888e-06+0j) [X5 Z6 Z7 X8 Y9 Y10] +
(-0.005923798336561347+0j) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(1.3304731886137888e-06+0j) [X5 Z6 Z7 Y8 Y9 X10] +
(0.005923798336561347+0j) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(0.01602460368917955+0j) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12] +
(-5.0714807360281715e-06+0j) [X5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-0.01602460368917955+0j) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12] +
(-5.0714807360281715e-06+0j) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(-6.631277927964912e-05+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-9.806102774489722e-06+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(-4.7346220384615475e-06+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(-0.3693708936615624+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.023145130929529113+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(-0.025637238296026835+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(3.3343312891416712e-06+0j) [X5 Z6 Z7 Z8 Z9 X11] +
(-0.009612634606847283+0j) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(-7.444344675437563e-06+0j) [X5 Z6 Z7 Z8 Z10 X11] +
(-0.04171881383982175+0j) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(-8.774817864051353e-06+0j) [X5 Z6 Z7 Z9 Z10 X11] +
(-0.04764261217638309+0j) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(-7.518362215235501e-06+0j) [X5 Z6 Z8 Z9 Z10 X11] +
(-0.039318051947197556+0j) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.8540608579572954e-06+0j) [X5 X7] +
(-6.290028432753968e-06+0j) [X5 Z7 Z8 Z9 Z10 X11] +
(-0.03956441632289337+0j) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2283337824815328e-06+0j) [Y5 X6 X7 Z8 Z9 Y10] +
(0.000246364375695814+0j) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-4.838052750988301e-07+0j) [Y5 X6 X8 Y9] +
(5.974311713057489e-06+0j) [Y5 X6 X10 Y11] +
(0.020175921723535464+0j) [Y5 X6 X10 Z11 Z12 Y13] +
(-0.011285190200840893+0j) [Y5 X6 X11 Y12] +
(-4.55656921776986e-06+0j) [Y5 X6 X12 Y13] +
(1.2283337824815328e-06+0j) [Y5 Y6 X7 Z8 Z9 X10] +
(-0.000246364375695814+0j) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-4.838052750988301e-07+0j) [Y5 Y6 Y8 Y9] +
(5.974311713057489e-06+0j) [Y5 Y6 Y10 Y11] +
(0.020175921723535464+0j) [Y5 Y6 Y10 Z11 Z12 Y13] +
(0.011285190200840893+0j) [Y5 Y6 X11 X12] +
(-4.55656921776986e-06+0j) [Y5 Y6 Y12 Y13] +
(0.00889073152269457+0j) [Y5 Z6 X7 X11 Z12 Y13] +
(-1.2260484988755717e-05+0j) [Y5 Z6 Y7] +
(-1.881850183190604e-06+0j) [Y5 Z6 Y7 Z8] +
(-1.398044908091774e-06+0j) [Y5 Z6 Y7 Z9] +
(-0.012215040997613925+0j) [Y5 Z6 Y7 X10 Z11 X12] +
(-0.012215040997613925+0j) [Y5 Z6 Y7 Y10 Z11 Y12] +
(4.281913884591407e-06+0j) [Y5 Z6 Y7 Z10] +
(-0.0009298507967730285+0j) [Y5 Z6 Y7 X11 Z12 X13] +
(0.007960880725921542+0j) [Y5 Z6 Y7 Y11 Z12 Y13] +
(-1.6923978284660816e-06+0j) [Y5 Z6 Y7 Z11] +
(-4.588855155360685e-06+0j) [Y5 Z6 Y7 Z12] +
(1.3304731886137888e-06+0j) [Y5 Z6 Z7 X8 X9 Y10] +
(0.005923798336561347+0j) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-1.3304731886137888e-06+0j) [Y5 Z6 Z7 Y8 X9 X10] +
(-0.005923798336561347+0j) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-0.01602460368917955+0j) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12] +
(-5.0714807360281715e-06+0j) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(0.01602460368917955+0j) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12] +
(-5.0714807360281715e-06+0j) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(-6.631277927964912e-05+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-9.806102774489722e-06+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(-4.7346220384615475e-06+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(-0.3693708936615624+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.023145130929529113+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(-0.025637238296026835+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(3.3343312891416712e-06+0j) [Y5 Z6 Z7 Z8 Z9 Y11] +
(-0.009612634606847283+0j) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(-7.444344675437563e-06+0j) [Y5 Z6 Z7 Z8 Z10 Y11] +
(-0.04171881383982175+0j) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(-8.774817864051353e-06+0j) [Y5 Z6 Z7 Z9 Z10 Y11] +
(-0.04764261217638309+0j) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(-7.518362215235501e-06+0j) [Y5 Z6 Z8 Z9 Z10 Y11] +
(-0.039318051947197556+0j) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.8540608579572954e-06+0j) [Y5 Y7] +
(-6.290028432753968e-06+0j) [Y5 Z7 Z8 Z9 Z10 Y11] +
(-0.03956441632289337+0j) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.203440228914563+0j) [Z5] +
(0.010960074940542472+0j) [Z5 X6 Z7 Z8 Z9 X10] +
(-1.9429468365564106e-06+0j) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.010960074940542472+0j) [Z5 Y6 Z7 Z8 Z9 Y10] +
(-1.9429468365564106e-06+0j) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.1489943057506553+0j) [Z5 Z6] +
(0.018266834869375456+0j) [Z5 X7 Z8 Z9 Z10 X11] +
(-1.654117476927522e-06+0j) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.018266834869375456+0j) [Z5 Y7 Z8 Z9 Z10 Y11] +
(-1.654117476927522e-06+0j) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.13701191674040736+0j) [Z5 Z7] +
(0.1567639617643098+0j) [Z5 Z8] +
(0.14960702684445287+0j) [Z5 Z9] +
(-1.8163031695938426e-06+0j) [Z5 X10 Z11 X12] +
(-1.8163031695938426e-06+0j) [Z5 Y10 Z11 Y12] +
(0.14257997712485754+0j) [Z5 Z10] +
(1.8782101246652918e-06+0j) [Z5 X11 Z12 X13] +
(1.8782101246652918e-06+0j) [Z5 Y11 Z12 Y13] +
(0.124899909172376+0j) [Z5 Z11] +
(0.15215040708869057+0j) [Z5 Z12] +
(0.11383573679388667+0j) [Z5 Z13] +
(-0.01387338174842612+0j) [X6 X7 Y8 Y9] +
(-0.017825140995786404+0j) [X6 X7 Y10 Y11] +
(-1.0358477601352874e-06+0j) [X6 X7 Y10 Z11 Z12 Y13] +
(-1.0358477601352874e-06+0j) [X6 X7 X11 X12] +
(0.01387338174842612+0j) [X6 Y7 Y8 X9] +
(0.017825140995786404+0j) [X6 Y7 Y10 X11] +
(1.0358477601352874e-06+0j) [X6 Y7 Y10 Z11 Z12 X13] +
(-1.0358477601352874e-06+0j) [X6 Y7 Y11 X12] +
(0.00029219862611108385+0j) [X6 Z7 X8 X9 Z10 X11] +
(-3.328139350592643e-07+0j) [X6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.00029219862611108385+0j) [X6 Z7 X8 Y9 Z10 Y11] +
(-3.328139350592643e-07+0j) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(0.2284810656491872+0j) [X6 Z7 Z8 Z9 X10] +
(3.313145499991033e-06+0j) [X6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(3.313145499991033e-06+0j) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(0.011307274008848154+0j) [X6 Z7 Z8 Z9 X10 Z11] +
(0.025104957138844492+0j) [X6 Z7 Z8 Z9 X10 Z12] +
(0.010540425907671508+0j) [X6 Z7 Z8 Z9 X10 Z13] +
(-0.014564531231172982+0j) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(0.014564531231172982+0j) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-2.5950860067262673e-05+0j) [X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.183932559166532e-06+0j) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-6.524373848100062e-06+0j) [X6 Z7 Z8 Z9 Z10 X12] +
(-3.2112283481090286e-06+0j) [X6 Z7 Z8 Z9 Z11 X12] +
(0.029812424517345677+0j) [X6 Z7 Z8 X10] +
(-3.277483195161045e-06+0j) [X6 Z7 Z8 Z10 Z11 X12] +
(0.030104623143456757+0j) [X6 Z7 Z9 X10] +
(-3.6102971302203094e-06+0j) [X6 Z7 Z9 Z10 Z11 X12] +
(0.030787505389143863+0j) [X6 Z8 Z9 X10] +
(-3.7696594515937515e-06+0j) [X6 Z8 Z9 Z10 Z11 X12] +
(0.01387338174842612+0j) [Y6 X7 X8 Y9] +
(0.017825140995786404+0j) [Y6 X7 X10 Y11] +
(1.0358477601352874e-06+0j) [Y6 X7 X10 Z11 Z12 Y13] +
(-1.0358477601352874e-06+0j) [Y6 X7 X11 Y12] +
(-0.01387338174842612+0j) [Y6 Y7 X8 X9] +
(-0.017825140995786404+0j) [Y6 Y7 X10 X11] +
(-1.0358477601352874e-06+0j) [Y6 Y7 X10 Z11 Z12 X13] +
(-1.0358477601352874e-06+0j) [Y6 Y7 Y11 Y12] +
(0.00029219862611108385+0j) [Y6 Z7 Y8 X9 Z10 X11] +
(-3.328139350592643e-07+0j) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.00029219862611108385+0j) [Y6 Z7 Y8 Y9 Z10 Y11] +
(-3.328139350592643e-07+0j) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(0.2284810656491872+0j) [Y6 Z7 Z8 Z9 Y10] +
(3.313145499991033e-06+0j) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(3.313145499991033e-06+0j) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(0.011307274008848154+0j) [Y6 Z7 Z8 Z9 Y10 Z11] +
(0.025104957138844492+0j) [Y6 Z7 Z8 Z9 Y10 Z12] +
(0.010540425907671508+0j) [Y6 Z7 Z8 Z9 Y10 Z13] +
(0.014564531231172982+0j) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-0.014564531231172982+0j) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-2.5950860067262673e-05+0j) [Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.183932559166532e-06+0j) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-6.524373848100062e-06+0j) [Y6 Z7 Z8 Z9 Z10 Y12] +
(-3.2112283481090286e-06+0j) [Y6 Z7 Z8 Z9 Z11 Y12] +
(0.029812424517345677+0j) [Y6 Z7 Z8 Y10] +
(-3.277483195161045e-06+0j) [Y6 Z7 Z8 Z10 Z11 Y12] +
(0.030104623143456757+0j) [Y6 Z7 Z9 Y10] +
(-3.6102971302203094e-06+0j) [Y6 Z7 Z9 Z10 Z11 Y12] +
(0.030787505389143863+0j) [Y6 Z8 Z9 Y10] +
(-3.7696594515937515e-06+0j) [Y6 Z8 Z9 Z10 Z11 Y12] +
(1.309686298861543+0j) [Z6] +
(0.030787505389143863+0j) [Z6 X7 Z8 Z9 Z10 X11] +
(-3.7696594515937515e-06+0j) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.030787505389143863+0j) [Z6 Y7 Z8 Z9 Z10 Y11] +
(-3.7696594515937515e-06+0j) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.1939253461327021+0j) [Z6 Z7] +
(0.1675665326546127+0j) [Z6 Z8] +
(0.1814399144030388+0j) [Z6 Z9] +
(-1.8551201212773364e-06+0j) [Z6 X10 Z11 X12] +
(-1.8551201212773364e-06+0j) [Z6 Y10 Z11 Y12] +
(0.11952438964682682+0j) [Z6 Z10] +
(-2.890967881412624e-06+0j) [Z6 X11 Z12 X13] +
(-2.890967881412624e-06+0j) [Z6 Y11 Z12 Y13] +
(0.13734953064261324+0j) [Z6 Z11] +
(0.13401715261963718+0j) [Z6 Z12] +
(0.15138327161428858+0j) [Z6 Z13] +
(-0.0002921986261110839+0j) [X7 X8 Y9 Y10] +
(3.328139350592643e-07+0j) [X7 X8 Y9 Z10 Z11 Y12] +
(0.0002921986261110839+0j) [X7 Y8 Y9 X10] +
(-3.328139350592643e-07+0j) [X7 Y8 Y9 Z10 Z11 X12] +
(-3.3131454999910333e-06+0j) [X7 Z8 Z9 X10 Y11 Y12] +
(-0.014564531231172982+0j) [X7 Z8 Z9 X10 X12 X13] +
(3.3131454999910333e-06+0j) [X7 Z8 Z9 Y10 Y11 X12] +
(-0.014564531231172982+0j) [X7 Z8 Z9 Y10 Y12 X13] +
(0.22848106564918716+0j) [X7 Z8 Z9 Z10 X11] +
(0.010540425907671508+0j) [X7 Z8 Z9 Z10 X11 Z12] +
(0.025104957138844492+0j) [X7 Z8 Z9 Z10 X11 Z13] +
(-2.5950860067262662e-05+0j) [X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.183932559166532e-06+0j) [X7 Z8 Z9 Z10 Z11 X13] +
(-3.2112283481090286e-06+0j) [X7 Z8 Z9 Z10 Z12 X13] +
(0.011307274008848154+0j) [X7 Z8 Z9 X11] +
(-6.524373848100062e-06+0j) [X7 Z8 Z9 Z11 Z12 X13] +
(0.030104623143456757+0j) [X7 Z8 Z10 X11] +
(-3.6102971302203094e-06+0j) [X7 Z8 Z10 Z11 Z12 X13] +
(0.029812424517345677+0j) [X7 Z9 Z10 X11] +
(-3.277483195161045e-06+0j) [X7 Z9 Z10 Z11 Z12 X13] +
(0.0002921986261110839+0j) [Y7 X8 X9 Y10] +
(-3.328139350592643e-07+0j) [Y7 X8 X9 Z10 Z11 Y12] +
(-0.0002921986261110839+0j) [Y7 Y8 X9 X10] +
(3.328139350592643e-07+0j) [Y7 Y8 X9 Z10 Z11 X12] +
(3.3131454999910333e-06+0j) [Y7 Z8 Z9 X10 X11 Y12] +
(-0.014564531231172982+0j) [Y7 Z8 Z9 X10 X12 Y13] +
(-3.3131454999910333e-06+0j) [Y7 Z8 Z9 Y10 X11 X12] +
(-0.014564531231172982+0j) [Y7 Z8 Z9 Y10 Y12 Y13] +
(0.22848106564918716+0j) [Y7 Z8 Z9 Z10 Y11] +
(0.010540425907671508+0j) [Y7 Z8 Z9 Z10 Y11 Z12] +
(0.025104957138844492+0j) [Y7 Z8 Z9 Z10 Y11 Z13] +
(-2.5950860067262662e-05+0j) [Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(4.183932559166532e-06+0j) [Y7 Z8 Z9 Z10 Z11 Y13] +
(-3.2112283481090286e-06+0j) [Y7 Z8 Z9 Z10 Z12 Y13] +
(0.011307274008848154+0j) [Y7 Z8 Z9 Y11] +
(-6.524373848100062e-06+0j) [Y7 Z8 Z9 Z11 Z12 Y13] +
(0.030104623143456757+0j) [Y7 Z8 Z10 Y11] +
(-3.6102971302203094e-06+0j) [Y7 Z8 Z10 Z11 Z12 Y13] +
(0.029812424517345677+0j) [Y7 Z9 Z10 Y11] +
(-3.277483195161045e-06+0j) [Y7 Z9 Z10 Z11 Z12 Y13] +
(1.3096862988615428+0j) [Z7] +
(0.1814399144030388+0j) [Z7 Z8] +
(0.1675665326546127+0j) [Z7 Z9] +
(-2.890967881412624e-06+0j) [Z7 X10 Z11 X12] +
(-2.890967881412624e-06+0j) [Z7 Y10 Z11 Y12] +
(0.13734953064261324+0j) [Z7 Z10] +
(-1.8551201212773364e-06+0j) [Z7 X11 Z12 X13] +
(-1.8551201212773364e-06+0j) [Z7 Y11 Z12 Y13] +
(0.11952438964682682+0j) [Z7 Z11] +
(0.15138327161428858+0j) [Z7 Z12] +
(0.13401715261963718+0j) [Z7 Z13] +
(-0.009560705729135964+0j) [X8 X9 Y10 Y11] +
(6.628614201362812e-07+0j) [X8 X9 Y10 Z11 Z12 Y13] +
(6.628614201362812e-07+0j) [X8 X9 X11 X12] +
(-0.006087822480561874+0j) [X8 X9 Y12 Y13] +
(0.009560705729135964+0j) [X8 Y9 Y10 X11] +
(-6.628614201362812e-07+0j) [X8 Y9 Y10 Z11 Z12 X13] +
(6.628614201362812e-07+0j) [X8 Y9 Y11 X12] +
(0.006087822480561874+0j) [X8 Y9 Y12 X13] +
(0.009560705729135964+0j) [Y8 X9 X10 Y11] +
(-6.628614201362812e-07+0j) [Y8 X9 X10 Z11 Z12 Y13] +
(6.628614201362812e-07+0j) [Y8 X9 X11 Y12] +
(0.006087822480561874+0j) [Y8 X9 X12 Y13] +
(-0.009560705729135964+0j) [Y8 Y9 X10 X11] +
(6.628614201362812e-07+0j) [Y8 Y9 X10 Z11 Z12 X13] +
(6.628614201362812e-07+0j) [Y8 Y9 Y11 Y12] +
(-0.006087822480561874+0j) [Y8 Y9 X12 X13] +
(1.3693525634718173+0j) [Z8] +
(0.2200397733437609+0j) [Z8 Z9] +
(-1.5973171975866097e-06+0j) [Z8 X10 Z11 X12] +
(-1.5973171975866097e-06+0j) [Z8 Y10 Z11 Y12] +
(0.13766872645852582+0j) [Z8 Z10] +
(-9.344557774503286e-07+0j) [Z8 X11 Z12 X13] +
(-9.344557774503286e-07+0j) [Z8 Y11 Z12 Y13] +
(0.14722943218766177+0j) [Z8 Z11] +
(0.14973486803496944+0j) [Z8 Z12] +
(0.15582269051553133+0j) [Z8 Z13] +
(1.3693525634718169+0j) [Z9] +
(-9.344557774503286e-07+0j) [Z9 X10 Z11 X12] +
(-9.344557774503286e-07+0j) [Z9 Y10 Z11 Y12] +
(0.14722943218766177+0j) [Z9 Z10] +
(-1.5973171975866097e-06+0j) [Z9 X11 Z12 X13] +
(-1.5973171975866097e-06+0j) [Z9 Y11 Z12 Y13] +
(0.13766872645852582+0j) [Z9 Z11] +
(0.15582269051553133+0j) [Z9 Z12] +
(0.14973486803496944+0j) [Z9 Z13] +
(-0.028685183716105903+0j) [X10 X11 Y12 Y13] +
(0.028685183716105903+0j) [X10 Y11 Y12 X13] +
(-1.0722312156674748e-05+0j) [X10 Z11 X12] +
(7.95441317576663e-06+0j) [X10 Z11 X12 Z13] +
(-8.194261371595923e-06+0j) [X10 X12] +
(0.028685183716105903+0j) [Y10 X11 X12 Y13] +
(-0.028685183716105903+0j) [Y10 Y11 X12 X13] +
(-1.0722312156674748e-05+0j) [Y10 Z11 Y12] +
(7.95441317576663e-06+0j) [Y10 Z11 Y12 Z13] +
(-8.194261371595923e-06+0j) [Y10 Y12] +
(0.78296617259502+0j) [Z10] +
(-8.194261371595923e-06+0j) [Z10 X11 Z12 X13] +
(-8.194261371595923e-06+0j) [Z10 Y11 Z12 Y13] +
(0.14926355147388917+0j) [Z10 Z11] +
(0.11270386920332234+0j) [Z10 Z12] +
(0.14138905291942827+0j) [Z10 Z13] +
(-1.0722312156674751e-05+0j) [X11 Z12 X13] +
(7.95441317576663e-06+0j) [X11 X13] +
(-1.0722312156674751e-05+0j) [Y11 Z12 Y13] +
(7.95441317576663e-06+0j) [Y11 Y13] +
(0.7829661725950201+0j) [Z11] +
(0.14138905291942827+0j) [Z11 Z12] +
(0.11270386920332234+0j) [Z11 Z13] +
(0.8084581961720506+0j) [Z12] +
(0.1543574865722366+0j) [Z12 Z13] +
(0.8084581961720505+0j) [Z13]
  (-46.46390678868895) [I0]
+ (0.782966172595019) [Z11]
+ (0.7829661725950193) [Z10]
+ (0.8084581961720487) [Z13]
+ (0.8084581961720491) [Z12]
+ (1.2034402289145627) [Z5]
+ (1.2034402289145634) [Z4]
+ (1.3096862988615456) [Z7]
+ (1.309686298861546) [Z6]
+ (1.3693525634718182) [Z8]
+ (1.3693525634718189) [Z9]
+ (1.653894222683171) [Z2]
+ (1.6538942226831714) [Z3]
+ (12.412630742111759) [Z0]
+ (12.412630742111759) [Z1]
+ (-8.19426137259929e-06) [Y10 Y12]
+ (-8.19426137259929e-06) [X10 X12]
+ (-1.8540608578559683e-06) [Y5 Y7]
+ (-1.8540608578559683e-06) [X5 X7]
+ (-7.764994118002307e-07) [Y3 Y5]
+ (-7.764994118002307e-07) [X3 X5]
+ (-5.929765815142931e-07) [Y4 Y6]
+ (-5.929765815142931e-07) [X4 X6]
+ (1.6021167406009305e-06) [Y2 Y4]
+ (1.6021167406009305e-06) [X2 X4]
+ (7.954413176534663e-06) [Y11 Y13]
+ (7.954413176534663e-06) [X11 X13]
+ (0.0032769719312315503) [Y1 Y3]
+ (0.0032769719312315503) [X1 X3]
+ (0.10433064780651372) [Y0 Y2]
+ (0.10433064780651372) [X0 X2]
+ (0.11270386920332204) [Z10 Z12]
+ (0.11270386920332204) [Z11 Z13]
+ (0.11383573679388648) [Z4 Z12]
+ (0.11383573679388648) [Z5 Z13]
+ (0.11952438964682675) [Z6 Z10]
+ (0.11952438964682675) [Z7 Z11]
+ (0.12489990917237587) [Z4 Z10]
+ (0.12489990917237587) [Z5 Z11]
+ (0.12495807739503209) [Z2 Z4]
+ (0.12495807739503209) [Z3 Z5]
+ (0.1279950249246841) [Z2 Z10]
+ (0.1279950249246841) [Z3 Z11]
+ (0.13401715261963706) [Z6 Z12]
+ (0.13401715261963706) [Z7 Z13]
+ (0.13701191674040752) [Z4 Z6]
+ (0.13701191674040752) [Z5 Z7]
+ (0.13734953064261313) [Z6 Z11]
+ (0.13734953064261313) [Z7 Z10]
+ (0.13739104762683235) [Z2 Z6]
+ (0.13739104762683235) [Z3 Z7]
+ (0.13766872645852568) [Z8 Z10]
+ (0.13766872645852568) [Z9 Z11]
+ (0.1401128986535481) [Z2 Z12]
+ (0.1401128986535481) [Z3 Z13]
+ (0.141389052919428) [Z10 Z13]
+ (0.141389052919428) [Z11 Z12]
+ (0.14257997712485743) [Z4 Z11]
+ (0.14257997712485743) [Z5 Z10]
+ (0.1472294321876616) [Z8 Z11]
+ (0.1472294321876616) [Z9 Z10]
+ (0.14899430575065548) [Z4 Z7]
+ (0.14899430575065548) [Z5 Z6]
+ (0.14926355147388887) [Z10 Z11]
+ (0.14960702684445287) [Z4 Z8]
+ (0.14960702684445287) [Z5 Z9]
+ (0.14973486803496922) [Z8 Z12]
+ (0.14973486803496922) [Z9 Z13]
+ (0.15071408121008287) [Z2 Z8]
+ (0.15071408121008287) [Z3 Z9]
+ (0.1513832716142884) [Z6 Z13]
+ (0.1513832716142884) [Z7 Z12]
+ (0.15215040708869032) [Z4 Z13]
+ (0.15215040708869032) [Z5 Z12]
+ (0.15337968243314148) [Z2 Z11]
+ (0.15337968243314148) [Z3 Z10]
+ (0.15435748657223616) [Z12 Z13]
+ (0.15569010671752453) [Z2 Z13]
+ (0.15569010671752453) [Z3 Z12]
+ (0.15582269051553108) [Z8 Z13]
+ (0.15582269051553108) [Z9 Z12]
+ (0.15676396176430982) [Z4 Z9]
+ (0.15676396176430982) [Z5 Z8]
+ (0.15755314797985648) [Z4 Z5]
+ (0.16079764534838556) [Z2 Z5]
+ (0.16079764534838556) [Z3 Z4]
+ (0.1675665326546128) [Z6 Z8]
+ (0.1675665326546128) [Z7 Z9]
+ (0.1685348656157995) [Z2 Z7]
+ (0.1685348656157995) [Z3 Z6]
+ (0.18143991440303892) [Z6 Z9]
+ (0.18143991440303892) [Z7 Z8]
+ (0.18189085790751364) [Z2 Z3]
+ (0.18690820476912548) [Z2 Z9]
+ (0.18690820476912548) [Z3 Z8]
+ (0.192997239353642) [Z0 Z10]
+ (0.192997239353642) [Z1 Z11]
+ (0.1939253461327023) [Z6 Z7]
+ (0.1966177089034211) [Z0 Z4]
+ (0.1966177089034211) [Z1 Z5]
+ (0.1993635453736079) [Z0 Z5]
+ (0.1993635453736079) [Z1 Z4]
+ (0.2007286646044173) [Z0 Z11]
+ (0.2007286646044173) [Z1 Z10]
+ (0.21102659849791489) [Z0 Z12]
+ (0.21102659849791489) [Z1 Z13]
+ (0.21631037498631783) [Z0 Z13]
+ (0.21631037498631783) [Z1 Z12]
+ (0.2200397733437609) [Z8 Z9]
+ (0.236710807838304) [Z0 Z2]
+ (0.236710807838304) [Z1 Z3]
+ (0.24164663936017203) [Z0 Z6]
+ (0.24164663936017203) [Z1 Z7]
+ (0.2485348337131426) [Z0 Z7]
+ (0.2485348337131426) [Z1 Z6]
+ (0.25129445674591666) [Z0 Z3]
+ (0.25129445674591666) [Z1 Z2]
+ (0.2723251830660565) [Z0 Z8]
+ (0.2723251830660565) [Z1 Z9]
+ (0.27883454426723375) [Z0 Z9]
+ (0.27883454426723375) [Z1 Z8]
+ (1.1861763734860475) [Z0 Z1]
+ (-1.2260484988467187e-05) [Y4 Z5 Y6]
+ (-1.2260484988467187e-05) [X4 Z5 X6]
+ (-1.2260484988467187e-05) [Y5 Z6 Y7]
+ (-1.2260484988467187e-05) [X5 Z6 X7]
+ (-1.0722312158253439e-05) [Y10 Z11 Y12]
+ (-1.0722312158253439e-05) [X10 Z11 X12]
+ (-1.0722312158253434e-05) [Y11 Z12 Y13]
+ (-1.0722312158253434e-05) [X11 Z12 X13]
+ (-3.887051671574338e-06) [Y3 Z4 Y5]
+ (-3.887051671574338e-06) [X3 Z4 X5]
+ (-3.887051671574337e-06) [Y2 Z3 Y4]
+ (-3.887051671574337e-06) [X2 Z3 X4]
+ (0.12507032579771904) [Y0 Z1 Y2]
+ (0.12507032579771904) [X0 Z1 X2]
+ (0.12507032579771907) [Y1 Z2 Y3]
+ (0.12507032579771907) [X1 Z2 X3]
+ (-0.038314670294803836) [Y4 Y5 X12 X13]
+ (-0.038314670294803836) [X4 X5 Y12 Y13]
+ (-0.036194123559042606) [Y2 Y3 X8 X9]
+ (-0.036194123559042606) [X2 X3 Y8 Y9]
+ (-0.03583956795335346) [Y2 Y3 X4 X5]
+ (-0.03583956795335346) [X2 X3 Y4 Y5]
+ (-0.03114381798896714) [Y2 Y3 X6 X7]
+ (-0.03114381798896714) [X2 X3 Y6 Y7]
+ (-0.028685183716105952) [Y10 Y11 X12 X13]
+ (-0.028685183716105952) [X10 X11 Y12 Y13]
+ (-0.025996177598021197) [Y3 Z4 Z5 Y7]
+ (-0.025996177598021197) [X3 Z4 Z5 X7]
+ (-0.025384657508457385) [Y2 Y3 X10 X11]
+ (-0.025384657508457385) [X2 X3 Y10 Y11]
+ (-0.01902824244384731) [Y3 Y4 X11 X12]
+ (-0.01902824244384731) [X3 X4 Y11 Y12]
+ (-0.017825140995786384) [Y6 Y7 X10 X11]
+ (-0.017825140995786384) [X6 X7 Y10 Y11]
+ (-0.017680067952481546) [Y4 Y5 X10 X11]
+ (-0.017680067952481546) [X4 X5 Y10 Y11]
+ (-0.01736611899465135) [Y6 Y7 X12 X13]
+ (-0.01736611899465135) [X6 X7 Y12 Y13]
+ (-0.015577208063976456) [Y2 Y3 X12 X13]
+ (-0.015577208063976456) [X2 X3 Y12 Y13]
+ (-0.01458364890761264) [Y0 Y1 X2 X3]
+ (-0.01458364890761264) [X0 X1 Y2 Y3]
+ (-0.013873381748426115) [Y6 Y7 X8 X9]
+ (-0.013873381748426115) [X6 X7 Y8 Y9]
+ (-0.011982389010247924) [Y4 Y5 X6 X7]
+ (-0.011982389010247924) [X4 X5 Y6 Y7]
+ (-0.01128519020084087) [Y5 X6 X11 Y12]
+ (-0.01128519020084087) [X5 Y6 Y11 X12]
+ (-0.00956070572913594) [Y8 Y9 X10 X11]
+ (-0.00956070572913594) [X8 X9 Y10 Y11]
+ (-0.008125251921381048) [Y1 X2 X8 Y9]
+ (-0.008125251921381048) [Y1 Y2 Y8 Y9]
+ (-0.008125251921381048) [X1 X2 X8 X9]
+ (-0.008125251921381048) [X1 Y2 Y8 X9]
+ (-0.007731425250775297) [Y0 Y1 X10 X11]
+ (-0.007731425250775297) [X0 X1 Y10 Y11]
+ (-0.0068881943529705714) [Y0 Y1 X6 X7]
+ (-0.0068881943529705714) [X0 X1 Y6 Y7]
+ (-0.006087822480561857) [Y8 Y9 X12 X13]
+ (-0.006087822480561857) [X8 X9 Y12 Y13]
+ (-0.005143391768825098) [Y3 X4 X5 Y6]
+ (-0.005143391768825098) [X3 Y4 Y5 X6]
+ (-0.004684903388155231) [Y1 X2 X6 Y7]
+ (-0.004684903388155231) [Y1 Y2 Y6 Y7]
+ (-0.004684903388155231) [X1 X2 X6 X7]
+ (-0.004684903388155231) [X1 Y2 Y6 X7]
+ (-0.004575007626639206) [Y1 X2 X12 Y13]
+ (-0.004575007626639206) [Y1 Y2 Y12 Y13]
+ (-0.004575007626639206) [X1 X2 X12 X13]
+ (-0.004575007626639206) [X1 Y2 Y12 X13]
+ (-0.004424855449441861) [Y1 X2 X4 Y5]
+ (-0.004424855449441861) [Y1 Y2 Y4 Y5]
+ (-0.004424855449441861) [X1 X2 X4 X5]
+ (-0.004424855449441861) [X1 Y2 Y4 X5]
+ (-0.0034795118903342727) [Y2 Z3 Z5 Y6]
+ (-0.0034795118903342727) [X2 Z3 Z5 X6]
+ (-0.0034795118903342727) [Y3 Z4 Z6 Y7]
+ (-0.0034795118903342727) [X3 Z4 Z6 X7]
+ (-0.0027458364701868094) [Y0 Y1 X4 X5]
+ (-0.0027458364701868094) [X0 X1 Y4 Y5]
+ (-0.001799219493663004) [Y1 X2 X10 Y11]
+ (-0.001799219493663004) [Y1 Y2 Y10 Y11]
+ (-0.001799219493663004) [X1 X2 X10 X11]
+ (-0.001799219493663004) [X1 Y2 Y10 X11]
+ (-0.00029219862611107046) [Y7 Y8 X9 X10]
+ (-0.00029219862611107046) [X7 X8 Y9 Y10]
+ (-8.19426137259929e-06) [Z10 Y11 Z12 Y13]
+ (-8.19426137259929e-06) [Z10 X11 Z12 X13]
+ (-7.80170750086841e-06) [Y2 Z3 Y4 Z11]
+ (-7.80170750086841e-06) [X2 Z3 X4 Z11]
+ (-7.80170750086841e-06) [Y3 Z4 Y5 Z10]
+ (-7.80170750086841e-06) [X3 Z4 X5 Z10]
+ (-4.6430510687212846e-06) [Y3 X4 X10 Y11]
+ (-4.6430510687212846e-06) [Y3 Y4 Y10 Y11]
+ (-4.6430510687212846e-06) [X3 X4 X10 X11]
+ (-4.6430510687212846e-06) [X3 Y4 Y10 X11]
+ (-4.58885515585455e-06) [Y4 Z5 Y6 Z13]
+ (-4.58885515585455e-06) [X4 Z5 X6 Z13]
+ (-4.58885515585455e-06) [Y5 Z6 Y7 Z12]
+ (-4.58885515585455e-06) [X5 Z6 X7 Z12]
+ (-4.5565692184402975e-06) [Y5 X6 X12 Y13]
+ (-4.5565692184402975e-06) [Y5 Y6 Y12 Y13]
+ (-4.5565692184402975e-06) [X5 X6 X12 X13]
+ (-4.5565692184402975e-06) [X5 Y6 Y12 X13]
+ (-3.6945132947365456e-06) [Y4 X5 X11 Y12]
+ (-3.6945132947365456e-06) [Y4 Y5 Y11 Y12]
+ (-3.6945132947365456e-06) [X4 X5 X11 X12]
+ (-3.6945132947365456e-06) [X4 Y5 Y11 X12]
+ (-3.3440815562926993e-06) [Z0 Y5 Z6 Y7]
+ (-3.3440815562926993e-06) [Z0 X5 Z6 X7]
+ (-3.3440815562926993e-06) [Z1 Y4 Z5 Y6]
+ (-3.3440815562926993e-06) [Z1 X4 Z5 X6]
+ (-3.1586564321471257e-06) [Y2 Z3 Y4 Z10]
+ (-3.1586564321471257e-06) [X2 Z3 X4 Z10]
+ (-3.1586564321471257e-06) [Y3 Z4 Y5 Z11]
+ (-3.1586564321471257e-06) [X3 Z4 X5 Z11]
+ (-3.0993492434383338e-06) [Z0 Y4 Z5 Y6]
+ (-3.0993492434383338e-06) [Z0 X4 Z5 X6]
+ (-3.0993492434383338e-06) [Z1 Y5 Z6 Y7]
+ (-3.0993492434383338e-06) [Z1 X5 Z6 X7]
+ (-2.8909678817607777e-06) [Z6 Y11 Z12 Y13]
+ (-2.8909678817607777e-06) [Z6 X11 Z12 X13]
+ (-2.8909678817607777e-06) [Z7 Y10 Z11 Y12]
+ (-2.8909678817607777e-06) [Z7 X10 Z11 X12]
+ (-2.177664605286025e-06) [Z0 Y10 Z11 Y12]
+ (-2.177664605286025e-06) [Z0 X10 Z11 X12]
+ (-2.177664605286025e-06) [Z1 Y11 Z12 Y13]
+ (-2.177664605286025e-06) [Z1 X11 Z12 X13]
+ (-1.881850183112444e-06) [Y4 Z5 Y6 Z9]
+ (-1.881850183112444e-06) [X4 Z5 X6 Z9]
+ (-1.881850183112444e-06) [Y5 Z6 Y7 Z8]
+ (-1.881850183112444e-06) [X5 Z6 X7 Z8]
+ (-1.8551201217330897e-06) [Z6 Y10 Z11 Y12]
+ (-1.8551201217330897e-06) [Z6 X10 Z11 X12]
+ (-1.8551201217330897e-06) [Z7 Y11 Z12 Y13]
+ (-1.8551201217330897e-06) [Z7 X11 Z12 X13]
+ (-1.8540608578559683e-06) [Y4 Z5 Y6 Z7]
+ (-1.8540608578559683e-06) [X4 Z5 X6 Z7]
+ (-1.8163031699054416e-06) [Z4 Y11 Z12 Y13]
+ (-1.8163031699054416e-06) [Z4 X11 Z12 X13]
+ (-1.8163031699054416e-06) [Z5 Y10 Z11 Y12]
+ (-1.8163031699054416e-06) [Z5 X10 Z11 X12]
+ (-1.6923978286399652e-06) [Y4 Z5 Y6 Z10]
+ (-1.6923978286399652e-06) [X4 Z5 X6 Z10]
+ (-1.6923978286399652e-06) [Y5 Z6 Y7 Z11]
+ (-1.6923978286399652e-06) [X5 Z6 X7 Z11]
+ (-1.6148794141319337e-06) [Z0 Y11 Z12 Y13]
+ (-1.6148794141319337e-06) [Z0 X11 Z12 X13]
+ (-1.6148794141319337e-06) [Z1 Y10 Z11 Y12]
+ (-1.6148794141319337e-06) [Z1 X10 Z11 X12]
+ (-1.597317197938434e-06) [Z8 Y10 Z11 Y12]
+ (-1.597317197938434e-06) [Z8 X10 Z11 X12]
+ (-1.597317197938434e-06) [Z9 Y11 Z12 Y13]
+ (-1.597317197938434e-06) [Z9 X11 Z12 X13]
+ (-1.454842448913947e-06) [Y3 X4 X6 Y7]
+ (-1.454842448913947e-06) [Y3 Y4 Y6 Y7]
+ (-1.454842448913947e-06) [X3 X4 X6 X7]
+ (-1.454842448913947e-06) [X3 Y4 Y6 X7]
+ (-1.3980449080673468e-06) [Y4 Z5 Y6 Z8]
+ (-1.3980449080673468e-06) [X4 Z5 X6 Z8]
+ (-1.3980449080673468e-06) [Y5 Z6 Y7 Z9]
+ (-1.3980449080673468e-06) [X5 Z6 X7 Z9]
+ (-1.195489009768799e-06) [Y2 Z3 Y4 Z7]
+ (-1.195489009768799e-06) [X2 Z3 X4 Z7]
+ (-1.195489009768799e-06) [Y3 Z4 Y5 Z6]
+ (-1.195489009768799e-06) [X3 Z4 X5 Z6]
+ (-1.190850808037421e-06) [Z0 Y3 Z4 Y5]
+ (-1.190850808037421e-06) [Z0 X3 Z4 X5]
+ (-1.190850808037421e-06) [Z1 Y2 Z3 Y4]
+ (-1.190850808037421e-06) [Z1 X2 Z3 X4]
+ (-1.1708301369883637e-06) [Z2 Y5 Z6 Y7]
+ (-1.1708301369883637e-06) [Z2 X5 Z6 X7]
+ (-1.1708301369883637e-06) [Z3 Y4 Z5 Y6]
+ (-1.1708301369883637e-06) [Z3 X4 Z5 X6]
+ (-1.063228342508572e-06) [Z2 Y10 Z11 Y12]
+ (-1.063228342508572e-06) [Z2 X10 Z11 X12]
+ (-1.063228342508572e-06) [Z3 Y11 Z12 Y13]
+ (-1.063228342508572e-06) [Z3 X11 Z12 X13]
+ (-1.035847760027688e-06) [Y6 X7 X11 Y12]
+ (-1.035847760027688e-06) [Y6 Y7 Y11 Y12]
+ (-1.035847760027688e-06) [X6 X7 X11 X12]
+ (-1.035847760027688e-06) [X6 Y7 Y11 X12]
+ (-9.50924975134113e-07) [Z2 Y4 Z5 Y6]
+ (-9.50924975134113e-07) [Z2 X4 Z5 X6]
+ (-9.50924975134113e-07) [Z3 Y5 Z6 Y7]
+ (-9.50924975134113e-07) [Z3 X5 Z6 X7]
+ (-9.344557777608474e-07) [Z8 Y11 Z12 Y13]
+ (-9.344557777608474e-07) [Z8 X11 Z12 X13]
+ (-9.344557777608474e-07) [Z9 Y10 Z11 Y12]
+ (-9.344557777608474e-07) [Z9 X10 Z11 X12]
+ (-8.33774675180268e-07) [Z0 Y2 Z3 Y4]
+ (-8.33774675180268e-07) [Z0 X2 Z3 X4]
+ (-8.33774675180268e-07) [Z1 Y3 Z4 Y5]
+ (-8.33774675180268e-07) [Z1 X3 Z4 X5]
+ (-7.956895371464832e-07) [Y3 X4 X8 Y9]
+ (-7.956895371464832e-07) [Y3 Y4 Y8 Y9]
+ (-7.956895371464832e-07) [X3 X4 X8 X9]
+ (-7.956895371464832e-07) [X3 Y4 Y8 X9]
+ (-7.764994118002305e-07) [Y2 Z3 Y4 Z5]
+ (-7.764994118002305e-07) [X2 Z3 X4 Z5]
+ (-5.929765815142931e-07) [Z4 Y5 Z6 Y7]
+ (-5.929765815142931e-07) [Z4 X5 Z6 X7]
+ (-5.77005299334946e-07) [Y2 Z3 Y4 Z9]
+ (-5.77005299334946e-07) [X2 Z3 X4 Z9]
+ (-5.77005299334946e-07) [Y3 Z4 Y5 Z8]
+ (-5.77005299334946e-07) [X3 Z4 X5 Z8]
+ (-5.471647744876414e-07) [Y1 Y2 X11 X12]
+ (-5.471647744876414e-07) [X1 X2 Y11 Y12]
+ (-4.838052750450973e-07) [Y5 X6 X8 Y9]
+ (-4.838052750450973e-07) [Y5 Y6 Y8 Y9]
+ (-4.838052750450973e-07) [X5 X6 X8 X9]
+ (-4.838052750450973e-07) [X5 Y6 Y8 X9]
+ (-3.5707613285715305e-07) [Y0 X1 X3 Y4]
+ (-3.5707613285715305e-07) [Y0 Y1 Y3 Y4]
+ (-3.5707613285715305e-07) [X0 X1 X3 X4]
+ (-3.5707613285715305e-07) [X0 Y1 Y3 X4]
+ (-2.447323128543655e-07) [Y0 X1 X5 Y6]
+ (-2.447323128543655e-07) [Y0 Y1 Y5 Y6]
+ (-2.447323128543655e-07) [X0 X1 X5 X6]
+ (-2.447323128543655e-07) [X0 Y1 Y5 X6]
+ (-2.1990516185425066e-07) [Y2 X3 X5 Y6]
+ (-2.1990516185425066e-07) [Y2 Y3 Y5 Y6]
+ (-2.1990516185425066e-07) [X2 X3 X5 X6]
+ (-2.1990516185425066e-07) [X2 Y3 Y5 X6]
+ (-1.9332412769016697e-07) [Y1 X2 X3 Y4]
+ (-1.9332412769016697e-07) [X1 Y2 Y3 X4]
+ (-1.2919694861335722e-07) [Y1 Z2 Z3 Y5]
+ (-1.2919694861335722e-07) [X1 Z2 Z3 X5]
+ (1.7379332622297848e-07) [Y0 Z1 Z3 Y4]
+ (1.7379332622297848e-07) [X0 Z1 Z3 X4]
+ (1.7379332622297848e-07) [Y1 Z2 Z4 Y5]
+ (1.7379332622297848e-07) [X1 Z2 Z4 X5]
+ (1.9332412769016697e-07) [Y1 Y2 X3 X4]
+ (1.9332412769016697e-07) [X1 X2 Y3 Y4]
+ (2.1868423781153717e-07) [Y2 Z3 Y4 Z8]
+ (2.1868423781153717e-07) [X2 Z3 X4 Z8]
+ (2.1868423781153717e-07) [Y3 Z4 Y5 Z9]
+ (2.1868423781153717e-07) [X3 Z4 X5 Z9]
+ (2.593534391451483e-07) [Y2 Z3 Y4 Z6]
+ (2.593534391451483e-07) [X2 Z3 X4 Z6]
+ (2.593534391451483e-07) [Y3 Z4 Y5 Z7]
+ (2.593534391451483e-07) [X3 Z4 X5 Z7]
+ (3.606071867578728e-07) [Y0 Z1 Z2 Y4]
+ (3.606071867578728e-07) [X0 Z1 Z2 X4]
+ (3.606071867578728e-07) [Y1 Z3 Z4 Y5]
+ (3.606071867578728e-07) [X1 Z3 Z4 X5]
+ (5.471647744876414e-07) [Y1 X2 X11 Y12]
+ (5.471647744876414e-07) [X1 Y2 Y11 X12]
+ (5.627851911540907e-07) [Y0 X1 X11 Y12]
+ (5.627851911540907e-07) [Y0 Y1 Y11 Y12]
+ (5.627851911540907e-07) [X0 X1 X11 X12]
+ (5.627851911540907e-07) [X0 Y1 Y11 X12]
+ (6.628614201775864e-07) [Y8 X9 X11 Y12]
+ (6.628614201775864e-07) [Y8 Y9 Y11 Y12]
+ (6.628614201775864e-07) [X8 X9 X11 X12]
+ (6.628614201775864e-07) [X8 Y9 Y11 X12]
+ (1.109440758995731e-06) [Z2 Y11 Z12 Y13]
+ (1.109440758995731e-06) [Z2 X11 Z12 X13]
+ (1.109440758995731e-06) [Z3 Y10 Z11 Y12]
+ (1.109440758995731e-06) [Z3 X10 Z11 X12]
+ (1.6021167406009305e-06) [Z2 Y3 Z4 Y5]
+ (1.6021167406009305e-06) [Z2 X3 Z4 X5]
+ (1.8782101248311038e-06) [Z4 Y10 Z11 Y12]
+ (1.8782101248311038e-06) [Z4 X10 Z11 X12]
+ (1.8782101248311038e-06) [Z5 Y11 Z12 Y13]
+ (1.8782101248311038e-06) [Z5 X11 Z12 X13]
+ (2.172669101504303e-06) [Y2 X3 X11 Y12]
+ (2.172669101504303e-06) [Y2 Y3 Y11 Y12]
+ (2.172669101504303e-06) [X2 X3 X11 X12]
+ (2.172669101504303e-06) [X2 Y3 Y11 X12]
+ (3.1174479456744726e-06) [Y0 Z2 Z3 Y4]
+ (3.1174479456744726e-06) [X0 Z2 Z3 X4]
+ (3.5390541847282602e-06) [Y2 Z3 Y4 Z12]
+ (3.5390541847282602e-06) [X2 Z3 X4 Z12]
+ (3.5390541847282602e-06) [Y3 Z4 Y5 Z13]
+ (3.5390541847282602e-06) [X3 Z4 X5 Z13]
+ (4.281913884990466e-06) [Y4 Z5 Y6 Z11]
+ (4.281913884990466e-06) [X4 Z5 X6 Z11]
+ (4.281913884990466e-06) [Y5 Z6 Y7 Z10]
+ (4.281913884990466e-06) [X5 Z6 X7 Z10]
+ (5.275883122304562e-06) [Y3 X4 X12 Y13]
+ (5.275883122304562e-06) [Y3 Y4 Y12 Y13]
+ (5.275883122304562e-06) [X3 X4 X12 X13]
+ (5.275883122304562e-06) [X3 Y4 Y12 X13]
+ (5.974311713630432e-06) [Y5 X6 X10 Y11]
+ (5.974311713630432e-06) [Y5 Y6 Y10 Y11]
+ (5.974311713630432e-06) [X5 X6 X10 X11]
+ (5.974311713630432e-06) [X5 Y6 Y10 X11]
+ (7.954413176534663e-06) [Y10 Z11 Y12 Z13]
+ (7.954413176534663e-06) [X10 Z11 X12 Z13]
+ (8.814937307032823e-06) [Y2 Z3 Y4 Z13]
+ (8.814937307032823e-06) [X2 Z3 X4 Z13]
+ (8.814937307032823e-06) [Y3 Z4 Y5 Z12]
+ (8.814937307032823e-06) [X3 Z4 X5 Z12]
+ (0.00029219862611107046) [Y7 X8 X9 Y10]
+ (0.00029219862611107046) [X7 Y8 Y9 X10]
+ (0.0004956762314917193) [Y2 Z4 Z5 Y6]
+ (0.0004956762314917193) [X2 Z4 Z5 X6]
+ (0.0011059037691895923) [Y0 Z1 Y2 Z5]
+ (0.0011059037691895923) [X0 Z1 X2 Z5]
+ (0.0011059037691895923) [Y1 Z2 Y3 Z4]
+ (0.0011059037691895923) [X1 Z2 X3 Z4]
+ (0.0016638798784908264) [Y2 Z3 Z4 Y6]
+ (0.0016638798784908264) [X2 Z3 Z4 X6]
+ (0.0016638798784908264) [Y3 Z5 Z6 Y7]
+ (0.0016638798784908264) [X3 Z5 Z6 X7]
+ (0.001756070701841157) [Y0 Z1 Y2 Z11]
+ (0.001756070701841157) [X0 Z1 X2 Z11]
+ (0.001756070701841157) [Y1 Z2 Y3 Z10]
+ (0.001756070701841157) [X1 Z2 X3 Z10]
+ (0.0023262306231579934) [Y0 Z1 Y2 Z13]
+ (0.0023262306231579934) [X0 Z1 X2 Z13]
+ (0.0023262306231579934) [Y1 Z2 Y3 Z12]
+ (0.0023262306231579934) [X1 Z2 X3 Z12]
+ (0.0027458364701868094) [Y0 X1 X4 Y5]
+ (0.0027458364701868094) [X0 Y1 Y4 X5]
+ (0.0029297686747509445) [Y0 Z1 Y2 Z9]
+ (0.0029297686747509445) [X0 Z1 X2 Z9]
+ (0.0029297686747509445) [Y1 Z2 Y3 Z8]
+ (0.0029297686747509445) [X1 Z2 X3 Z8]
+ (0.0032769719312315507) [Y0 Z1 Y2 Z3]
+ (0.0032769719312315507) [X0 Z1 X2 Z3]
+ (0.003347617530666088) [Y0 Z1 Y2 Z7]
+ (0.003347617530666088) [X0 Z1 X2 Z7]
+ (0.003347617530666088) [Y1 Z2 Y3 Z6]
+ (0.003347617530666088) [X1 Z2 X3 Z6]
+ (0.0035552901955041615) [Y0 Z1 Y2 Z10]
+ (0.0035552901955041615) [X0 Z1 X2 Z10]
+ (0.0035552901955041615) [Y1 Z2 Y3 Z11]
+ (0.0035552901955041615) [X1 Z2 X3 Z11]
+ (0.005143391768825098) [Y3 Y4 X5 X6]
+ (0.005143391768825098) [X3 X4 Y5 Y6]
+ (0.005530759218631453) [Y0 Z1 Y2 Z4]
+ (0.005530759218631453) [X0 Z1 X2 Z4]
+ (0.005530759218631453) [Y1 Z2 Y3 Z5]
+ (0.005530759218631453) [X1 Z2 X3 Z5]
+ (0.006087822480561857) [Y8 X9 X12 Y13]
+ (0.006087822480561857) [X8 Y9 Y12 X13]
+ (0.0068881943529705714) [Y0 X1 X6 Y7]
+ (0.0068881943529705714) [X0 Y1 Y6 X7]
+ (0.006901238249797199) [Y0 Z1 Y2 Z12]
+ (0.006901238249797199) [X0 Z1 X2 Z12]
+ (0.006901238249797199) [Y1 Z2 Y3 Z13]
+ (0.006901238249797199) [X1 Z2 X3 Z13]
+ (0.007731425250775297) [Y0 X1 X10 Y11]
+ (0.007731425250775297) [X0 Y1 Y10 X11]
+ (0.008032520918821319) [Y0 Z1 Y2 Z6]
+ (0.008032520918821319) [X0 Z1 X2 Z6]
+ (0.008032520918821319) [Y1 Z2 Y3 Z7]
+ (0.008032520918821319) [X1 Z2 X3 Z7]
+ (0.00956070572913594) [Y8 X9 X10 Y11]
+ (0.00956070572913594) [X8 Y9 Y10 X11]
+ (0.011055020596131991) [Y0 Z1 Y2 Z8]
+ (0.011055020596131991) [X0 Z1 X2 Z8]
+ (0.011055020596131991) [Y1 Z2 Y3 Z9]
+ (0.011055020596131991) [X1 Z2 X3 Z9]
+ (0.01128519020084087) [Y5 Y6 X11 X12]
+ (0.01128519020084087) [X5 X6 Y11 Y12]
+ (0.011307274008848123) [Y7 Z8 Z9 Y11]
+ (0.011307274008848123) [X7 Z8 Z9 X11]
+ (0.011982389010247924) [Y4 X5 X6 Y7]
+ (0.011982389010247924) [X4 Y5 Y6 X7]
+ (0.013873381748426115) [Y6 X7 X8 Y9]
+ (0.013873381748426115) [X6 Y7 Y8 X9]
+ (0.01458364890761264) [Y0 X1 X2 Y3]
+ (0.01458364890761264) [X0 Y1 Y2 X3]
+ (0.015577208063976456) [Y2 X3 X12 Y13]
+ (0.015577208063976456) [X2 Y3 Y12 X13]
+ (0.01736611899465135) [Y6 X7 X12 Y13]
+ (0.01736611899465135) [X6 Y7 Y12 X13]
+ (0.017680067952481546) [Y4 X5 X10 Y11]
+ (0.017680067952481546) [X4 Y5 Y10 X11]
+ (0.017825140995786384) [Y6 X7 X10 Y11]
+ (0.017825140995786384) [X6 Y7 Y10 X11]
+ (0.01902824244384731) [Y3 X4 X11 Y12]
+ (0.01902824244384731) [X3 Y4 Y11 X12]
+ (0.025384657508457385) [Y2 X3 X10 Y11]
+ (0.025384657508457385) [X2 Y3 Y10 X11]
+ (0.028685183716105952) [Y10 X11 X12 Y13]
+ (0.028685183716105952) [X10 Y11 Y12 X13]
+ (0.029812424517345688) [Y6 Z7 Z8 Y10]
+ (0.029812424517345688) [X6 Z7 Z8 X10]
+ (0.029812424517345688) [Y7 Z9 Z10 Y11]
+ (0.029812424517345688) [X7 Z9 Z10 X11]
+ (0.03010462314345676) [Y6 Z7 Z9 Y10]
+ (0.03010462314345676) [X6 Z7 Z9 X10]
+ (0.03010462314345676) [Y7 Z8 Z10 Y11]
+ (0.03010462314345676) [X7 Z8 Z10 X11]
+ (0.03078750538914391) [Y6 Z8 Z9 Y10]
+ (0.03078750538914391) [X6 Z8 Z9 X10]
+ (0.03114381798896714) [Y2 X3 X6 Y7]
+ (0.03114381798896714) [X2 Y3 Y6 X7]
+ (0.03583956795335346) [Y2 X3 X4 Y5]
+ (0.03583956795335346) [X2 Y3 Y4 X5]
+ (0.036194123559042606) [Y2 X3 X8 Y9]
+ (0.036194123559042606) [X2 Y3 Y8 X9]
+ (0.038314670294803836) [Y4 X5 X12 Y13]
+ (0.038314670294803836) [X4 Y5 Y12 X13]
+ (0.10433064780651372) [Z0 Y1 Z2 Y3]
+ (0.10433064780651372) [Z0 X1 Z2 X3]
+ (-0.12133276911042343) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042343) [X2 Z3 Z4 Z5 X6]
+ (-0.1213327691104234) [Y3 Z4 Z5 Z6 Y7]
+ (-0.1213327691104234) [X3 Z4 Z5 Z6 X7]
+ (3.2020768795009174e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.2020768795009174e-06) [X0 Z1 Z2 Z3 X4]
+ (3.2020768795009182e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.2020768795009182e-06) [X1 Z2 Z3 Z4 X5]
+ (0.22848106564918708) [Y6 Z7 Z8 Z9 Y10]
+ (0.22848106564918708) [X6 Z7 Z8 Z9 X10]
+ (0.22848106564918716) [Y7 Z8 Z9 Z10 Y11]
+ (0.22848106564918716) [X7 Z8 Z9 Z10 X11]
+ (-0.032767657823290344) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.032767657823290344) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.032767657823290344) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.032767657823290344) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273017) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.027115036845273017) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273017) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.027115036845273017) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.0259961775980212) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.0259961775980212) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.017561202409646093) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.017561202409646093) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.017561202409646093) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.017561202409646093) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.014564531231172973) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564531231172973) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564531231172973) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564531231172973) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.012215040997613912) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.012215040997613912) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.012215040997613912) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.012215040997613912) [X4 Z5 X6 X11 Z12 X13]
+ (-0.012215040997613912) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.012215040997613912) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.012215040997613912) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.012215040997613912) [X5 Z6 X7 X10 Z11 X12]
+ (-0.011756013419819234) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.011756013419819234) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.011756013419819234) [X3 Z4 Z5 X6 X8 X9]
+ (-0.011756013419819234) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.00876482757568879) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.00876482757568879) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.00876482757568879) [X2 Z3 Z4 X5 X11 X12]
+ (-0.00876482757568879) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.00876482757568879) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.00876482757568879) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.00876482757568879) [X3 X4 X10 Z11 Z12 X13]
+ (-0.00876482757568879) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.008125251921381048) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125251921381048) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.007306759928832958) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306759928832958) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306759928832958) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306759928832958) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.00580518898982686) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.00580518898982686) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.00580518898982686) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.00580518898982686) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.005652620978017331) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.005652620978017331) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.005652620978017331) [X0 X1 X3 Z4 Z5 X6]
+ (-0.005652620978017331) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.005143391768825098) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.005143391768825098) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.005143391768825098) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.005143391768825098) [X2 Z3 X4 X5 Z6 X7]
+ (-0.004684903388155231) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.004684903388155231) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.0046686203187763) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.0046686203187763) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.004575007626639206) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.004575007626639206) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424855449441861) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424855449441861) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.004158797381840053) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.004158797381840053) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.004158797381840053) [X3 Z4 Z5 X6 X12 X13]
+ (-0.004158797381840053) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.003493790359890111) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.003493790359890111) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.003493790359890111) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.003493790359890111) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.002779026799025575) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.002779026799025575) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.0022939566113524632) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.0022939566113524632) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.001799219493663004) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.001799219493663004) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278753941369638) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.0017278753941369638) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298507967730378) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298507967730378) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298507967730378) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298507967730378) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533856254125488) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533856254125488) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008145313270956096) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008145313270956096) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008145313270956096) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008145313270956096) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.735036880590922e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735036880590922e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735036880590922e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735036880590922e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.774817864992034e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.774817864992034e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.774817864992034e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.774817864992034e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.5183622160180715e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.5183622160180715e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.5183622160180715e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.5183622160180715e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.444344676305111e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.444344676305111e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.444344676305111e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.444344676305111e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.5243738488478806e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.5243738488478806e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.5243738488478806e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.5243738488478806e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.290028433638855e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.290028433638855e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.290028433638855e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.290028433638855e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.9743117136304315e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.9743117136304315e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.275883122304563e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.275883122304563e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.6430510687212846e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.6430510687212846e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.5565692184402975e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.5565692184402975e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.253224225702135e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.253224225702135e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.769659452323561e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.769659452323561e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.6945132947365456e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.6945132947365456e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.610297130904696e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.610297130904696e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.610297130904696e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.610297130904696e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.3131455001666867e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.3131455001666867e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.2774831958675723e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.2774831958675723e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.2774831958675723e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.2774831958675723e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.2112283486811934e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.2112283486811934e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.2112283486811934e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.2112283486811934e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.1513463113331298e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.1513463113331298e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.0882507115012374e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.0882507115012374e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.172669101504303e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.172669101504303e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.454842448913947e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.454842448913947e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.330473188686923e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.330473188686923e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2283337823792163e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2283337823792163e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.035847760027688e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.035847760027688e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956895371464832e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956895371464832e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733197742995295e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733197742995295e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733197742995295e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733197742995295e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.628614201775864e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.628614201775864e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.556281914836179e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.556281914836179e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.556281914836179e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.556281914836179e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.418291574885833e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.418291574885833e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.418291574885833e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.418291574885833e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.927453083382652e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927453083382652e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927453083382652e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927453083382652e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627851911540907e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627851911540907e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660624941918e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660624941918e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660624941918e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660624941918e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660624941918e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660624941918e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660624941918e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660624941918e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.838052750450973e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.838052750450973e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.570761328571531e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.570761328571531e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.328139350371237e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.328139350371237e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.086826564985899e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.086826564985899e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.086826564985899e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.086826564985899e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.447323128543655e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.447323128543655e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.371328947592172e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.371328947592172e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.371328947592172e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.371328947592172e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.199051618542507e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.199051618542507e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.9332412769016697e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.9332412769016697e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.9332412769016697e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.9332412769016697e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.8394209152537656e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.8394209152537656e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.8394209152537656e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.8394209152537656e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.551053917567891e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.551053917567891e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.551053917567891e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.551053917567891e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.3807781479775886e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.3807781479775886e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.3807781479775886e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.3807781479775886e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.3807781479775886e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.3807781479775886e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.3807781479775886e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.3807781479775886e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.3807781479775886e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.3807781479775886e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.3807781479775886e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.3807781479775886e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.2919694861335722e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.2919694861335722e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.1076325599518384e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.1076325599518384e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.1076325599518384e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.1076325599518384e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.1076325599518384e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.1076325599518384e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.1076325599518384e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.1076325599518384e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.057446596126435e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057446596126435e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057446596126435e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057446596126435e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649310133826711e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649310133826711e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649310133826711e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649310133826711e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.8394209152537659e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.8394209152537659e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.8394209152537659e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.8394209152537659e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.199051618542507e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.199051618542507e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.447323128543655e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.447323128543655e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.2362599609748426e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.2362599609748426e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.2362599609748426e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.2362599609748426e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.328139350371237e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.328139350371237e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.570761328571531e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.570761328571531e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.838052750450973e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.838052750450973e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.627851911540907e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627851911540907e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.628614201775864e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.628614201775864e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956895371464832e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956895371464832e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.30653665209193e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.30653665209193e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.30653665209193e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.30653665209193e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.035847760027688e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.035847760027688e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2283337823792163e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2283337823792163e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.2393363217077827e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393363217077827e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393363217077827e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393363217077827e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.330473188686923e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.330473188686923e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.454842448913947e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.454842448913947e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.172669101504303e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.172669101504303e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.0882507115012374e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.0882507115012374e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.1174479456744726e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.1174479456744726e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.1513463113331298e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.1513463113331298e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.3131455001666867e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.3131455001666867e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.334331289563947e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.334331289563947e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.6945132947365456e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.6945132947365456e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.18393255948081e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.18393255948081e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.5565692184402975e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.5565692184402975e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.6430510687212846e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.6430510687212846e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.275883122304563e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.275883122304563e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.9743117136304315e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.9743117136304315e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.00029219862611107046) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.00029219862611107046) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.00029219862611107046) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.00029219862611107046) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004956762314917193) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004956762314917193) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.000665007021949943) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.000665007021949943) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.000665007021949943) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.000665007021949943) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533856254125488) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533856254125488) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.0016095313817213637) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.0016095313817213637) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.0016095313817213637) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.0016095313817213637) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0016676041811440547) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.0016676041811440547) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.0016676041811440547) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.0016676041811440547) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.0017278753941369638) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.0017278753941369638) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.001799219493663004) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.001799219493663004) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.0022939566113524632) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.0022939566113524632) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.0024629170071339126) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.0024629170071339126) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.0024629170071339126) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.0024629170071339126) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.0039615607924965174) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.0039615607924965174) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.0039615607924965174) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.0039615607924965174) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.004424855449441861) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424855449441861) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004575007626639206) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.004575007626639206) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.0046686203187763) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.0046686203187763) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.004684903388155231) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.004684903388155231) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.00532483523422169) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.00532483523422169) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.00532483523422169) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.00532483523422169) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.005368659358109512) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368659358109512) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368659358109512) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368659358109512) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.007960880725921535) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.007960880725921535) [X4 Z5 X6 X10 Z11 X12]
+ (0.007960880725921535) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.007960880725921535) [X5 Z6 X7 X11 Z12 X13]
+ (0.008125251921381048) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125251921381048) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.00889073152269457) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.00889073152269457) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.00889073152269457) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.00889073152269457) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.010263414868158521) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.010263414868158521) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.010263414868158521) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.010263414868158521) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.010540425907671487) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.010540425907671487) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.010540425907671487) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.010540425907671487) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010960074940542543) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010960074940542543) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010960074940542543) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010960074940542543) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307274008848124) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307274008848124) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.014411099430130987) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411099430130987) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411099430130987) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411099430130987) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.015225630757226596) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.015225630757226596) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.015225630757226596) [X3 Z4 Z5 X6 X10 X11]
+ (0.015225630757226596) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.015588250102380215) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.015588250102380215) [X2 Z3 X4 X10 Z11 X12]
+ (0.015588250102380215) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.015588250102380215) [X3 Z4 X5 X11 Z12 X13]
+ (0.018266834869375505) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.018266834869375505) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.018266834869375505) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.018266834869375505) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.01902042317303989) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.01902042317303989) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.01902042317303989) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.01902042317303989) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.02017592172353544) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.02017592172353544) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.02017592172353544) [X4 Z5 Z6 X7 X11 X12]
+ (0.02017592172353544) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.02017592172353544) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.02017592172353544) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.02017592172353544) [X5 X6 X10 Z11 Z12 X13]
+ (0.02017592172353544) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.024353077678068998) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.024353077678068998) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.024353077678068998) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.024353077678068998) [X2 Z3 X4 X11 Z12 X13]
+ (0.024353077678068998) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.024353077678068998) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.024353077678068998) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.024353077678068998) [X3 Z4 X5 X10 Z11 X12]
+ (0.0243890825311494) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.0243890825311494) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.0243890825311494) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.0243890825311494) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.02510495713884446) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.02510495713884446) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.02510495713884446) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.02510495713884446) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.03078750538914391) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.03078750538914391) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.04587947078129789) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.04587947078129789) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.05600733087780748) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.05600733087780748) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.05600733087780748) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.05600733087780748) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661339) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.05608468124661339) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661339) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.05608468124661339) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.63127792878291e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.63127792878291e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.631277928782906e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631277928782906e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.595086007246201e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.595086007246201e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.5950860072461998e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.5950860072461998e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.042743277013782804) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.042743277013782804) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (0.042743277013782804) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.042743277013782804) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.04764261217638305) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.04764261217638305) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.04764261217638305) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.04764261217638305) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.041718813839821706) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.041718813839821706) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.041718813839821706) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.041718813839821706) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.039564416322893405) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.039564416322893405) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.039564416322893405) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.039564416322893405) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.039359168022053075) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.039359168022053075) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.039359168022053075) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.039359168022053075) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.03931805194719757) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03931805194719757) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.03931805194719757) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03931805194719757) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03560837898831252) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.03560837898831252) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.029903789512624797) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.029903789512624797) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.029903789512624797) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.029903789512624797) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.028730779551905478) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.028730779551905478) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.028730779551905478) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.028730779551905478) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.02563723829602675) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.02563723829602675) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.02563723829602675) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.02563723829602675) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.024755463292890946) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.024755463292890946) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.024755463292890946) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.024755463292890946) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.024282117354692843) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.024282117354692843) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.02314513092952899) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.02314513092952899) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.022528440196012918) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.022528440196012918) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.021433810721600964) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.021433810721600964) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.021433810721600964) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.021433810721600964) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.01925750509525156) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.01925750509525156) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.01902824244384731) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.01902824244384731) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.018889030304942843) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.018889030304942843) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.018889030304942843) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.018889030304942843) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.01602460368917958) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.01602460368917958) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.015225630757226598) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225630757226598) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.014603704729162129) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.014603704729162129) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564531231172973) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564531231172973) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.011756013419819234) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.011756013419819234) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.01128519020084087) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.01128519020084087) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.009841749246962635) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009841749246962635) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.009612634606847172) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612634606847172) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612634606847172) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612634606847172) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.008469978791023833) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.008469978791023833) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.007306759928832958) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.007306759928832958) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005923798336561342) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.005923798336561342) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.005652620978017331) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.005652620978017331) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368659358109513) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368659358109513) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.004158797381840053) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.004158797381840053) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.003356670563832891) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.003356670563832891) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.003356670563832891) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.003356670563832891) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.003267513854423554) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.003267513854423554) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.003267513854423554) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.003267513854423554) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.0027790267990255753) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.0027790267990255753) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.0026860409778066124) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.0026860409778066124) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.0026860409778066124) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.0026860409778066124) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.0022939566113524637) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524637) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.0022939566113524637) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524637) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.0009581655836696489) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.0009581655836696489) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0009581655836696489) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.0009581655836696489) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.0009581655836696489) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.0009581655836696489) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0009581655836696489) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.0009581655836696489) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.0002463643756958392) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0002463643756958392) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303549024) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.00013840177303549024) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.00013840177303549024) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.00013840177303549024) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.735036880590923e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735036880590923e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.610358530646844e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.610358530646844e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.610358530646844e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.610358530646844e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808796042788e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.5316808796042788e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808796042788e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5316808796042788e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.80610277560026e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.80610277560026e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.80610277560026e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.80610277560026e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.0897994677925714e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.0897994677925714e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.0897994677925714e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.0897994677925714e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.652209670144717e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.652209670144717e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.652209670144717e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.652209670144717e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851834613072e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.481851834613072e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851834613072e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.481851834613072e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.07148073664138e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.07148073664138e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.07148073664138e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.07148073664138e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.7346220389588795e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.7346220389588795e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.7346220389588795e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.7346220389588795e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.7288431474269845e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.7288431474269845e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.7288431474269845e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.7288431474269845e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.253224225702135e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.253224225702135e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.769659452323561e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.769659452323561e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.544395429426347e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.544395429426347e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.544395429426347e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.544395429426347e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.544395429426347e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.544395429426347e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.544395429426347e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.544395429426347e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.360956320365587e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.360956320365587e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.360956320365587e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.360956320365587e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.103215604897516e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.103215604897516e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.103215604897516e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.103215604897516e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220985487584e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.0111220985487584e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220985487584e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.0111220985487584e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.942946836744034e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.942946836744034e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.942946836744034e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.942946836744034e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174773417407e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.6541174773417407e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174773417407e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.6541174773417407e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.5224930677143175e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224930677143175e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224930677143175e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224930677143175e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224930677143175e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930677143175e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930677143175e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224930677143175e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.2283337823792163e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337823792163e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.2283337823792163e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337823792163e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.988770287866558e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988770287866558e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988770287866558e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988770287866558e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.867765104256556e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104256556e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104256556e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.867765104256556e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.189990975731963e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.189990975731963e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246207177646e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246207177646e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.471647744876414e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.471647744876414e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.5614471792510923e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.5614471792510923e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.5614471792510923e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.5614471792510923e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.523389678301106e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.523389678301106e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.4273231086154645e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.4273231086154645e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.4273231086154645e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.4273231086154645e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.328139350371237e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350371237e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.328139350371237e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350371237e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.086826564985899e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.086826564985899e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.888293594022934e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.888293594022934e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.888293594022934e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.888293594022934e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.3713289475921725e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.3713289475921725e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.839420915253766e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.839420915253766e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.057446596126435e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057446596126435e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.5371780961982426e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.5371780961982426e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.5371780961982426e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.5371780961982426e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.057446596126435e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057446596126435e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.209350634875778e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350634875778e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350634875778e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.209350634875778e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.703578355316448e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.703578355316448e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.703578355316448e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.703578355316448e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.839420915253766e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.839420915253766e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.3713289475921725e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.3713289475921725e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.086826564985899e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.086826564985899e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.523389678301106e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.523389678301106e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.471647744876414e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.471647744876414e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246207177646e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246207177646e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189990975731963e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.189990975731963e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.330473188686923e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.330473188686923e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.330473188686923e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.330473188686923e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288532436188117e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288532436188117e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288532436188117e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288532436188117e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.689348951568172e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.689348951568172e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.689348951568172e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.689348951568172e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.745518400639691e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.745518400639691e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.745518400639691e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.745518400639691e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.745518400639691e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.745518400639691e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.745518400639691e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.745518400639691e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.2118420192824897e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420192824897e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.2118420192824897e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420192824897e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.2118420192824897e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420192824897e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.2118420192824897e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420192824897e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.3131455001666867e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.3131455001666867e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.3131455001666867e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.3131455001666867e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.3343312895639464e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.3343312895639464e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.18393255948081e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.18393255948081e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.735036880590923e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735036880590923e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.0002463643756958392) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.0002463643756958392) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00044585351288409095) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.00044585351288409095) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.00044585351288409095) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.00044585351288409095) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940221543005184) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005184) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005184) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005184) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005184) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005184) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940221543005184) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005184) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533856254125489) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533856254125489) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533856254125489) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533856254125489) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0010435246534907408) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435246534907408) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435246534907408) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435246534907408) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.001280306097349649) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.001280306097349649) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.001280306097349649) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.001280306097349649) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.0013038004788127038) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.0013038004788127038) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.0013038004788127038) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.0013038004788127038) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.0022619660624823524) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.0022619660624823524) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.0022619660624823524) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.0022619660624823524) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.0022619660624823524) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.0022619660624823524) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.0022619660624823524) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.0022619660624823524) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.003989841456619317) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.003989841456619317) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.003989841456619317) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.003989841456619317) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.004158797381840053) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.004158797381840053) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.004311038507914295) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.004311038507914295) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.004311038507914295) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.004311038507914295) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.00463697666118254) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.00463697666118254) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.00463697666118254) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.00463697666118254) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.00511447383166039) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.00511447383166039) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.00511447383166039) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.00511447383166039) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.00511447383166039) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.00511447383166039) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.00511447383166039) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.00511447383166039) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005241535382803865) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.005241535382803865) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.005241535382803865) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.005241535382803865) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.005262642473076818) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262642473076818) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262642473076818) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262642473076818) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368659358109513) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368659358109513) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005379937155839355) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.005379937155839355) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.005379937155839355) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.005379937155839355) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.005652620978017331) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.005652620978017331) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.005708495985960909) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708495985960909) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708495985960909) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708495985960909) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.005923798336561342) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.005923798336561342) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.007306759928832958) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.007306759928832958) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469978791023833) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.008469978791023833) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.009841749246962635) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.009841749246962635) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.01128519020084087) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.01128519020084087) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.011756013419819234) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.011756013419819234) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.014564531231172973) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564531231172973) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603704729162129) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.014603704729162129) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.015225630757226598) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225630757226598) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.01602460368917958) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.01602460368917958) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.01902824244384731) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.01902824244384731) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.01925750509525156) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.01925750509525156) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.04587947078129789) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04587947078129789) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.3693708936615612) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.3693708936615612) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.36937089366156095) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.36937089366156095) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.28164257767022943) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.28164257767022943) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.2816425776702293) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.2816425776702293) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.09065144207036457) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.09065144207036457) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.09065144207036457) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.09065144207036457) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863607) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684737589863607) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863607) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684737589863607) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635021950635007) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635021950635007) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635021950635007) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635021950635007) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.03560837898831252) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.03560837898831252) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0349033433736617) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0349033433736617) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0349033433736617) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0349033433736617) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883829933) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.024591860883829933) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883829933) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.024591860883829933) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.024282117354692843) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.024282117354692843) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.023145130929528992) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.023145130929528992) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.022528440196012918) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.022528440196012918) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0195380503113147) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.0195380503113147) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.0195380503113147) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.0195380503113147) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.01709155315589882) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.01709155315589882) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.01709155315589882) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.01709155315589882) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.01602460368917958) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.01602460368917958) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.01602460368917958) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.01602460368917958) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.010311482489831766) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831766) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831766) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311482489831766) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.009841749246962636) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962636) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.009841749246962636) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962636) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209862) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209862) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209862) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008826368514209862) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008541996625454825) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454825) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454825) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454825) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454825) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454825) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454825) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008541996625454825) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008469978791023833) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023833) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.008469978791023833) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023833) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.0046686203187763) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0046686203187763) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.003876470899336959) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.003876470899336959) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.00380406617172854) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00380406617172854) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00380406617172854) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.00380406617172854) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178865) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178865) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003356670563832891) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.003356670563832891) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.003267513854423554) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.003267513854423554) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0021413612231016054) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.0021413612231016054) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.0017278753941369638) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.0017278753941369638) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.0016407548553123942) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0016407548553123942) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214169146) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0014528843214169146) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214169146) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0014528843214169146) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870896771024472) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007870896771024472) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192743499487813) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192743499487813) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.00019400857029757196) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00019400857029757196) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303549024) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.00013840177303549024) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.141625221158585e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.141625221158585e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.141625221158585e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.141625221158585e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.07148073664138e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.07148073664138e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.1513463113331298e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.1513463113331298e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0882507115012374e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.0882507115012374e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.9885117061776683e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.9885117061776683e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8742990714620754e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.8742990714620754e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.3609563203655865e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.3609563203655865e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.3002946562867098e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3002946562867098e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.1468376508201053e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.1468376508201053e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.1468376508201053e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.1468376508201053e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.352332103720533e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.352332103720533e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.352332103720533e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.352332103720533e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091637199608668e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637199608668e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637199608668e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637199608668e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637199608668e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637199608668e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637199608668e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091637199608668e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.074305986449416e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.074305986449416e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.074305986449416e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.074305986449416e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128986893491e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128986893491e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128986893491e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128986893491e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867765104256556e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.867765104256556e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560692465426513e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692465426513e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692465426513e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692465426513e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692465426513e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692465426513e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692465426513e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560692465426513e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.997018422399085e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.997018422399085e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.997018422399085e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.997018422399085e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.997018422399085e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.997018422399085e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.997018422399085e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.997018422399085e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.5682475213075634e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.5682475213075634e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.5682475213075634e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.5682475213075634e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393085923826e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393085923826e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393085923826e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393085923826e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393085923826e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393085923826e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.3767393085923826e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393085923826e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.888293594022934e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.888293594022934e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.686381547484513e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.686381547484513e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.703578355316448e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.703578355316448e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.209350634875778e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.209350634875778e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773245057362e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773245057362e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773245057362e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773245057362e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773245057362e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773245057362e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773245057362e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773245057362e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253798554218e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253798554218e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9742253798554218e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9742253798554218e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0474716556302092e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.0474716556302092e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.0474716556302092e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.0474716556302092e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.209350634875778e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.209350634875778e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.071728218557012e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.071728218557012e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.071728218557012e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.071728218557012e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.200428749398477e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.200428749398477e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.200428749398477e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.200428749398477e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.703578355316448e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.703578355316448e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.31209430528931e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.31209430528931e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.31209430528931e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.31209430528931e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.686381547484513e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.686381547484513e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.888293594022934e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.888293594022934e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.0922506163152357e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506163152357e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.0922506163152357e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506163152357e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.0922506163152357e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506163152357e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.0922506163152357e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506163152357e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.4445978540341327e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.4445978540341327e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.4445978540341327e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.4445978540341327e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.6849150952669065e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.6849150952669065e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.6849150952669065e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.6849150952669065e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246974425734657e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246974425734657e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246974425734657e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246974425734657e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246974425734657e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246974425734657e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246974425734657e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246974425734657e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.867765104256556e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.867765104256556e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3002946562867098e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3002946562867098e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.3609563203655865e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.3609563203655865e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.8742990714620754e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.8742990714620754e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.8836765760439756e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.8836765760439756e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.947356011858643e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.947356011858643e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.947356011858643e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.947356011858643e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117061776683e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9885117061776683e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.0882507115012374e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.0882507115012374e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.1513463113331298e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.1513463113331298e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.84620167135811e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.84620167135811e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.84620167135811e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.84620167135811e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.07148073664138e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.07148073664138e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.105526722061267e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.105526722061267e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.105526722061267e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.105526722061267e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.1464963276448205e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.1464963276448205e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.1464963276448205e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.1464963276448205e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159350501903633e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159350501903633e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159350501903633e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159350501903633e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427988656652084e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427988656652084e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427988656652084e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427988656652084e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935867718036311e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935867718036311e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935867718036311e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935867718036311e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348206342e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.253273348206342e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979825793523343e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979825793523343e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979825793523343e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979825793523343e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.2055484112176605e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.2055484112176605e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.2055484112176605e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.2055484112176605e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00013840177303549024) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.00013840177303549024) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.00018787053389547937) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.00018787053389547937) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.00018787053389547937) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00018787053389547937) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00019400857029757196) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00019400857029757196) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0002463643756958392) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0002463643756958392) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0002463643756958392) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0002463643756958392) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192743499487813) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192743499487813) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156734248908614) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156734248908614) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156734248908614) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156734248908614) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024472) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007870896771024472) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.0015324835230730125) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.0015324835230730125) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.0015324835230730125) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.0015324835230730125) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.0016407548553123942) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0016407548553123942) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0017278753941369638) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.0017278753941369638) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.0024464971554158756) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.0024464971554158756) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.0024464971554158756) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.0024464971554158756) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.003267513854423554) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.003267513854423554) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.003356670563832891) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.003356670563832891) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.0034841573002178865) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0034841573002178865) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.003876470899336959) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.003876470899336959) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.0046686203187763) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.0046686203187763) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.0047672721882780905) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.0047672721882780905) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.0047672721882780905) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.0047672721882780905) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.005286546538226872) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.005286546538226872) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.005286546538226872) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.005286546538226872) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.005408954422409972) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.005408954422409972) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.005408954422409972) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.005408954422409972) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.005923798336561342) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561342) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.005923798336561342) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561342) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.010715508469796735) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010715508469796735) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010715508469796735) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010715508469796735) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.010757563953908911) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010757563953908911) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010757563953908911) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010757563953908911) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.014603704729162129) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.014603704729162129) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.014603704729162129) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.014603704729162129) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.019299560579363738) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363738) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363738) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363738) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363738) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363738) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.019299560579363738) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363738) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.05859198873386194) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.05859198873386194) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.7759505272839906e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.7759505272839906e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.775950527283991e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.775950527283991e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.07165035181002738) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.07165035181002738) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0716503518100274) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0716503518100274) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.01925750509525156) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.01925750509525156) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010311482489831766) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311482489831766) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008826368514209862) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209862) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.007597464029770589) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.007597464029770589) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.007597464029770589) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.007597464029770589) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311883) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311883) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311883) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311883) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311883) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311883) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311883) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311883) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676592) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005348051582676592) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005348051582676592) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676592) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00380406617172854) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00380406617172854) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219442) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.0029841661681219442) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.0029841661681219442) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.0029841661681219442) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.0024464971554158756) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.0024464971554158756) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.0022494124470939965) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0022494124470939965) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022494124470939965) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0022494124470939965) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0021413612231016054) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.0021413612231016054) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.0018638942824587056) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587056) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587056) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587056) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587056) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587056) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0018638942824587056) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587056) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0016407548553123942) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553123942) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0016407548553123942) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553123942) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0012223378081538327) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538327) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538327) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538327) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538327) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538327) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538327) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0012223378081538327) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0010283292378562606) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0010283292378562606) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0010283292378562606) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0010283292378562606) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1463061453388531e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1463061453388531e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.8742990714620754e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990714620754e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.8742990714620754e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990714620754e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3002946562867098e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3002946562867098e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3002946562867098e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3002946562867098e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.0444941298025236e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.0444941298025236e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.0444941298025236e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.0444941298025236e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.956079230016368e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.956079230016368e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.956079230016368e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.956079230016368e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105515037204046e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105515037204046e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105515037204046e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105515037204046e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.66134721315416e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.66134721315416e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.66134721315416e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.66134721315416e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341413937906e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341413937906e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189990975731963e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.189990975731963e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876621658190371e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876621658190371e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876621658190371e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876621658190371e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246207177646e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246207177646e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.523389678301106e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.523389678301106e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0767325315138247e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.0767325315138247e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0767325315138247e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.0767325315138247e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.013471458991327e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.013471458991327e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.9045998840873287e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.9045998840873287e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.9045998840873287e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.9045998840873287e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.6667317540212087e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.6667317540212087e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.6667317540212087e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6667317540212087e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.850564192812321e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.850564192812321e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309317689547e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6569309317689547e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309317689547e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6569309317689547e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.850564192812321e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.850564192812321e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.686381547484513e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.686381547484513e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.686381547484513e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.686381547484513e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458991327e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.013471458991327e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.523389678301106e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.523389678301106e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.6704023907602814e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.6704023907602814e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.6704023907602814e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.6704023907602814e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246207177646e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246207177646e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189990975731963e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.189990975731963e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341413937906e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341413937906e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949476488001356e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949476488001356e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.7924939577556751e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939577556751e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939577556751e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.7924939577556751e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.883676576043975e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.883676576043975e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.9885117061776683e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117061776683e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117061776683e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117061776683e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348206342e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253273348206342e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4017109735508586e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4017109735508586e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4017109735508586e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4017109735508586e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.580960369326426e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.580960369326426e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.580960369326426e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.580960369326426e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192743499487813) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487813) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192743499487813) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487813) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.0007870896771024472) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024472) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024472) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024472) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0011726348316441878) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0011726348316441878) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0011726348316441878) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0011726348316441878) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.001236647801924511) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.001236647801924511) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.001236647801924511) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.001236647801924511) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.002200964069500449) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002200964069500449) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002200964069500449) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002200964069500449) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798021) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798021) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798021) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798021) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798021) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798021) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002394972639798021) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798021) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0024464971554158756) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.0024464971554158756) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.00380406617172854) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00380406617172854) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0038764708993369594) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.0038764708993369594) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.0038764708993369594) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.0038764708993369594) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.004220813970046456) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.004220813970046456) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.004220813970046456) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.004220813970046456) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.008826368514209862) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.008826368514209862) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.010311482489831766) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311482489831766) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01925750509525156) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01925750509525156) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.05859198873386194) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.05859198873386194) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3987009015958443e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.3987009015958443e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.3987009015958443e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3987009015958443e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178865) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178865) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219442) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.0029841661681219442) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.00019400857029757196) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00019400857029757196) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453388531e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453388531e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7924939577556747e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7924939577556747e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540341413937906e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413937906e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540341413937906e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413937906e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.850564192812321e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.850564192812321e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.850564192812321e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.850564192812321e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458991327e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471458991327e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458991327e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471458991327e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949476488001356e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949476488001356e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.7924939577556747e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7924939577556747e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029757196) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029757196) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0029841661681219442) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.0029841661681219442) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.0034841573002178865) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0034841573002178865) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
  (-73.1387323135253) [I0]
+ (-0.18066792656583353) [Z7]
+ (-0.1806679265658335) [Z6]
+ (-0.15961432501809877) [Z5]
+ (-0.15961432501809875) [Z4]
+ (0.17419956155055763) [Z2]
+ (0.17419956155055763) [Z3]
+ (0.2275726900545344) [Z0]
+ (0.22757269005453448) [Z1]
+ (-8.194261372310019e-06) [Y4 Y6]
+ (-8.194261372310019e-06) [X4 X6]
+ (7.954413176205555e-06) [Y5 Y7]
+ (7.954413176205555e-06) [X5 X7]
+ (0.11270386920332215) [Z4 Z6]
+ (0.11270386920332215) [Z5 Z7]
+ (0.1195243896468266) [Z0 Z4]
+ (0.1195243896468266) [Z1 Z5]
+ (0.13401715261963695) [Z0 Z6]
+ (0.13401715261963695) [Z1 Z7]
+ (0.13734953064261307) [Z0 Z5]
+ (0.13734953064261307) [Z1 Z4]
+ (0.14138905291942805) [Z4 Z7]
+ (0.14138905291942805) [Z5 Z6]
+ (0.14722943218766169) [Z2 Z5]
+ (0.14722943218766169) [Z3 Z4]
+ (0.14926355147388895) [Z4 Z5]
+ (0.1497348680349693) [Z2 Z6]
+ (0.1497348680349693) [Z3 Z7]
+ (0.15138327161428836) [Z0 Z7]
+ (0.15138327161428836) [Z1 Z6]
+ (0.1543574865722363) [Z6 Z7]
+ (0.15582269051553116) [Z2 Z7]
+ (0.15582269051553116) [Z3 Z6]
+ (0.16756653265461258) [Z0 Z2]
+ (0.16756653265461258) [Z1 Z3]
+ (0.18143991440303867) [Z0 Z3]
+ (0.18143991440303867) [Z1 Z2]
+ (0.19392534613270174) [Z0 Z1]
+ (-7.037887510663516e-06) [Y4 Z5 Y6]
+ (-7.037887510663516e-06) [X4 Z5 X6]
+ (-7.0378875106635146e-06) [Y5 Z6 Y7]
+ (-7.0378875106635146e-06) [X5 Z6 X7]
+ (-0.028685183716105896) [Y4 Y5 X6 X7]
+ (-0.028685183716105896) [X4 X5 Y6 Y7]
+ (-0.017366118994651392) [Y0 Y1 X6 X7]
+ (-0.017366118994651392) [X0 X1 Y6 Y7]
+ (-0.013873381748426101) [Y0 Y1 X2 X3]
+ (-0.013873381748426101) [X0 X1 Y2 Y3]
+ (-0.009560705729135935) [Y2 Y3 X4 X5]
+ (-0.009560705729135935) [X2 X3 Y4 Y5]
+ (-0.0060878224805618626) [Y2 Y3 X6 X7]
+ (-0.0060878224805618626) [X2 X3 Y6 Y7]
+ (-0.00029219862611105696) [Y1 Y2 X3 X4]
+ (-0.00029219862611105696) [X1 X2 Y3 Y4]
+ (-8.194261372310019e-06) [Z4 Y5 Z6 Y7]
+ (-8.194261372310019e-06) [Z4 X5 Z6 X7]
+ (-2.8909678817571316e-06) [Z0 Y5 Z6 Y7]
+ (-2.8909678817571316e-06) [Z0 X5 Z6 X7]
+ (-2.8909678817571316e-06) [Z1 Y4 Z5 Y6]
+ (-2.8909678817571316e-06) [Z1 X4 Z5 X6]
+ (-1.8551201215436896e-06) [Z0 Y4 Z5 Y6]
+ (-1.8551201215436896e-06) [Z0 X4 Z5 X6]
+ (-1.8551201215436896e-06) [Z1 Y5 Z6 Y7]
+ (-1.8551201215436896e-06) [Z1 X5 Z6 X7]
+ (-1.5973171978333711e-06) [Z2 Y4 Z5 Y6]
+ (-1.5973171978333711e-06) [Z2 X4 Z5 X6]
+ (-1.5973171978333711e-06) [Z3 Y5 Z6 Y7]
+ (-1.5973171978333711e-06) [Z3 X5 Z6 X7]
+ (-1.0358477602134423e-06) [Y0 X1 X5 Y6]
+ (-1.0358477602134423e-06) [Y0 Y1 Y5 Y6]
+ (-1.0358477602134423e-06) [X0 X1 X5 X6]
+ (-1.0358477602134423e-06) [X0 Y1 Y5 X6]
+ (-9.344557776645561e-07) [Z2 Y5 Z6 Y7]
+ (-9.344557776645561e-07) [Z2 X5 Z6 X7]
+ (-9.344557776645561e-07) [Z3 Y4 Z5 Y6]
+ (-9.344557776645561e-07) [Z3 X4 Z5 X6]
+ (6.628614201688151e-07) [Y2 X3 X5 Y6]
+ (6.628614201688151e-07) [Y2 Y3 Y5 Y6]
+ (6.628614201688151e-07) [X2 X3 X5 X6]
+ (6.628614201688151e-07) [X2 Y3 Y5 X6]
+ (7.954413176205555e-06) [Y4 Z5 Y6 Z7]
+ (7.954413176205555e-06) [X4 Z5 X6 Z7]
+ (0.00029219862611105696) [Y1 X2 X3 Y4]
+ (0.00029219862611105696) [X1 Y2 Y3 X4]
+ (0.0060878224805618626) [Y2 X3 X6 Y7]
+ (0.0060878224805618626) [X2 Y3 Y6 X7]
+ (0.009560705729135935) [Y2 X3 X4 Y5]
+ (0.009560705729135935) [X2 Y3 Y4 X5]
+ (0.011307274008848197) [Y1 Z2 Z3 Y5]
+ (0.011307274008848197) [X1 Z2 Z3 X5]
+ (0.013873381748426101) [Y0 X1 X2 Y3]
+ (0.013873381748426101) [X0 Y1 Y2 X3]
+ (0.017366118994651392) [Y0 X1 X6 Y7]
+ (0.017366118994651392) [X0 Y1 Y6 X7]
+ (0.028685183716105896) [Y4 X5 X6 Y7]
+ (0.028685183716105896) [X4 Y5 Y6 X7]
+ (0.02981242451734579) [Y0 Z1 Z2 Y4]
+ (0.02981242451734579) [X0 Z1 Z2 X4]
+ (0.02981242451734579) [Y1 Z3 Z4 Y5]
+ (0.02981242451734579) [X1 Z3 Z4 X5]
+ (0.030104623143456848) [Y0 Z1 Z3 Y4]
+ (0.030104623143456848) [X0 Z1 Z3 X4]
+ (0.030104623143456848) [Y1 Z2 Z4 Y5]
+ (0.030104623143456848) [X1 Z2 Z4 X5]
+ (0.030787505389143918) [Y0 Z2 Z3 Y4]
+ (0.030787505389143918) [X0 Z2 Z3 X4]
+ (0.04375263801065985) [Y1 Z2 Z3 Z4 Y5]
+ (0.04375263801065985) [X1 Z2 Z3 Z4 X5]
+ (0.04375263801065986) [Y0 Z1 Z2 Z3 Y4]
+ (0.04375263801065986) [X0 Z1 Z2 Z3 X4]
+ (-0.014564531231172986) [Y1 Z2 Z3 X4 X6 Y7]
+ (-0.014564531231172986) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-0.014564531231172986) [X1 Z2 Z3 X4 X6 X7]
+ (-0.014564531231172986) [X1 Z2 Z3 Y4 Y6 X7]
+ (-6.524373848534984e-06) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (-6.524373848534984e-06) [X0 Z1 Z2 Z3 Z4 X6]
+ (-6.524373848534984e-06) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (-6.524373848534984e-06) [X1 Z2 Z3 Z5 Z6 X7]
+ (-3.769659451858428e-06) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (-3.769659451858428e-06) [X0 Z2 Z3 Z4 Z5 X6]
+ (-3.610297130457601e-06) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (-3.610297130457601e-06) [X0 Z1 Z3 Z4 Z5 X6]
+ (-3.610297130457601e-06) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (-3.610297130457601e-06) [X1 Z2 Z4 Z5 Z6 X7]
+ (-3.313145500200347e-06) [Y1 Z2 Z3 Y4 X5 X6]
+ (-3.313145500200347e-06) [X1 Z2 Z3 X4 Y5 Y6]
+ (-3.277483195383544e-06) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (-3.277483195383544e-06) [X0 Z1 Z2 Z4 Z5 X6]
+ (-3.277483195383544e-06) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (-3.277483195383544e-06) [X1 Z3 Z4 Z5 Z6 X7]
+ (-3.211228348334637e-06) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (-3.211228348334637e-06) [X0 Z1 Z2 Z3 Z5 X6]
+ (-3.211228348334637e-06) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (-3.211228348334637e-06) [X1 Z2 Z3 Z4 Z6 X7]
+ (-1.0358477602134423e-06) [Y0 Y1 X4 Z5 Z6 X7]
+ (-1.0358477602134423e-06) [X0 X1 Y4 Z5 Z6 Y7]
+ (-6.628614201688151e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (-6.628614201688151e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (-3.328139350740571e-07) [Y1 X2 X3 Z4 Z5 Y6]
+ (-3.328139350740571e-07) [X1 Y2 Y3 Z4 Z5 X6]
+ (3.328139350740571e-07) [Y1 Y2 X3 Z4 Z5 X6]
+ (3.328139350740571e-07) [X1 X2 Y3 Z4 Z5 Y6]
+ (6.628614201688151e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (6.628614201688151e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (1.0358477602134423e-06) [Y0 X1 X4 Z5 Z6 Y7]
+ (1.0358477602134423e-06) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.313145500200347e-06) [Y1 Z2 Z3 X4 X5 Y6]
+ (3.313145500200347e-06) [X1 Z2 Z3 Y4 Y5 X6]
+ (4.183932559477494e-06) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (4.183932559477494e-06) [X1 Z2 Z3 Z4 Z5 X7]
+ (0.00029219862611105696) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (0.00029219862611105696) [Y0 Z1 Y2 X3 Z4 X5]
+ (0.00029219862611105696) [X0 Z1 X2 Y3 Z4 Y5]
+ (0.00029219862611105696) [X0 Z1 X2 X3 Z4 X5]
+ (0.010540425907671548) [Y0 Z1 Z2 Z3 Y4 Z7]
+ (0.010540425907671548) [X0 Z1 Z2 Z3 X4 Z7]
+ (0.010540425907671548) [Y1 Z2 Z3 Z4 Y5 Z6]
+ (0.010540425907671548) [X1 Z2 Z3 Z4 X5 Z6]
+ (0.011307274008848196) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (0.011307274008848196) [X0 Z1 Z2 Z3 X4 Z5]
+ (0.025104957138844534) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (0.025104957138844534) [X0 Z1 Z2 Z3 X4 Z6]
+ (0.025104957138844534) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (0.025104957138844534) [X1 Z2 Z3 Z4 X5 Z7]
+ (0.030787505389143918) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (0.030787505389143918) [Z0 X1 Z2 Z3 Z4 X5]
+ (-5.105396549480814e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-5.105396549480814e-06) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-5.10539654948081e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (-5.10539654948081e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (-0.014564531231172989) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-0.014564531231172989) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-3.769659451858428e-06) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-3.769659451858428e-06) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-3.328139350740571e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-3.328139350740571e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-3.328139350740571e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-3.328139350740571e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (3.313145500200347e-06) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (3.313145500200347e-06) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (3.313145500200347e-06) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (3.313145500200347e-06) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (4.183932559477495e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (4.183932559477495e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (0.014564531231172989) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (0.014564531231172989) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
 </code>
 </pre>
 </details>

---

## 10. tutorial_adaptive_circuits.html <a name="demo9"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_adaptive_circuits.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Excitation : [0, 1, 2, 3], Gradient: -0.012782175157666994
Excitation : [0, 1, 2, 5], Gradient: 9.486769009248161e-20
Excitation : [0, 1, 2, 7], Gradient: -6.098637220230962e-20
Excitation : [0, 1, 2, 9], Gradient: 0.034264511701685985
Excitation : [0, 1, 3, 4], Gradient: -6.776263578034409e-20
Excitation : [0, 1, 3, 6], Gradient: 1.0164395367051604e-19
Excitation : [0, 1, 3, 8], Gradient: -0.034264511701686
Excitation : [0, 1, 4, 5], Gradient: -0.02358152902067784
Excitation : [0, 1, 5, 8], Gradient: 6.77626357803446e-20
Excitation : [0, 1, 6, 7], Gradient: -0.023581529020677863
Excitation : [0, 1, 7, 8], Gradient: -2.0328790734102582e-20
Excitation : [0, 1, 8, 9], Gradient: -0.12362273485598727
[[0, 1, 2, 3], [0, 1, 2, 9], [0, 1, 3, 8], [0, 1, 4, 5], [0, 1, 6, 7], [0, 1, 8, 9]]
Excitation : [0, 2], Gradient: -0.005062536239329972
Excitation : [0, 4], Gradient: -1.4772079341377727e-17
Excitation : [0, 6], Gradient: -3.3912033120964774e-18
Excitation : [0, 8], Gradient: -0.000944804462578046
Excitation : [1, 3], Gradient: 0.004926616876998486
Excitation : [1, 5], Gradient: -8.639649228184764e-20
Excitation : [1, 7], Gradient: 3.1769473282856554e-19
Excitation : [1, 9], Gradient: 0.001453553485404249
[[0, 2], [0, 8], [1, 3], [1, 9]]
n = 0,  E = -7.86266587 H, t = 3.11 s
n = 1,  E = -7.87094621 H, t = 3.17 s
n = 2,  E = -7.87563100 H, t = 2.59 s
n = 3,  E = -7.87829146 H, t = 3.19 s
n = 4,  E = -7.87981705 H, t = 2.63 s
n = 5,  E = -7.88070477 H, t = 3.17 s
n = 6,  E = -7.88123143 H, t = 2.61 s
n = 7,  E = -7.88155161 H, t = 3.13 s
n = 8,  E = -7.88175217 H, t = 2.60 s
n = 9,  E = -7.88188237 H, t = 3.15 s
n = 10,  E = -7.88197041 H, t = 2.63 s
n = 11,  E = -7.88203267 H, t = 3.16 s
n = 12,  E = -7.88207879 H, t = 3.22 s
n = 13,  E = -7.88211452 H, t = 2.65 s
n = 14,  E = -7.88214335 H, t = 3.21 s
n = 15,  E = -7.88216743 H, t = 2.66 s
n = 16,  E = -7.88218814 H, t = 3.19 s
n = 17,  E = -7.88220634 H, t = 2.63 s
n = 18,  E = -7.88222261 H, t = 3.17 s
n = 19,  E = -7.88223734 H, t = 2.65 s
n = 0,  E = -7.86266587 H, t = 0.14 s
n = 1,  E = -7.87094621 H, t = 0.14 s
n = 2,  E = -7.87563100 H, t = 0.14 s
n = 3,  E = -7.87829146 H, t = 0.14 s
n = 4,  E = -7.87981705 H, t = 0.14 s
n = 5,  E = -7.88070477 H, t = 0.14 s
n = 6,  E = -7.88123143 H, t = 0.14 s
n = 7,  E = -7.88155161 H, t = 0.14 s
n = 8,  E = -7.88175217 H, t = 0.15 s
n = 9,  E = -7.88188237 H, t = 0.14 s
n = 10,  E = -7.88197041 H, t = 0.14 s
n = 11,  E = -7.88203267 H, t = 0.14 s
n = 12,  E = -7.88207879 H, t = 0.15 s
n = 13,  E = -7.88211452 H, t = 0.15 s
n = 14,  E = -7.88214335 H, t = 0.15 s
n = 15,  E = -7.88216743 H, t = 0.15 s
n = 16,  E = -7.88218814 H, t = 0.15 s
n = 17,  E = -7.88220634 H, t = 0.15 s
n = 18,  E = -7.88222261 H, t = 0.15 s
n = 19,  E = -7.88223734 H, t = 0.15 s
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
Excitation : [0, 1, 2, 9], Gradient: 0.0342645117016873
Excitation : [0, 1, 3, 4], Gradient: 0.0
Excitation : [0, 1, 3, 6], Gradient: 0.0
Excitation : [0, 1, 3, 8], Gradient: -0.008566127925421914
Excitation : [0, 1, 4, 5], Gradient: 0.0
Excitation : [0, 1, 5, 8], Gradient: 0.0
Excitation : [0, 1, 6, 7], Gradient: 0.0
Excitation : [0, 1, 7, 8], Gradient: 0.0
Excitation : [0, 1, 8, 9], Gradient: 0.0
[[0, 1, 2, 9], [0, 1, 3, 8]]
Excitation : [0, 2], Gradient: -0.013361843799981067
Excitation : [0, 4], Gradient: 0.0
Excitation : [0, 6], Gradient: 0.0
Excitation : [0, 8], Gradient: 0.008127419311874549
Excitation : [1, 3], Gradient: 9.609881567180091e-06
Excitation : [1, 5], Gradient: -0.004875127086708457
Excitation : [1, 7], Gradient: -0.004875127086708458
Excitation : [1, 9], Gradient: -0.007509748822110066
[[0, 2], [0, 8], [1, 5], [1, 7], [1, 9]]
n = 0,  E = -7.85513767 H, t = 1.79 s
n = 1,  E = -7.85585993 H, t = 1.76 s
n = 2,  E = -7.85642249 H, t = 1.76 s
n = 3,  E = -7.85686535 H, t = 2.01 s
n = 4,  E = -7.85721832 H, t = 1.81 s
n = 5,  E = -7.85750361 H, t = 1.55 s
n = 6,  E = -7.85773773 H, t = 2.00 s
n = 7,  E = -7.85793296 H, t = 1.80 s
n = 8,  E = -7.85809846 H, t = 1.55 s
n = 9,  E = -7.85824102 H, t = 2.00 s
n = 10,  E = -7.85836572 H, t = 1.81 s
n = 11,  E = -7.85847636 H, t = 1.55 s
n = 12,  E = -7.85857579 H, t = 2.00 s
n = 13,  E = -7.85866614 H, t = 1.81 s
n = 14,  E = -7.85874902 H, t = 1.81 s
n = 15,  E = -7.85882566 H, t = 1.55 s
n = 16,  E = -7.85889701 H, t = 2.01 s
n = 17,  E = -7.85896378 H, t = 1.55 s
n = 18,  E = -7.85902654 H, t = 2.00 s
n = 19,  E = -7.85908573 H, t = 1.81 s
n = 0,  E = -7.86266587 H, t = 0.08 s
n = 1,  E = -7.86373056 H, t = 0.08 s
n = 2,  E = -7.86443636 H, t = 0.08 s
n = 3,  E = -7.86490587 H, t = 0.08 s
n = 4,  E = -7.86521992 H, t = 0.08 s
n = 5,  E = -7.86543166 H, t = 0.08 s
n = 6,  E = -7.86557597 H, t = 0.08 s
n = 7,  E = -7.86567575 H, t = 0.08 s
n = 8,  E = -7.86574604 H, t = 0.08 s
n = 9,  E = -7.86579669 H, t = 0.08 s
n = 10,  E = -7.86583418 H, t = 0.08 s
n = 11,  E = -7.86586277 H, t = 0.08 s
n = 12,  E = -7.86588528 H, t = 0.08 s
n = 13,  E = -7.86590357 H, t = 0.08 s
n = 14,  E = -7.86591886 H, t = 0.08 s
n = 15,  E = -7.86593199 H, t = 0.08 s
n = 16,  E = -7.86594350 H, t = 0.08 s
n = 17,  E = -7.86595377 H, t = 0.08 s
n = 18,  E = -7.86596307 H, t = 0.08 s
n = 19,  E = -7.86597156 H, t = 0.08 s
 </code>
 </pre>
 </details>

---

## 11. tutorial_error_mitigation.html <a name="demo10"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_error_mitigation.html):

```
2: ──RY(4.05)───RY(-4.05)──RY(4.05)──╭C──────────RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)──╰Z──RY(-3.32)──╭C──RY(-4.05)──┤
3: ──RY(3.51)────────────────────────╰Z──────────RY(3.66)───RY(-3.66)──────────────────────────────────────╰Z──RY(-3.51)──┤
0: ──RY(4.56)──╭C──RY(5.93)──RY(-5.93)────────────────────────────────────────────────────╭C──RY(-4.56)──┤
1: ──RY(3.6)───╰Z──RY(5.9)────────────────────────╭C──RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z──RY(-3.6)───┤
2: ──RY(4.05)──╭C──RY(3.32)──RY(-3.32)──RY(3.32)──╰Z──RY(1.07)──RY(-1.07)──╰Z──RY(-3.32)──╭C──RY(-4.05)──┤
3: ──RY(3.51)──╰Z──RY(3.66)──RY(-3.66)────────────────────────────────────────────────────╰Z──RY(-3.51)──┤
0.9679379497825091
0: ──RY(4.56)──╭C──RY(5.93)───RY(-5.93)──────────────────────────────────────╭C──RY(-4.56)─────────────────────┤
1: ──RY(3.6)───╰Z──RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z──RY(-3.6)───RY(3.6)──RY(-3.6)──┤
2: ──RY(4.05)──╭C──RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)──╰Z──RY(-3.32)──╭C──RY(-4.05)─────────────────────┤
3: ──RY(3.51)──╰Z──RY(3.66)───RY(-3.66)──────────────────────────────────────╰Z──RY(-3.51)─────────────────────┤
0: ──RY(4.56)──╭C──────────RY(5.93)───RY(-5.93)────────────────────────────────────────────────────────────╭C──RY(-4.56)──┤
1: ──RY(3.6)───╰Z──────────RY(5.9)─────────────────────────╭C──────────RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z──RY(-3.6)───┤
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_error_mitigation.html):

```
2: ──RY(4.05)──╭C──RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)──╰Z──RY(-3.32)───RY(3.32)──RY(-3.32)──╭C──RY(-4.05)──┤
3: ──RY(3.51)──╰Z──RY(3.66)───RY(-3.66)────────────────────────────────────────────────────────────╰Z──RY(-3.51)──┤
0: ──RY(4.56)──╭C──RY(5.93)───RY(-5.93)──────────────────────────────────────╭C──RY(-4.56)──┤
1: ──RY(3.6)───╰Z──RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z──RY(-3.6)───┤
2: ──RY(4.05)──╭C──RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)──╰Z──RY(-3.32)──╭C──RY(-4.05)──┤
3: ──RY(3.51)──╰Z──RY(3.66)───RY(-3.66)──RY(3.66)──RY(-3.66)─────────────────╰Z──RY(-3.51)──┤
0.966148861047503
0: ──RY(4.56)──╭C──RY(5.93)───RY(-5.93)───────────────────────────────────────────────────────────╭C──RY(-4.56)──┤
1: ──RY(3.6)───╰Z──RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)───────────────────────╭C──RY(-5.9)───╰Z──RY(-3.6)───┤
2: ──RY(4.05)──╭C──RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)──RY(1.07)──RY(-1.07)──╰Z──RY(-3.32)──╭C──RY(-4.05)──┤
3: ──RY(3.51)──╰Z──RY(3.66)───RY(-3.66)───────────────────────────────────────────────────────────╰Z──RY(-3.51)──┤
0: ──RY(4.56)──╭C──RY(5.93)───RY(-5.93)──────────────────────────────────────╭C─────────RY(-4.56)─────────────────┤
1: ──RY(3.6)───╰Z──RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z─────────RY(-3.6)──────────────────┤
```

---

## 12. tutorial_expressivity_fourier_series.html <a name="demo11"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_expressivity_fourier_series.html):

```
Cost at step  10: 0.04735694890119537
Cost at step  20: 0.04193426710325428
Cost at step  30: 0.0056074793613251055
Cost at step  40: 0.004608455923986289
Cost at step  50: 0.0016064517040624577
Cost at step  10: 0.022632957627087866
Cost at step  20: 0.001888527061988235
Cost at step  30: 0.0017982806807008216
Cost at step  40: 0.0007504225639151788
Cost at step  50: 0.0006664901287576844
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_expressivity_fourier_series.html):

```
Cost at step  10: 0.03212041720004561
Cost at step  20: 0.01385356188302471
Cost at step  30: 0.004049396436389439
Cost at step  40: 0.0005624933894468292
Cost at step  50: 8.145777333271201e-05
Cost at step  10: 0.017166449445319296
Cost at step  20: 0.005497199314425934
Cost at step  30: 0.004784402394898768
Cost at step  40: 0.004015481434557096
Cost at step  50: 0.0013998102989800916
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
Cost function value: -0.09248671036774045
   (-46.463906788688945) [I0]
+ (0.7829661725950177) [Z11]
+ (0.7829661725950178) [Z10]
+ (0.8084581961720478) [Z12]
+ (0.8084581961720481) [Z13]
+ (1.2034402289145636) [Z4]
+ (1.2034402289145636) [Z5]
+ (1.3096862988615412) [Z6]
+ (1.3096862988615414) [Z7]
+ (1.3693525634718173) [Z9]
+ (1.3693525634718178) [Z8]
+ (1.653894222683168) [Z2]
+ (1.6538942226831683) [Z3]
+ (12.412630742111773) [Z0]
+ (12.412630742111773) [Z1]
+ (-8.194261372247218e-06) [Y10 Y12]
+ (-8.194261372247218e-06) [X10 X12]
+ (-1.854060857998293e-06) [Y5 Y7]
+ (-1.854060857998293e-06) [X5 X7]
+ (-7.764994119866044e-07) [Y3 Y5]
+ (-7.764994119866044e-07) [X3 X5]
+ (-5.929765816048471e-07) [Y4 Y6]
+ (-5.929765816048471e-07) [X4 X6]
+ (1.602116740596601e-06) [Y2 Y4]
+ (1.602116740596601e-06) [X2 X4]
+ (7.954413176350869e-06) [Y11 Y13]
+ (7.954413176350869e-06) [X11 X13]
+ (0.003276971931231632) [Y1 Y3]
+ (0.003276971931231632) [X1 X3]
+ (0.10433064780651408) [Y0 Y2]
+ (0.10433064780651408) [X0 X2]
+ (0.11270386920332226) [Z10 Z12]
+ (0.11270386920332226) [Z11 Z13]
+ (0.11383573679388678) [Z4 Z12]
+ (0.11383573679388678) [Z5 Z13]
+ (0.11952438964682689) [Z6 Z10]
+ (0.11952438964682689) [Z7 Z11]
+ (0.12489990917237624) [Z4 Z10]
+ (0.12489990917237624) [Z5 Z11]
+ (0.12495807739503223) [Z2 Z4]
+ (0.12495807739503223) [Z3 Z5]
+ (0.1279950249246841) [Z2 Z10]
+ (0.1279950249246841) [Z3 Z11]
+ (0.13401715261963723) [Z6 Z12]
+ (0.13401715261963723) [Z7 Z13]
+ (0.13701191674040775) [Z4 Z6]
+ (0.13701191674040775) [Z5 Z7]
+ (0.13734953064261335) [Z6 Z11]
+ (0.13734953064261335) [Z7 Z10]
+ (0.13739104762683224) [Z2 Z6]
+ (0.13739104762683224) [Z3 Z7]
+ (0.1376687264585258) [Z8 Z10]
+ (0.1376687264585258) [Z9 Z11]
+ (0.14011289865354815) [Z2 Z12]
+ (0.14011289865354815) [Z3 Z13]
+ (0.14138905291942822) [Z10 Z13]
+ (0.14138905291942822) [Z11 Z12]
+ (0.14257997712485776) [Z4 Z11]
+ (0.14257997712485776) [Z5 Z10]
+ (0.14722943218766177) [Z8 Z11]
+ (0.14722943218766177) [Z9 Z10]
+ (0.1489943057506557) [Z4 Z7]
+ (0.1489943057506557) [Z5 Z6]
+ (0.14926355147388917) [Z10 Z11]
+ (0.14960702684445312) [Z4 Z8]
+ (0.14960702684445312) [Z5 Z9]
+ (0.14973486803496938) [Z8 Z12]
+ (0.14973486803496938) [Z9 Z13]
+ (0.15071408121008284) [Z2 Z8]
+ (0.15071408121008284) [Z3 Z9]
+ (0.1513832716142886) [Z6 Z13]
+ (0.1513832716142886) [Z7 Z12]
+ (0.15215040708869074) [Z4 Z13]
+ (0.15215040708869074) [Z5 Z12]
+ (0.1533796824331415) [Z2 Z11]
+ (0.1533796824331415) [Z3 Z10]
+ (0.15435748657223652) [Z12 Z13]
+ (0.1556901067175246) [Z2 Z13]
+ (0.1556901067175246) [Z3 Z12]
+ (0.15582269051553121) [Z8 Z13]
+ (0.15582269051553121) [Z9 Z12]
+ (0.1567639617643101) [Z4 Z9]
+ (0.1567639617643101) [Z5 Z8]
+ (0.15755314797985698) [Z4 Z5]
+ (0.16079764534838575) [Z2 Z5]
+ (0.16079764534838575) [Z3 Z4]
+ (0.1675665326546128) [Z6 Z8]
+ (0.1675665326546128) [Z7 Z9]
+ (0.18143991440303892) [Z6 Z9]
+ (0.18143991440303892) [Z7 Z8]
+ (0.18189085790751341) [Z2 Z3]
+ (0.1869082047691254) [Z2 Z9]
+ (0.1869082047691254) [Z3 Z8]
+ (0.1929972393536426) [Z0 Z10]
+ (0.1929972393536426) [Z1 Z11]
+ (0.19392534613270226) [Z6 Z7]
+ (0.1966177089034218) [Z0 Z4]
+ (0.1966177089034218) [Z1 Z5]
+ (0.19936354537360862) [Z0 Z5]
+ (0.19936354537360862) [Z1 Z4]
+ (0.2007286646044179) [Z0 Z11]
+ (0.2007286646044179) [Z1 Z10]
+ (0.21102659849791533) [Z0 Z12]
+ (0.21102659849791533) [Z1 Z13]
+ (0.21631037498631828) [Z0 Z13]
+ (0.21631037498631828) [Z1 Z12]
+ (0.23671080783830417) [Z0 Z2]
+ (0.23671080783830417) [Z1 Z3]
+ (0.2512944567459168) [Z0 Z3]
+ (0.2512944567459168) [Z1 Z2]
+ (0.27232518306605696) [Z0 Z8]
+ (0.27232518306605696) [Z1 Z9]
+ (0.2788345442672342) [Z0 Z9]
+ (0.2788345442672342) [Z1 Z8]
+ (1.1861763734860513) [Z0 Z1]
+ (-1.2260484989164329e-05) [Y5 Z6 Y7]
+ (-1.2260484989164329e-05) [X5 Z6 X7]
+ (-1.2260484989164325e-05) [Y4 Z5 Y6]
+ (-1.2260484989164325e-05) [X4 Z5 X6]
+ (-1.072231215723367e-05) [Y11 Z12 Y13]
+ (-1.072231215723367e-05) [X11 Z12 X13]
+ (-1.0722312157233669e-05) [Y10 Z11 Y12]
+ (-1.0722312157233669e-05) [X10 Z11 X12]
+ (-3.887051674136427e-06) [Y3 Z4 Y5]
+ (-3.887051674136427e-06) [X3 Z4 X5]
+ (-3.887051674136426e-06) [Y2 Z3 Y4]
+ (-3.887051674136426e-06) [X2 Z3 X4]
+ (0.12507032579771937) [Y0 Z1 Y2]
+ (0.12507032579771937) [X0 Z1 X2]
+ (0.12507032579771937) [Y1 Z2 Y3]
+ (0.12507032579771937) [X1 Z2 X3]
+ (-0.03831467029480395) [Y4 Y5 X12 X13]
+ (-0.03831467029480395) [X4 X5 Y12 Y13]
+ (-0.03619412355904258) [Y2 Y3 X8 X9]
+ (-0.03619412355904258) [X2 X3 Y8 Y9]
+ (-0.03583956795335352) [Y2 Y3 X4 X5]
+ (-0.03583956795335352) [X2 X3 Y4 Y5]
+ (-0.031143817988967173) [Y2 Y3 X6 X7]
+ (-0.031143817988967173) [X2 X3 Y6 Y7]
+ (-0.028685183716105962) [Y10 Y11 X12 X13]
+ (-0.028685183716105962) [X10 X11 Y12 Y13]
+ (-0.02599617759802124) [Y3 Z4 Z5 Y7]
+ (-0.02599617759802124) [X3 Z4 Z5 X7]
+ (-0.025384657508457413) [Y2 Y3 X10 X11]
+ (-0.025384657508457413) [X2 X3 Y10 Y11]
+ (-0.019028242443847338) [Y3 Y4 X11 X12]
+ (-0.019028242443847338) [X3 X4 Y11 Y12]
+ (-0.017825140995786446) [Y6 Y7 X10 X11]
+ (-0.017825140995786446) [X6 X7 Y10 Y11]
+ (-0.01768006795248152) [Y4 Y5 X10 X11]
+ (-0.01768006795248152) [X4 X5 Y10 Y11]
+ (-0.017366118994651385) [Y6 Y7 X12 X13]
+ (-0.017366118994651385) [X6 X7 Y12 Y13]
+ (-0.01557720806397646) [Y2 Y3 X12 X13]
+ (-0.01557720806397646) [X2 X3 Y12 Y13]
+ (-0.014583648907612658) [Y0 Y1 X2 X3]
+ (-0.014583648907612658) [X0 X1 Y2 Y3]
+ (-0.013873381748426106) [Y6 Y7 X8 X9]
+ (-0.013873381748426106) [X6 X7 Y8 Y9]
+ (-0.011982389010247953) [Y4 Y5 X6 X7]
+ (-0.011982389010247953) [X4 X5 Y6 Y7]
+ (-0.011285190200840886) [Y5 X6 X11 Y12]
+ (-0.011285190200840886) [X5 Y6 Y11 X12]
+ (-0.009560705729135971) [Y8 Y9 X10 X11]
+ (-0.009560705729135971) [X8 X9 Y10 Y11]
+ (-0.00812525192138103) [Y1 X2 X8 Y9]
+ (-0.00812525192138103) [Y1 Y2 Y8 Y9]
+ (-0.00812525192138103) [X1 X2 X8 X9]
+ (-0.00812525192138103) [X1 Y2 Y8 X9]
+ (-0.007731425250775324) [Y0 Y1 X10 X11]
+ (-0.007731425250775324) [X0 X1 Y10 Y11]
+ (-0.0071569349198569564) [Y4 Y5 X8 X9]
+ (-0.0071569349198569564) [X4 X5 Y8 Y9]
+ (-0.0068881943529705714) [Y0 Y1 X6 X7]
+ (-0.0068881943529705714) [X0 X1 Y6 Y7]
+ (-0.006509361201177245) [Y0 Y1 X8 X9]
+ (-0.006509361201177245) [X0 X1 Y8 Y9]
+ (-0.006087822480561857) [Y8 Y9 X12 X13]
+ (-0.006087822480561857) [X8 X9 Y12 Y13]
+ (-0.005283776488402967) [Y0 Y1 X12 X13]
+ (-0.005283776488402967) [X0 X1 Y12 Y13]
+ (-0.005143391768825109) [Y3 X4 X5 Y6]
+ (-0.005143391768825109) [X3 Y4 Y5 X6]
+ (-0.004684903388155222) [Y1 X2 X6 Y7]
+ (-0.004684903388155222) [Y1 Y2 Y6 Y7]
+ (-0.004684903388155222) [X1 X2 X6 X7]
+ (-0.004684903388155222) [X1 Y2 Y6 X7]
+ (-0.004575007626639203) [Y1 X2 X12 Y13]
+ (-0.004575007626639203) [Y1 Y2 Y12 Y13]
+ (-0.004575007626639203) [X1 X2 X12 X13]
+ (-0.004575007626639203) [X1 Y2 Y12 X13]
+ (-0.004424855449441865) [Y1 X2 X4 Y5]
+ (-0.004424855449441865) [Y1 Y2 Y4 Y5]
+ (-0.004424855449441865) [X1 X2 X4 X5]
+ (-0.004424855449441865) [X1 Y2 Y4 X5]
+ (-0.0034795118903343295) [Y2 Z3 Z5 Y6]
+ (-0.0034795118903343295) [X2 Z3 Z5 X6]
+ (-0.0034795118903343295) [Y3 Z4 Z6 Y7]
+ (-0.0034795118903343295) [X3 Z4 Z6 X7]
+ (-0.0027458364701868185) [Y0 Y1 X4 X5]
+ (-0.0027458364701868185) [X0 X1 Y4 Y5]
+ (-0.0017992194936630008) [Y1 X2 X10 Y11]
+ (-0.0017992194936630008) [Y1 Y2 Y10 Y11]
+ (-0.0017992194936630008) [X1 X2 X10 X11]
+ (-0.0017992194936630008) [X1 Y2 Y10 X11]
+ (-0.00029219862611106964) [Y7 Y8 X9 X10]
+ (-0.00029219862611106964) [X7 X8 Y9 Y10]
+ (-8.194261372247218e-06) [Z10 Y11 Z12 Y13]
+ (-8.194261372247218e-06) [Z10 X11 Z12 X13]
+ (-7.8017075006512e-06) [Y2 Z3 Y4 Z11]
+ (-7.8017075006512e-06) [X2 Z3 X4 Z11]
+ (-7.8017075006512e-06) [Y3 Z4 Y5 Z10]
+ (-7.8017075006512e-06) [X3 Z4 X5 Z10]
+ (-4.643051068541529e-06) [Y3 X4 X10 Y11]
+ (-4.643051068541529e-06) [Y3 Y4 Y10 Y11]
+ (-4.643051068541529e-06) [X3 X4 X10 X11]
+ (-4.643051068541529e-06) [X3 Y4 Y10 X11]
+ (-4.588855155750191e-06) [Y4 Z5 Y6 Z13]
+ (-4.588855155750191e-06) [X4 Z5 X6 Z13]
+ (-4.588855155750191e-06) [Y5 Z6 Y7 Z12]
+ (-4.588855155750191e-06) [X5 Z6 X7 Z12]
+ (-4.5565692182324534e-06) [Y5 X6 X12 Y13]
+ (-4.5565692182324534e-06) [Y5 Y6 Y12 Y13]
+ (-4.5565692182324534e-06) [X5 X6 X12 X13]
+ (-4.5565692182324534e-06) [X5 Y6 Y12 X13]
+ (-3.6945132945578865e-06) [Y4 X5 X11 Y12]
+ (-3.6945132945578865e-06) [Y4 Y5 Y11 Y12]
+ (-3.6945132945578865e-06) [X4 X5 X11 X12]
+ (-3.6945132945578865e-06) [X4 Y5 Y11 X12]
+ (-3.3440815564823174e-06) [Z0 Y5 Z6 Y7]
+ (-3.3440815564823174e-06) [Z0 X5 Z6 X7]
+ (-3.3440815564823174e-06) [Z1 Y4 Z5 Y6]
+ (-3.3440815564823174e-06) [Z1 X4 Z5 X6]
+ (-3.1586564321096703e-06) [Y2 Z3 Y4 Z10]
+ (-3.1586564321096703e-06) [X2 Z3 X4 Z10]
+ (-3.1586564321096703e-06) [Y3 Z4 Y5 Z11]
+ (-3.1586564321096703e-06) [X3 Z4 X5 Z11]
+ (-3.0993492435993306e-06) [Z0 Y4 Z5 Y6]
+ (-3.0993492435993306e-06) [Z0 X4 Z5 X6]
+ (-3.0993492435993306e-06) [Z1 Y5 Z6 Y7]
+ (-3.0993492435993306e-06) [Z1 X5 Z6 X7]
+ (-2.8909678815789858e-06) [Z6 Y11 Z12 Y13]
+ (-2.8909678815789858e-06) [Z6 X11 Z12 X13]
+ (-2.8909678815789858e-06) [Z7 Y10 Z11 Y12]
+ (-2.8909678815789858e-06) [Z7 X10 Z11 X12]
+ (-2.1776646049106297e-06) [Z0 Y10 Z11 Y12]
+ (-2.1776646049106297e-06) [Z0 X10 Z11 X12]
+ (-2.1776646049106297e-06) [Z1 Y11 Z12 Y13]
+ (-2.1776646049106297e-06) [Z1 X11 Z12 X13]
+ (-1.881850183230135e-06) [Y4 Z5 Y6 Z9]
+ (-1.881850183230135e-06) [X4 Z5 X6 Z9]
+ (-1.881850183230135e-06) [Y5 Z6 Y7 Z8]
+ (-1.881850183230135e-06) [X5 Z6 X7 Z8]
+ (-1.8551201214819788e-06) [Z6 Y10 Z11 Y12]
+ (-1.8551201214819788e-06) [Z6 X10 Z11 X12]
+ (-1.8551201214819788e-06) [Z7 Y11 Z12 Y13]
+ (-1.8551201214819788e-06) [Z7 X11 Z12 X13]
+ (-1.854060857998293e-06) [Y4 Z5 Y6 Z7]
+ (-1.854060857998293e-06) [X4 Z5 X6 Z7]
+ (-1.8163031697529117e-06) [Z4 Y11 Z12 Y13]
+ (-1.8163031697529117e-06) [Z4 X11 Z12 X13]
+ (-1.8163031697529117e-06) [Z5 Y10 Z11 Y12]
+ (-1.8163031697529117e-06) [Z5 X10 Z11 X12]
+ (-1.69239782861523e-06) [Y4 Z5 Y6 Z10]
+ (-1.69239782861523e-06) [X4 Z5 X6 Z10]
+ (-1.69239782861523e-06) [Y5 Z6 Y7 Z11]
+ (-1.69239782861523e-06) [X5 Z6 X7 Z11]
+ (-1.6148794137591985e-06) [Z0 Y11 Z12 Y13]
+ (-1.6148794137591985e-06) [Z0 X11 Z12 X13]
+ (-1.6148794137591985e-06) [Z1 Y10 Z11 Y12]
+ (-1.6148794137591985e-06) [Z1 X10 Z11 X12]
+ (-1.5973171977351043e-06) [Z8 Y10 Z11 Y12]
+ (-1.5973171977351043e-06) [Z8 X10 Z11 X12]
+ (-1.5973171977351043e-06) [Z9 Y11 Z12 Y13]
+ (-1.5973171977351043e-06) [Z9 X11 Z12 X13]
+ (-1.45484244902995e-06) [Y3 X4 X6 Y7]
+ (-1.45484244902995e-06) [Y3 Y4 Y6 Y7]
+ (-1.45484244902995e-06) [X3 X4 X6 X7]
+ (-1.45484244902995e-06) [X3 Y4 Y6 X7]
+ (-1.398044908136441e-06) [Y4 Z5 Y6 Z8]
+ (-1.398044908136441e-06) [X4 Z5 X6 Z8]
+ (-1.398044908136441e-06) [Y5 Z6 Y7 Z9]
+ (-1.398044908136441e-06) [X5 Z6 X7 Z9]
+ (-1.1954890099786675e-06) [Y2 Z3 Y4 Z7]
+ (-1.1954890099786675e-06) [X2 Z3 X4 Z7]
+ (-1.1954890099786675e-06) [Y3 Z4 Y5 Z6]
+ (-1.1954890099786675e-06) [X3 Z4 X5 Z6]
+ (-1.190850808485132e-06) [Z0 Y3 Z4 Y5]
+ (-1.190850808485132e-06) [Z0 X3 Z4 X5]
+ (-1.190850808485132e-06) [Z1 Y2 Z3 Y4]
+ (-1.190850808485132e-06) [Z1 X2 Z3 X4]
+ (-1.1708301370606323e-06) [Z2 Y5 Z6 Y7]
+ (-1.1708301370606323e-06) [Z2 X5 Z6 X7]
+ (-1.1708301370606323e-06) [Z3 Y4 Z5 Y6]
+ (-1.1708301370606323e-06) [Z3 X4 Z5 X6]
+ (-1.0632283423178147e-06) [Z2 Y10 Z11 Y12]
+ (-1.0632283423178147e-06) [Z2 X10 Z11 X12]
+ (-1.0632283423178147e-06) [Z3 Y11 Z12 Y13]
+ (-1.0632283423178147e-06) [Z3 X11 Z12 X13]
+ (-1.0358477600970068e-06) [Y6 X7 X11 Y12]
+ (-1.0358477600970068e-06) [Y6 Y7 Y11 Y12]
+ (-1.0358477600970068e-06) [X6 X7 X11 X12]
+ (-1.0358477600970068e-06) [X6 Y7 Y11 X12]
+ (-9.50924975177282e-07) [Z2 Y4 Z5 Y6]
+ (-9.50924975177282e-07) [Z2 X4 Z5 X6]
+ (-9.50924975177282e-07) [Z3 Y5 Z6 Y7]
+ (-9.50924975177282e-07) [Z3 X5 Z6 X7]
+ (-9.344557775542793e-07) [Z8 Y11 Z12 Y13]
+ (-9.344557775542793e-07) [Z8 X11 Z12 X13]
+ (-9.344557775542793e-07) [Z9 Y10 Z11 Y12]
+ (-9.344557775542793e-07) [Z9 X10 Z11 X12]
+ (-8.337746755587313e-07) [Z0 Y2 Z3 Y4]
+ (-8.337746755587313e-07) [Z0 X2 Z3 X4]
+ (-8.337746755587313e-07) [Z1 Y3 Z4 Y5]
+ (-8.337746755587313e-07) [Z1 X3 Z4 X5]
+ (-7.956895372781423e-07) [Y3 X4 X8 Y9]
+ (-7.956895372781423e-07) [Y3 Y4 Y8 Y9]
+ (-7.956895372781423e-07) [X3 X4 X8 X9]
+ (-7.956895372781423e-07) [X3 Y4 Y8 X9]
+ (-7.764994119866045e-07) [Y2 Z3 Y4 Z5]
+ (-7.764994119866045e-07) [X2 Z3 X4 Z5]
+ (-5.929765816048471e-07) [Z4 Y5 Z6 Y7]
+ (-5.929765816048471e-07) [Z4 X5 Z6 X7]
+ (-5.770052995778244e-07) [Y2 Z3 Y4 Z9]
+ (-5.770052995778244e-07) [X2 Z3 X4 Z9]
+ (-5.770052995778244e-07) [Y3 Z4 Y5 Z8]
+ (-5.770052995778244e-07) [X3 Z4 X5 Z8]
+ (-5.471647744636925e-07) [Y1 Y2 X11 X12]
+ (-5.471647744636925e-07) [X1 X2 Y11 Y12]
+ (-4.838052750936942e-07) [Y5 X6 X8 Y9]
+ (-4.838052750936942e-07) [Y5 Y6 Y8 Y9]
+ (-4.838052750936942e-07) [X5 X6 X8 X9]
+ (-4.838052750936942e-07) [X5 Y6 Y8 X9]
+ (-3.570761329264007e-07) [Y0 X1 X3 Y4]
+ (-3.570761329264007e-07) [Y0 Y1 Y3 Y4]
+ (-3.570761329264007e-07) [X0 X1 X3 X4]
+ (-3.570761329264007e-07) [X0 Y1 Y3 X4]
+ (-2.4473231288298644e-07) [Y0 X1 X5 Y6]
+ (-2.4473231288298644e-07) [Y0 Y1 Y5 Y6]
+ (-2.4473231288298644e-07) [X0 X1 X5 X6]
+ (-2.4473231288298644e-07) [X0 Y1 Y5 X6]
+ (-2.1990516188335037e-07) [Y2 X3 X5 Y6]
+ (-2.1990516188335037e-07) [Y2 Y3 Y5 Y6]
+ (-2.1990516188335037e-07) [X2 X3 X5 X6]
+ (-2.1990516188335037e-07) [X2 Y3 Y5 X6]
+ (-1.9332412771080281e-07) [Y1 X2 X3 Y4]
+ (-1.9332412771080281e-07) [X1 Y2 Y3 X4]
+ (-1.2919694862865143e-07) [Y1 Z2 Z3 Y5]
+ (-1.2919694862865143e-07) [X1 Z2 Z3 X5]
+ (1.7379332624525264e-07) [Y0 Z1 Z3 Y4]
+ (1.7379332624525264e-07) [X0 Z1 Z3 X4]
+ (1.7379332624525264e-07) [Y1 Z2 Z4 Y5]
+ (1.7379332624525264e-07) [X1 Z2 Z4 X5]
+ (1.9332412771080281e-07) [Y1 Y2 X3 X4]
+ (1.9332412771080281e-07) [X1 X2 Y3 Y4]
+ (2.1868423770031812e-07) [Y2 Z3 Y4 Z8]
+ (2.1868423770031812e-07) [X2 Z3 X4 Z8]
+ (2.1868423770031812e-07) [Y3 Z4 Y5 Z9]
+ (2.1868423770031812e-07) [X3 Z4 X5 Z9]
+ (2.5935343905128243e-07) [Y2 Z3 Y4 Z6]
+ (2.5935343905128243e-07) [X2 Z3 X4 Z6]
+ (2.5935343905128243e-07) [Y3 Z4 Y5 Z7]
+ (2.5935343905128243e-07) [X3 Z4 X5 Z7]
+ (3.6060718680071947e-07) [Y0 Z1 Z2 Y4]
+ (3.6060718680071947e-07) [X0 Z1 Z2 X4]
+ (3.6060718680071947e-07) [Y1 Z3 Z4 Y5]
+ (3.6060718680071947e-07) [X1 Z3 Z4 X5]
+ (5.471647744636925e-07) [Y1 X2 X11 Y12]
+ (5.471647744636925e-07) [X1 Y2 Y11 X12]
+ (5.62785191151431e-07) [Y0 X1 X11 Y12]
+ (5.62785191151431e-07) [Y0 Y1 Y11 Y12]
+ (5.62785191151431e-07) [X0 X1 X11 X12]
+ (5.62785191151431e-07) [X0 Y1 Y11 X12]
+ (6.62861420180825e-07) [Y8 X9 X11 Y12]
+ (6.62861420180825e-07) [Y8 Y9 Y11 Y12]
+ (6.62861420180825e-07) [X8 X9 X11 X12]
+ (6.62861420180825e-07) [X8 Y9 Y11 X12]
+ (1.1094407592313514e-06) [Z2 Y11 Z12 Y13]
+ (1.1094407592313514e-06) [Z2 X11 Z12 X13]
+ (1.1094407592313514e-06) [Z3 Y10 Z11 Y12]
+ (1.1094407592313514e-06) [Z3 X10 Z11 X12]
+ (1.602116740596601e-06) [Z2 Y3 Z4 Y5]
+ (1.602116740596601e-06) [Z2 X3 Z4 X5]
+ (1.8782101248049748e-06) [Z4 Y10 Z11 Y12]
+ (1.8782101248049748e-06) [Z4 X10 Z11 X12]
+ (1.8782101248049748e-06) [Z5 Y11 Z12 Y13]
+ (1.8782101248049748e-06) [Z5 X11 Z12 X13]
+ (2.172669101549166e-06) [Y2 X3 X11 Y12]
+ (2.172669101549166e-06) [Y2 Y3 Y11 Y12]
+ (2.172669101549166e-06) [X2 X3 X11 X12]
+ (2.172669101549166e-06) [X2 Y3 Y11 X12]
+ (3.1174479462072385e-06) [Y0 Z2 Z3 Y4]
+ (3.1174479462072385e-06) [X0 Z2 Z3 X4]
+ (3.5390541844740314e-06) [Y2 Z3 Y4 Z12]
+ (3.5390541844740314e-06) [X2 Z3 X4 Z12]
+ (3.5390541844740314e-06) [Y3 Z4 Y5 Z13]
+ (3.5390541844740314e-06) [X3 Z4 X5 Z13]
+ (4.28191388485626e-06) [Y4 Z5 Y6 Z11]
+ (4.28191388485626e-06) [X4 Z5 X6 Z11]
+ (4.28191388485626e-06) [Y5 Z6 Y7 Z10]
+ (4.28191388485626e-06) [X5 Z6 X7 Z10]
+ (5.275883122184724e-06) [Y3 X4 X12 Y13]
+ (5.275883122184724e-06) [Y3 Y4 Y12 Y13]
+ (5.275883122184724e-06) [X3 X4 X12 X13]
+ (5.275883122184724e-06) [X3 Y4 Y12 X13]
+ (5.97431171347149e-06) [Y5 X6 X10 Y11]
+ (5.97431171347149e-06) [Y5 Y6 Y10 Y11]
+ (5.97431171347149e-06) [X5 X6 X10 X11]
+ (5.97431171347149e-06) [X5 Y6 Y10 X11]
+ (7.954413176350869e-06) [Y10 Z11 Y12 Z13]
+ (7.954413176350869e-06) [X10 Z11 X12 Z13]
+ (8.814937306658755e-06) [Y2 Z3 Y4 Z13]
+ (8.814937306658755e-06) [X2 Z3 X4 Z13]
+ (8.814937306658755e-06) [Y3 Z4 Y5 Z12]
+ (8.814937306658755e-06) [X3 Z4 X5 Z12]
+ (0.00029219862611106964) [Y7 X8 X9 Y10]
+ (0.00029219862611106964) [X7 Y8 Y9 X10]
+ (0.0004956762314916415) [Y2 Z4 Z5 Y6]
+ (0.0004956762314916415) [X2 Z4 Z5 X6]
+ (0.001105903769189667) [Y0 Z1 Y2 Z5]
+ (0.001105903769189667) [X0 Z1 X2 Z5]
+ (0.001105903769189667) [Y1 Z2 Y3 Z4]
+ (0.001105903769189667) [X1 Z2 X3 Z4]
+ (0.0016638798784907797) [Y2 Z3 Z4 Y6]
+ (0.0016638798784907797) [X2 Z3 Z4 X6]
+ (0.0016638798784907797) [Y3 Z5 Z6 Y7]
+ (0.0016638798784907797) [X3 Z5 Z6 X7]
+ (0.001756070701841226) [Y0 Z1 Y2 Z11]
+ (0.001756070701841226) [X0 Z1 X2 Z11]
+ (0.001756070701841226) [Y1 Z2 Y3 Z10]
+ (0.001756070701841226) [X1 Z2 X3 Z10]
+ (0.002326230623158056) [Y0 Z1 Y2 Z13]
+ (0.002326230623158056) [X0 Z1 X2 Z13]
+ (0.002326230623158056) [Y1 Z2 Y3 Z12]
+ (0.002326230623158056) [X1 Z2 X3 Z12]
+ (0.0027458364701868185) [Y0 X1 X4 Y5]
+ (0.0027458364701868185) [X0 Y1 Y4 X5]
+ (0.0029297686747510256) [Y0 Z1 Y2 Z9]
+ (0.0029297686747510256) [X0 Z1 X2 Z9]
+ (0.0029297686747510256) [Y1 Z2 Y3 Z8]
+ (0.0029297686747510256) [X1 Z2 X3 Z8]
+ (0.003276971931231632) [Y0 Z1 Y2 Z3]
+ (0.003276971931231632) [X0 Z1 X2 Z3]
+ (0.0033476175306661636) [Y0 Z1 Y2 Z7]
+ (0.0033476175306661636) [X0 Z1 X2 Z7]
+ (0.0033476175306661636) [Y1 Z2 Y3 Z6]
+ (0.0033476175306661636) [X1 Z2 X3 Z6]
+ (0.0035552901955042265) [Y0 Z1 Y2 Z10]
+ (0.0035552901955042265) [X0 Z1 X2 Z10]
+ (0.0035552901955042265) [Y1 Z2 Y3 Z11]
+ (0.0035552901955042265) [X1 Z2 X3 Z11]
+ (0.005143391768825109) [Y3 Y4 X5 X6]
+ (0.005143391768825109) [X3 X4 Y5 Y6]
+ (0.005283776488402967) [Y0 X1 X12 Y13]
+ (0.005283776488402967) [X0 Y1 Y12 X13]
+ (0.005530759218631533) [Y0 Z1 Y2 Z4]
+ (0.005530759218631533) [X0 Z1 X2 Z4]
+ (0.005530759218631533) [Y1 Z2 Y3 Z5]
+ (0.005530759218631533) [X1 Z2 X3 Z5]
+ (0.006087822480561857) [Y8 X9 X12 Y13]
+ (0.006087822480561857) [X8 Y9 Y12 X13]
+ (0.006509361201177245) [Y0 X1 X8 Y9]
+ (0.006509361201177245) [X0 Y1 Y8 X9]
+ (0.0068881943529705714) [Y0 X1 X6 Y7]
+ (0.0068881943529705714) [X0 Y1 Y6 X7]
+ (0.006901238249797258) [Y0 Z1 Y2 Z12]
+ (0.006901238249797258) [X0 Z1 X2 Z12]
+ (0.006901238249797258) [Y1 Z2 Y3 Z13]
+ (0.006901238249797258) [X1 Z2 X3 Z13]
+ (0.0071569349198569564) [Y4 X5 X8 Y9]
+ (0.0071569349198569564) [X4 Y5 Y8 X9]
+ (0.007731425250775324) [Y0 X1 X10 Y11]
+ (0.007731425250775324) [X0 Y1 Y10 X11]
+ (0.008032520918821385) [Y0 Z1 Y2 Z6]
+ (0.008032520918821385) [X0 Z1 X2 Z6]
+ (0.008032520918821385) [Y1 Z2 Y3 Z7]
+ (0.008032520918821385) [X1 Z2 X3 Z7]
+ (0.009560705729135971) [Y8 X9 X10 Y11]
+ (0.009560705729135971) [X8 Y9 Y10 X11]
+ (0.011055020596132057) [Y0 Z1 Y2 Z8]
+ (0.011055020596132057) [X0 Z1 X2 Z8]
+ (0.011055020596132057) [Y1 Z2 Y3 Z9]
+ (0.011055020596132057) [X1 Z2 X3 Z9]
+ (0.011285190200840886) [Y5 Y6 X11 X12]
+ (0.011285190200840886) [X5 X6 Y11 Y12]
+ (0.01130727400884824) [Y7 Z8 Z9 Y11]
+ (0.01130727400884824) [X7 Z8 Z9 X11]
+ (0.011982389010247953) [Y4 X5 X6 Y7]
+ (0.011982389010247953) [X4 Y5 Y6 X7]
+ (0.013873381748426106) [Y6 X7 X8 Y9]
+ (0.013873381748426106) [X6 Y7 Y8 X9]
+ (0.014583648907612658) [Y0 X1 X2 Y3]
+ (0.014583648907612658) [X0 Y1 Y2 X3]
+ (0.01557720806397646) [Y2 X3 X12 Y13]
+ (0.01557720806397646) [X2 Y3 Y12 X13]
+ (0.017366118994651385) [Y6 X7 X12 Y13]
+ (0.017366118994651385) [X6 Y7 Y12 X13]
+ (0.01768006795248152) [Y4 X5 X10 Y11]
+ (0.01768006795248152) [X4 Y5 Y10 X11]
+ (0.017825140995786446) [Y6 X7 X10 Y11]
+ (0.017825140995786446) [X6 Y7 Y10 X11]
+ (0.019028242443847338) [Y3 X4 X11 Y12]
+ (0.019028242443847338) [X3 Y4 Y11 X12]
+ (0.025384657508457413) [Y2 X3 X10 Y11]
+ (0.025384657508457413) [X2 Y3 Y10 X11]
+ (0.028685183716105962) [Y10 X11 X12 Y13]
+ (0.028685183716105962) [X10 Y11 Y12 X13]
+ (0.02981242451734576) [Y6 Z7 Z8 Y10]
+ (0.02981242451734576) [X6 Z7 Z8 X10]
+ (0.02981242451734576) [Y7 Z9 Z10 Y11]
+ (0.02981242451734576) [X7 Z9 Z10 X11]
+ (0.030104623143456834) [Y6 Z7 Z9 Y10]
+ (0.030104623143456834) [X6 Z7 Z9 X10]
+ (0.030104623143456834) [Y7 Z8 Z10 Y11]
+ (0.030104623143456834) [X7 Z8 Z10 X11]
+ (0.030787505389143953) [Y6 Z8 Z9 Y10]
+ (0.030787505389143953) [X6 Z8 Z9 X10]
+ (0.031143817988967173) [Y2 X3 X6 Y7]
+ (0.031143817988967173) [X2 Y3 Y6 X7]
+ (0.03583956795335352) [Y2 X3 X4 Y5]
+ (0.03583956795335352) [X2 Y3 Y4 X5]
+ (0.03619412355904258) [Y2 X3 X8 Y9]
+ (0.03619412355904258) [X2 Y3 Y8 X9]
+ (0.03831467029480395) [Y4 X5 X12 Y13]
+ (0.03831467029480395) [X4 Y5 Y12 X13]
+ (0.10433064780651408) [Z0 Y1 Z2 Y3]
+ (0.10433064780651408) [Z0 X1 Z2 X3]
+ (-0.12133276911042329) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133276911042329) [X3 Z4 Z5 Z6 X7]
+ (-0.12133276911042327) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042327) [X2 Z3 Z4 Z5 X6]
+ (3.202076880808083e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.202076880808083e-06) [X0 Z1 Z2 Z3 X4]
+ (3.2020768808080836e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.2020768808080836e-06) [X1 Z2 Z3 Z4 X5]
+ (0.22848106564918777) [Y6 Z7 Z8 Z9 Y10]
+ (0.22848106564918777) [X6 Z7 Z8 Z9 X10]
+ (0.22848106564918777) [Y7 Z8 Z9 Z10 Y11]
+ (0.22848106564918777) [X7 Z8 Z9 Z10 X11]
+ (-0.0327676578232905) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.0327676578232905) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.0327676578232905) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.0327676578232905) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.02711503684527315) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.02711503684527315) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.02711503684527315) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.02711503684527315) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.02599617759802124) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.02599617759802124) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.01756120240964616) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.01756120240964616) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.01756120240964616) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.01756120240964616) [X3 Z4 Z5 Z6 X7 Z8]
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
+ (-0.01175601341981924) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.01175601341981924) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.01175601341981924) [X3 Z4 Z5 X6 X8 X9]
+ (-0.01175601341981924) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.008764827575688808) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.008764827575688808) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.008764827575688808) [X2 Z3 Z4 X5 X11 X12]
+ (-0.008764827575688808) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.008764827575688808) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.008764827575688808) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.008764827575688808) [X3 X4 X10 Z11 Z12 X13]
+ (-0.008764827575688808) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.00812525192138103) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.00812525192138103) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.007306759928832954) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306759928832954) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306759928832954) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306759928832954) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.0058051889898269185) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.0058051889898269185) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.0058051889898269185) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.0058051889898269185) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.005652620978017345) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.005652620978017345) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.005652620978017345) [X0 X1 X3 Z4 Z5 X6]
+ (-0.005652620978017345) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.00514339176882511) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.00514339176882511) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.00514339176882511) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.00514339176882511) [X2 Z3 X4 X5 Z6 X7]
+ (-0.004684903388155222) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.004684903388155222) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668620318776299) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668620318776299) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.004575007626639204) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.004575007626639204) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424855449441865) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424855449441865) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.004158797381840032) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.004158797381840032) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.004158797381840032) [X3 Z4 Z5 X6 X12 X13]
+ (-0.004158797381840032) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.003493790359890125) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.003493790359890125) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.003493790359890125) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.003493790359890125) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.0027790267990255536) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.0027790267990255536) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.0022939566113524606) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.0022939566113524606) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.0017992194936630008) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.0017992194936630008) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.001727875394136973) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.001727875394136973) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298507967730487) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298507967730487) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298507967730487) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298507967730487) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533856254125502) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533856254125502) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008145313270956664) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008145313270956664) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008145313270956664) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008145313270956664) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.735036880591229e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735036880591229e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735036880591229e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735036880591229e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.774817864735922e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.774817864735922e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.774817864735922e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.774817864735922e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.518362215802099e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.518362215802099e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.518362215802099e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.518362215802099e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.44434467604316e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.44434467604316e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.44434467604316e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.44434467604316e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.5243738486544775e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.5243738486544775e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.5243738486544775e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.5243738486544775e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.2900284333507945e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.2900284333507945e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.2900284333507945e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.2900284333507945e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.97431171347149e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.97431171347149e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.275883122184724e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.275883122184724e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.643051068541529e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.643051068541529e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.5565692182324534e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.5565692182324534e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.253224225643258e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.253224225643258e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.7696594520760436e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.7696594520760436e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.6945132945578865e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.6945132945578865e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.6102971306782175e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.6102971306782175e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.6102971306782175e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.6102971306782175e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.313145500181932e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.313145500181932e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.2774831956251002e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.2774831956251002e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.2774831956251002e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.2774831956251002e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.211228348472546e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.211228348472546e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.211228348472546e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.211228348472546e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.1513463111600517e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.1513463111600517e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.0882507113704623e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.0882507113704623e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.172669101549166e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.172669101549166e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.45484244902995e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.45484244902995e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.3304731886927621e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.3304731886927621e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.228333782451305e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.228333782451305e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.0358477600970068e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.0358477600970068e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956895372781423e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956895372781423e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733197742373106e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733197742373106e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733197742373106e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733197742373106e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.62861420180825e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.62861420180825e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.556281914559407e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.556281914559407e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.556281914559407e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.556281914559407e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.418291574595618e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.418291574595618e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.418291574595618e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.418291574595618e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.927453082854051e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927453082854051e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927453082854051e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927453082854051e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.62785191151431e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.62785191151431e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660624686048e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660624686048e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660624686048e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660624686048e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660624686048e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660624686048e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660624686048e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660624686048e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.838052750936942e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.838052750936942e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.570761329264007e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.570761329264007e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.328139350531173e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.328139350531173e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.086826565188776e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.086826565188776e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.086826565188776e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.086826565188776e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.4473231288298644e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.4473231288298644e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.3713289479585635e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.3713289479585635e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.3713289479585635e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.3713289479585635e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.1990516188335037e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.1990516188335037e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.9332412771080284e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.9332412771080284e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.9332412771080284e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.9332412771080284e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.8394209154756715e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.8394209154756715e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.8394209154756715e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.8394209154756715e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.5510539175746453e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.5510539175746453e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.5510539175746453e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.5510539175746453e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.380778148063582e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.380778148063582e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.380778148063582e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.380778148063582e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.380778148063582e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.380778148063582e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.380778148063582e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.380778148063582e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.380778148063582e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.380778148063582e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.380778148063582e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.380778148063582e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.2919694862865143e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.2919694862865143e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.1076325599160258e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.1076325599160258e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.1076325599160258e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.1076325599160258e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.1076325599160258e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.1076325599160258e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.1076325599160258e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.1076325599160258e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.057446595190556e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057446595190556e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057446595190556e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057446595190556e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649310135641421e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649310135641421e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649310135641421e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649310135641421e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.8394209154756715e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.8394209154756715e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.8394209154756715e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.8394209154756715e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.1990516188335037e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.1990516188335037e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.4473231288298644e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.4473231288298644e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.236259961522705e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.236259961522705e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.236259961522705e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.236259961522705e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.328139350531173e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.328139350531173e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.570761329264007e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.570761329264007e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.838052750936942e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.838052750936942e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.62785191151431e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.62785191151431e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.62861420180825e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.62861420180825e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956895372781423e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956895372781423e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.306536652020523e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.306536652020523e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.306536652020523e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.306536652020523e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.0358477600970068e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.0358477600970068e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.228333782451305e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.228333782451305e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.23933632172093e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.23933632172093e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.23933632172093e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.23933632172093e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.3304731886927621e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.3304731886927621e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.45484244902995e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.45484244902995e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.172669101549166e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.172669101549166e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.0882507113704623e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.0882507113704623e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.1174479462072385e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.1174479462072385e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.1513463111600517e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.1513463111600517e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.313145500181932e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.313145500181932e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.3343312894059807e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.3343312894059807e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.6945132945578865e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.6945132945578865e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.183932559376533e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.183932559376533e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.5565692182324534e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.5565692182324534e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.643051068541529e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.643051068541529e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.275883122184724e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.275883122184724e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.97431171347149e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.97431171347149e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.00029219862611106964) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.00029219862611106964) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.00029219862611106964) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.00029219862611106964) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004956762314916415) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004956762314916415) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650070219499076) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650070219499076) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650070219499076) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650070219499076) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533856254125502) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533856254125502) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.0016095313817213771) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.0016095313817213771) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.0016095313817213771) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.0016095313817213771) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0016676041811440534) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.0016676041811440534) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.0016676041811440534) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.0016676041811440534) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.001727875394136973) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.001727875394136973) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.0017992194936630008) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.0017992194936630008) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.0022939566113524606) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.0022939566113524606) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.0024629170071339274) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.0024629170071339274) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.0024629170071339274) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.0024629170071339274) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.003961560792496514) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.003961560792496514) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.003961560792496514) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.003961560792496514) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.004424855449441865) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424855449441865) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004575007626639204) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.004575007626639204) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668620318776299) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668620318776299) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.004684903388155222) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.004684903388155222) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005324835234221697) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.005324835234221697) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.005324835234221697) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.005324835234221697) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.005368659358109505) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368659358109505) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368659358109505) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368659358109505) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.007960880725921556) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.007960880725921556) [X4 Z5 X6 X10 Z11 X12]
+ (0.007960880725921556) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.007960880725921556) [X5 Z6 X7 X11 Z12 X13]
+ (0.00812525192138103) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.00812525192138103) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.008890731522694605) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.008890731522694605) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.008890731522694605) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.008890731522694605) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.010263414868158523) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.010263414868158523) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.010263414868158523) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.010263414868158523) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.010540425907671545) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.010540425907671545) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.010540425907671545) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.010540425907671545) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.01096007494054264) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.01096007494054264) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.01096007494054264) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.01096007494054264) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.01130727400884824) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.01130727400884824) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.014411099430130947) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411099430130947) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411099430130947) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411099430130947) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.015225630757226612) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.015225630757226612) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.015225630757226612) [X3 Z4 Z5 X6 X10 X11]
+ (0.015225630757226612) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.015588250102380224) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.015588250102380224) [X2 Z3 X4 X10 Z11 X12]
+ (0.015588250102380224) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.015588250102380224) [X3 Z4 X5 X11 Z12 X13]
+ (0.018266834869375595) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.018266834869375595) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.018266834869375595) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.018266834869375595) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.01902042317303998) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.01902042317303998) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.01902042317303998) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.01902042317303998) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.020175921723535488) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.020175921723535488) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.020175921723535488) [X4 Z5 Z6 X7 X11 X12]
+ (0.020175921723535488) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.020175921723535488) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.020175921723535488) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.020175921723535488) [X5 X6 X10 Z11 Z12 X13]
+ (0.020175921723535488) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.024353077678069036) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.024353077678069036) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.024353077678069036) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.024353077678069036) [X2 Z3 X4 X11 Z12 X13]
+ (0.024353077678069036) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.024353077678069036) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.024353077678069036) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.024353077678069036) [X3 Z4 X5 X10 Z11 X12]
+ (0.024389082531149485) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.024389082531149485) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.024389082531149485) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.024389082531149485) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.02510495713884456) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.02510495713884456) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.02510495713884456) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.02510495713884456) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.030787505389143953) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.030787505389143953) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.05600733087780768) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.05600733087780768) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.05600733087780768) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.05600733087780768) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.056084681246613595) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.056084681246613595) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.056084681246613595) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.056084681246613595) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.631277928504226e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631277928504226e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.631277928504223e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631277928504223e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.5950860070870918e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.5950860070870918e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.5950860070870914e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.5950860070870914e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.04274327701378251) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04274327701378251) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (0.04274327701378253) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.04274327701378253) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (-0.047642612176383145) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.047642612176383145) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.047642612176383145) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.047642612176383145) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.0417188138398218) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.0417188138398218) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.0417188138398218) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.0417188138398218) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.03956441632289345) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.03956441632289345) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.03956441632289345) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03956441632289345) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.039359168022053095) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.039359168022053095) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.039359168022053095) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.039359168022053095) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.03931805194719763) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03931805194719763) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.03931805194719763) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03931805194719763) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.035608378988312483) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.035608378988312483) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.028730779551905526) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.028730779551905526) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.028730779551905526) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.028730779551905526) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.025637238296026835) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.025637238296026835) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.025637238296026835) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.025637238296026835) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.02475546329289095) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.02475546329289095) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.02475546329289095) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.02475546329289095) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.02428211735469297) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.02428211735469297) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.023145130929529065) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.023145130929529065) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.02252844019601307) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.02252844019601307) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.021433810721600968) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.021433810721600968) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.021433810721600968) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.021433810721600968) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251617) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.019257505095251617) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019028242443847338) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.019028242443847338) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.018889030304942916) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.018889030304942916) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.018889030304942916) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.018889030304942916) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.016024603689179552) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.016024603689179552) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.015225630757226612) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225630757226612) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.014603704729162146) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.014603704729162146) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564531231173017) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564531231173017) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.011756013419819238) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.011756013419819238) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.011285190200840886) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.011285190200840886) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.009841749246962607) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009841749246962607) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.009612634606847286) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612634606847286) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612634606847286) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612634606847286) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.008469978791023864) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.008469978791023864) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.0073067599288329545) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.0073067599288329545) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005923798336561349) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.005923798336561349) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.005652620978017345) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.005652620978017345) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368659358109504) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368659358109504) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.004158797381840032) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.004158797381840032) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0033566705638328875) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.0033566705638328875) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.0033566705638328875) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.0033566705638328875) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.0032675138544235515) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.0032675138544235515) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.0032675138544235515) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.0032675138544235515) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.002779026799025553) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.002779026799025553) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.002686040977806633) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.002686040977806633) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.002686040977806633) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.002686040977806633) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.0022939566113524606) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524606) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.0022939566113524606) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524606) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.0009581655836696602) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.0009581655836696602) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0009581655836696602) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.0009581655836696602) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.0009581655836696602) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.0009581655836696602) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0009581655836696602) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.0009581655836696602) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.0002463643756958201) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0002463643756958201) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303549664) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.00013840177303549664) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.00013840177303549664) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.00013840177303549664) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.735036880591229e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735036880591229e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.610358530599823e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.610358530599823e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.610358530599823e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.610358530599823e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.531680879556182e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.531680879556182e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.531680879556182e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.531680879556182e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.806102775284862e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.806102775284862e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.806102775284862e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.806102775284862e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.0897994676692765e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.0897994676692765e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.0897994676692765e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.0897994676692765e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.652209669659424e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.652209669659424e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.652209669659424e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.652209669659424e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851834118426e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.481851834118426e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851834118426e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.481851834118426e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.071480736420529e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.071480736420529e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.071480736420529e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.071480736420529e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.734622038864331e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.734622038864331e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.734622038864331e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.734622038864331e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.728843147294602e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.728843147294602e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.728843147294602e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.728843147294602e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.253224225643258e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.253224225643258e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.769659452076044e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.769659452076044e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.5443954293450877e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.5443954293450877e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.5443954293450877e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.5443954293450877e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.5443954293450877e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.5443954293450877e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.5443954293450877e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.5443954293450877e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.3609563203746747e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563203746747e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563203746747e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.3609563203746747e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.1032156047422722e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.1032156047422722e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.1032156047422722e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.1032156047422722e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220983090993e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.0111220983090993e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220983090993e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.0111220983090993e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.942946836721921e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.942946836721921e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.942946836721921e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.942946836721921e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.654117477204732e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.654117477204732e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.654117477204732e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.654117477204732e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.5224930676134973e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224930676134973e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224930676134973e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224930676134973e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224930676134973e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930676134973e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930676134973e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224930676134973e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.2283337824513048e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337824513048e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.2283337824513048e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337824513048e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.988770288556096e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988770288556096e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988770288556096e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988770288556096e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.867765104364124e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104364124e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104364124e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.867765104364124e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.189990975414762e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.189990975414762e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246207033744e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246207033744e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.471647744636925e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.471647744636925e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.561447179746253e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.561447179746253e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.561447179746253e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.561447179746253e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.5233896778789067e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.5233896778789067e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.427323108809844e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.427323108809844e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.427323108809844e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.427323108809844e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.328139350531173e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350531173e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.328139350531173e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350531173e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.086826565188776e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.086826565188776e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.888293595171888e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.888293595171888e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.888293595171888e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.888293595171888e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.371328947958563e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.371328947958563e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.8394209154756715e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.8394209154756715e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.057446595190556e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057446595190556e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.537178094703968e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.537178094703968e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.537178094703968e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.537178094703968e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.057446595190556e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057446595190556e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.209350643317303e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350643317303e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350643317303e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.209350643317303e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.703578355409977e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.703578355409977e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.703578355409977e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.703578355409977e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.8394209154756715e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.8394209154756715e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.371328947958563e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.371328947958563e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.086826565188776e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.086826565188776e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.5233896778789067e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.5233896778789067e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.471647744636925e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.471647744636925e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246207033744e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246207033744e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189990975414762e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.189990975414762e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.3304731886927621e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.3304731886927621e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.3304731886927621e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.3304731886927621e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.628853243546555e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.628853243546555e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.628853243546555e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.628853243546555e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.6893489514763477e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.6893489514763477e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.6893489514763477e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.6893489514763477e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.7455184004894784e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.7455184004894784e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.7455184004894784e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.7455184004894784e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.7455184004894784e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.7455184004894784e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.7455184004894784e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.7455184004894784e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.211842019089844e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.211842019089844e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.211842019089844e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.211842019089844e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.211842019089844e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.211842019089844e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.211842019089844e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.211842019089844e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.313145500181932e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.313145500181932e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.313145500181932e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.313145500181932e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.3343312894059802e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.3343312894059802e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.183932559376533e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.183932559376533e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.735036880591229e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735036880591229e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.0002463643756958201) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.0002463643756958201) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00044585351288409627) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.00044585351288409627) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.00044585351288409627) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.00044585351288409627) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940221543005354) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005354) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005354) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005354) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005354) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005354) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940221543005354) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005354) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533856254125501) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533856254125501) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533856254125501) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533856254125501) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0010435246534907555) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435246534907555) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435246534907555) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435246534907555) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.001280306097349664) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.001280306097349664) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.001280306097349664) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.001280306097349664) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.0013038004788126969) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.0013038004788126969) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.0013038004788126969) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.0013038004788126969) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.002261966062482357) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.002261966062482357) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.002261966062482357) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.002261966062482357) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.002261966062482357) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.002261966062482357) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.002261966062482357) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.002261966062482357) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.00398984145661933) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.00398984145661933) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.00398984145661933) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.00398984145661933) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.004158797381840032) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.004158797381840032) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.004311038507914308) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.004311038507914308) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.004311038507914308) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.004311038507914308) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.0046369766611825515) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.0046369766611825515) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.0046369766611825515) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.0046369766611825515) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.005114473831660395) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.005114473831660395) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.005114473831660395) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.005114473831660395) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.005114473831660395) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660395) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660395) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005114473831660395) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005241535382803879) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.005241535382803879) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.005241535382803879) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.005241535382803879) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.005262642473076834) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262642473076834) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262642473076834) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262642473076834) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368659358109504) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368659358109504) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005379937155839376) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.005379937155839376) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.005379937155839376) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.005379937155839376) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.005652620978017345) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.005652620978017345) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.00570849598596093) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.00570849598596093) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.00570849598596093) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.00570849598596093) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.005923798336561349) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.005923798336561349) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.0073067599288329545) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.0073067599288329545) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469978791023864) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.008469978791023864) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.009841749246962607) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.009841749246962607) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.011285190200840886) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.011285190200840886) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.011756013419819238) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.011756013419819238) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.014564531231173017) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564531231173017) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603704729162146) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.014603704729162146) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.015225630757226612) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225630757226612) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.016024603689179552) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.016024603689179552) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.019028242443847338) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.019028242443847338) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.019257505095251617) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.019257505095251617) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.36937089366156145) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.36937089366156145) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.36937089366156134) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.36937089366156134) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.2816425776702296) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.2816425776702296) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.2816425776702295) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.2816425776702295) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0906514420703649) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0906514420703649) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0906514420703649) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0906514420703649) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863634) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684737589863634) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863634) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684737589863634) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635021950635021) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635021950635021) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635021950635021) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635021950635021) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214032) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.06752385099214032) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214032) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.06752385099214032) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.035608378988312483) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.035608378988312483) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.024591860883830006) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.024591860883830006) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883830006) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.024591860883830006) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.024282117354692972) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.024282117354692972) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.02314513092952906) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.02314513092952906) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.02252844019601307) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.02252844019601307) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019538050311314736) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.019538050311314736) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.019538050311314736) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.019538050311314736) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.017091553155898838) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.017091553155898838) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.017091553155898838) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.017091553155898838) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.016024603689179552) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.016024603689179552) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.016024603689179552) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.016024603689179552) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.010311482489831754) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831754) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831754) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311482489831754) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.009841749246962607) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962607) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.009841749246962607) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962607) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209875) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209875) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209875) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008826368514209875) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.00854199662545485) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00854199662545485) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00854199662545485) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00854199662545485) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.00854199662545485) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00854199662545485) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00854199662545485) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00854199662545485) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008469978791023864) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023864) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.008469978791023864) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023864) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.004668620318776299) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668620318776299) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.003876470899336959) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.003876470899336959) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.003804066171728548) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728548) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728548) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003804066171728548) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484157300217887) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484157300217887) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0033566705638328875) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.0033566705638328875) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.0032675138544235524) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.0032675138544235524) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.002141361223101581) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.002141361223101581) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.001727875394136973) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.001727875394136973) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.0016407548553124159) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0016407548553124159) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214168855) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0014528843214168855) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214168855) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0014528843214168855) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870896771024523) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007870896771024523) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192743499487834) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192743499487834) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.00019400857029756608) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00019400857029756608) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303549664) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.00013840177303549664) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.141625221154563e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.141625221154563e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.141625221154563e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.141625221154563e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.07148073642053e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.07148073642053e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.1513463111600517e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.1513463111600517e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0882507113704623e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.0882507113704623e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.9885117063531888e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.9885117063531888e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8742990714107655e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.8742990714107655e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.3609563203746747e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.3609563203746747e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.300294656272283e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.300294656272283e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.146837650761164e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.146837650761164e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.146837650761164e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.146837650761164e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.352332103260146e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.352332103260146e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.352332103260146e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.352332103260146e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091637199096191e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637199096191e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637199096191e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637199096191e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637199096191e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637199096191e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637199096191e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091637199096191e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.074305986060878e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.074305986060878e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.074305986060878e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.074305986060878e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128986394362e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128986394362e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128986394362e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128986394362e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867765104364124e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.867765104364124e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.56069246501884e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.56069246501884e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.56069246501884e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.56069246501884e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.56069246501884e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.56069246501884e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.56069246501884e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.56069246501884e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.997018422167419e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.997018422167419e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.997018422167419e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.997018422167419e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.997018422167419e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.997018422167419e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.997018422167419e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.997018422167419e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.568247521217284e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.568247521217284e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.568247521217284e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.568247521217284e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393085154524e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393085154524e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393085154524e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393085154524e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393085154524e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393085154524e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.3767393085154524e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393085154524e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.888293595171888e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.888293595171888e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.686381546034053e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.686381546034053e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035783554099775e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7035783554099775e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.209350643317303e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.209350643317303e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244437126e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773244437126e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244437126e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773244437126e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244437126e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773244437126e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773244437126e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773244437126e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.974225379594075e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.974225379594075e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.974225379594075e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.974225379594075e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.047471655582987e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.047471655582987e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.047471655582987e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.047471655582987e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.209350643317303e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.209350643317303e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0717282184461648e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0717282184461648e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0717282184461648e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0717282184461648e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.2004287493841073e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.2004287493841073e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.2004287493841073e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.2004287493841073e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.7035783554099775e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7035783554099775e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.3120943052436862e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.3120943052436862e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.3120943052436862e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.3120943052436862e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.686381546034053e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.686381546034053e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.888293595171888e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.888293595171888e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.092250616152609e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.092250616152609e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.092250616152609e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.092250616152609e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.092250616152609e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.092250616152609e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.092250616152609e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.092250616152609e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.444597854153451e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.444597854153451e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.444597854153451e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.444597854153451e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.6849150952760566e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.6849150952760566e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.6849150952760566e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.6849150952760566e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.24697442547991e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.24697442547991e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.24697442547991e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.24697442547991e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.24697442547991e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.24697442547991e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.24697442547991e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.24697442547991e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.867765104364124e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.867765104364124e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.300294656272283e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.300294656272283e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.3609563203746747e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.3609563203746747e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.8742990714107655e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.8742990714107655e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.8836765760891847e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.8836765760891847e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.94735601172216e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.94735601172216e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.94735601172216e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.94735601172216e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117063531888e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9885117063531888e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.0882507113704623e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.0882507113704623e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.1513463111600517e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.1513463111600517e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.846201671302226e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846201671302226e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846201671302226e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846201671302226e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.07148073642053e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.07148073642053e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.105526722043197e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.105526722043197e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.105526722043197e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.105526722043197e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146496327574509e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146496327574509e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146496327574509e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146496327574509e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159350501944748e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159350501944748e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159350501944748e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159350501944748e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427988656548153e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427988656548153e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427988656548153e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427988656548153e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935867718075349e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935867718075349e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935867718075349e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935867718075349e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.2532733481428165e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.2532733481428165e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979825793453962e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979825793453962e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979825793453962e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979825793453962e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.205548411216446e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.205548411216446e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.205548411216446e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.205548411216446e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00013840177303549664) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.00013840177303549664) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.00018787053389553027) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.00018787053389553027) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.00018787053389553027) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00018787053389553027) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00019400857029756608) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00019400857029756608) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0002463643756958201) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0002463643756958201) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0002463643756958201) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0002463643756958201) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192743499487834) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192743499487834) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156734248909066) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156734248909066) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156734248909066) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156734248909066) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024523) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007870896771024523) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.0015324835230730602) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.0015324835230730602) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.0015324835230730602) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.0015324835230730602) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.0016407548553124159) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0016407548553124159) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.001727875394136973) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.001727875394136973) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.002446497155415898) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.002446497155415898) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.002446497155415898) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.002446497155415898) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.0032675138544235524) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.0032675138544235524) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.0033566705638328875) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.0033566705638328875) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.003484157300217887) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.003484157300217887) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.003876470899336959) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.003876470899336959) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.004668620318776299) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668620318776299) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278134) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.004767272188278134) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.004767272188278134) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278134) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.005286546538226917) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.005286546538226917) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.005286546538226917) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.005286546538226917) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.00540895442241002) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.00540895442241002) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.00540895442241002) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.00540895442241002) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.005923798336561349) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561349) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.005923798336561349) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561349) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.010715508469796766) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010715508469796766) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010715508469796766) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010715508469796766) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.010757563953908929) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010757563953908929) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010757563953908929) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010757563953908929) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.014603704729162146) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.014603704729162146) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.014603704729162146) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.014603704729162146) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.01929956057936378) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01929956057936378) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01929956057936378) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01929956057936378) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01929956057936378) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01929956057936378) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.01929956057936378) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01929956057936378) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.05859198873386222) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.05859198873386222) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.775950527360815e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.775950527360815e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.775950527360815e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.775950527360815e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.07165035181002735) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07165035181002735) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0716503518100274) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0716503518100274) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251617) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019257505095251617) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010311482489831754) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311482489831754) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008826368514209875) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209875) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.007597464029770612) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.007597464029770612) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.007597464029770612) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.007597464029770612) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311881) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311881) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311881) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311881) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311881) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311881) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311881) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311881) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003804066171728548) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003804066171728548) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984166168121942) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.002984166168121942) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.002984166168121942) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.002984166168121942) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.0024464971554158986) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.0024464971554158986) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.002249412447093995) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.002249412447093995) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.002249412447093995) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002249412447093995) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.002141361223101581) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.002141361223101581) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.0018638942824587305) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587305) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587305) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587305) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587305) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587305) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0018638942824587305) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587305) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0016407548553124156) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124156) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0016407548553124156) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124156) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0012223378081538366) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538366) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538366) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538366) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538366) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538366) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538366) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0012223378081538366) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0010283292378562702) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0010283292378562702) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0010283292378562702) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0010283292378562702) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1463061453108639e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1463061453108639e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.8742990714107655e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990714107655e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.8742990714107655e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990714107655e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.300294656272283e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.300294656272283e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.300294656272283e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.300294656272283e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.0444941297953128e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.0444941297953128e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.0444941297953128e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.0444941297953128e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.956079229946106e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.956079229946106e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.956079229946106e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.956079229946106e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105515037080542e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105515037080542e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105515037080542e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105515037080542e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661347213050225e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661347213050225e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661347213050225e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661347213050225e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341413757459e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341413757459e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189990975414762e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.189990975414762e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876621658097278e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876621658097278e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876621658097278e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876621658097278e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246207033744e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246207033744e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.5233896778789067e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.5233896778789067e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0767325319271344e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.0767325319271344e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0767325319271344e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.0767325319271344e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.0134714589125093e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0134714589125093e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.9045998841956705e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.9045998841956705e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.9045998841956705e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.9045998841956705e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.6667317545958676e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.6667317545958676e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.6667317545958676e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6667317545958676e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641928655638e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.8505641928655638e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309317479104e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6569309317479104e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309317479104e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6569309317479104e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.8505641928655638e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.8505641928655638e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815460340534e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815460340534e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.6863815460340534e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815460340534e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714589125093e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.0134714589125093e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.5233896778789067e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.5233896778789067e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.6704023906604194e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.6704023906604194e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.6704023906604194e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.6704023906604194e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246207033744e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246207033744e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189990975414762e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.189990975414762e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341413757459e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341413757459e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949476487537631e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949476487537631e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.792493957713563e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.792493957713563e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.792493957713563e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.792493957713563e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.8836765760891847e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.8836765760891847e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.9885117063531888e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117063531888e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117063531888e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117063531888e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348142816e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253273348142816e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4017109735337636e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4017109735337636e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4017109735337636e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4017109735337636e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.58096036930512e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.58096036930512e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.58096036930512e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.58096036930512e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192743499487834) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487834) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192743499487834) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487834) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.0007870896771024523) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024523) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024523) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024523) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0011726348316441874) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0011726348316441874) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0011726348316441874) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0011726348316441874) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012366478019245502) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.0012366478019245502) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.0012366478019245502) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.0012366478019245502) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.002200964069500458) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002200964069500458) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002200964069500458) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002200964069500458) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980236) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980236) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0023949726397980236) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980236) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0023949726397980236) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0023949726397980236) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0023949726397980236) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0023949726397980236) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0024464971554158986) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.0024464971554158986) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.003804066171728548) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.003804066171728548) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00387647089933696) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.00387647089933696) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.00387647089933696) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.00387647089933696) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.004220813970046492) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.004220813970046492) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.004220813970046492) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.004220813970046492) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.008826368514209875) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.008826368514209875) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.010311482489831754) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311482489831754) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257505095251617) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019257505095251617) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.05859198873386222) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.05859198873386222) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3987009015216433e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3987009015216433e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.398700901521643e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.398700901521643e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484157300217887) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003484157300217887) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984166168121942) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.002984166168121942) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.00019400857029756608) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00019400857029756608) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453108639e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453108639e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7924939577135632e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7924939577135632e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540341413757459e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413757459e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540341413757459e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413757459e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.8505641928655638e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641928655638e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641928655638e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641928655638e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.01347145891251e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.01347145891251e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.01347145891251e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.01347145891251e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949476487537631e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949476487537631e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.7924939577135632e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7924939577135632e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756608) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756608) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002984166168121942) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.002984166168121942) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.003484157300217887) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.003484157300217887) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
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
Cost function value: -0.544996018090293
   (-46.46390678868899) [I0]
+ (0.7829661725950187) [Z10]
+ (0.7829661725950188) [Z11]
+ (0.8084581961720481) [Z12]
+ (0.8084581961720482) [Z13]
+ (1.2034402289145611) [Z4]
+ (1.2034402289145614) [Z5]
+ (1.3096862988615416) [Z7]
+ (1.3096862988615419) [Z6]
+ (1.3693525634718173) [Z8]
+ (1.3693525634718176) [Z9]
+ (1.6538942226831705) [Z2]
+ (1.6538942226831705) [Z3]
+ (12.412630742111787) [Z0]
+ (12.412630742111787) [Z1]
+ (-8.194261373215782e-06) [Y10 Y12]
+ (-8.194261373215782e-06) [X10 X12]
+ (-1.8540608577689437e-06) [Y5 Y7]
+ (-1.8540608577689437e-06) [X5 X7]
+ (-7.764994117078864e-07) [Y3 Y5]
+ (-7.764994117078864e-07) [X3 X5]
+ (-5.929765815194202e-07) [Y4 Y6]
+ (-5.929765815194202e-07) [X4 X6]
+ (1.602116740542133e-06) [Y2 Y4]
+ (1.602116740542133e-06) [X2 X4]
+ (7.95441317720968e-06) [Y11 Y13]
+ (7.95441317720968e-06) [X11 X13]
+ (0.0032769719312316604) [Y1 Y3]
+ (0.0032769719312316604) [X1 X3]
+ (0.10433064780651427) [Y0 Y2]
+ (0.10433064780651427) [X0 X2]
+ (0.11270386920332218) [Z10 Z12]
+ (0.11270386920332218) [Z11 Z13]
+ (0.11383573679388662) [Z4 Z12]
+ (0.11383573679388662) [Z5 Z13]
+ (0.11952438964682674) [Z6 Z10]
+ (0.11952438964682674) [Z7 Z11]
+ (0.12489990917237592) [Z4 Z10]
+ (0.12489990917237592) [Z5 Z11]
+ (0.1249580773950321) [Z2 Z4]
+ (0.1249580773950321) [Z3 Z5]
+ (0.12799502492468406) [Z2 Z10]
+ (0.12799502492468406) [Z3 Z11]
+ (0.13401715261963712) [Z6 Z12]
+ (0.13401715261963712) [Z7 Z13]
+ (0.1370119167404074) [Z4 Z6]
+ (0.1370119167404074) [Z5 Z7]
+ (0.13734953064261315) [Z6 Z11]
+ (0.13734953064261315) [Z7 Z10]
+ (0.13739104762683227) [Z2 Z6]
+ (0.13739104762683227) [Z3 Z7]
+ (0.1376687264585257) [Z8 Z10]
+ (0.1376687264585257) [Z9 Z11]
+ (0.14011289865354817) [Z2 Z12]
+ (0.14011289865354817) [Z3 Z13]
+ (0.14138905291942816) [Z10 Z13]
+ (0.14138905291942816) [Z11 Z12]
+ (0.14257997712485748) [Z4 Z11]
+ (0.14257997712485748) [Z5 Z10]
+ (0.14722943218766166) [Z8 Z11]
+ (0.14722943218766166) [Z9 Z10]
+ (0.14899430575065536) [Z4 Z7]
+ (0.14899430575065536) [Z5 Z6]
+ (0.14926355147388906) [Z10 Z11]
+ (0.1496070268444529) [Z4 Z8]
+ (0.1496070268444529) [Z5 Z9]
+ (0.1497348680349693) [Z8 Z12]
+ (0.1497348680349693) [Z9 Z13]
+ (0.1507140812100829) [Z2 Z8]
+ (0.1507140812100829) [Z3 Z9]
+ (0.15138327161428847) [Z6 Z13]
+ (0.15138327161428847) [Z7 Z12]
+ (0.15215040708869046) [Z4 Z13]
+ (0.15215040708869046) [Z5 Z12]
+ (0.15337968243314148) [Z2 Z11]
+ (0.15337968243314148) [Z3 Z10]
+ (0.1543574865722364) [Z12 Z13]
+ (0.15569010671752465) [Z2 Z13]
+ (0.15569010671752465) [Z3 Z12]
+ (0.15582269051553116) [Z8 Z13]
+ (0.15582269051553116) [Z9 Z12]
+ (0.15676396176430984) [Z4 Z9]
+ (0.15676396176430984) [Z5 Z8]
+ (0.15755314797985648) [Z4 Z5]
+ (0.16079764534838553) [Z2 Z5]
+ (0.16079764534838553) [Z3 Z4]
+ (0.1675665326546127) [Z6 Z8]
+ (0.1675665326546127) [Z7 Z9]
+ (0.1814399144030388) [Z6 Z9]
+ (0.1814399144030388) [Z7 Z8]
+ (0.18189085790751366) [Z2 Z3]
+ (0.18690820476912554) [Z2 Z9]
+ (0.18690820476912554) [Z3 Z8]
+ (0.19299723935364246) [Z0 Z10]
+ (0.19299723935364246) [Z1 Z11]
+ (0.19392534613270204) [Z6 Z7]
+ (0.19661770890342153) [Z0 Z4]
+ (0.19661770890342153) [Z1 Z5]
+ (0.19936354537360834) [Z0 Z5]
+ (0.19936354537360834) [Z1 Z4]
+ (0.20072866460441774) [Z0 Z11]
+ (0.20072866460441774) [Z1 Z10]
+ (0.21102659849791555) [Z0 Z12]
+ (0.21102659849791555) [Z1 Z13]
+ (0.2163103749863185) [Z0 Z13]
+ (0.2163103749863185) [Z1 Z12]
+ (0.23671080783830453) [Z0 Z2]
+ (0.23671080783830453) [Z1 Z3]
+ (0.2512944567459172) [Z0 Z3]
+ (0.2512944567459172) [Z1 Z2]
+ (0.27232518306605713) [Z0 Z8]
+ (0.27232518306605713) [Z1 Z9]
+ (0.2788345442672344) [Z0 Z9]
+ (0.2788345442672344) [Z1 Z8]
+ (1.1861763734860524) [Z0 Z1]
+ (-1.2260484987843362e-05) [Y5 Z6 Y7]
+ (-1.2260484987843362e-05) [X5 Z6 X7]
+ (-1.226048498784336e-05) [Y4 Z5 Y6]
+ (-1.226048498784336e-05) [X4 Z5 X6]
+ (-1.0722312157811711e-05) [Y11 Z12 Y13]
+ (-1.0722312157811711e-05) [X11 Z12 X13]
+ (-1.0722312157811708e-05) [Y10 Z11 Y12]
+ (-1.0722312157811708e-05) [X10 Z11 X12]
+ (-3.8870516713760455e-06) [Y2 Z3 Y4]
+ (-3.8870516713760455e-06) [X2 Z3 X4]
+ (-3.887051671376044e-06) [Y3 Z4 Y5]
+ (-3.887051671376044e-06) [X3 Z4 X5]
+ (0.1250703257977198) [Y1 Z2 Y3]
+ (0.1250703257977198) [X1 Z2 X3]
+ (0.12507032579771984) [Y0 Z1 Y2]
+ (0.12507032579771984) [X0 Z1 X2]
+ (-0.03831467029480385) [Y4 Y5 X12 X13]
+ (-0.03831467029480385) [X4 X5 Y12 Y13]
+ (-0.03619412355904262) [Y2 Y3 X8 X9]
+ (-0.03619412355904262) [X2 X3 Y8 Y9]
+ (-0.035839567953353434) [Y2 Y3 X4 X5]
+ (-0.035839567953353434) [X2 X3 Y4 Y5]
+ (-0.03114381798896714) [Y2 Y3 X6 X7]
+ (-0.03114381798896714) [X2 X3 Y6 Y7]
+ (-0.028685183716106004) [Y10 Y11 X12 X13]
+ (-0.028685183716106004) [X10 X11 Y12 Y13]
+ (-0.025996177598021142) [Y3 Z4 Z5 Y7]
+ (-0.025996177598021142) [X3 Z4 Z5 X7]
+ (-0.025384657508457396) [Y2 Y3 X10 X11]
+ (-0.025384657508457396) [X2 X3 Y10 Y11]
+ (-0.019028242443847314) [Y3 Y4 X11 X12]
+ (-0.019028242443847314) [X3 X4 Y11 Y12]
+ (-0.017825140995786404) [Y6 Y7 X10 X11]
+ (-0.017825140995786404) [X6 X7 Y10 Y11]
+ (-0.017680067952481563) [Y4 Y5 X10 X11]
+ (-0.017680067952481563) [X4 X5 Y10 Y11]
+ (-0.01736611899465135) [Y6 Y7 X12 X13]
+ (-0.01736611899465135) [X6 X7 Y12 Y13]
+ (-0.01557720806397647) [Y2 Y3 X12 X13]
+ (-0.01557720806397647) [X2 X3 Y12 Y13]
+ (-0.014583648907612694) [Y0 Y1 X2 X3]
+ (-0.014583648907612694) [X0 X1 Y2 Y3]
+ (-0.013873381748426089) [Y6 Y7 X8 X9]
+ (-0.013873381748426089) [X6 X7 Y8 Y9]
+ (-0.011982389010247924) [Y4 Y5 X6 X7]
+ (-0.011982389010247924) [X4 X5 Y6 Y7]
+ (-0.011285190200840893) [Y5 X6 X11 Y12]
+ (-0.011285190200840893) [X5 Y6 Y11 X12]
+ (-0.00956070572913595) [Y8 Y9 X10 X11]
+ (-0.00956070572913595) [X8 X9 Y10 Y11]
+ (-0.008125251921381034) [Y1 X2 X8 Y9]
+ (-0.008125251921381034) [Y1 Y2 Y8 Y9]
+ (-0.008125251921381034) [X1 X2 X8 X9]
+ (-0.008125251921381034) [X1 Y2 Y8 X9]
+ (-0.007731425250775277) [Y0 Y1 X10 X11]
+ (-0.007731425250775277) [X0 X1 Y10 Y11]
+ (-0.0071569349198569365) [Y4 Y5 X8 X9]
+ (-0.0071569349198569365) [X4 X5 Y8 Y9]
+ (-0.006888194352970571) [Y0 Y1 X6 X7]
+ (-0.006888194352970571) [X0 X1 Y6 Y7]
+ (-0.0065093612011772415) [Y0 Y1 X8 X9]
+ (-0.0065093612011772415) [X0 X1 Y8 Y9]
+ (-0.006087822480561862) [Y8 Y9 X12 X13]
+ (-0.006087822480561862) [X8 X9 Y12 Y13]
+ (-0.005283776488402968) [Y0 Y1 X12 X13]
+ (-0.005283776488402968) [X0 X1 Y12 Y13]
+ (-0.005143391768825112) [Y3 X4 X5 Y6]
+ (-0.005143391768825112) [X3 Y4 Y5 X6]
+ (-0.004684903388155221) [Y1 X2 X6 Y7]
+ (-0.004684903388155221) [Y1 Y2 Y6 Y7]
+ (-0.004684903388155221) [X1 X2 X6 X7]
+ (-0.004684903388155221) [X1 Y2 Y6 X7]
+ (-0.004575007626639209) [Y1 X2 X12 Y13]
+ (-0.004575007626639209) [Y1 Y2 Y12 Y13]
+ (-0.004575007626639209) [X1 X2 X12 X13]
+ (-0.004575007626639209) [X1 Y2 Y12 X13]
+ (-0.004424855449441851) [Y1 X2 X4 Y5]
+ (-0.004424855449441851) [Y1 Y2 Y4 Y5]
+ (-0.004424855449441851) [X1 X2 X4 X5]
+ (-0.004424855449441851) [X1 Y2 Y4 X5]
+ (-0.0034795118903342658) [Y2 Z3 Z5 Y6]
+ (-0.0034795118903342658) [X2 Z3 Z5 X6]
+ (-0.0034795118903342658) [Y3 Z4 Z6 Y7]
+ (-0.0034795118903342658) [X3 Z4 Z6 X7]
+ (-0.0027458364701868094) [Y0 Y1 X4 X5]
+ (-0.0027458364701868094) [X0 X1 Y4 Y5]
+ (-0.0017992194936630125) [Y1 X2 X10 Y11]
+ (-0.0017992194936630125) [Y1 Y2 Y10 Y11]
+ (-0.0017992194936630125) [X1 X2 X10 X11]
+ (-0.0017992194936630125) [X1 Y2 Y10 X11]
+ (-0.00029219862611106357) [Y7 Y8 X9 X10]
+ (-0.00029219862611106357) [X7 X8 Y9 Y10]
+ (-8.194261373215782e-06) [Z10 Y11 Z12 Y13]
+ (-8.194261373215782e-06) [Z10 X11 Z12 X13]
+ (-7.801707501533473e-06) [Y2 Z3 Y4 Z11]
+ (-7.801707501533473e-06) [X2 Z3 X4 Z11]
+ (-7.801707501533473e-06) [Y3 Z4 Y5 Z10]
+ (-7.801707501533473e-06) [X3 Z4 X5 Z10]
+ (-4.643051069093798e-06) [Y3 X4 X10 Y11]
+ (-4.643051069093798e-06) [Y3 Y4 Y10 Y11]
+ (-4.643051069093798e-06) [X3 X4 X10 X11]
+ (-4.643051069093798e-06) [X3 Y4 Y10 X11]
+ (-4.588855156135963e-06) [Y4 Z5 Y6 Z13]
+ (-4.588855156135963e-06) [X4 Z5 X6 Z13]
+ (-4.588855156135963e-06) [Y5 Z6 Y7 Z12]
+ (-4.588855156135963e-06) [X5 Z6 X7 Z12]
+ (-4.556569218860123e-06) [Y5 X6 X12 Y13]
+ (-4.556569218860123e-06) [Y5 Y6 Y12 Y13]
+ (-4.556569218860123e-06) [X5 X6 X12 X13]
+ (-4.556569218860123e-06) [X5 Y6 Y12 X13]
+ (-3.6945132949405883e-06) [Y4 X5 X11 Y12]
+ (-3.6945132949405883e-06) [Y4 Y5 Y11 Y12]
+ (-3.6945132949405883e-06) [X4 X5 X11 X12]
+ (-3.6945132949405883e-06) [X4 Y5 Y11 X12]
+ (-3.344081556165591e-06) [Z0 Y5 Z6 Y7]
+ (-3.344081556165591e-06) [Z0 X5 Z6 X7]
+ (-3.344081556165591e-06) [Z1 Y4 Z5 Y6]
+ (-3.344081556165591e-06) [Z1 X4 Z5 X6]
+ (-3.1586564324396764e-06) [Y2 Z3 Y4 Z10]
+ (-3.1586564324396764e-06) [X2 Z3 X4 Z10]
+ (-3.1586564324396764e-06) [Y3 Z4 Y5 Z11]
+ (-3.1586564324396764e-06) [X3 Z4 X5 Z11]
+ (-3.0993492433138292e-06) [Z0 Y4 Z5 Y6]
+ (-3.0993492433138292e-06) [Z0 X4 Z5 X6]
+ (-3.0993492433138292e-06) [Z1 Y5 Z6 Y7]
+ (-3.0993492433138292e-06) [Z1 X5 Z6 X7]
+ (-2.890967881838843e-06) [Z6 Y11 Z12 Y13]
+ (-2.890967881838843e-06) [Z6 X11 Z12 X13]
+ (-2.890967881838843e-06) [Z7 Y10 Z11 Y12]
+ (-2.890967881838843e-06) [Z7 X10 Z11 X12]
+ (-2.177664605310716e-06) [Z0 Y10 Z11 Y12]
+ (-2.177664605310716e-06) [Z0 X10 Z11 X12]
+ (-2.177664605310716e-06) [Z1 Y11 Z12 Y13]
+ (-2.177664605310716e-06) [Z1 X11 Z12 X13]
+ (-1.8818501830277464e-06) [Y4 Z5 Y6 Z9]
+ (-1.8818501830277464e-06) [X4 Z5 X6 Z9]
+ (-1.8818501830277464e-06) [Y5 Z6 Y7 Z8]
+ (-1.8818501830277464e-06) [X5 Z6 X7 Z8]
+ (-1.8551201218160778e-06) [Z6 Y10 Z11 Y12]
+ (-1.8551201218160778e-06) [Z6 X10 Z11 X12]
+ (-1.8551201218160778e-06) [Z7 Y11 Z12 Y13]
+ (-1.8551201218160778e-06) [Z7 X11 Z12 X13]
+ (-1.8540608577689437e-06) [Y4 Z5 Y6 Z7]
+ (-1.8540608577689437e-06) [X4 Z5 X6 Z7]
+ (-1.816303169841323e-06) [Z4 Y11 Z12 Y13]
+ (-1.816303169841323e-06) [Z4 X11 Z12 X13]
+ (-1.816303169841323e-06) [Z5 Y10 Z11 Y12]
+ (-1.816303169841323e-06) [Z5 X10 Z11 X12]
+ (-1.6923978286999178e-06) [Y4 Z5 Y6 Z10]
+ (-1.6923978286999178e-06) [X4 Z5 X6 Z10]
+ (-1.6923978286999178e-06) [Y5 Z6 Y7 Z11]
+ (-1.6923978286999178e-06) [X5 Z6 X7 Z11]
+ (-1.6148794141197644e-06) [Z0 Y11 Z12 Y13]
+ (-1.6148794141197644e-06) [Z0 X11 Z12 X13]
+ (-1.6148794141197644e-06) [Z1 Y10 Z11 Y12]
+ (-1.6148794141197644e-06) [Z1 X10 Z11 X12]
+ (-1.5973171980101626e-06) [Z8 Y10 Z11 Y12]
+ (-1.5973171980101626e-06) [Z8 X10 Z11 X12]
+ (-1.5973171980101626e-06) [Z9 Y11 Z12 Y13]
+ (-1.5973171980101626e-06) [Z9 X11 Z12 X13]
+ (-1.4548424488846644e-06) [Y3 X4 X6 Y7]
+ (-1.4548424488846644e-06) [Y3 Y4 Y6 Y7]
+ (-1.4548424488846644e-06) [X3 X4 X6 X7]
+ (-1.4548424488846644e-06) [X3 Y4 Y6 X7]
+ (-1.3980449079928253e-06) [Y4 Z5 Y6 Z8]
+ (-1.3980449079928253e-06) [X4 Z5 X6 Z8]
+ (-1.3980449079928253e-06) [Y5 Z6 Y7 Z9]
+ (-1.3980449079928253e-06) [X5 Z6 X7 Z9]
+ (-1.195489009756221e-06) [Y2 Z3 Y4 Z7]
+ (-1.195489009756221e-06) [X2 Z3 X4 Z7]
+ (-1.195489009756221e-06) [Y3 Z4 Y5 Z6]
+ (-1.195489009756221e-06) [X3 Z4 X5 Z6]
+ (-1.1908508081036334e-06) [Z0 Y3 Z4 Y5]
+ (-1.1908508081036334e-06) [Z0 X3 Z4 X5]
+ (-1.1908508081036334e-06) [Z1 Y2 Z3 Y4]
+ (-1.1908508081036334e-06) [Z1 X2 Z3 X4]
+ (-1.1708301369282445e-06) [Z2 Y5 Z6 Y7]
+ (-1.1708301369282445e-06) [Z2 X5 Z6 X7]
+ (-1.1708301369282445e-06) [Z3 Y4 Z5 Y6]
+ (-1.1708301369282445e-06) [Z3 X4 Z5 X6]
+ (-1.0632283426204833e-06) [Z2 Y10 Z11 Y12]
+ (-1.0632283426204833e-06) [Z2 X10 Z11 X12]
+ (-1.0632283426204833e-06) [Z3 Y11 Z12 Y13]
+ (-1.0632283426204833e-06) [Z3 X11 Z12 X13]
+ (-1.0358477600227656e-06) [Y6 X7 X11 Y12]
+ (-1.0358477600227656e-06) [Y6 Y7 Y11 Y12]
+ (-1.0358477600227656e-06) [X6 X7 X11 X12]
+ (-1.0358477600227656e-06) [X6 Y7 Y11 X12]
+ (-9.509249750803807e-07) [Z2 Y4 Z5 Y6]
+ (-9.509249750803807e-07) [Z2 X4 Z5 X6]
+ (-9.509249750803807e-07) [Z3 Y5 Z6 Y7]
+ (-9.509249750803807e-07) [Z3 X5 Z6 X7]
+ (-9.344557777648903e-07) [Z8 Y11 Z12 Y13]
+ (-9.344557777648903e-07) [Z8 X11 Z12 X13]
+ (-9.344557777648903e-07) [Z9 Y10 Z11 Y12]
+ (-9.344557777648903e-07) [Z9 X10 Z11 X12]
+ (-8.337746752378834e-07) [Z0 Y2 Z3 Y4]
+ (-8.337746752378834e-07) [Z0 X2 Z3 X4]
+ (-8.337746752378834e-07) [Z1 Y3 Z4 Y5]
+ (-8.337746752378834e-07) [Z1 X3 Z4 X5]
+ (-7.956895371441828e-07) [Y3 X4 X8 Y9]
+ (-7.956895371441828e-07) [Y3 Y4 Y8 Y9]
+ (-7.956895371441828e-07) [X3 X4 X8 X9]
+ (-7.956895371441828e-07) [X3 Y4 Y8 X9]
+ (-7.764994117078864e-07) [Y2 Z3 Y4 Z5]
+ (-7.764994117078864e-07) [X2 Z3 X4 Z5]
+ (-5.929765815194203e-07) [Z4 Y5 Z6 Y7]
+ (-5.929765815194203e-07) [Z4 X5 Z6 X7]
+ (-5.7700529933912e-07) [Y2 Z3 Y4 Z9]
+ (-5.7700529933912e-07) [X2 Z3 X4 Z9]
+ (-5.7700529933912e-07) [Y3 Z4 Y5 Z8]
+ (-5.7700529933912e-07) [X3 Z4 X5 Z8]
+ (-5.471647745138482e-07) [Y1 Y2 X11 X12]
+ (-5.471647745138482e-07) [X1 X2 Y11 Y12]
+ (-4.83805275034921e-07) [Y5 X6 X8 Y9]
+ (-4.83805275034921e-07) [Y5 Y6 Y8 Y9]
+ (-4.83805275034921e-07) [X5 X6 X8 X9]
+ (-4.83805275034921e-07) [X5 Y6 Y8 X9]
+ (-3.570761328657502e-07) [Y0 X1 X3 Y4]
+ (-3.570761328657502e-07) [Y0 Y1 Y3 Y4]
+ (-3.570761328657502e-07) [X0 X1 X3 X4]
+ (-3.570761328657502e-07) [X0 Y1 Y3 X4]
+ (-2.4473231285176163e-07) [Y0 X1 X5 Y6]
+ (-2.4473231285176163e-07) [Y0 Y1 Y5 Y6]
+ (-2.4473231285176163e-07) [X0 X1 X5 X6]
+ (-2.4473231285176163e-07) [X0 Y1 Y5 X6]
+ (-2.199051618478637e-07) [Y2 X3 X5 Y6]
+ (-2.199051618478637e-07) [Y2 Y3 Y5 Y6]
+ (-2.199051618478637e-07) [X2 X3 X5 X6]
+ (-2.199051618478637e-07) [X2 Y3 Y5 X6]
+ (-1.9332412768653423e-07) [Y1 X2 X3 Y4]
+ (-1.9332412768653423e-07) [X1 Y2 Y3 X4]
+ (-1.2919694860340715e-07) [Y1 Z2 Z3 Y5]
+ (-1.2919694860340715e-07) [X1 Z2 Z3 X5]
+ (1.7379332622189502e-07) [Y0 Z1 Z3 Y4]
+ (1.7379332622189502e-07) [X0 Z1 Z3 X4]
+ (1.7379332622189502e-07) [Y1 Z2 Z4 Y5]
+ (1.7379332622189502e-07) [X1 Z2 Z4 X5]
+ (1.9332412768653423e-07) [Y1 Y2 X3 X4]
+ (1.9332412768653423e-07) [X1 X2 Y3 Y4]
+ (2.1868423780506266e-07) [Y2 Z3 Y4 Z8]
+ (2.1868423780506266e-07) [X2 Z3 X4 Z8]
+ (2.1868423780506266e-07) [Y3 Z4 Y5 Z9]
+ (2.1868423780506266e-07) [X3 Z4 X5 Z9]
+ (2.5935343912844374e-07) [Y2 Z3 Y4 Z6]
+ (2.5935343912844374e-07) [X2 Z3 X4 Z6]
+ (2.5935343912844374e-07) [Y3 Z4 Y5 Z7]
+ (2.5935343912844374e-07) [X3 Z4 X5 Z7]
+ (3.606071867535166e-07) [Y0 Z1 Z2 Y4]
+ (3.606071867535166e-07) [X0 Z1 Z2 X4]
+ (3.606071867535166e-07) [Y1 Z3 Z4 Y5]
+ (3.606071867535166e-07) [X1 Z3 Z4 X5]
+ (5.471647745138482e-07) [Y1 X2 X11 Y12]
+ (5.471647745138482e-07) [X1 Y2 Y11 X12]
+ (5.627851911909516e-07) [Y0 X1 X11 Y12]
+ (5.627851911909516e-07) [Y0 Y1 Y11 Y12]
+ (5.627851911909516e-07) [X0 X1 X11 X12]
+ (5.627851911909516e-07) [X0 Y1 Y11 X12]
+ (6.628614202452722e-07) [Y8 X9 X11 Y12]
+ (6.628614202452722e-07) [Y8 Y9 Y11 Y12]
+ (6.628614202452722e-07) [X8 X9 X11 X12]
+ (6.628614202452722e-07) [X8 Y9 Y11 X12]
+ (1.1094407590392819e-06) [Z2 Y11 Z12 Y13]
+ (1.1094407590392819e-06) [Z2 X11 Z12 X13]
+ (1.1094407590392819e-06) [Z3 Y10 Z11 Y12]
+ (1.1094407590392819e-06) [Z3 X10 Z11 X12]
+ (1.602116740542133e-06) [Z2 Y3 Z4 Y5]
+ (1.602116740542133e-06) [Z2 X3 Z4 X5]
+ (1.8782101250992658e-06) [Z4 Y10 Z11 Y12]
+ (1.8782101250992658e-06) [Z4 X10 Z11 X12]
+ (1.8782101250992658e-06) [Z5 Y11 Z12 Y13]
+ (1.8782101250992658e-06) [Z5 X11 Z12 X13]
+ (2.1726691016597654e-06) [Y2 X3 X11 Y12]
+ (2.1726691016597654e-06) [Y2 Y3 Y11 Y12]
+ (2.1726691016597654e-06) [X2 X3 X11 X12]
+ (2.1726691016597654e-06) [X2 Y3 Y11 X12]
+ (3.1174479457008704e-06) [Y0 Z2 Z3 Y4]
+ (3.1174479457008704e-06) [X0 Z2 Z3 X4]
+ (3.5390541850189658e-06) [Y2 Z3 Y4 Z12]
+ (3.5390541850189658e-06) [X2 Z3 X4 Z12]
+ (3.5390541850189658e-06) [Y3 Z4 Y5 Z13]
+ (3.5390541850189658e-06) [X3 Z4 X5 Z13]
+ (4.281913885346095e-06) [Y4 Z5 Y6 Z11]
+ (4.281913885346095e-06) [X4 Z5 X6 Z11]
+ (4.281913885346095e-06) [Y5 Z6 Y7 Z10]
+ (4.281913885346095e-06) [X5 Z6 X7 Z10]
+ (5.275883122691318e-06) [Y3 X4 X12 Y13]
+ (5.275883122691318e-06) [Y3 Y4 Y12 Y13]
+ (5.275883122691318e-06) [X3 X4 X12 X13]
+ (5.275883122691318e-06) [X3 Y4 Y12 X13]
+ (5.974311714046013e-06) [Y5 X6 X10 Y11]
+ (5.974311714046013e-06) [Y5 Y6 Y10 Y11]
+ (5.974311714046013e-06) [X5 X6 X10 X11]
+ (5.974311714046013e-06) [X5 Y6 Y10 X11]
+ (7.95441317720968e-06) [Y10 Z11 Y12 Z13]
+ (7.95441317720968e-06) [X10 Z11 X12 Z13]
+ (8.814937307710284e-06) [Y2 Z3 Y4 Z13]
+ (8.814937307710284e-06) [X2 Z3 X4 Z13]
+ (8.814937307710284e-06) [Y3 Z4 Y5 Z12]
+ (8.814937307710284e-06) [X3 Z4 X5 Z12]
+ (0.00029219862611106357) [Y7 X8 X9 Y10]
+ (0.00029219862611106357) [X7 Y8 Y9 X10]
+ (0.0004956762314917208) [Y2 Z4 Z5 Y6]
+ (0.0004956762314917208) [X2 Z4 Z5 X6]
+ (0.001105903769189683) [Y0 Z1 Y2 Z5]
+ (0.001105903769189683) [X0 Z1 X2 Z5]
+ (0.001105903769189683) [Y1 Z2 Y3 Z4]
+ (0.001105903769189683) [X1 Z2 X3 Z4]
+ (0.0016638798784908457) [Y2 Z3 Z4 Y6]
+ (0.0016638798784908457) [X2 Z3 Z4 X6]
+ (0.0016638798784908457) [Y3 Z5 Z6 Y7]
+ (0.0016638798784908457) [X3 Z5 Z6 X7]
+ (0.0017560707018412336) [Y0 Z1 Y2 Z11]
+ (0.0017560707018412336) [X0 Z1 X2 Z11]
+ (0.0017560707018412336) [Y1 Z2 Y3 Z10]
+ (0.0017560707018412336) [X1 Z2 X3 Z10]
+ (0.00232623062315808) [Y0 Z1 Y2 Z13]
+ (0.00232623062315808) [X0 Z1 X2 Z13]
+ (0.00232623062315808) [Y1 Z2 Y3 Z12]
+ (0.00232623062315808) [X1 Z2 X3 Z12]
+ (0.0027458364701868094) [Y0 X1 X4 Y5]
+ (0.0027458364701868094) [X0 Y1 Y4 X5]
+ (0.0029297686747510507) [Y0 Z1 Y2 Z9]
+ (0.0029297686747510507) [X0 Z1 X2 Z9]
+ (0.0029297686747510507) [Y1 Z2 Y3 Z8]
+ (0.0029297686747510507) [X1 Z2 X3 Z8]
+ (0.0032769719312316604) [Y0 Z1 Y2 Z3]
+ (0.0032769719312316604) [X0 Z1 X2 Z3]
+ (0.0033476175306661857) [Y0 Z1 Y2 Z7]
+ (0.0033476175306661857) [X0 Z1 X2 Z7]
+ (0.0033476175306661857) [Y1 Z2 Y3 Z6]
+ (0.0033476175306661857) [X1 Z2 X3 Z6]
+ (0.0035552901955042465) [Y0 Z1 Y2 Z10]
+ (0.0035552901955042465) [X0 Z1 X2 Z10]
+ (0.0035552901955042465) [Y1 Z2 Y3 Z11]
+ (0.0035552901955042465) [X1 Z2 X3 Z11]
+ (0.005143391768825112) [Y3 Y4 X5 X6]
+ (0.005143391768825112) [X3 X4 Y5 Y6]
+ (0.005283776488402968) [Y0 X1 X12 Y13]
+ (0.005283776488402968) [X0 Y1 Y12 X13]
+ (0.005530759218631534) [Y0 Z1 Y2 Z4]
+ (0.005530759218631534) [X0 Z1 X2 Z4]
+ (0.005530759218631534) [Y1 Z2 Y3 Z5]
+ (0.005530759218631534) [X1 Z2 X3 Z5]
+ (0.006087822480561862) [Y8 X9 X12 Y13]
+ (0.006087822480561862) [X8 Y9 Y12 X13]
+ (0.0065093612011772415) [Y0 X1 X8 Y9]
+ (0.0065093612011772415) [X0 Y1 Y8 X9]
+ (0.006888194352970571) [Y0 X1 X6 Y7]
+ (0.006888194352970571) [X0 Y1 Y6 X7]
+ (0.006901238249797291) [Y0 Z1 Y2 Z12]
+ (0.006901238249797291) [X0 Z1 X2 Z12]
+ (0.006901238249797291) [Y1 Z2 Y3 Z13]
+ (0.006901238249797291) [X1 Z2 X3 Z13]
+ (0.0071569349198569365) [Y4 X5 X8 Y9]
+ (0.0071569349198569365) [X4 Y5 Y8 X9]
+ (0.007731425250775277) [Y0 X1 X10 Y11]
+ (0.007731425250775277) [X0 Y1 Y10 X11]
+ (0.008032520918821407) [Y0 Z1 Y2 Z6]
+ (0.008032520918821407) [X0 Z1 X2 Z6]
+ (0.008032520918821407) [Y1 Z2 Y3 Z7]
+ (0.008032520918821407) [X1 Z2 X3 Z7]
+ (0.00956070572913595) [Y8 X9 X10 Y11]
+ (0.00956070572913595) [X8 Y9 Y10 X11]
+ (0.011055020596132085) [Y0 Z1 Y2 Z8]
+ (0.011055020596132085) [X0 Z1 X2 Z8]
+ (0.011055020596132085) [Y1 Z2 Y3 Z9]
+ (0.011055020596132085) [X1 Z2 X3 Z9]
+ (0.011285190200840893) [Y5 Y6 X11 X12]
+ (0.011285190200840893) [X5 X6 Y11 Y12]
+ (0.01130727400884809) [Y7 Z8 Z9 Y11]
+ (0.01130727400884809) [X7 Z8 Z9 X11]
+ (0.011982389010247924) [Y4 X5 X6 Y7]
+ (0.011982389010247924) [X4 Y5 Y6 X7]
+ (0.013873381748426089) [Y6 X7 X8 Y9]
+ (0.013873381748426089) [X6 Y7 Y8 X9]
+ (0.014583648907612694) [Y0 X1 X2 Y3]
+ (0.014583648907612694) [X0 Y1 Y2 X3]
+ (0.01557720806397647) [Y2 X3 X12 Y13]
+ (0.01557720806397647) [X2 Y3 Y12 X13]
+ (0.01736611899465135) [Y6 X7 X12 Y13]
+ (0.01736611899465135) [X6 Y7 Y12 X13]
+ (0.017680067952481563) [Y4 X5 X10 Y11]
+ (0.017680067952481563) [X4 Y5 Y10 X11]
+ (0.017825140995786404) [Y6 X7 X10 Y11]
+ (0.017825140995786404) [X6 Y7 Y10 X11]
+ (0.019028242443847314) [Y3 X4 X11 Y12]
+ (0.019028242443847314) [X3 Y4 Y11 X12]
+ (0.025384657508457396) [Y2 X3 X10 Y11]
+ (0.025384657508457396) [X2 Y3 Y10 X11]
+ (0.028685183716106004) [Y10 X11 X12 Y13]
+ (0.028685183716106004) [X10 Y11 Y12 X13]
+ (0.029812424517345684) [Y6 Z7 Z8 Y10]
+ (0.029812424517345684) [X6 Z7 Z8 X10]
+ (0.029812424517345684) [Y7 Z9 Z10 Y11]
+ (0.029812424517345684) [X7 Z9 Z10 X11]
+ (0.030104623143456744) [Y6 Z7 Z9 Y10]
+ (0.030104623143456744) [X6 Z7 Z9 X10]
+ (0.030104623143456744) [Y7 Z8 Z10 Y11]
+ (0.030104623143456744) [X7 Z8 Z10 X11]
+ (0.030787505389143835) [Y6 Z8 Z9 Y10]
+ (0.030787505389143835) [X6 Z8 Z9 X10]
+ (0.03114381798896714) [Y2 X3 X6 Y7]
+ (0.03114381798896714) [X2 Y3 Y6 X7]
+ (0.035839567953353434) [Y2 X3 X4 Y5]
+ (0.035839567953353434) [X2 Y3 Y4 X5]
+ (0.03619412355904262) [Y2 X3 X8 Y9]
+ (0.03619412355904262) [X2 Y3 Y8 X9]
+ (0.03831467029480385) [Y4 X5 X12 Y13]
+ (0.03831467029480385) [X4 Y5 Y12 X13]
+ (0.10433064780651426) [Z0 Y1 Z2 Y3]
+ (0.10433064780651426) [Z0 X1 Z2 X3]
+ (-0.12133276911042284) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042284) [X2 Z3 Z4 Z5 X6]
+ (-0.12133276911042279) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133276911042279) [X3 Z4 Z5 Z6 X7]
+ (3.2020768798713626e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.2020768798713626e-06) [X0 Z1 Z2 Z3 X4]
+ (3.2020768798713626e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.2020768798713626e-06) [X1 Z2 Z3 Z4 X5]
+ (0.22848106564918733) [Y6 Z7 Z8 Z9 Y10]
+ (0.22848106564918733) [X6 Z7 Z8 Z9 X10]
+ (0.22848106564918738) [Y7 Z8 Z9 Z10 Y11]
+ (0.22848106564918738) [X7 Z8 Z9 Z10 X11]
+ (-0.032767657823290386) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.032767657823290386) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.032767657823290386) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.032767657823290386) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273045) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.027115036845273045) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273045) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.027115036845273045) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.025996177598021142) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.025996177598021142) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.01756120240964607) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.01756120240964607) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.01756120240964607) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.01756120240964607) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.014564531231173) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564531231173) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564531231173) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564531231173) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.012215040997613927) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.012215040997613927) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.012215040997613927) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.012215040997613927) [X4 Z5 X6 X11 Z12 X13]
+ (-0.012215040997613927) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.012215040997613927) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.012215040997613927) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.012215040997613927) [X5 Z6 X7 X10 Z11 X12]
+ (-0.011756013419819217) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.011756013419819217) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.011756013419819217) [X3 Z4 Z5 X6 X8 X9]
+ (-0.011756013419819217) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.00876482757568878) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.00876482757568878) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.00876482757568878) [X2 Z3 Z4 X5 X11 X12]
+ (-0.00876482757568878) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.00876482757568878) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.00876482757568878) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.00876482757568878) [X3 X4 X10 Z11 Z12 X13]
+ (-0.00876482757568878) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.008125251921381034) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125251921381034) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.00730675992883297) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.00730675992883297) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.00730675992883297) [X4 X5 X7 Z8 Z9 X10]
+ (-0.00730675992883297) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005805188989826851) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.005805188989826851) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.005805188989826851) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.005805188989826851) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.005652620978017343) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.005652620978017343) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.005652620978017343) [X0 X1 X3 Z4 Z5 X6]
+ (-0.005652620978017343) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.005143391768825112) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.005143391768825112) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.005143391768825112) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.005143391768825112) [X2 Z3 X4 X5 Z6 X7]
+ (-0.004684903388155221) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.004684903388155221) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668620318776298) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668620318776298) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.00457500762663921) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.00457500762663921) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424855449441851) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424855449441851) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.0041587973818400705) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.0041587973818400705) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.0041587973818400705) [X3 Z4 Z5 X6 X12 X13]
+ (-0.0041587973818400705) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.003493790359890106) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.003493790359890106) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.003493790359890106) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.003493790359890106) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.00277902679902554) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.00277902679902554) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.0022939566113524667) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.0022939566113524667) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.0017992194936630125) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.0017992194936630125) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278753941369492) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.0017278753941369492) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298507967730352) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298507967730352) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298507967730352) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298507967730352) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533856254125446) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533856254125446) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008145313270955831) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008145313270955831) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008145313270955831) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008145313270955831) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.735036880589926e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735036880589926e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735036880589926e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735036880589926e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.774817865588843e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.774817865588843e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.774817865588843e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.774817865588843e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.51836221649247e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.51836221649247e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.51836221649247e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.51836221649247e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.444344676831776e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.444344676831776e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.444344676831776e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.444344676831776e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.524373849447511e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.524373849447511e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.524373849447511e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.524373849447511e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.2900284341358975e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.2900284341358975e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.2900284341358975e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.2900284341358975e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.974311714046014e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.974311714046014e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.275883122691318e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.275883122691318e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.643051069093797e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.643051069093797e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.556569218860123e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.556569218860123e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.25322422605756e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.25322422605756e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.7696594527476908e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.7696594527476908e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.6945132949405883e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.6945132949405883e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.610297131341925e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.610297131341925e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.610297131341925e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.610297131341925e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.3131455004412266e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.3131455004412266e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.277483196293917e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.277483196293917e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.277483196293917e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.277483196293917e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.211228349006285e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.211228349006285e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.211228349006285e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.211228349006285e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.1513463115769914e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.1513463115769914e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.0882507117374444e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.0882507117374444e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.1726691016597654e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.1726691016597654e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.4548424488846648e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.4548424488846648e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.3304731887570663e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.3304731887570663e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.228333782356573e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.228333782356573e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.0358477600227656e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.0358477600227656e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956895371441827e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956895371441827e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733197743676836e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733197743676836e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733197743676836e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733197743676836e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.628614202452722e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.628614202452722e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.556281915171815e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.556281915171815e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.556281915171815e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.556281915171815e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.418291575186248e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.418291575186248e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.418291575186248e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.418291575186248e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.927453083930772e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927453083930772e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927453083930772e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927453083930772e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627851911909516e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627851911909516e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660625163983e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660625163983e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660625163983e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660625163983e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660625163983e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660625163983e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660625163983e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660625163983e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.83805275034921e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.83805275034921e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.570761328657502e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.570761328657502e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.3281393504800803e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.3281393504800803e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.086826564997497e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.086826564997497e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.086826564997497e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.086826564997497e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.4473231285176163e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.4473231285176163e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.3713289475967311e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.3713289475967311e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.3713289475967311e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.3713289475967311e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.199051618478637e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.199051618478637e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.933241276865342e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.933241276865342e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.933241276865342e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.933241276865342e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.839420915228539e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.839420915228539e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.839420915228539e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.839420915228539e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.5510539174822682e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.5510539174822682e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.5510539174822682e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.5510539174822682e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.380778147898382e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.380778147898382e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.380778147898382e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.380778147898382e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.380778147898382e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.380778147898382e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.380778147898382e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.380778147898382e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.380778147898382e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.380778147898382e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.380778147898382e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.380778147898382e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.2919694860340715e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.2919694860340715e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.1076325600277984e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.1076325600277984e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.1076325600277984e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.1076325600277984e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.1076325600277984e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.1076325600277984e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.1076325600277984e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.1076325600277984e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.057446597460642e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057446597460642e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057446597460642e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057446597460642e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.64931013388902e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.64931013388902e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.64931013388902e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.64931013388902e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.839420915228539e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.839420915228539e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.839420915228539e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.839420915228539e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.199051618478637e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.199051618478637e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.4473231285176163e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.4473231285176163e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.236259960985633e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.236259960985633e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.236259960985633e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.236259960985633e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.3281393504800803e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.3281393504800803e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.570761328657502e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.570761328657502e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.83805275034921e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.83805275034921e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.627851911909516e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627851911909516e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.628614202452722e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.628614202452722e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956895371441827e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956895371441827e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.306536652806455e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.306536652806455e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.306536652806455e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.306536652806455e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.0358477600227656e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.0358477600227656e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.228333782356573e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.228333782356573e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.2393363217803953e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393363217803953e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393363217803953e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393363217803953e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.3304731887570663e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.3304731887570663e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.4548424488846648e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.4548424488846648e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.1726691016597654e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.1726691016597654e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.0882507117374444e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.0882507117374444e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.11744794570087e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.11744794570087e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.1513463115769914e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.1513463115769914e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.3131455004412266e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.3131455004412266e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.3343312899109224e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.3343312899109224e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.6945132949405883e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.6945132949405883e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.183932559710562e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.183932559710562e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.556569218860123e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.556569218860123e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.643051069093797e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.643051069093797e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.275883122691318e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.275883122691318e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.974311714046014e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.974311714046014e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.0002921986261110635) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.0002921986261110635) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.0002921986261110635) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.0002921986261110635) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004956762314917208) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004956762314917208) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650070219499645) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650070219499645) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650070219499645) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650070219499645) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533856254125446) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533856254125446) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.001609531381721385) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.001609531381721385) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.001609531381721385) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.001609531381721385) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0016676041811440672) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.0016676041811440672) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.0016676041811440672) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.0016676041811440672) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.0017278753941369492) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.0017278753941369492) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.0017992194936630125) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.0017992194936630125) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.0022939566113524667) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.0022939566113524667) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.0024629170071339295) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.0024629170071339295) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.0024629170071339295) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.0024629170071339295) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.003961560792496534) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.003961560792496534) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.003961560792496534) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.003961560792496534) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.004424855449441851) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424855449441851) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.00457500762663921) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.00457500762663921) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668620318776298) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668620318776298) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.004684903388155221) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.004684903388155221) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.0053248352342216915) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.0053248352342216915) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.0053248352342216915) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.0053248352342216915) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.005368659358109535) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368659358109535) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368659358109535) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368659358109535) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.007960880725921524) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.007960880725921524) [X4 Z5 X6 X10 Z11 X12]
+ (0.007960880725921524) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.007960880725921524) [X5 Z6 X7 X11 Z12 X13]
+ (0.008125251921381034) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125251921381034) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.00889073152269456) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.00889073152269456) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.00889073152269456) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.00889073152269456) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.010263414868158531) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.010263414868158531) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.010263414868158531) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.010263414868158531) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.010540425907671493) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.010540425907671493) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.010540425907671493) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.010540425907671493) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010960074940542517) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010960074940542517) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010960074940542517) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010960074940542517) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307274008848092) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307274008848092) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.014411099430131018) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411099430131018) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411099430131018) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411099430131018) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.015225630757226601) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.015225630757226601) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.015225630757226601) [X3 Z4 Z5 X6 X10 X11]
+ (0.015225630757226601) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.01558825010238022) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.01558825010238022) [X2 Z3 X4 X10 Z11 X12]
+ (0.01558825010238022) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.01558825010238022) [X3 Z4 X5 X11 Z12 X13]
+ (0.01826683486937549) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.01826683486937549) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.01826683486937549) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.01826683486937549) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.019020423173039924) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.019020423173039924) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.019020423173039924) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.019020423173039924) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.020175921723535453) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.020175921723535453) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.020175921723535453) [X4 Z5 Z6 X7 X11 X12]
+ (0.020175921723535453) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.020175921723535453) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.020175921723535453) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.020175921723535453) [X5 X6 X10 Z11 Z12 X13]
+ (0.020175921723535453) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.024353077678069) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.024353077678069) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.024353077678069) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.024353077678069) [X2 Z3 X4 X11 Z12 X13]
+ (0.024353077678069) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.024353077678069) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.024353077678069) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.024353077678069) [X3 Z4 X5 X10 Z11 X12]
+ (0.024389082531149457) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.024389082531149457) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.024389082531149457) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.024389082531149457) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.025104957138844496) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.025104957138844496) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.025104957138844496) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.025104957138844496) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.030787505389143835) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.030787505389143835) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.05600733087780761) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.05600733087780761) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.05600733087780761) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.05600733087780761) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661352) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.05608468124661352) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661352) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.05608468124661352) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.631277929185038e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631277929185038e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.631277929185037e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631277929185037e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.595086007584171e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.595086007584171e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.595086007584169e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.595086007584169e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.04274327701378291) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.04274327701378291) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (0.042743277013782936) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.042743277013782936) [X1 Z2 Z3 Z4 Z5 Z6 X7]
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
+ (-0.039359168022053116) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.039359168022053116) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.039359168022053116) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.039359168022053116) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.03931805194719755) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03931805194719755) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.03931805194719755) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03931805194719755) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03560837898831255) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.03560837898831255) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.028730779551905502) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.028730779551905502) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.028730779551905502) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.028730779551905502) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.025637238296026803) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.025637238296026803) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.025637238296026803) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.025637238296026803) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.02475546329289098) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.02475546329289098) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.02475546329289098) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.02475546329289098) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.024282117354692906) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.024282117354692906) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.02314513092952904) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.02314513092952904) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.02252844019601294) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.02252844019601294) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.021433810721601013) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.021433810721601013) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.021433810721601013) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.021433810721601013) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251558) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.019257505095251558) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019028242443847314) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.019028242443847314) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.01888903030494287) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.01888903030494287) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.01888903030494287) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.01888903030494287) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.016024603689179615) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.016024603689179615) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.015225630757226598) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225630757226598) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.014603704729162139) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.014603704729162139) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564531231173) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564531231173) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.011756013419819219) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.011756013419819219) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.011285190200840893) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.011285190200840893) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.009841749246962631) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009841749246962631) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.00961263460684719) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.00961263460684719) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.00961263460684719) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.00961263460684719) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.008469978791023822) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.008469978791023822) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.007306759928832969) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.007306759928832969) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005923798336561343) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.005923798336561343) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.005652620978017343) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.005652620978017343) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368659358109535) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368659358109535) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.0041587973818400705) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.0041587973818400705) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0033566705638328805) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.0033566705638328805) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.0033566705638328805) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.0033566705638328805) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.0032675138544235476) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.0032675138544235476) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.0032675138544235476) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.0032675138544235476) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.00277902679902554) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.00277902679902554) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.002686040977806602) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.002686040977806602) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.002686040977806602) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.002686040977806602) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.0022939566113524667) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524667) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.0022939566113524667) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524667) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.0009581655836696525) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.0009581655836696525) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0009581655836696525) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.0009581655836696525) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.0009581655836696525) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.0009581655836696525) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0009581655836696525) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.0009581655836696525) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.0002463643756958197) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0002463643756958197) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303548352) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.00013840177303548352) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.00013840177303548352) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.00013840177303548352) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.735036880589926e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735036880589926e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.610358530761968e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.610358530761968e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.610358530761968e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.610358530761968e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808797140935e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.5316808797140935e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808797140935e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5316808797140935e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.806102776322577e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.806102776322577e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.806102776322577e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.806102776322577e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.0897994682170815e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.0897994682170815e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.0897994682170815e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.0897994682170815e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.6522096709199025e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.6522096709199025e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.6522096709199025e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.6522096709199025e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851835391308e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.481851835391308e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851835391308e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.481851835391308e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.0714807369914775e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.0714807369914775e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.0714807369914775e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.0714807369914775e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.7346220393311e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.7346220393311e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.7346220393311e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.7346220393311e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.728843147741842e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.728843147741842e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.728843147741842e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.728843147741842e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.25322422605756e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.25322422605756e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.7696594527476908e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.7696594527476908e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.5443954296782317e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.5443954296782317e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.5443954296782317e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.5443954296782317e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.5443954296782317e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.5443954296782317e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.5443954296782317e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.5443954296782317e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.3609563204752396e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563204752396e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563204752396e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.3609563204752396e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.1032156051819005e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.1032156051819005e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.1032156051819005e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.1032156051819005e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220989320353e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.0111220989320353e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220989320353e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.0111220989320353e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.942946836840773e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.942946836840773e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.942946836840773e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.942946836840773e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174776050887e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.6541174776050887e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174776050887e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.6541174776050887e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.522493067816124e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.522493067816124e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.522493067816124e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.522493067816124e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.522493067816124e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.522493067816124e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.522493067816124e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.522493067816124e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.228333782356573e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.228333782356573e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.228333782356573e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.228333782356573e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.988770288316374e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988770288316374e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988770288316374e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988770288316374e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.867765104787441e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104787441e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104787441e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.867765104787441e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.189990976390523e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.189990976390523e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246207610794e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246207610794e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.471647745138482e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.471647745138482e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.5614471794078723e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.5614471794078723e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.5614471794078723e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.5614471794078723e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.52338967876964e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.52338967876964e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.427323108908502e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.427323108908502e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.427323108908502e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.427323108908502e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.3281393504800803e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.3281393504800803e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.3281393504800803e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.3281393504800803e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.0868265649974963e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.0868265649974963e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.888293592356848e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.888293592356848e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.888293592356848e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.888293592356848e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.3713289475967311e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.3713289475967311e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.839420915228539e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.839420915228539e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.057446597460642e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057446597460642e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.5371780967581313e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.5371780967581313e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.5371780967581313e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.5371780967581313e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.057446597460642e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057446597460642e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.209350624986533e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350624986533e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350624986533e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.209350624986533e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783552859454e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783552859454e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783552859454e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783552859454e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.839420915228539e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.839420915228539e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.3713289475967311e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.3713289475967311e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.0868265649974963e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.0868265649974963e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.52338967876964e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.52338967876964e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.471647745138482e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.471647745138482e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246207610794e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246207610794e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189990976390523e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.189990976390523e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.3304731887570663e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.3304731887570663e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.3304731887570663e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.3304731887570663e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288532437608678e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288532437608678e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288532437608678e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288532437608678e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.6893489517182406e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.6893489517182406e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.6893489517182406e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.6893489517182406e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.7455184008465947e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.7455184008465947e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.7455184008465947e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.7455184008465947e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.7455184008465947e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.7455184008465947e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.7455184008465947e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.7455184008465947e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.211842019534364e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.211842019534364e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.211842019534364e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.211842019534364e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.211842019534364e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.211842019534364e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.211842019534364e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.211842019534364e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.3131455004412266e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.3131455004412266e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.3131455004412266e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.3131455004412266e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.334331289910923e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.334331289910923e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.183932559710562e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.183932559710562e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.735036880589926e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735036880589926e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.0002463643756958197) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.0002463643756958197) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00044585351288408336) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.00044585351288408336) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.00044585351288408336) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.00044585351288408336) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940221543005391) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005391) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005391) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005391) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005391) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005391) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940221543005391) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005391) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533856254125446) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533856254125446) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533856254125446) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533856254125446) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.001043524653490763) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.001043524653490763) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.001043524653490763) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.001043524653490763) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.0012803060973496758) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.0012803060973496758) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.0012803060973496758) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.0012803060973496758) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.0013038004788127) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.0013038004788127) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.0013038004788127) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.0013038004788127) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.002261966062482353) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.002261966062482353) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.002261966062482353) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.002261966062482353) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.002261966062482353) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.002261966062482353) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.002261966062482353) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.002261966062482353) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.003989841456619303) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.003989841456619303) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.003989841456619303) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.003989841456619303) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.0041587973818400705) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.0041587973818400705) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.00431103850791431) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.00431103850791431) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.00431103850791431) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.00431103850791431) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.004636976661182555) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.004636976661182555) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.004636976661182555) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.004636976661182555) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.005114473831660381) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.005114473831660381) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.005114473831660381) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.005114473831660381) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.005114473831660381) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660381) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660381) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005114473831660381) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005241535382803877) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.005241535382803877) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.005241535382803877) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.005241535382803877) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.005262642473076836) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262642473076836) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262642473076836) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262642473076836) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368659358109535) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368659358109535) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005379937155839361) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.005379937155839361) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.005379937155839361) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.005379937155839361) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.005652620978017343) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.005652620978017343) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.005708495985960919) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708495985960919) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708495985960919) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708495985960919) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.005923798336561343) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.005923798336561343) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.007306759928832969) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.007306759928832969) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469978791023822) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.008469978791023822) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.009841749246962631) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.009841749246962631) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.011285190200840893) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.011285190200840893) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.011756013419819219) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.011756013419819219) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.014564531231173) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564531231173) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603704729162139) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.014603704729162139) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.015225630757226598) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225630757226598) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.016024603689179615) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.016024603689179615) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.019028242443847314) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.019028242443847314) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.019257505095251558) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.019257505095251558) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.3693708936615606) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.3693708936615606) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.3693708936615606) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.3693708936615606) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.28164257767022943) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.28164257767022943) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.2816425776702293) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.2816425776702293) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.09065144207036485) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.09065144207036485) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.09065144207036485) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.09065144207036485) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863632) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684737589863632) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863632) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684737589863632) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635021950635026) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635021950635026) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635021950635026) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635021950635026) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0675238509921404) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0675238509921404) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0675238509921404) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0675238509921404) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03560837898831255) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.03560837898831255) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.024591860883829957) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.024591860883829957) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883829957) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.024591860883829957) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.024282117354692902) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.024282117354692902) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.023145130929529037) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.023145130929529037) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.02252844019601294) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.02252844019601294) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.01953805031131472) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.01953805031131472) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.01953805031131472) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.01953805031131472) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.017091553155898873) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.017091553155898873) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.017091553155898873) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.017091553155898873) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.016024603689179615) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.016024603689179615) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.016024603689179615) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.016024603689179615) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.010311482489831807) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831807) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831807) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311482489831807) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.009841749246962631) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962631) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.009841749246962631) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962631) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.00882636851420986) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.00882636851420986) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.00882636851420986) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.00882636851420986) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008541996625454813) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454813) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454813) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454813) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454813) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454813) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454813) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008541996625454813) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008469978791023822) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023822) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.008469978791023822) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023822) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.004668620318776298) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668620318776298) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.0038764708993369546) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.0038764708993369546) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.0038040661717285425) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0038040661717285425) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0038040661717285425) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0038040661717285425) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178847) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178847) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0033566705638328805) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.0033566705638328805) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.003267513854423547) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.003267513854423547) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0021413612231016123) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.0021413612231016123) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.0017278753941369492) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.0017278753941369492) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.0016407548553124176) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0016407548553124176) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214169367) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0014528843214169367) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214169367) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0014528843214169367) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870896771024469) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007870896771024469) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192743499487894) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192743499487894) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.00019400857029756372) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00019400857029756372) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303548352) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.00013840177303548352) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.141625221158181e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.141625221158181e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.141625221158181e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.141625221158181e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.0714807369914775e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.0714807369914775e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.1513463115769914e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.1513463115769914e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0882507117374444e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.0882507117374444e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.9885117063399056e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.9885117063399056e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.87429907167618e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.87429907167618e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.3609563204752396e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.3609563204752396e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.300294656505395e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.300294656505395e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.1468376509177009e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.1468376509177009e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.1468376509177009e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.1468376509177009e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.352332104450538e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.352332104450538e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.352332104450538e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.352332104450538e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091637200386659e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637200386659e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637200386659e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637200386659e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637200386659e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637200386659e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637200386659e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091637200386659e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.074305987165272e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.074305987165272e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.074305987165272e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.074305987165272e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128987559984e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128987559984e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128987559984e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128987559984e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867765104787441e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.867765104787441e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560692466105783e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692466105783e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692466105783e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692466105783e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692466105783e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692466105783e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692466105783e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560692466105783e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.99701842285567e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.99701842285567e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.99701842285567e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.99701842285567e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.99701842285567e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.99701842285567e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.99701842285567e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.99701842285567e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.5682475216170216e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.5682475216170216e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.5682475216170216e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.5682475216170216e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393087903453e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393087903453e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393087903453e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393087903453e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393087903453e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393087903453e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.3767393087903453e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393087903453e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.888293592356848e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.888293592356848e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.686381548443272e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.686381548443272e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035783552859454e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7035783552859454e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.209350624986533e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.209350624986533e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.37977324559752e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.37977324559752e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.37977324559752e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.37977324559752e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.37977324559752e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.37977324559752e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.37977324559752e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.37977324559752e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.974225380005005e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.974225380005005e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.974225380005005e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.974225380005005e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0474716557228401e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.0474716557228401e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.0474716557228401e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.0474716557228401e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.209350624986533e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.209350624986533e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0717282187504582e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0717282187504582e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0717282187504582e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0717282187504582e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.200428749470743e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.200428749470743e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.200428749470743e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.200428749470743e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.7035783552859454e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7035783552859454e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.312094305472392e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.312094305472392e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.312094305472392e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.312094305472392e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.686381548443272e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.686381548443272e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.888293592356848e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.888293592356848e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.0922506165810426e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506165810426e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.0922506165810426e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506165810426e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.0922506165810426e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506165810426e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.0922506165810426e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506165810426e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.444597854164006e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.444597854164006e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.444597854164006e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.444597854164006e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.6849150955394324e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.6849150955394324e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.6849150955394324e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.6849150955394324e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246974426361251e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246974426361251e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246974426361251e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246974426361251e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246974426361251e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246974426361251e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246974426361251e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246974426361251e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.867765104787441e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.867765104787441e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.300294656505395e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.300294656505395e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.3609563204752396e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.3609563204752396e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.87429907167618e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.87429907167618e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.8836765762775716e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.8836765762775716e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.9473560121620493e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9473560121620493e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.9473560121620493e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9473560121620493e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117063399056e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9885117063399056e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.0882507117374444e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.0882507117374444e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.1513463115769914e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.1513463115769914e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.846201671619488e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846201671619488e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846201671619488e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846201671619488e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.0714807369914775e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.0714807369914775e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.1055267224290356e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.1055267224290356e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.1055267224290356e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.1055267224290356e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146496328124883e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146496328124883e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146496328124883e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146496328124883e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159350502234614e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159350502234614e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159350502234614e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159350502234614e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427988657078942e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427988657078942e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427988657078942e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427988657078942e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935867718501955e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935867718501955e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935867718501955e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935867718501955e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348731213e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.253273348731213e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979825794105215e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979825794105215e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979825794105215e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979825794105215e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.205548411216793e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.205548411216793e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.205548411216793e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.205548411216793e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00013840177303548352) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.00013840177303548352) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.00018787053389548067) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.00018787053389548067) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.00018787053389548067) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00018787053389548067) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00019400857029756372) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00019400857029756372) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0002463643756958197) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0002463643756958197) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0002463643756958197) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0002463643756958197) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192743499487894) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192743499487894) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156734248908649) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156734248908649) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156734248908649) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156734248908649) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024469) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007870896771024469) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.0015324835230730058) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.0015324835230730058) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.0015324835230730058) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.0015324835230730058) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.0016407548553124176) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0016407548553124176) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0017278753941369492) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.0017278753941369492) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.0024464971554158453) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.0024464971554158453) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.0024464971554158453) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.0024464971554158453) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.003267513854423547) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.003267513854423547) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.0033566705638328805) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.0033566705638328805) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.0034841573002178847) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0034841573002178847) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0038764708993369546) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.0038764708993369546) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.004668620318776298) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668620318776298) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278072) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.004767272188278072) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.004767272188278072) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278072) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.005286546538226862) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.005286546538226862) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.005286546538226862) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.005286546538226862) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.00540895442240996) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.00540895442240996) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.00540895442240996) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.00540895442240996) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.005923798336561343) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561343) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.005923798336561343) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561343) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.01071550846979674) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01071550846979674) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.01071550846979674) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01071550846979674) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01075756395390891) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01075756395390891) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.01075756395390891) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01075756395390891) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.014603704729162139) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.014603704729162139) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.014603704729162139) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.014603704729162139) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.01929956057936372) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01929956057936372) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01929956057936372) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01929956057936372) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01929956057936372) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01929956057936372) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.01929956057936372) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01929956057936372) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.05859198873386191) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.05859198873386191) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.7759505277526376e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.7759505277526376e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.77595052775264e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.77595052775264e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.07165035181002469) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07165035181002469) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.07165035181002469) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.07165035181002469) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251558) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019257505095251558) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010311482489831807) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311482489831807) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00882636851420986) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.00882636851420986) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0075974640297706095) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0075974640297706095) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0075974640297706095) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0075974640297706095) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311874) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311874) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311874) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311874) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311874) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311874) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311874) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311874) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0038040661717285425) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0038040661717285425) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219416) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.0029841661681219416) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.0029841661681219416) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.0029841661681219416) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.002446497155415845) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.002446497155415845) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.0022494124470939913) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0022494124470939913) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022494124470939913) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0022494124470939913) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0021413612231016123) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.0021413612231016123) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.0018638942824587335) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587335) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587335) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587335) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587335) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587335) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0018638942824587335) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587335) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0016407548553124176) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124176) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0016407548553124176) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124176) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.00122233780815383) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00122233780815383) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00122233780815383) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00122233780815383) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.00122233780815383) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00122233780815383) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00122233780815383) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00122233780815383) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0010283292378562667) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0010283292378562667) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0010283292378562667) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0010283292378562667) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1463061454126173e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1463061454126173e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.87429907167618e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.87429907167618e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.87429907167618e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.87429907167618e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.300294656505395e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.300294656505395e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.300294656505395e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.300294656505395e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.0444941298797141e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.0444941298797141e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.0444941298797141e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.0444941298797141e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.956079230766826e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.956079230766826e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.956079230766826e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.956079230766826e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.10551503787684e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.10551503787684e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.10551503787684e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.10551503787684e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661347213713238e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661347213713238e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661347213713238e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661347213713238e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341414512851e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341414512851e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189990976390523e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.189990976390523e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876621658722003e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876621658722003e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876621658722003e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876621658722003e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246207610794e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246207610794e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.52338967876964e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.52338967876964e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0767325316028176e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.0767325316028176e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0767325316028176e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.0767325316028176e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.013471459258496e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.013471459258496e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.904599884284289e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.904599884284289e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.904599884284289e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.904599884284289e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.666731754033662e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.666731754033662e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.666731754033662e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.666731754033662e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641928899872e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.8505641928899872e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309316046207e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6569309316046207e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309316046207e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6569309316046207e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.8505641928899872e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.8505641928899872e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.686381548443272e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.686381548443272e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.686381548443272e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.686381548443272e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471459258496e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.013471459258496e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.52338967876964e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.52338967876964e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.6704023908631165e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.6704023908631165e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.6704023908631165e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.6704023908631165e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246207610794e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246207610794e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189990976390523e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.189990976390523e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341414512851e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341414512851e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949476488623898e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949476488623898e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.79249395786412e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.79249395786412e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.79249395786412e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.79249395786412e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.8836765762775716e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.8836765762775716e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.9885117063399056e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117063399056e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117063399056e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117063399056e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348731213e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253273348731213e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4017109736463524e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4017109736463524e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4017109736463524e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4017109736463524e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603694327645e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.5809603694327645e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603694327645e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.5809603694327645e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192743499487894) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487894) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192743499487894) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487894) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.0007870896771024469) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024469) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024469) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024469) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0011726348316441855) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0011726348316441855) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0011726348316441855) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0011726348316441855) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012366478019245097) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.0012366478019245097) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.0012366478019245097) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.0012366478019245097) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.002200964069500452) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002200964069500452) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002200964069500452) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002200964069500452) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980153) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980153) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0023949726397980153) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980153) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0023949726397980153) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0023949726397980153) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0023949726397980153) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0023949726397980153) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002446497155415845) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.002446497155415845) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.0038040661717285425) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0038040661717285425) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0038764708993369546) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.0038764708993369546) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.0038764708993369546) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.0038764708993369546) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.0042208139700464515) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.0042208139700464515) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.0042208139700464515) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.0042208139700464515) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.00882636851420986) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00882636851420986) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.010311482489831807) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311482489831807) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257505095251558) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019257505095251558) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.05859198873386191) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.05859198873386191) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3987009015497003e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3987009015497003e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.3987009015497e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.3987009015497e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178847) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178847) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219416) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.0029841661681219416) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.00019400857029756372) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00019400857029756372) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061454126173e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061454126173e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.79249395786412e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.79249395786412e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540341414512851e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341414512851e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540341414512851e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341414512851e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.8505641928899872e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641928899872e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641928899872e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641928899872e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471459258496e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471459258496e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471459258496e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471459258496e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949476488623897e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949476488623897e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.79249395786412e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.79249395786412e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756372) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756372) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0029841661681219416) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.0029841661681219416) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.0034841573002178847) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0034841573002178847) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
 </code>
 </pre>
 </details>

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
   16384/11490434 [..............................] - ETA: 1s
  417792/11490434 [>.............................] - ETA: 1s
 6373376/11490434 [===============>..............] - ETA: 0s
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
 4145152/11490434 [=========>....................] - ETA: 0s
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

---

## 15. tutorial_quantum_analytic_descent.html <a name="demo14"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quantum_analytic_descent.html):

```
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
Epoch   50: Model cost = -0.7981  at relative parameters [-0.2397  2.0158]
True energy at the minimum of the model: -0.7358296722728767
New reference parameters: [2.4222 6.0741]
Epoch   50: Model cost = -1.0084  at relative parameters [0.6908 0.2794]
True energy at the minimum of the model: -0.9971225971605668
New reference parameters: [3.113  6.3535]
Epoch   50: Model cost = -1.0  at relative parameters [ 0.0272 -0.0685]
True energy at the minimum of the model: -0.9999975843757788
New reference parameters: [3.1403 6.285 ]
```

---

## 16. tutorial_vqe_parallel.html <a name="demo15"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_vqe_parallel.html):

```
Speed up: 2.79
Evaluation time: 321.67 s
Evaluation time: 115.25 s
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_vqe_parallel.html):

```
Speed up: 3.01
Evaluation time: 270.50 s
Evaluation time: 89.83 s
```

---

## 17. tutorial_qnn_module_tf.html <a name="demo16"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_qnn_module_tf.html):

```
30/30 - 11s - loss: 0.4997 - accuracy: 0.5000 - val_loss: 0.5081 - val_accuracy: 0.4400
30/30 - 11s - loss: 0.4673 - accuracy: 0.6200 - val_loss: 0.4488 - val_accuracy: 0.6200
30/30 - 11s - loss: 0.3230 - accuracy: 0.8267 - val_loss: 0.2562 - val_accuracy: 0.8400
30/30 - 11s - loss: 0.2124 - accuracy: 0.8867 - val_loss: 0.1997 - val_accuracy: 0.8400
30/30 - 11s - loss: 0.1800 - accuracy: 0.8933 - val_loss: 0.1841 - val_accuracy: 0.8400
30/30 - 11s - loss: 0.1593 - accuracy: 0.8667 - val_loss: 0.2177 - val_accuracy: 0.8400
30/30 - 22s - loss: 0.5189 - accuracy: 0.4000 - val_loss: 0.4945 - val_accuracy: 0.5400
30/30 - 22s - loss: 0.4822 - accuracy: 0.6200 - val_loss: 0.4412 - val_accuracy: 0.7200
30/30 - 22s - loss: 0.3850 - accuracy: 0.7133 - val_loss: 0.2898 - val_accuracy: 0.7800
30/30 - 23s - loss: 0.2720 - accuracy: 0.7867 - val_loss: 0.2185 - val_accuracy: 0.8200
30/30 - 22s - loss: 0.2056 - accuracy: 0.8400 - val_loss: 0.1893 - val_accuracy: 0.8400
30/30 - 22s - loss: 0.1753 - accuracy: 0.8400 - val_loss: 0.1856 - val_accuracy: 0.8400
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_qnn_module_tf.html):

```
30/30 - 9s - loss: 0.4997 - accuracy: 0.5000 - val_loss: 0.5081 - val_accuracy: 0.4400
30/30 - 9s - loss: 0.4673 - accuracy: 0.6200 - val_loss: 0.4488 - val_accuracy: 0.6200
30/30 - 9s - loss: 0.3230 - accuracy: 0.8267 - val_loss: 0.2562 - val_accuracy: 0.8400
30/30 - 9s - loss: 0.2124 - accuracy: 0.8867 - val_loss: 0.1997 - val_accuracy: 0.8400
30/30 - 9s - loss: 0.1800 - accuracy: 0.8933 - val_loss: 0.1841 - val_accuracy: 0.8400
30/30 - 9s - loss: 0.1593 - accuracy: 0.8667 - val_loss: 0.2177 - val_accuracy: 0.8400
30/30 - 18s - loss: 0.5189 - accuracy: 0.4000 - val_loss: 0.4945 - val_accuracy: 0.5400
30/30 - 18s - loss: 0.4822 - accuracy: 0.6200 - val_loss: 0.4412 - val_accuracy: 0.7200
30/30 - 18s - loss: 0.3850 - accuracy: 0.7133 - val_loss: 0.2898 - val_accuracy: 0.7800
30/30 - 18s - loss: 0.2720 - accuracy: 0.7867 - val_loss: 0.2185 - val_accuracy: 0.8200
30/30 - 18s - loss: 0.2056 - accuracy: 0.8400 - val_loss: 0.1893 - val_accuracy: 0.8400
30/30 - 18s - loss: 0.1753 - accuracy: 0.8400 - val_loss: 0.1856 - val_accuracy: 0.8400
```

---

## 18. tutorial_ensemble_multi_qpu.html <a name="demo17"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_ensemble_multi_qpu.html):

```
Training accuracy (ensemble): 0.8
Test accuracy (QPU1):  0.36
Choices: [1 0 1 0 0 0 0 1 0 0 0 1 0 0 0 1 0 0 1 1 0 0 1 0 0 0 0 0 1 0 0 0 1 1 0 1 0
Choices counts: Counter({0: 110, 1: 40})
Counter({2: 55, 0: 55})
Counter({1: 36, 0: 4})
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_ensemble_multi_qpu.html):

```
Training accuracy (ensemble): 0.808
Test accuracy (QPU1):  0.4
Choices: [1 0 1 0 0 0 0 1 0 0 0 1 1 0 0 1 0 0 1 1 0 0 1 0 0 0 0 0 1 0 0 0 1 1 0 1 0
Choices counts: Counter({0: 109, 1: 41})
Counter({0: 55, 2: 54})
Counter({1: 37, 0: 4})
```

---

