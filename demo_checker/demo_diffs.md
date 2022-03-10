Last update: 2022-03-10  03:21:04 (All times shown in Eastern time)
# List of differences in demonstration outputs

# Table of contents

1. [tutorial_general_parshift.html](#demo0)
2. [tutorial_vqt.html](#demo1)
3. [tutorial_expressivity_fourier_series.html](#demo2)
4. [tutorial_noisy_circuits.html](#demo3)
5. [tutorial_QGAN.html](#demo4)
6. [tutorial_rosalin.html](#demo5)
7. [tutorial_quanvolution.html](#demo6)
8. [tutorial_quantum_analytic_descent.html](#demo7)
9. [tutorial_noisy_circuit_optimization.html](#demo8)
10. [tutorial_error_mitigation.html](#demo9)
11. [tutorial_doubly_stochastic.html](#demo10)
12. [tutorial_quantum_transfer_learning.html](#demo11)
13. [tutorial_quantum_metrology.html](#demo12)
14. [tutorial_qaoa_intro.html](#demo13)
15. [tutorial_backprop.html](#demo14)
16. [tutorial_adaptive_circuits.html](#demo15)
17. [tutorial_jax_transformations.html](#demo16)
18. [tutorial_measurement_optimize.html](#demo17)
19. [tutorial_kernel_based_training.html](#demo18)
20. [tutorial_qnn_module_tf.html](#demo19)
21. [tutorial_quantum_chemistry.html](#demo20)
22. [tutorial_gbs.html](#demo21)


Number of demos different/all demos: 22/57

## 1. tutorial_general_parshift.html <a name="demo0"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_general_parshift.html):

```
Second-order finite difference:    [ 0.26814   1.696854 -2.055919 -7.236954]
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_general_parshift.html):

```
Second-order finite difference:    [ 0.26814   1.696854 -2.055918 -7.236954]
```

---

## 2. tutorial_vqt.html <a name="demo1"></a>

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
H0 =
[[ 4.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j
   0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  2.+0.j
   0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  2.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j
   0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j
   0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  2.+0.j
   0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j -4.+0.j  2.+0.j  0.+0.j  0.+0.j
   2.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j
   0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j
   0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j
   2.+0.j -4.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j
   0.+0.j  0.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j
   0.+0.j  0.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j
   0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  4.+0.j]]
Cost at Step 0: -0.6605354666522008
Cost at Step 50: -2.869994162243927
Cost at Step 100: -4.64206718424408
Cost at Step 150: -5.127022428216529
Cost at Step 200: -6.561237930042809
Cost at Step 250: -7.058251609076517
Cost at Step 300: -8.037625694857155
Cost at Step 350: -9.52547334894263
Cost at Step 400: -10.484319085913086
Cost at Step 450: -10.606829196526334
Cost at Step 500: -11.393207441595656
Cost at Step 550: -12.035185481360603
Cost at Step 600: -12.35223595627993
Cost at Step 650: -12.611003042349648
Cost at Step 700: -12.817651448645346
Cost at Step 750: -12.999662452066135
Cost at Step 800: -13.536823965606517
Cost at Step 850: -13.604628321371722
Cost at Step 900: -13.708923136713517
Cost at Step 950: -13.851219173473567
Cost at Step 1000: -13.990610394065657
Cost at Step 1050: -14.103484513831358
Cost at Step 1100: -14.206499698642727
Cost at Step 1150: -14.33009816352317
Cost at Step 1200: -14.442591667696115
Cost at Step 1250: -14.488105306008986
Cost at Step 1300: -14.624999975340113
Cost at Step 1350: -14.711770163783243
Cost at Step 1400: -14.786751168636771
Cost at Step 1450: -14.804684082337417
Cost at Step 1500: -14.870638186084873
Cost at Step 1550: -14.936918418196457
Trace Distance: 0.07096501502412955
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
0: ──X─────────RZ(1.00)──RY(1.00)──RX(1.00)─╭C────────────────────────────╭RX(1.00)──RZ(1.00)
1: ──RZ(1.00)──RY(1.00)──RX(1.00)───────────╰RX(1.00)─╭C──────────────────│──────────RZ(1.00)
2: ──X─────────RZ(1.00)──RY(1.00)──RX(1.00)───────────╰RX(1.00)─╭C────────│──────────RZ(1.00)
3: ──RZ(1.00)──RY(1.00)──RX(1.00)───────────────────────────────╰RX(1.00)─╰C─────────RZ(1.00)
───RY(1.00)──RX(1.00)─╭C────────────────────────────╭RX(1.00)──RZ(1.00)──RY(1.00)──RX(1.00)
───RY(1.00)──RX(1.00)─╰RX(1.00)─╭C──────────────────│──────────RZ(1.00)──RY(1.00)──RX(1.00)
───RY(1.00)──RX(1.00)───────────╰RX(1.00)─╭C────────│──────────RZ(1.00)──RY(1.00)──RX(1.00)
───RY(1.00)──RX(1.00)─────────────────────╰RX(1.00)─╰C─────────RZ(1.00)──RY(1.00)──RX(1.00)
──╭C────────────────────────────╭RX(1.00)──RZ(1.00)──RY(1.00)──RX(1.00)─╭C─────────────────
──╰RX(1.00)─╭C──────────────────│──────────RZ(1.00)──RY(1.00)──RX(1.00)─╰RX(1.00)─╭C───────
────────────╰RX(1.00)─╭C────────│──────────RZ(1.00)──RY(1.00)──RX(1.00)───────────╰RX(1.00)
──────────────────────╰RX(1.00)─╰C─────────RZ(1.00)──RY(1.00)──RX(1.00)────────────────────
────────────╭RX(1.00)─┤ ╭<𝓗(M0)>
────────────│─────────┤ ├<𝓗(M0)>
──╭C────────│─────────┤ ├<𝓗(M0)>
──╰RX(1.00)─╰C────────┤ ╰<𝓗(M0)>
M0 =
[[ 4.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j
   0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  2.+0.j
   0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  2.+0.j
   0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j -4.+0.j  2.+0.j  0.+0.j  0.+0.j
   2.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  2.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j
   0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j
   0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j
   2.+0.j -4.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j
   0.+0.j  0.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j
   0.+0.j  2.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j
   0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j
   0.+0.j  0.+0.j  0.+0.j  0.+0.j  2.+0.j  0.+0.j  0.+0.j]
 [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j
   0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  4.+0.j]]
Cost at Step 0: -0.6605354666522008
Cost at Step 50: -2.869994162243926
Cost at Step 100: -4.642067184244079
Cost at Step 150: -5.127022428216538
Cost at Step 200: -6.5612379300427826
Cost at Step 250: -7.0582516090763665
Cost at Step 300: -7.640510120945974
Cost at Step 350: -8.257197960457098
Cost at Step 400: -9.063307267167723
Cost at Step 450: -9.807942741615197
Cost at Step 500: -10.444113688870624
Cost at Step 550: -10.55728830931337
Cost at Step 600: -11.024616702766755
Cost at Step 650: -11.653848423245002
Cost at Step 700: -12.076997848630482
Cost at Step 750: -12.314711165521649
Cost at Step 800: -12.423843768774816
Cost at Step 850: -12.774457419438855
Cost at Step 900: -12.993846556186526
Cost at Step 950: -13.225424165914621
Cost at Step 1000: -13.299919020159663
 </code>
 </pre>
 </details>

---

## 3. tutorial_expressivity_fourier_series.html <a name="demo2"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_expressivity_fourier_series.html):

```
0: ──Rot(2.35, 5.97, 4.6)──RX(6)──Rot(3.76, 0.98, 0.98)──┤ ⟨Z⟩
2: ──Rot(3.72, 5.18, 2.19)───────╰X─────────────────────────╰C──Rot(5.34, 5.45, 4.45)──╰X──────╰C──┤
0: ──Rot(1.38, 4.29, 0.478)──╭C─────────────────────────────╭X──Rot(4.26, 3.55, 1.68)──╭C──╭X──────┤ ⟨I⟩
1: ──Rot(5.35, 3.11, 3.02)───╰X──╭C──Rot(5.52, 5.01, 4.14)──│──────────────────────────│───╰C──╭X──┤
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_expressivity_fourier_series.html):

```
0: ──Rot(2.35,5.97,4.60)──RX(6.00)──Rot(3.76,0.98,0.98)─┤  <Z>
2: ──Rot(3.72,5.18,2.19)────╰X─╰C──Rot(5.34,5.45,4.45)─╰X────╰C─┤
0: ──Rot(1.38,4.29,0.48)─╭C────╭X──Rot(4.26,3.55,1.68)─╭C─╭X────┤  <I>
1: ──Rot(5.35,3.11,3.02)─╰X─╭C─│───Rot(5.52,5.01,4.14)─│──╰C─╭X─┤
```

---

## 4. tutorial_noisy_circuits.html <a name="demo3"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_noisy_circuits.html):

```
Step: 25    Cost: 0.006192562764640602
Step: 30    Cost: 6.427645677603198e-07
Step: 34    Cost: 1.1072988376257744e-09
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_noisy_circuits.html):

```
Step: 25    Cost: 0.00619256273608413
Step: 30    Cost: 6.427645721531149e-07
Step: 34    Cost: 1.1072988471351325e-09
```

---

## 5. tutorial_QGAN.html <a name="demo4"></a>

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
Step 10: cost = -0.9784243106842041
Step 15: cost = -0.9946483373641968
Step 20: cost = -0.9984995424747467
Step 25: cost = -0.9995636343955994
Step 30: cost = -0.999871701002121
Step 35: cost = -0.9999619424343109
Step 40: cost = -0.9999886155128479
Step 45: cost = -0.9999965131282806
Prob(fake classified as real):  0.9999986290931702
Discriminator cost:  0.0014115869998931885
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
Step 0: cost = -0.057276904582977295
Step 5: cost = -0.2634810581803322
Step 10: cost = -0.42739149928092957
Step 15: cost = -0.4726157709956169
Step 20: cost = -0.48406896740198135
Step 25: cost = -0.4894639253616333
Step 30: cost = -0.4928186237812042
Step 35: cost = -0.4949491620063782
Step 40: cost = -0.4962702840566635
Step 45: cost = -0.4970718026161194
Prob(real classified as real):  0.9985870718955994
Prob(fake classified as real):  0.5011128038167953
Step 0: cost = -0.5833386406302452
Step 5: cost = -0.8915732204914093
Step 10: cost = -0.9784242212772369
Step 15: cost = -0.9946482181549072
Step 20: cost = -0.9984995126724243
Step 25: cost = -0.9995637834072113
Step 30: cost = -0.9998717606067657
Step 35: cost = -0.999961793422699
Step 40: cost = -0.9999887347221375
Step 45: cost = -0.9999964535236359
Prob(fake classified as real):  0.9999987185001373
Discriminator cost:  0.0014116466045379639
Generator Bloch vector: [-0.28404659  0.41893226 -0.86244404]
 </code>
 </pre>
 </details>

---

## 6. tutorial_rosalin.html <a name="demo5"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_rosalin.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Step 9: cost = -7.43238804878266 shots_used = 24000
Step 10: cost = -7.374281342889537 shots_used = 26400
Step 11: cost = -7.2878455754894835 shots_used = 28800
Step 12: cost = -7.211212391636272 shots_used = 31200
Step 13: cost = -7.168136331225393 shots_used = 33600
Step 14: cost = -7.171989037816058 shots_used = 36000
Step 15: cost = -7.215331794272855 shots_used = 38400
Step 16: cost = -7.278024044019375 shots_used = 40800
Step 17: cost = -7.361449931178438 shots_used = 43200
Step 18: cost = -7.442410269187307 shots_used = 45600
Step 22: cost = -7.608322534779552 shots_used = 55200
Step 23: cost = -7.607043067486308 shots_used = 57600
Step 24: cost = -7.594076978496339 shots_used = 60000
Step 25: cost = -7.5791531795788 shots_used = 62400
Step 27: cost = -7.568439746440855 shots_used = 67200
Step 28: cost = -7.581935968178152 shots_used = 69600
Step 29: cost = -7.610907153836916 shots_used = 72000
Step 30: cost = -7.651198088153218 shots_used = 74400
Step 31: cost = -7.697526604943235 shots_used = 76800
Step 32: cost = -7.7469033971024 shots_used = 79200
Step 33: cost = -7.787293318189748 shots_used = 81600
Step 35: cost = -7.8407299133653945 shots_used = 86400
Step 36: cost = -7.858055514435193 shots_used = 88800
Step 38: cost = -7.877775001403506 shots_used = 93600
Step 39: cost = -7.884473822847106 shots_used = 96000
Step 40: cost = -7.887248861002907 shots_used = 98400
Step 41: cost = -7.8859305197679666 shots_used = 100800
Step 42: cost = -7.881002890765943 shots_used = 103200
Step 44: cost = -7.866129786750201 shots_used = 108000
Step 45: cost = -7.850581153251574 shots_used = 110400
Step 46: cost = -7.843337695237988 shots_used = 112800
Step 47: cost = -7.845453624375395 shots_used = 115200
Step 48: cost = -7.853444576995694 shots_used = 117600
Step 49: cost = -7.858018368114795 shots_used = 120000
Step 50: cost = -7.858043805938923 shots_used = 122400
Step 51: cost = -7.855559046474576 shots_used = 124800
Step 52: cost = -7.850626102015815 shots_used = 127200
Step 53: cost = -7.848969631273947 shots_used = 129600
Step 54: cost = -7.852176020039671 shots_used = 132000
Step 55: cost = -7.861427874163019 shots_used = 134400
Step 56: cost = -7.868403253322002 shots_used = 136800
Step 58: cost = -7.88046548748952 shots_used = 141600
Step 59: cost = -7.88002615475712 shots_used = 144000
Step 60: cost = -7.877251725772389 shots_used = 146400
Step 61: cost = -7.870289689150822 shots_used = 148800
Step 63: cost = -7.862715331323385 shots_used = 153600
Step 64: cost = -7.861607002909471 shots_used = 156000
Step 65: cost = -7.866735539612959 shots_used = 158400
Step 66: cost = -7.867386735902062 shots_used = 160800
Step 67: cost = -7.867196452121144 shots_used = 163200
Step 70: cost = -7.86657829219507 shots_used = 170400
Step 71: cost = -7.857706098037314 shots_used = 172800
Step 72: cost = -7.8558663452092246 shots_used = 175200
Step 73: cost = -7.858187887358694 shots_used = 177600
Step 75: cost = -7.864825877239141 shots_used = 182400
Step 76: cost = -7.8635708249475975 shots_used = 184800
Step 77: cost = -7.863497614169013 shots_used = 187200
Step 78: cost = -7.860326845355643 shots_used = 189600
Step 79: cost = -7.855301086551648 shots_used = 192000
Step 80: cost = -7.855890069918929 shots_used = 194400
Step 82: cost = -7.863450868072558 shots_used = 199200
Step 83: cost = -7.870058142679087 shots_used = 201600
Step 85: cost = -7.883842326350927 shots_used = 206400
Step 86: cost = -7.882633952688103 shots_used = 208800
Step 87: cost = -7.879224942149157 shots_used = 211200
Step 88: cost = -7.872184015334469 shots_used = 213600
Step 89: cost = -7.864992896022884 shots_used = 216000
Step 91: cost = -7.859810127594738 shots_used = 220800
Step 92: cost = -7.86306566068536 shots_used = 223200
Step 93: cost = -7.868586359942562 shots_used = 225600
Step 94: cost = -7.874757156105922 shots_used = 228000
Step 95: cost = -7.879893808186249 shots_used = 230400
Step 96: cost = -7.8833173990872165 shots_used = 232800
Step 97: cost = -7.883690480348113 shots_used = 235200
Step 98: cost = -7.8820611003810965 shots_used = 237600
Step 99: cost = -7.8789076755031635 shots_used = 240000
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
Step 9: cost = -7.432388048782661 shots_used = 24000
Step 10: cost = -7.374281342889539 shots_used = 26400
Step 11: cost = -7.287845575489478 shots_used = 28800
Step 12: cost = -7.211212391636275 shots_used = 31200
Step 13: cost = -7.168136331225396 shots_used = 33600
Step 14: cost = -7.171989037816061 shots_used = 36000
Step 15: cost = -7.215331794272854 shots_used = 38400
Step 16: cost = -7.278024044019372 shots_used = 40800
Step 17: cost = -7.36144993117844 shots_used = 43200
Step 18: cost = -7.442410269187304 shots_used = 45600
Step 22: cost = -7.6083225347795524 shots_used = 55200
Step 23: cost = -7.6070430674863045 shots_used = 57600
Step 24: cost = -7.594076978496342 shots_used = 60000
Step 25: cost = -7.579153179578801 shots_used = 62400
Step 27: cost = -7.5684397464408555 shots_used = 67200
Step 28: cost = -7.5819359681781515 shots_used = 69600
Step 29: cost = -7.610907153836915 shots_used = 72000
Step 30: cost = -7.651198088153216 shots_used = 74400
Step 31: cost = -7.697526604943237 shots_used = 76800
Step 32: cost = -7.746903397102402 shots_used = 79200
Step 33: cost = -7.787293318189752 shots_used = 81600
Step 35: cost = -7.840729913365394 shots_used = 86400
Step 36: cost = -7.858055514435198 shots_used = 88800
Step 38: cost = -7.87777500140351 shots_used = 93600
Step 39: cost = -7.884473822847109 shots_used = 96000
Step 40: cost = -7.887248861002908 shots_used = 98400
Step 41: cost = -7.885930519767962 shots_used = 100800
Step 42: cost = -7.8810028907659415 shots_used = 103200
Step 44: cost = -7.866129786750197 shots_used = 108000
Step 45: cost = -7.8505811532515715 shots_used = 110400
Step 46: cost = -7.8433376952379845 shots_used = 112800
Step 47: cost = -7.845453624375393 shots_used = 115200
Step 48: cost = -7.8534445769956935 shots_used = 117600
Step 49: cost = -7.8580183681147915 shots_used = 120000
Step 50: cost = -7.858043805938924 shots_used = 122400
Step 51: cost = -7.855559046474577 shots_used = 124800
Step 52: cost = -7.850626102015814 shots_used = 127200
Step 53: cost = -7.8489696312739445 shots_used = 129600
Step 54: cost = -7.852176020039672 shots_used = 132000
Step 55: cost = -7.86142787416302 shots_used = 134400
Step 56: cost = -7.868403253322004 shots_used = 136800
Step 58: cost = -7.880465487489517 shots_used = 141600
Step 59: cost = -7.880026154757117 shots_used = 144000
Step 60: cost = -7.877251725772391 shots_used = 146400
Step 61: cost = -7.87028968915082 shots_used = 148800
Step 63: cost = -7.862715331323388 shots_used = 153600
Step 64: cost = -7.861607002909468 shots_used = 156000
Step 65: cost = -7.866735539612957 shots_used = 158400
Step 66: cost = -7.8673867359020635 shots_used = 160800
Step 67: cost = -7.867196452121142 shots_used = 163200
Step 70: cost = -7.866578292195072 shots_used = 170400
Step 71: cost = -7.857706098037319 shots_used = 172800
Step 72: cost = -7.855866345209227 shots_used = 175200
Step 73: cost = -7.858187887358697 shots_used = 177600
Step 75: cost = -7.864825877239139 shots_used = 182400
Step 76: cost = -7.863570824947598 shots_used = 184800
Step 77: cost = -7.86349761416901 shots_used = 187200
Step 78: cost = -7.8603268453556385 shots_used = 189600
Step 79: cost = -7.855301086551645 shots_used = 192000
Step 80: cost = -7.855890069918927 shots_used = 194400
Step 82: cost = -7.863450868072559 shots_used = 199200
Step 83: cost = -7.870058142679085 shots_used = 201600
Step 85: cost = -7.883842326350928 shots_used = 206400
Step 86: cost = -7.882633952688098 shots_used = 208800
Step 87: cost = -7.87922494214916 shots_used = 211200
Step 88: cost = -7.872184015334465 shots_used = 213600
Step 89: cost = -7.864992896022885 shots_used = 216000
Step 91: cost = -7.85981012759474 shots_used = 220800
Step 92: cost = -7.863065660685359 shots_used = 223200
Step 93: cost = -7.868586359942557 shots_used = 225600
Step 94: cost = -7.8747571561059235 shots_used = 228000
Step 95: cost = -7.879893808186248 shots_used = 230400
Step 96: cost = -7.883317399087219 shots_used = 232800
Step 97: cost = -7.883690480348111 shots_used = 235200
Step 98: cost = -7.882061100381098 shots_used = 237600
Step 99: cost = -7.878907675503154 shots_used = 240000
 </code>
 </pre>
 </details>

---

## 7. tutorial_quanvolution.html <a name="demo6"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quanvolution.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
   16384/11490434 [..............................] - ETA: 3s
   90112/11490434 [..............................] - ETA: 7s
  417792/11490434 [>.............................] - ETA: 2s
 1900544/11490434 [===>..........................] - ETA: 0s
 7471104/11490434 [==================>...........] - ETA: 0s
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
 1548288/11490434 [===>..........................] - ETA: 0s
10944512/11490434 [===========================>..] - ETA: 0s
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
13/13 - 0s - loss: 3.0160 - accuracy: 0.1000 - val_loss: 2.0646 - val_accuracy: 0.2000 - 448ms/epoch - 34ms/step
Epoch 2/30
13/13 - 0s - loss: 2.2510 - accuracy: 0.1800 - val_loss: 1.9801 - val_accuracy: 0.3333 - 50ms/epoch - 4ms/step
Epoch 3/30
13/13 - 0s - loss: 1.7851 - accuracy: 0.4000 - val_loss: 1.8177 - val_accuracy: 0.2667 - 49ms/epoch - 4ms/step
Epoch 4/30
13/13 - 0s - loss: 1.3652 - accuracy: 0.5400 - val_loss: 1.6107 - val_accuracy: 0.4667 - 49ms/epoch - 4ms/step
Epoch 5/30
13/13 - 0s - loss: 1.1317 - accuracy: 0.7800 - val_loss: 1.4723 - val_accuracy: 0.6000 - 36ms/epoch - 3ms/step
Epoch 6/30
13/13 - 0s - loss: 0.9360 - accuracy: 0.8600 - val_loss: 1.4686 - val_accuracy: 0.5333 - 49ms/epoch - 4ms/step
Epoch 7/30
13/13 - 0s - loss: 0.7383 - accuracy: 0.9400 - val_loss: 1.3536 - val_accuracy: 0.5667 - 35ms/epoch - 3ms/step
Epoch 8/30
13/13 - 0s - loss: 0.5846 - accuracy: 0.9800 - val_loss: 1.2785 - val_accuracy: 0.6667 - 33ms/epoch - 3ms/step
Epoch 9/30
13/13 - 0s - loss: 0.4987 - accuracy: 0.9800 - val_loss: 1.2253 - val_accuracy: 0.6333 - 33ms/epoch - 3ms/step
Epoch 10/30
13/13 - 0s - loss: 0.3921 - accuracy: 1.0000 - val_loss: 1.2655 - val_accuracy: 0.6333 - 33ms/epoch - 3ms/step
Epoch 11/30
13/13 - 0s - loss: 0.3617 - accuracy: 1.0000 - val_loss: 1.1555 - val_accuracy: 0.7000 - 48ms/epoch - 4ms/step
Epoch 12/30
13/13 - 0s - loss: 0.3078 - accuracy: 1.0000 - val_loss: 1.2107 - val_accuracy: 0.6667 - 35ms/epoch - 3ms/step
Epoch 13/30
13/13 - 0s - loss: 0.2618 - accuracy: 1.0000 - val_loss: 1.1166 - val_accuracy: 0.7333 - 47ms/epoch - 4ms/step
Epoch 14/30
13/13 - 0s - loss: 0.2463 - accuracy: 1.0000 - val_loss: 1.0624 - val_accuracy: 0.7000 - 48ms/epoch - 4ms/step
Epoch 15/30
13/13 - 0s - loss: 0.2033 - accuracy: 1.0000 - val_loss: 1.0904 - val_accuracy: 0.7333 - 50ms/epoch - 4ms/step
Epoch 16/30
13/13 - 0s - loss: 0.1799 - accuracy: 1.0000 - val_loss: 1.0865 - val_accuracy: 0.7333 - 34ms/epoch - 3ms/step
Epoch 17/30
13/13 - 0s - loss: 0.1682 - accuracy: 1.0000 - val_loss: 1.0385 - val_accuracy: 0.7333 - 50ms/epoch - 4ms/step
Epoch 18/30
13/13 - 0s - loss: 0.1484 - accuracy: 1.0000 - val_loss: 1.0676 - val_accuracy: 0.7000 - 49ms/epoch - 4ms/step
Epoch 19/30
13/13 - 0s - loss: 0.1349 - accuracy: 1.0000 - val_loss: 1.0447 - val_accuracy: 0.7000 - 49ms/epoch - 4ms/step
Epoch 20/30
13/13 - 0s - loss: 0.1255 - accuracy: 1.0000 - val_loss: 0.9935 - val_accuracy: 0.7333 - 48ms/epoch - 4ms/step
Epoch 21/30
13/13 - 0s - loss: 0.1135 - accuracy: 1.0000 - val_loss: 1.0451 - val_accuracy: 0.7333 - 49ms/epoch - 4ms/step
Epoch 22/30
13/13 - 0s - loss: 0.1041 - accuracy: 1.0000 - val_loss: 1.0142 - val_accuracy: 0.7333 - 48ms/epoch - 4ms/step
Epoch 23/30
13/13 - 0s - loss: 0.0983 - accuracy: 1.0000 - val_loss: 0.9893 - val_accuracy: 0.7333 - 46ms/epoch - 4ms/step
Epoch 24/30
13/13 - 0s - loss: 0.0913 - accuracy: 1.0000 - val_loss: 0.9807 - val_accuracy: 0.7000 - 33ms/epoch - 3ms/step
Epoch 25/30
13/13 - 0s - loss: 0.0868 - accuracy: 1.0000 - val_loss: 0.9715 - val_accuracy: 0.7333 - 48ms/epoch - 4ms/step
Epoch 26/30
13/13 - 0s - loss: 0.0788 - accuracy: 1.0000 - val_loss: 0.9850 - val_accuracy: 0.7333 - 32ms/epoch - 2ms/step
Epoch 27/30
13/13 - 0s - loss: 0.0749 - accuracy: 1.0000 - val_loss: 0.9750 - val_accuracy: 0.7333 - 45ms/epoch - 3ms/step
Epoch 28/30
13/13 - 0s - loss: 0.0730 - accuracy: 1.0000 - val_loss: 0.9570 - val_accuracy: 0.7667 - 36ms/epoch - 3ms/step
Epoch 29/30
13/13 - 0s - loss: 0.0681 - accuracy: 1.0000 - val_loss: 0.9895 - val_accuracy: 0.7333 - 49ms/epoch - 4ms/step
Epoch 30/30
13/13 - 0s - loss: 0.0635 - accuracy: 1.0000 - val_loss: 0.9560 - val_accuracy: 0.7333 - 38ms/epoch - 3ms/step
Epoch 1/30
13/13 - 0s - loss: 2.3619 - accuracy: 0.1400 - val_loss: 2.0567 - val_accuracy: 0.3667 - 388ms/epoch - 30ms/step
Epoch 2/30
13/13 - 0s - loss: 1.9696 - accuracy: 0.4200 - val_loss: 1.9381 - val_accuracy: 0.4667 - 48ms/epoch - 4ms/step
Epoch 3/30
13/13 - 0s - loss: 1.6671 - accuracy: 0.6400 - val_loss: 1.8300 - val_accuracy: 0.4333 - 48ms/epoch - 4ms/step
Epoch 4/30
13/13 - 0s - loss: 1.4340 - accuracy: 0.7400 - val_loss: 1.7113 - val_accuracy: 0.4333 - 33ms/epoch - 3ms/step
Epoch 5/30
13/13 - 0s - loss: 1.2342 - accuracy: 0.7600 - val_loss: 1.6044 - val_accuracy: 0.5000 - 36ms/epoch - 3ms/step
Epoch 6/30
13/13 - 0s - loss: 1.0721 - accuracy: 0.8600 - val_loss: 1.5232 - val_accuracy: 0.5333 - 36ms/epoch - 3ms/step
Epoch 7/30
13/13 - 0s - loss: 0.9348 - accuracy: 0.9000 - val_loss: 1.4596 - val_accuracy: 0.5667 - 49ms/epoch - 4ms/step
Epoch 8/30
13/13 - 0s - loss: 0.8178 - accuracy: 0.9200 - val_loss: 1.3921 - val_accuracy: 0.6000 - 33ms/epoch - 3ms/step
Epoch 9/30
13/13 - 0s - loss: 0.7223 - accuracy: 0.9400 - val_loss: 1.3404 - val_accuracy: 0.6333 - 35ms/epoch - 3ms/step
Epoch 10/30
13/13 - 0s - loss: 0.6404 - accuracy: 0.9600 - val_loss: 1.3065 - val_accuracy: 0.6667 - 36ms/epoch - 3ms/step
Epoch 11/30
13/13 - 0s - loss: 0.5772 - accuracy: 1.0000 - val_loss: 1.2644 - val_accuracy: 0.6333 - 49ms/epoch - 4ms/step
Epoch 12/30
13/13 - 0s - loss: 0.5199 - accuracy: 1.0000 - val_loss: 1.2558 - val_accuracy: 0.6667 - 35ms/epoch - 3ms/step
Epoch 13/30
13/13 - 0s - loss: 0.4695 - accuracy: 1.0000 - val_loss: 1.2258 - val_accuracy: 0.6667 - 35ms/epoch - 3ms/step
Epoch 14/30
13/13 - 0s - loss: 0.4238 - accuracy: 1.0000 - val_loss: 1.1897 - val_accuracy: 0.6667 - 49ms/epoch - 4ms/step
Epoch 15/30
13/13 - 0s - loss: 0.3848 - accuracy: 1.0000 - val_loss: 1.1651 - val_accuracy: 0.6667 - 35ms/epoch - 3ms/step
Epoch 16/30
13/13 - 0s - loss: 0.3525 - accuracy: 1.0000 - val_loss: 1.1503 - val_accuracy: 0.7333 - 35ms/epoch - 3ms/step
Epoch 17/30
13/13 - 0s - loss: 0.3245 - accuracy: 1.0000 - val_loss: 1.1374 - val_accuracy: 0.7000 - 48ms/epoch - 4ms/step
Epoch 18/30
13/13 - 0s - loss: 0.2992 - accuracy: 1.0000 - val_loss: 1.1174 - val_accuracy: 0.6667 - 34ms/epoch - 3ms/step
Epoch 19/30
13/13 - 0s - loss: 0.2745 - accuracy: 1.0000 - val_loss: 1.1119 - val_accuracy: 0.6667 - 48ms/epoch - 4ms/step
Epoch 20/30
13/13 - 0s - loss: 0.2551 - accuracy: 1.0000 - val_loss: 1.0903 - val_accuracy: 0.7000 - 48ms/epoch - 4ms/step
Epoch 21/30
13/13 - 0s - loss: 0.2370 - accuracy: 1.0000 - val_loss: 1.0877 - val_accuracy: 0.6667 - 46ms/epoch - 4ms/step
Epoch 22/30
13/13 - 0s - loss: 0.2199 - accuracy: 1.0000 - val_loss: 1.0776 - val_accuracy: 0.6667 - 48ms/epoch - 4ms/step
Epoch 23/30
13/13 - 0s - loss: 0.2050 - accuracy: 1.0000 - val_loss: 1.0675 - val_accuracy: 0.7000 - 48ms/epoch - 4ms/step
Epoch 24/30
13/13 - 0s - loss: 0.1919 - accuracy: 1.0000 - val_loss: 1.0592 - val_accuracy: 0.7000 - 48ms/epoch - 4ms/step
Epoch 25/30
13/13 - 0s - loss: 0.1812 - accuracy: 1.0000 - val_loss: 1.0554 - val_accuracy: 0.6667 - 49ms/epoch - 4ms/step
Epoch 26/30
13/13 - 0s - loss: 0.1691 - accuracy: 1.0000 - val_loss: 1.0477 - val_accuracy: 0.7000 - 46ms/epoch - 4ms/step
Epoch 27/30
13/13 - 0s - loss: 0.1599 - accuracy: 1.0000 - val_loss: 1.0377 - val_accuracy: 0.7000 - 46ms/epoch - 4ms/step
Epoch 28/30
13/13 - 0s - loss: 0.1515 - accuracy: 1.0000 - val_loss: 1.0341 - val_accuracy: 0.6667 - 34ms/epoch - 3ms/step
Epoch 29/30
13/13 - 0s - loss: 0.1426 - accuracy: 1.0000 - val_loss: 1.0291 - val_accuracy: 0.7000 - 49ms/epoch - 4ms/step
Epoch 30/30
13/13 - 0s - loss: 0.1344 - accuracy: 1.0000 - val_loss: 1.0264 - val_accuracy: 0.7000 - 34ms/epoch - 3ms/step
 </code>
 </pre>
 </details>

---

## 8. tutorial_quantum_analytic_descent.html <a name="demo7"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quantum_analytic_descent.html):

```
True energy at the minimum of the model: -0.7358296722728767
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_quantum_analytic_descent.html):

```
True energy at the minimum of the model: -0.735829672272876
```

---

## 9. tutorial_noisy_circuit_optimization.html <a name="demo8"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_noisy_circuit_optimization.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Expectation value: 0.856
Step     5. Cost:  0.9540000; Noisy Cost:  0.5740000
Step    10. Cost: -0.0040000; Noisy Cost:  0.5320000
Step    15. Cost: -0.9760000; Noisy Cost:  0.2160000
Step    20. Cost: -1.0000000; Noisy Cost: -0.4220000
Step    25. Cost: -1.0000000; Noisy Cost: -0.5760000
Step    30. Cost: -1.0000000; Noisy Cost: -0.6040000
Step    35. Cost: -1.0000000; Noisy Cost: -0.5940000
Step    40. Cost: -1.0000000; Noisy Cost: -0.5900000
Step    45. Cost: -1.0000000; Noisy Cost: -0.6080000
Step    50. Cost: -1.0000000; Noisy Cost: -0.5680000
Step    55. Cost: -1.0000000; Noisy Cost: -0.5660000
Step    60. Cost: -1.0000000; Noisy Cost: -0.6120000
Step    65. Cost: -1.0000000; Noisy Cost: -0.5780000
Step    70. Cost: -1.0000000; Noisy Cost: -0.6280000
Step    75. Cost: -1.0000000; Noisy Cost: -0.5920000
Step    80. Cost: -1.0000000; Noisy Cost: -0.5120000
Step    85. Cost: -1.0000000; Noisy Cost: -0.5720000
Step    90. Cost: -1.0000000; Noisy Cost: -0.6000000
Step    95. Cost: -1.0000000; Noisy Cost: -0.5660000
Step   100. Cost: -1.0000000; Noisy Cost: -0.6240000
( 0.0186000,  3.1390000)
(-0.0182000,  3.1126000)
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
Expectation value: 0.862
Step     5. Cost:  0.9520000; Noisy Cost:  0.5900000
Step    10. Cost:  0.0480000; Noisy Cost:  0.5400000
Step    15. Cost: -0.9540000; Noisy Cost:  0.3320000
Step    20. Cost: -1.0000000; Noisy Cost: -0.1720000
Step    25. Cost: -0.9980000; Noisy Cost: -0.5300000
Step    30. Cost: -1.0000000; Noisy Cost: -0.6400000
Step    35. Cost: -1.0000000; Noisy Cost: -0.5980000
Step    40. Cost: -1.0000000; Noisy Cost: -0.5960000
Step    45. Cost: -1.0000000; Noisy Cost: -0.6120000
Step    50. Cost: -1.0000000; Noisy Cost: -0.6400000
Step    55. Cost: -1.0000000; Noisy Cost: -0.6000000
Step    60. Cost: -1.0000000; Noisy Cost: -0.5920000
Step    65. Cost: -1.0000000; Noisy Cost: -0.5820000
Step    70. Cost: -1.0000000; Noisy Cost: -0.5840000
Step    75. Cost: -1.0000000; Noisy Cost: -0.5460000
Step    80. Cost: -0.9980000; Noisy Cost: -0.6020000
Step    85. Cost: -1.0000000; Noisy Cost: -0.6120000
Step    90. Cost: -1.0000000; Noisy Cost: -0.6020000
Step    95. Cost: -1.0000000; Noisy Cost: -0.6080000
Step   100. Cost: -1.0000000; Noisy Cost: -0.5800000
(-0.0078000,  3.1210000)
(-0.0266000,  3.1430000)
 </code>
 </pre>
 </details>

---

## 10. tutorial_error_mitigation.html <a name="demo9"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_error_mitigation.html):

```
0: ──RY(4.56)──╭C──RY(5.93)───RY(-5.93)──────────────────────────────────────╭C──RY(-4.56)──┤ ⟨Z⟩
1: ──RY(3.6)───╰Z──RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z──RY(-3.6)───┤
2: ──RY(4.05)──╭C──RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)──╰Z──RY(-3.32)──╭C──RY(-4.05)──┤
3: ──RY(3.51)──╰Z──RY(3.66)───RY(-3.66)──────────────────────────────────────╰Z──RY(-3.51)──┤
0.8985196547410969
ZNE result: 0.8985196547410969
0.9766035030623954
0: ──RY(4.56)──╭C──RY(5.93)───RY(-5.93)──────────────────────────────────────╭C──RY(-4.56)─────────────────────┤
1: ──RY(3.6)───╰Z──RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z──RY(-3.6)───RY(3.6)──RY(-3.6)──┤
2: ──RY(4.05)──╭C──RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)──╰Z──RY(-3.32)──╭C──RY(-4.05)─────────────────────┤
3: ──RY(3.51)──╰Z──RY(3.66)───RY(-3.66)──────────────────────────────────────╰Z──RY(-3.51)─────────────────────┤
0: ──RY(4.56)──╭C──RY(5.93)───RY(-5.93)──────────────────────────────────────╭C──RY(-4.56)──RY(4.56)──RY(-4.56)──┤
1: ──RY(3.6)───╰Z──RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z──RY(-3.6)────────────────────────┤
2: ──RY(4.05)──╭C──RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)──╰Z──RY(-3.32)──╭C──RY(-4.05)───────────────────────┤
3: ──RY(3.51)──╰Z──RY(3.66)───RY(-3.66)──────────────────────────────────────╰Z──RY(-3.51)───────────────────────┤
0: ──RY(4.56)──╭C──RY(5.93)───RY(-5.93)──────────────────────────────────────╭C──╭C──────────╭C──RY(-4.56)──┤
1: ──RY(3.6)───╰Z──RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z──╰Z──────────╰Z──RY(-3.6)───┤
2: ──RY(4.05)──╭C──RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)──╰Z──RY(-3.32)──╭C───RY(-4.05)─────────────────┤
3: ──RY(3.51)──╰Z──RY(3.66)───RY(-3.66)──────────────────────────────────────╰Z───RY(-3.51)─────────────────┤
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_error_mitigation.html):

```
0: ──RY(4.56)─╭C──RY(5.93)──RY(-5.93)────────────────────────────────────╭C──RY(-4.56)─┤  <Z>
1: ──RY(3.60)─╰Z──RY(5.90)─╭C──────────RY(5.18)──RY(-5.18)─╭C──RY(-5.90)─╰Z──RY(-3.60)─┤
2: ──RY(4.05)─╭C──RY(3.32)─╰Z──────────RY(1.07)──RY(-1.07)─╰Z──RY(-3.32)─╭C──RY(-4.05)─┤
3: ──RY(3.51)─╰Z──RY(3.66)──RY(-3.66)────────────────────────────────────╰Z──RY(-3.51)─┤
0.8985196547410967
ZNE result: 0.8985196547410967
0.9633064166763504
0: ──RY(4.56)──╭C──RY(5.93)───RY(-5.93)───────────────────────────────────────────────────────────╭C──RY(-4.56)──┤
1: ──RY(3.6)───╰Z──RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)──RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z──RY(-3.6)───┤
2: ──RY(4.05)──╭C──RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)───────────────────────╰Z──RY(-3.32)──╭C──RY(-4.05)──┤
3: ──RY(3.51)──╰Z──RY(3.66)───RY(-3.66)───────────────────────────────────────────────────────────╰Z──RY(-3.51)──┤
0: ──RY(4.56)──╭C──RY(5.93)──RY(-5.93)───────────────────────────────────────────────────╭C──RY(-4.56)──┤
1: ──RY(3.6)───╰Z──RY(5.9)───RY(-5.9)───RY(5.9)──╭C──RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z──RY(-3.6)───┤
2: ──RY(4.05)──╭C──RY(3.32)──────────────────────╰Z──RY(1.07)──RY(-1.07)──╰Z──RY(-3.32)──╭C──RY(-4.05)──┤
3: ──RY(3.51)──╰Z──RY(3.66)──RY(-3.66)───────────────────────────────────────────────────╰Z──RY(-3.51)──┤
0: ──RY(4.56)──╭C──RY(5.93)───RY(-5.93)──────────────────────────────────────╭C───RY(-4.56)─────────────────┤
1: ──RY(3.6)───╰Z──RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z───RY(-3.6)──────────────────┤
2: ──RY(4.05)──╭C──RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)──╰Z──RY(-3.32)──╭C──╭C──────────╭C──RY(-4.05)──┤
3: ──RY(3.51)──╰Z──RY(3.66)───RY(-3.66)──────────────────────────────────────╰Z──╰Z──────────╰Z──RY(-3.51)──┤
```

---

## 11. tutorial_doubly_stochastic.html <a name="demo10"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_doubly_stochastic.html):

```
Stochastic gradient descent (shots=100) min energy =  -4.600655176916145
Stochastic gradient descent (shots=1) min energy =  -4.457668962761634
Doubly stochastic gradient descent min energy =  -4.4990195930951575
Adaptive QSGD min energy =  -4.59254874161316
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_doubly_stochastic.html):

```
Stochastic gradient descent (shots=100) min energy =  -4.600655176916143
Stochastic gradient descent (shots=1) min energy =  -4.457668962761635
Doubly stochastic gradient descent min energy =  -4.499019593095155
Adaptive QSGD min energy =  -4.592548741613159
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
  8%|8         | 3.75M/44.7M [00:00<00:01, 38.7MB/s]
 17%|#6        | 7.45M/44.7M [00:00<00:01, 28.9MB/s]
 23%|##3       | 10.4M/44.7M [00:00<00:01, 29.6MB/s]
 30%|##9       | 13.4M/44.7M [00:00<00:01, 25.4MB/s]
 36%|###5      | 15.9M/44.7M [00:00<00:01, 25.2MB/s]
 44%|####3     | 19.6M/44.7M [00:00<00:00, 29.2MB/s]
 52%|#####2    | 23.3M/44.7M [00:00<00:00, 32.2MB/s]
 59%|#####9    | 26.5M/44.7M [00:00<00:00, 31.9MB/s]
 68%|######7   | 30.3M/44.7M [00:01<00:00, 34.2MB/s]
 75%|#######5  | 33.6M/44.7M [00:01<00:00, 26.0MB/s]
 83%|########3 | 37.1M/44.7M [00:01<00:00, 28.2MB/s]
 96%|#########5| 42.7M/44.7M [00:01<00:00, 35.8MB/s]
100%|##########| 44.7M/44.7M [00:01<00:00, 31.8MB/s]
Training started:
Phase: train Epoch: 1/3 Iter: 1/62 Batch time: 0.2378
Phase: train Epoch: 1/3 Iter: 2/62 Batch time: 0.2072
Phase: train Epoch: 1/3 Iter: 3/62 Batch time: 0.2119
Phase: train Epoch: 1/3 Iter: 4/62 Batch time: 0.2097
Phase: train Epoch: 1/3 Iter: 5/62 Batch time: 0.2105
Phase: train Epoch: 1/3 Iter: 6/62 Batch time: 0.2101
Phase: train Epoch: 1/3 Iter: 7/62 Batch time: 0.2101
Phase: train Epoch: 1/3 Iter: 8/62 Batch time: 0.2113
Phase: train Epoch: 1/3 Iter: 9/62 Batch time: 0.2091
Phase: train Epoch: 1/3 Iter: 10/62 Batch time: 0.2135
Phase: train Epoch: 1/3 Iter: 11/62 Batch time: 0.2093
Phase: train Epoch: 1/3 Iter: 12/62 Batch time: 0.2092
Phase: train Epoch: 1/3 Iter: 13/62 Batch time: 0.2115
Phase: train Epoch: 1/3 Iter: 14/62 Batch time: 0.2108
Phase: train Epoch: 1/3 Iter: 15/62 Batch time: 0.2148
Phase: train Epoch: 1/3 Iter: 16/62 Batch time: 0.2103
Phase: train Epoch: 1/3 Iter: 17/62 Batch time: 0.2107
Phase: train Epoch: 1/3 Iter: 18/62 Batch time: 0.2093
Phase: train Epoch: 1/3 Iter: 19/62 Batch time: 0.2115
Phase: train Epoch: 1/3 Iter: 20/62 Batch time: 0.2100
Phase: train Epoch: 1/3 Iter: 21/62 Batch time: 0.2102
Phase: train Epoch: 1/3 Iter: 22/62 Batch time: 0.2097
Phase: train Epoch: 1/3 Iter: 23/62 Batch time: 0.2117
Phase: train Epoch: 1/3 Iter: 24/62 Batch time: 0.2109
Phase: train Epoch: 1/3 Iter: 25/62 Batch time: 0.2100
Phase: train Epoch: 1/3 Iter: 26/62 Batch time: 0.2099
Phase: train Epoch: 1/3 Iter: 27/62 Batch time: 0.2101
Phase: train Epoch: 1/3 Iter: 28/62 Batch time: 0.2096
Phase: train Epoch: 1/3 Iter: 29/62 Batch time: 0.2095
Phase: train Epoch: 1/3 Iter: 30/62 Batch time: 0.2116
Phase: train Epoch: 1/3 Iter: 31/62 Batch time: 0.2099
Phase: train Epoch: 1/3 Iter: 32/62 Batch time: 0.2103
Phase: train Epoch: 1/3 Iter: 33/62 Batch time: 0.2123
Phase: train Epoch: 1/3 Iter: 34/62 Batch time: 0.2105
Phase: train Epoch: 1/3 Iter: 35/62 Batch time: 0.2095
Phase: train Epoch: 1/3 Iter: 36/62 Batch time: 0.2107
Phase: train Epoch: 1/3 Iter: 37/62 Batch time: 0.2106
Phase: train Epoch: 1/3 Iter: 38/62 Batch time: 0.2106
Phase: train Epoch: 1/3 Iter: 39/62 Batch time: 0.2116
Phase: train Epoch: 1/3 Iter: 40/62 Batch time: 0.2121
Phase: train Epoch: 1/3 Iter: 41/62 Batch time: 0.2126
Phase: train Epoch: 1/3 Iter: 42/62 Batch time: 0.2122
Phase: train Epoch: 1/3 Iter: 43/62 Batch time: 0.2113
Phase: train Epoch: 1/3 Iter: 44/62 Batch time: 0.2092
Phase: train Epoch: 1/3 Iter: 45/62 Batch time: 0.2099
Phase: train Epoch: 1/3 Iter: 46/62 Batch time: 0.2105
Phase: train Epoch: 1/3 Iter: 47/62 Batch time: 0.2102
Phase: train Epoch: 1/3 Iter: 48/62 Batch time: 0.2145
Phase: train Epoch: 1/3 Iter: 49/62 Batch time: 0.2140
Phase: train Epoch: 1/3 Iter: 50/62 Batch time: 0.2148
Phase: train Epoch: 1/3 Iter: 51/62 Batch time: 0.2135
Phase: train Epoch: 1/3 Iter: 52/62 Batch time: 0.2162
Phase: train Epoch: 1/3 Iter: 53/62 Batch time: 0.2099
Phase: train Epoch: 1/3 Iter: 54/62 Batch time: 0.2089
Phase: train Epoch: 1/3 Iter: 55/62 Batch time: 0.2114
Phase: train Epoch: 1/3 Iter: 56/62 Batch time: 0.2095
Phase: train Epoch: 1/3 Iter: 57/62 Batch time: 0.2103
Phase: train Epoch: 1/3 Iter: 58/62 Batch time: 0.2277
Phase: train Epoch: 1/3 Iter: 59/62 Batch time: 0.2253
Phase: train Epoch: 1/3 Iter: 60/62 Batch time: 0.2100
Phase: train Epoch: 1/3 Iter: 61/62 Batch time: 0.2121
Phase: train Epoch: 1/3 Loss: 0.6993 Acc: 0.5246
Phase: validation Epoch: 1/3 Iter: 1/39 Batch time: 0.1616
Phase: validation Epoch: 1/3 Iter: 2/39 Batch time: 0.1576
Phase: validation Epoch: 1/3 Iter: 3/39 Batch time: 0.1554
Phase: validation Epoch: 1/3 Iter: 4/39 Batch time: 0.1546
Phase: validation Epoch: 1/3 Iter: 5/39 Batch time: 0.1538
Phase: validation Epoch: 1/3 Iter: 6/39 Batch time: 0.1582
Phase: validation Epoch: 1/3 Iter: 7/39 Batch time: 0.1527
Phase: validation Epoch: 1/3 Iter: 8/39 Batch time: 0.1543
Phase: validation Epoch: 1/3 Iter: 9/39 Batch time: 0.1533
Phase: validation Epoch: 1/3 Iter: 10/39 Batch time: 0.1526
Phase: validation Epoch: 1/3 Iter: 11/39 Batch time: 0.1540
Phase: validation Epoch: 1/3 Iter: 12/39 Batch time: 0.1542
Phase: validation Epoch: 1/3 Iter: 13/39 Batch time: 0.1523
Phase: validation Epoch: 1/3 Iter: 14/39 Batch time: 0.1609
Phase: validation Epoch: 1/3 Iter: 15/39 Batch time: 0.1523
Phase: validation Epoch: 1/3 Iter: 16/39 Batch time: 0.1539
Phase: validation Epoch: 1/3 Iter: 17/39 Batch time: 0.1523
Phase: validation Epoch: 1/3 Iter: 18/39 Batch time: 0.1552
Phase: validation Epoch: 1/3 Iter: 19/39 Batch time: 0.1520
Phase: validation Epoch: 1/3 Iter: 20/39 Batch time: 0.1528
Phase: validation Epoch: 1/3 Iter: 21/39 Batch time: 0.1530
Phase: validation Epoch: 1/3 Iter: 22/39 Batch time: 0.1575
Phase: validation Epoch: 1/3 Iter: 23/39 Batch time: 0.1525
Phase: validation Epoch: 1/3 Iter: 24/39 Batch time: 0.1537
Phase: validation Epoch: 1/3 Iter: 25/39 Batch time: 0.1548
Phase: validation Epoch: 1/3 Iter: 26/39 Batch time: 0.1534
Phase: validation Epoch: 1/3 Iter: 27/39 Batch time: 0.1526
Phase: validation Epoch: 1/3 Iter: 28/39 Batch time: 0.1545
Phase: validation Epoch: 1/3 Iter: 29/39 Batch time: 0.1527
Phase: validation Epoch: 1/3 Iter: 30/39 Batch time: 0.1554
Phase: validation Epoch: 1/3 Iter: 31/39 Batch time: 0.1524
Phase: validation Epoch: 1/3 Iter: 32/39 Batch time: 0.1523
Phase: validation Epoch: 1/3 Iter: 33/39 Batch time: 0.1526
Phase: validation Epoch: 1/3 Iter: 34/39 Batch time: 0.1518
Phase: validation Epoch: 1/3 Iter: 35/39 Batch time: 0.1516
Phase: validation Epoch: 1/3 Iter: 36/39 Batch time: 0.1539
Phase: validation Epoch: 1/3 Iter: 37/39 Batch time: 0.1537
Phase: validation Epoch: 1/3 Iter: 38/39 Batch time: 0.1521
Phase: validation Epoch: 1/3 Iter: 39/39 Batch time: 0.0536
Phase: validation   Epoch: 1/3 Loss: 0.6432 Acc: 0.6536
Phase: train Epoch: 2/3 Iter: 1/62 Batch time: 0.2031
Phase: train Epoch: 2/3 Iter: 2/62 Batch time: 0.2058
Phase: train Epoch: 2/3 Iter: 3/62 Batch time: 0.2063
Phase: train Epoch: 2/3 Iter: 4/62 Batch time: 0.2076
Phase: train Epoch: 2/3 Iter: 5/62 Batch time: 0.2061
Phase: train Epoch: 2/3 Iter: 6/62 Batch time: 0.2089
Phase: train Epoch: 2/3 Iter: 7/62 Batch time: 0.2105
Phase: train Epoch: 2/3 Iter: 8/62 Batch time: 0.2077
Phase: train Epoch: 2/3 Iter: 9/62 Batch time: 0.2059
Phase: train Epoch: 2/3 Iter: 10/62 Batch time: 0.2093
Phase: train Epoch: 2/3 Iter: 11/62 Batch time: 0.2091
Phase: train Epoch: 2/3 Iter: 12/62 Batch time: 0.2090
Phase: train Epoch: 2/3 Iter: 13/62 Batch time: 0.2079
Phase: train Epoch: 2/3 Iter: 14/62 Batch time: 0.2116
Phase: train Epoch: 2/3 Iter: 15/62 Batch time: 0.2074
Phase: train Epoch: 2/3 Iter: 16/62 Batch time: 0.2061
Phase: train Epoch: 2/3 Iter: 17/62 Batch time: 0.2080
Phase: train Epoch: 2/3 Iter: 18/62 Batch time: 0.2085
Phase: train Epoch: 2/3 Iter: 19/62 Batch time: 0.2058
Phase: train Epoch: 2/3 Iter: 20/62 Batch time: 0.2098
Phase: train Epoch: 2/3 Iter: 21/62 Batch time: 0.2066
Phase: train Epoch: 2/3 Iter: 22/62 Batch time: 0.2082
Phase: train Epoch: 2/3 Iter: 23/62 Batch time: 0.2089
Phase: train Epoch: 2/3 Iter: 24/62 Batch time: 0.2095
Phase: train Epoch: 2/3 Iter: 25/62 Batch time: 0.2060
Phase: train Epoch: 2/3 Iter: 26/62 Batch time: 0.2096
Phase: train Epoch: 2/3 Iter: 27/62 Batch time: 0.2127
Phase: train Epoch: 2/3 Iter: 28/62 Batch time: 0.2085
Phase: train Epoch: 2/3 Iter: 29/62 Batch time: 0.2071
Phase: train Epoch: 2/3 Iter: 30/62 Batch time: 0.2084
Phase: train Epoch: 2/3 Iter: 31/62 Batch time: 0.2130
Phase: train Epoch: 2/3 Iter: 32/62 Batch time: 0.2146
Phase: train Epoch: 2/3 Iter: 33/62 Batch time: 0.2061
Phase: train Epoch: 2/3 Iter: 34/62 Batch time: 0.2073
Phase: train Epoch: 2/3 Iter: 35/62 Batch time: 0.2069
Phase: train Epoch: 2/3 Iter: 36/62 Batch time: 0.2067
Phase: train Epoch: 2/3 Iter: 37/62 Batch time: 0.2094
Phase: train Epoch: 2/3 Iter: 38/62 Batch time: 0.2089
Phase: train Epoch: 2/3 Iter: 39/62 Batch time: 0.2063
Phase: train Epoch: 2/3 Iter: 40/62 Batch time: 0.2083
Phase: train Epoch: 2/3 Iter: 41/62 Batch time: 0.2068
Phase: train Epoch: 2/3 Iter: 42/62 Batch time: 0.2084
Phase: train Epoch: 2/3 Iter: 43/62 Batch time: 0.2051
Phase: train Epoch: 2/3 Iter: 44/62 Batch time: 0.2090
Phase: train Epoch: 2/3 Iter: 45/62 Batch time: 0.2068
Phase: train Epoch: 2/3 Iter: 46/62 Batch time: 0.2101
Phase: train Epoch: 2/3 Iter: 47/62 Batch time: 0.2088
Phase: train Epoch: 2/3 Iter: 48/62 Batch time: 0.2149
Phase: train Epoch: 2/3 Iter: 49/62 Batch time: 0.2057
Phase: train Epoch: 2/3 Iter: 50/62 Batch time: 0.2113
Phase: train Epoch: 2/3 Iter: 51/62 Batch time: 0.2083
Phase: train Epoch: 2/3 Iter: 52/62 Batch time: 0.2104
Phase: train Epoch: 2/3 Iter: 53/62 Batch time: 0.2072
Phase: train Epoch: 2/3 Iter: 54/62 Batch time: 0.2109
Phase: train Epoch: 2/3 Iter: 55/62 Batch time: 0.2076
Phase: train Epoch: 2/3 Iter: 56/62 Batch time: 0.2092
Phase: train Epoch: 2/3 Iter: 57/62 Batch time: 0.2099
Phase: train Epoch: 2/3 Iter: 58/62 Batch time: 0.2095
Phase: train Epoch: 2/3 Iter: 59/62 Batch time: 0.2050
Phase: train Epoch: 2/3 Iter: 60/62 Batch time: 0.2097
Phase: train Epoch: 2/3 Iter: 61/62 Batch time: 0.2061
Phase: train Epoch: 2/3 Loss: 0.6141 Acc: 0.7049
Phase: validation Epoch: 2/3 Iter: 1/39 Batch time: 0.1571
Phase: validation Epoch: 2/3 Iter: 2/39 Batch time: 0.1552
Phase: validation Epoch: 2/3 Iter: 3/39 Batch time: 0.1543
Phase: validation Epoch: 2/3 Iter: 4/39 Batch time: 0.1646
Phase: validation Epoch: 2/3 Iter: 5/39 Batch time: 0.1609
Phase: validation Epoch: 2/3 Iter: 6/39 Batch time: 0.1561
Phase: validation Epoch: 2/3 Iter: 7/39 Batch time: 0.1550
Phase: validation Epoch: 2/3 Iter: 8/39 Batch time: 0.1564
Phase: validation Epoch: 2/3 Iter: 9/39 Batch time: 0.1560
Phase: validation Epoch: 2/3 Iter: 10/39 Batch time: 0.1540
Phase: validation Epoch: 2/3 Iter: 11/39 Batch time: 0.1546
Phase: validation Epoch: 2/3 Iter: 12/39 Batch time: 0.1562
Phase: validation Epoch: 2/3 Iter: 13/39 Batch time: 0.1571
Phase: validation Epoch: 2/3 Iter: 14/39 Batch time: 0.1570
Phase: validation Epoch: 2/3 Iter: 15/39 Batch time: 0.1560
Phase: validation Epoch: 2/3 Iter: 16/39 Batch time: 0.1556
Phase: validation Epoch: 2/3 Iter: 17/39 Batch time: 0.1559
Phase: validation Epoch: 2/3 Iter: 18/39 Batch time: 0.1677
Phase: validation Epoch: 2/3 Iter: 19/39 Batch time: 0.1685
Phase: validation Epoch: 2/3 Iter: 20/39 Batch time: 0.1681
Phase: validation Epoch: 2/3 Iter: 21/39 Batch time: 0.1680
Phase: validation Epoch: 2/3 Iter: 22/39 Batch time: 0.1737
Phase: validation Epoch: 2/3 Iter: 23/39 Batch time: 0.1677
Phase: validation Epoch: 2/3 Iter: 24/39 Batch time: 0.1685
Phase: validation Epoch: 2/3 Iter: 25/39 Batch time: 0.1688
Phase: validation Epoch: 2/3 Iter: 26/39 Batch time: 0.1680
Phase: validation Epoch: 2/3 Iter: 27/39 Batch time: 0.1680
Phase: validation Epoch: 2/3 Iter: 28/39 Batch time: 0.1679
Phase: validation Epoch: 2/3 Iter: 29/39 Batch time: 0.1695
Phase: validation Epoch: 2/3 Iter: 30/39 Batch time: 0.1673
Phase: validation Epoch: 2/3 Iter: 31/39 Batch time: 0.1700
Phase: validation Epoch: 2/3 Iter: 32/39 Batch time: 0.1666
Phase: validation Epoch: 2/3 Iter: 33/39 Batch time: 0.1686
Phase: validation Epoch: 2/3 Iter: 34/39 Batch time: 0.1677
Phase: validation Epoch: 2/3 Iter: 35/39 Batch time: 0.1673
Phase: validation Epoch: 2/3 Iter: 36/39 Batch time: 0.1716
Phase: validation Epoch: 2/3 Iter: 37/39 Batch time: 0.1696
Phase: validation Epoch: 2/3 Iter: 38/39 Batch time: 0.1693
Phase: validation Epoch: 2/3 Iter: 39/39 Batch time: 0.0523
Phase: validation   Epoch: 2/3 Loss: 0.5392 Acc: 0.8235
Phase: train Epoch: 3/3 Iter: 1/62 Batch time: 0.2189
Phase: train Epoch: 3/3 Iter: 2/62 Batch time: 0.2234
Phase: train Epoch: 3/3 Iter: 3/62 Batch time: 0.2239
Phase: train Epoch: 3/3 Iter: 4/62 Batch time: 0.2252
Phase: train Epoch: 3/3 Iter: 5/62 Batch time: 0.2249
Phase: train Epoch: 3/3 Iter: 6/62 Batch time: 0.2239
Phase: train Epoch: 3/3 Iter: 7/62 Batch time: 0.2250
Phase: train Epoch: 3/3 Iter: 8/62 Batch time: 0.2245
Phase: train Epoch: 3/3 Iter: 9/62 Batch time: 0.2266
Phase: train Epoch: 3/3 Iter: 10/62 Batch time: 0.2247
Phase: train Epoch: 3/3 Iter: 11/62 Batch time: 0.2262
Phase: train Epoch: 3/3 Iter: 12/62 Batch time: 0.2255
Phase: train Epoch: 3/3 Iter: 13/62 Batch time: 0.2241
Phase: train Epoch: 3/3 Iter: 14/62 Batch time: 0.2259
Phase: train Epoch: 3/3 Iter: 15/62 Batch time: 0.2244
Phase: train Epoch: 3/3 Iter: 16/62 Batch time: 0.2189
Phase: train Epoch: 3/3 Iter: 17/62 Batch time: 0.2143
Phase: train Epoch: 3/3 Iter: 18/62 Batch time: 0.2140
Phase: train Epoch: 3/3 Iter: 19/62 Batch time: 0.2147
Phase: train Epoch: 3/3 Iter: 20/62 Batch time: 0.2146
Phase: train Epoch: 3/3 Iter: 21/62 Batch time: 0.2127
Phase: train Epoch: 3/3 Iter: 22/62 Batch time: 0.2092
Phase: train Epoch: 3/3 Iter: 23/62 Batch time: 0.2056
Phase: train Epoch: 3/3 Iter: 24/62 Batch time: 0.2102
Phase: train Epoch: 3/3 Iter: 25/62 Batch time: 0.2080
Phase: train Epoch: 3/3 Iter: 26/62 Batch time: 0.2063
Phase: train Epoch: 3/3 Iter: 27/62 Batch time: 0.2048
Phase: train Epoch: 3/3 Iter: 28/62 Batch time: 0.2055
Phase: train Epoch: 3/3 Iter: 29/62 Batch time: 0.2089
Phase: train Epoch: 3/3 Iter: 30/62 Batch time: 0.2100
Phase: train Epoch: 3/3 Iter: 31/62 Batch time: 0.2042
Phase: train Epoch: 3/3 Iter: 32/62 Batch time: 0.2066
Phase: train Epoch: 3/3 Iter: 33/62 Batch time: 0.2097
Phase: train Epoch: 3/3 Iter: 34/62 Batch time: 0.2113
Phase: train Epoch: 3/3 Iter: 35/62 Batch time: 0.2059
Phase: train Epoch: 3/3 Iter: 36/62 Batch time: 0.2107
Phase: train Epoch: 3/3 Iter: 37/62 Batch time: 0.2095
Phase: train Epoch: 3/3 Iter: 38/62 Batch time: 0.2080
Phase: train Epoch: 3/3 Iter: 39/62 Batch time: 0.2077
Phase: train Epoch: 3/3 Iter: 40/62 Batch time: 0.2107
Phase: train Epoch: 3/3 Iter: 41/62 Batch time: 0.2044
Phase: train Epoch: 3/3 Iter: 42/62 Batch time: 0.2088
Phase: train Epoch: 3/3 Iter: 43/62 Batch time: 0.2079
Phase: train Epoch: 3/3 Iter: 44/62 Batch time: 0.2068
Phase: train Epoch: 3/3 Iter: 45/62 Batch time: 0.2054
Phase: train Epoch: 3/3 Iter: 46/62 Batch time: 0.2122
Phase: train Epoch: 3/3 Iter: 47/62 Batch time: 0.2080
Phase: train Epoch: 3/3 Iter: 48/62 Batch time: 0.2055
Phase: train Epoch: 3/3 Iter: 49/62 Batch time: 0.2061
Phase: train Epoch: 3/3 Iter: 50/62 Batch time: 0.2078
Phase: train Epoch: 3/3 Iter: 51/62 Batch time: 0.2099
Phase: train Epoch: 3/3 Iter: 52/62 Batch time: 0.2129
Phase: train Epoch: 3/3 Iter: 53/62 Batch time: 0.2126
Phase: train Epoch: 3/3 Iter: 54/62 Batch time: 0.2236
Phase: train Epoch: 3/3 Iter: 55/62 Batch time: 0.2145
Phase: train Epoch: 3/3 Iter: 56/62 Batch time: 0.2157
Phase: train Epoch: 3/3 Iter: 57/62 Batch time: 0.2149
Phase: train Epoch: 3/3 Iter: 58/62 Batch time: 0.2078
Phase: train Epoch: 3/3 Iter: 59/62 Batch time: 0.2064
Phase: train Epoch: 3/3 Iter: 60/62 Batch time: 0.2080
Phase: train Epoch: 3/3 Iter: 61/62 Batch time: 0.2077
Phase: train Epoch: 3/3 Loss: 0.5652 Acc: 0.7336
Phase: validation Epoch: 3/3 Iter: 1/39 Batch time: 0.1643
Phase: validation Epoch: 3/3 Iter: 2/39 Batch time: 0.1549
Phase: validation Epoch: 3/3 Iter: 3/39 Batch time: 0.1558
Phase: validation Epoch: 3/3 Iter: 4/39 Batch time: 0.1539
Phase: validation Epoch: 3/3 Iter: 5/39 Batch time: 0.1550
Phase: validation Epoch: 3/3 Iter: 6/39 Batch time: 0.1538
Phase: validation Epoch: 3/3 Iter: 7/39 Batch time: 0.1571
Phase: validation Epoch: 3/3 Iter: 8/39 Batch time: 0.1568
Phase: validation Epoch: 3/3 Iter: 9/39 Batch time: 0.1559
Phase: validation Epoch: 3/3 Iter: 10/39 Batch time: 0.1535
Phase: validation Epoch: 3/3 Iter: 11/39 Batch time: 0.1557
Phase: validation Epoch: 3/3 Iter: 12/39 Batch time: 0.1545
Phase: validation Epoch: 3/3 Iter: 13/39 Batch time: 0.1576
Phase: validation Epoch: 3/3 Iter: 14/39 Batch time: 0.1549
Phase: validation Epoch: 3/3 Iter: 15/39 Batch time: 0.1555
Phase: validation Epoch: 3/3 Iter: 16/39 Batch time: 0.1525
Phase: validation Epoch: 3/3 Iter: 17/39 Batch time: 0.1549
Phase: validation Epoch: 3/3 Iter: 18/39 Batch time: 0.1527
Phase: validation Epoch: 3/3 Iter: 19/39 Batch time: 0.1560
Phase: validation Epoch: 3/3 Iter: 20/39 Batch time: 0.1530
Phase: validation Epoch: 3/3 Iter: 21/39 Batch time: 0.1553
Phase: validation Epoch: 3/3 Iter: 22/39 Batch time: 0.1537
Phase: validation Epoch: 3/3 Iter: 23/39 Batch time: 0.1539
Phase: validation Epoch: 3/3 Iter: 24/39 Batch time: 0.1532
Phase: validation Epoch: 3/3 Iter: 25/39 Batch time: 0.1571
Phase: validation Epoch: 3/3 Iter: 26/39 Batch time: 0.1543
Phase: validation Epoch: 3/3 Iter: 27/39 Batch time: 0.1585
Phase: validation Epoch: 3/3 Iter: 28/39 Batch time: 0.1551
Phase: validation Epoch: 3/3 Iter: 29/39 Batch time: 0.1543
Phase: validation Epoch: 3/3 Iter: 30/39 Batch time: 0.1536
Phase: validation Epoch: 3/3 Iter: 31/39 Batch time: 0.1554
Phase: validation Epoch: 3/3 Iter: 32/39 Batch time: 0.1556
Phase: validation Epoch: 3/3 Iter: 33/39 Batch time: 0.1569
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
 13%|#2        | 5.76M/44.7M [00:00<00:00, 60.3MB/s]
 26%|##6       | 11.8M/44.7M [00:00<00:00, 62.0MB/s]
 80%|#######9  | 35.6M/44.7M [00:00<00:00, 147MB/s]
100%|##########| 44.7M/44.7M [00:00<00:00, 139MB/s]
Training started:
Phase: train Epoch: 1/3 Iter: 1/62 Batch time: 0.2269
Phase: train Epoch: 1/3 Iter: 2/62 Batch time: 0.2061
Phase: train Epoch: 1/3 Iter: 3/62 Batch time: 0.2090
Phase: train Epoch: 1/3 Iter: 4/62 Batch time: 0.2017
Phase: train Epoch: 1/3 Iter: 5/62 Batch time: 0.2043
Phase: train Epoch: 1/3 Iter: 6/62 Batch time: 0.2130
Phase: train Epoch: 1/3 Iter: 7/62 Batch time: 0.2045
Phase: train Epoch: 1/3 Iter: 8/62 Batch time: 0.2054
Phase: train Epoch: 1/3 Iter: 9/62 Batch time: 0.2048
Phase: train Epoch: 1/3 Iter: 10/62 Batch time: 0.2041
Phase: train Epoch: 1/3 Iter: 11/62 Batch time: 0.2205
Phase: train Epoch: 1/3 Iter: 12/62 Batch time: 0.2092
Phase: train Epoch: 1/3 Iter: 13/62 Batch time: 0.2050
Phase: train Epoch: 1/3 Iter: 14/62 Batch time: 0.2330
Phase: train Epoch: 1/3 Iter: 15/62 Batch time: 0.2022
Phase: train Epoch: 1/3 Iter: 16/62 Batch time: 0.2023
Phase: train Epoch: 1/3 Iter: 17/62 Batch time: 0.2039
Phase: train Epoch: 1/3 Iter: 18/62 Batch time: 0.2004
Phase: train Epoch: 1/3 Iter: 19/62 Batch time: 0.2246
Phase: train Epoch: 1/3 Iter: 20/62 Batch time: 0.2053
Phase: train Epoch: 1/3 Iter: 21/62 Batch time: 0.2114
Phase: train Epoch: 1/3 Iter: 22/62 Batch time: 0.2009
Phase: train Epoch: 1/3 Iter: 23/62 Batch time: 0.2000
Phase: train Epoch: 1/3 Iter: 24/62 Batch time: 0.2098
Phase: train Epoch: 1/3 Iter: 25/62 Batch time: 0.2061
Phase: train Epoch: 1/3 Iter: 26/62 Batch time: 0.1990
Phase: train Epoch: 1/3 Iter: 27/62 Batch time: 0.2099
Phase: train Epoch: 1/3 Iter: 28/62 Batch time: 0.2317
Phase: train Epoch: 1/3 Iter: 29/62 Batch time: 0.2242
Phase: train Epoch: 1/3 Iter: 30/62 Batch time: 0.2226
Phase: train Epoch: 1/3 Iter: 31/62 Batch time: 0.2354
Phase: train Epoch: 1/3 Iter: 32/62 Batch time: 0.2498
Phase: train Epoch: 1/3 Iter: 33/62 Batch time: 0.2494
Phase: train Epoch: 1/3 Iter: 34/62 Batch time: 0.2521
Phase: train Epoch: 1/3 Iter: 35/62 Batch time: 0.2137
Phase: train Epoch: 1/3 Iter: 36/62 Batch time: 0.2373
Phase: train Epoch: 1/3 Iter: 37/62 Batch time: 0.2358
Phase: train Epoch: 1/3 Iter: 38/62 Batch time: 0.2300
Phase: train Epoch: 1/3 Iter: 39/62 Batch time: 0.2352
Phase: train Epoch: 1/3 Iter: 40/62 Batch time: 0.2330
Phase: train Epoch: 1/3 Iter: 41/62 Batch time: 0.2243
Phase: train Epoch: 1/3 Iter: 42/62 Batch time: 0.2303
Phase: train Epoch: 1/3 Iter: 43/62 Batch time: 0.2248
Phase: train Epoch: 1/3 Iter: 44/62 Batch time: 0.2369
Phase: train Epoch: 1/3 Iter: 45/62 Batch time: 0.2285
Phase: train Epoch: 1/3 Iter: 46/62 Batch time: 0.2216
Phase: train Epoch: 1/3 Iter: 47/62 Batch time: 0.2298
Phase: train Epoch: 1/3 Iter: 48/62 Batch time: 0.2336
Phase: train Epoch: 1/3 Iter: 49/62 Batch time: 0.2410
Phase: train Epoch: 1/3 Iter: 50/62 Batch time: 0.2412
Phase: train Epoch: 1/3 Iter: 51/62 Batch time: 0.2512
Phase: train Epoch: 1/3 Iter: 52/62 Batch time: 0.2504
Phase: train Epoch: 1/3 Iter: 53/62 Batch time: 0.2450
Phase: train Epoch: 1/3 Iter: 54/62 Batch time: 0.2458
Phase: train Epoch: 1/3 Iter: 55/62 Batch time: 0.2437
Phase: train Epoch: 1/3 Iter: 56/62 Batch time: 0.2416
Phase: train Epoch: 1/3 Iter: 57/62 Batch time: 0.2434
Phase: train Epoch: 1/3 Iter: 58/62 Batch time: 0.2324
Phase: train Epoch: 1/3 Iter: 59/62 Batch time: 0.2422
Phase: train Epoch: 1/3 Iter: 60/62 Batch time: 0.2341
Phase: train Epoch: 1/3 Iter: 61/62 Batch time: 0.2385
Phase: train Epoch: 1/3 Loss: 0.6993 Acc: 0.5246
Phase: validation Epoch: 1/3 Iter: 1/39 Batch time: 0.1844
Phase: validation Epoch: 1/3 Iter: 2/39 Batch time: 0.1877
Phase: validation Epoch: 1/3 Iter: 3/39 Batch time: 0.1899
Phase: validation Epoch: 1/3 Iter: 4/39 Batch time: 0.1898
Phase: validation Epoch: 1/3 Iter: 5/39 Batch time: 0.1861
Phase: validation Epoch: 1/3 Iter: 6/39 Batch time: 0.1657
Phase: validation Epoch: 1/3 Iter: 7/39 Batch time: 0.1660
Phase: validation Epoch: 1/3 Iter: 8/39 Batch time: 0.1767
Phase: validation Epoch: 1/3 Iter: 9/39 Batch time: 0.1684
Phase: validation Epoch: 1/3 Iter: 10/39 Batch time: 0.1901
Phase: validation Epoch: 1/3 Iter: 11/39 Batch time: 0.2148
Phase: validation Epoch: 1/3 Iter: 12/39 Batch time: 0.1791
Phase: validation Epoch: 1/3 Iter: 13/39 Batch time: 0.1975
Phase: validation Epoch: 1/3 Iter: 14/39 Batch time: 0.1719
Phase: validation Epoch: 1/3 Iter: 15/39 Batch time: 0.1755
Phase: validation Epoch: 1/3 Iter: 16/39 Batch time: 0.1714
Phase: validation Epoch: 1/3 Iter: 17/39 Batch time: 0.1650
Phase: validation Epoch: 1/3 Iter: 18/39 Batch time: 0.1760
Phase: validation Epoch: 1/3 Iter: 19/39 Batch time: 0.1699
Phase: validation Epoch: 1/3 Iter: 20/39 Batch time: 0.1543
Phase: validation Epoch: 1/3 Iter: 21/39 Batch time: 0.1521
Phase: validation Epoch: 1/3 Iter: 22/39 Batch time: 0.1574
Phase: validation Epoch: 1/3 Iter: 23/39 Batch time: 0.1505
Phase: validation Epoch: 1/3 Iter: 24/39 Batch time: 0.1601
Phase: validation Epoch: 1/3 Iter: 25/39 Batch time: 0.1769
Phase: validation Epoch: 1/3 Iter: 26/39 Batch time: 0.1765
Phase: validation Epoch: 1/3 Iter: 27/39 Batch time: 0.1738
Phase: validation Epoch: 1/3 Iter: 28/39 Batch time: 0.1706
Phase: validation Epoch: 1/3 Iter: 29/39 Batch time: 0.1784
Phase: validation Epoch: 1/3 Iter: 30/39 Batch time: 0.1845
Phase: validation Epoch: 1/3 Iter: 31/39 Batch time: 0.1872
Phase: validation Epoch: 1/3 Iter: 32/39 Batch time: 0.1914
Phase: validation Epoch: 1/3 Iter: 33/39 Batch time: 0.1943
Phase: validation Epoch: 1/3 Iter: 34/39 Batch time: 0.2047
Phase: validation Epoch: 1/3 Iter: 35/39 Batch time: 0.2070
Phase: validation Epoch: 1/3 Iter: 36/39 Batch time: 0.1691
Phase: validation Epoch: 1/3 Iter: 37/39 Batch time: 0.1838
Phase: validation Epoch: 1/3 Iter: 38/39 Batch time: 0.1739
Phase: validation Epoch: 1/3 Iter: 39/39 Batch time: 0.0596
Phase: validation   Epoch: 1/3 Loss: 0.6432 Acc: 0.6536
Phase: train Epoch: 2/3 Iter: 1/62 Batch time: 0.2312
Phase: train Epoch: 2/3 Iter: 2/62 Batch time: 0.2218
Phase: train Epoch: 2/3 Iter: 3/62 Batch time: 0.2109
Phase: train Epoch: 2/3 Iter: 4/62 Batch time: 0.2034
Phase: train Epoch: 2/3 Iter: 5/62 Batch time: 0.2035
Phase: train Epoch: 2/3 Iter: 6/62 Batch time: 0.2033
Phase: train Epoch: 2/3 Iter: 7/62 Batch time: 0.1989
Phase: train Epoch: 2/3 Iter: 8/62 Batch time: 0.2008
Phase: train Epoch: 2/3 Iter: 9/62 Batch time: 0.2085
Phase: train Epoch: 2/3 Iter: 10/62 Batch time: 0.2009
Phase: train Epoch: 2/3 Iter: 11/62 Batch time: 0.2006
Phase: train Epoch: 2/3 Iter: 12/62 Batch time: 0.2061
Phase: train Epoch: 2/3 Iter: 13/62 Batch time: 0.2006
Phase: train Epoch: 2/3 Iter: 14/62 Batch time: 0.2215
Phase: train Epoch: 2/3 Iter: 15/62 Batch time: 0.2266
Phase: train Epoch: 2/3 Iter: 16/62 Batch time: 0.2327
Phase: train Epoch: 2/3 Iter: 17/62 Batch time: 0.2130
Phase: train Epoch: 2/3 Iter: 18/62 Batch time: 0.2083
Phase: train Epoch: 2/3 Iter: 19/62 Batch time: 0.2046
Phase: train Epoch: 2/3 Iter: 20/62 Batch time: 0.2019
Phase: train Epoch: 2/3 Iter: 21/62 Batch time: 0.2121
Phase: train Epoch: 2/3 Iter: 22/62 Batch time: 0.2074
Phase: train Epoch: 2/3 Iter: 23/62 Batch time: 0.2191
Phase: train Epoch: 2/3 Iter: 24/62 Batch time: 0.2441
Phase: train Epoch: 2/3 Iter: 25/62 Batch time: 0.2037
Phase: train Epoch: 2/3 Iter: 26/62 Batch time: 0.2054
Phase: train Epoch: 2/3 Iter: 27/62 Batch time: 0.2135
Phase: train Epoch: 2/3 Iter: 28/62 Batch time: 0.2582
Phase: train Epoch: 2/3 Iter: 29/62 Batch time: 0.2317
Phase: train Epoch: 2/3 Iter: 30/62 Batch time: 0.2275
Phase: train Epoch: 2/3 Iter: 31/62 Batch time: 0.2103
Phase: train Epoch: 2/3 Iter: 32/62 Batch time: 0.2050
Phase: train Epoch: 2/3 Iter: 33/62 Batch time: 0.2007
Phase: train Epoch: 2/3 Iter: 34/62 Batch time: 0.2006
Phase: train Epoch: 2/3 Iter: 35/62 Batch time: 0.2053
Phase: train Epoch: 2/3 Iter: 36/62 Batch time: 0.2017
Phase: train Epoch: 2/3 Iter: 37/62 Batch time: 0.2070
Phase: train Epoch: 2/3 Iter: 38/62 Batch time: 0.2029
Phase: train Epoch: 2/3 Iter: 39/62 Batch time: 0.2244
Phase: train Epoch: 2/3 Iter: 40/62 Batch time: 0.2086
Phase: train Epoch: 2/3 Iter: 41/62 Batch time: 0.2044
Phase: train Epoch: 2/3 Iter: 42/62 Batch time: 0.2031
Phase: train Epoch: 2/3 Iter: 43/62 Batch time: 0.2006
Phase: train Epoch: 2/3 Iter: 44/62 Batch time: 0.2184
Phase: train Epoch: 2/3 Iter: 45/62 Batch time: 0.2183
Phase: train Epoch: 2/3 Iter: 46/62 Batch time: 0.2226
Phase: train Epoch: 2/3 Iter: 47/62 Batch time: 0.2251
Phase: train Epoch: 2/3 Iter: 48/62 Batch time: 0.2189
Phase: train Epoch: 2/3 Iter: 49/62 Batch time: 0.2037
Phase: train Epoch: 2/3 Iter: 50/62 Batch time: 0.2015
Phase: train Epoch: 2/3 Iter: 51/62 Batch time: 0.2033
Phase: train Epoch: 2/3 Iter: 52/62 Batch time: 0.2222
Phase: train Epoch: 2/3 Iter: 53/62 Batch time: 0.2224
Phase: train Epoch: 2/3 Iter: 54/62 Batch time: 0.2264
Phase: train Epoch: 2/3 Iter: 55/62 Batch time: 0.2236
Phase: train Epoch: 2/3 Iter: 56/62 Batch time: 0.2198
Phase: train Epoch: 2/3 Iter: 57/62 Batch time: 0.2016
Phase: train Epoch: 2/3 Iter: 58/62 Batch time: 0.2280
Phase: train Epoch: 2/3 Iter: 59/62 Batch time: 0.2208
Phase: train Epoch: 2/3 Iter: 60/62 Batch time: 0.2229
Phase: train Epoch: 2/3 Iter: 61/62 Batch time: 0.2244
Phase: train Epoch: 2/3 Loss: 0.6141 Acc: 0.7049
Phase: validation Epoch: 2/3 Iter: 1/39 Batch time: 0.1592
Phase: validation Epoch: 2/3 Iter: 2/39 Batch time: 0.1514
Phase: validation Epoch: 2/3 Iter: 3/39 Batch time: 0.1501
Phase: validation Epoch: 2/3 Iter: 4/39 Batch time: 0.1550
Phase: validation Epoch: 2/3 Iter: 5/39 Batch time: 0.1506
Phase: validation Epoch: 2/3 Iter: 6/39 Batch time: 0.1571
Phase: validation Epoch: 2/3 Iter: 7/39 Batch time: 0.1509
Phase: validation Epoch: 2/3 Iter: 8/39 Batch time: 0.1580
Phase: validation Epoch: 2/3 Iter: 9/39 Batch time: 0.1546
Phase: validation Epoch: 2/3 Iter: 10/39 Batch time: 0.1568
Phase: validation Epoch: 2/3 Iter: 11/39 Batch time: 0.1535
Phase: validation Epoch: 2/3 Iter: 12/39 Batch time: 0.1555
Phase: validation Epoch: 2/3 Iter: 13/39 Batch time: 0.1493
Phase: validation Epoch: 2/3 Iter: 14/39 Batch time: 0.1564
Phase: validation Epoch: 2/3 Iter: 15/39 Batch time: 0.1607
Phase: validation Epoch: 2/3 Iter: 16/39 Batch time: 0.1639
Phase: validation Epoch: 2/3 Iter: 17/39 Batch time: 0.1679
Phase: validation Epoch: 2/3 Iter: 18/39 Batch time: 0.1529
Phase: validation Epoch: 2/3 Iter: 19/39 Batch time: 0.1575
Phase: validation Epoch: 2/3 Iter: 20/39 Batch time: 0.1527
Phase: validation Epoch: 2/3 Iter: 21/39 Batch time: 0.1535
Phase: validation Epoch: 2/3 Iter: 22/39 Batch time: 0.1542
Phase: validation Epoch: 2/3 Iter: 23/39 Batch time: 0.1538
Phase: validation Epoch: 2/3 Iter: 24/39 Batch time: 0.1515
Phase: validation Epoch: 2/3 Iter: 25/39 Batch time: 0.1520
Phase: validation Epoch: 2/3 Iter: 26/39 Batch time: 0.1565
Phase: validation Epoch: 2/3 Iter: 27/39 Batch time: 0.1546
Phase: validation Epoch: 2/3 Iter: 28/39 Batch time: 0.1524
Phase: validation Epoch: 2/3 Iter: 29/39 Batch time: 0.1728
Phase: validation Epoch: 2/3 Iter: 30/39 Batch time: 0.1601
Phase: validation Epoch: 2/3 Iter: 31/39 Batch time: 0.1796
Phase: validation Epoch: 2/3 Iter: 32/39 Batch time: 0.1531
Phase: validation Epoch: 2/3 Iter: 33/39 Batch time: 0.1524
Phase: validation Epoch: 2/3 Iter: 34/39 Batch time: 0.1511
Phase: validation Epoch: 2/3 Iter: 35/39 Batch time: 0.1509
Phase: validation Epoch: 2/3 Iter: 36/39 Batch time: 0.1577
Phase: validation Epoch: 2/3 Iter: 37/39 Batch time: 0.1527
Phase: validation Epoch: 2/3 Iter: 38/39 Batch time: 0.1514
Phase: validation Epoch: 2/3 Iter: 39/39 Batch time: 0.0500
Phase: validation   Epoch: 2/3 Loss: 0.5392 Acc: 0.8235
Phase: train Epoch: 3/3 Iter: 1/62 Batch time: 0.2002
Phase: train Epoch: 3/3 Iter: 2/62 Batch time: 0.1996
Phase: train Epoch: 3/3 Iter: 3/62 Batch time: 0.2025
Phase: train Epoch: 3/3 Iter: 4/62 Batch time: 0.2313
Phase: train Epoch: 3/3 Iter: 5/62 Batch time: 0.2254
Phase: train Epoch: 3/3 Iter: 6/62 Batch time: 0.2307
Phase: train Epoch: 3/3 Iter: 7/62 Batch time: 0.2226
Phase: train Epoch: 3/3 Iter: 8/62 Batch time: 0.2479
Phase: train Epoch: 3/3 Iter: 9/62 Batch time: 0.2499
Phase: train Epoch: 3/3 Iter: 10/62 Batch time: 0.2525
Phase: train Epoch: 3/3 Iter: 11/62 Batch time: 0.2286
Phase: train Epoch: 3/3 Iter: 12/62 Batch time: 0.2347
Phase: train Epoch: 3/3 Iter: 13/62 Batch time: 0.2410
Phase: train Epoch: 3/3 Iter: 14/62 Batch time: 0.2306
Phase: train Epoch: 3/3 Iter: 15/62 Batch time: 0.2342
Phase: train Epoch: 3/3 Iter: 16/62 Batch time: 0.2343
Phase: train Epoch: 3/3 Iter: 17/62 Batch time: 0.2263
Phase: train Epoch: 3/3 Iter: 18/62 Batch time: 0.2360
Phase: train Epoch: 3/3 Iter: 19/62 Batch time: 0.2285
Phase: train Epoch: 3/3 Iter: 20/62 Batch time: 0.2268
Phase: train Epoch: 3/3 Iter: 21/62 Batch time: 0.2331
Phase: train Epoch: 3/3 Iter: 22/62 Batch time: 0.2253
Phase: train Epoch: 3/3 Iter: 23/62 Batch time: 0.2338
Phase: train Epoch: 3/3 Iter: 24/62 Batch time: 0.2182
Phase: train Epoch: 3/3 Iter: 25/62 Batch time: 0.2176
Phase: train Epoch: 3/3 Iter: 26/62 Batch time: 0.2284
Phase: train Epoch: 3/3 Iter: 27/62 Batch time: 0.2198
Phase: train Epoch: 3/3 Iter: 28/62 Batch time: 0.2382
Phase: train Epoch: 3/3 Iter: 29/62 Batch time: 0.2328
Phase: train Epoch: 3/3 Iter: 30/62 Batch time: 0.2452
Phase: train Epoch: 3/3 Iter: 31/62 Batch time: 0.2318
Phase: train Epoch: 3/3 Iter: 32/62 Batch time: 0.2137
Phase: train Epoch: 3/3 Iter: 33/62 Batch time: 0.2383
Phase: train Epoch: 3/3 Iter: 34/62 Batch time: 0.2239
Phase: train Epoch: 3/3 Iter: 35/62 Batch time: 0.2012
Phase: train Epoch: 3/3 Iter: 36/62 Batch time: 0.2046
Phase: train Epoch: 3/3 Iter: 37/62 Batch time: 0.2096
Phase: train Epoch: 3/3 Iter: 38/62 Batch time: 0.2049
Phase: train Epoch: 3/3 Iter: 39/62 Batch time: 0.2037
Phase: train Epoch: 3/3 Iter: 40/62 Batch time: 0.2030
Phase: train Epoch: 3/3 Iter: 41/62 Batch time: 0.2005
Phase: train Epoch: 3/3 Iter: 42/62 Batch time: 0.2099
Phase: train Epoch: 3/3 Iter: 43/62 Batch time: 0.2052
Phase: train Epoch: 3/3 Iter: 44/62 Batch time: 0.2175
Phase: train Epoch: 3/3 Iter: 45/62 Batch time: 0.2054
Phase: train Epoch: 3/3 Iter: 46/62 Batch time: 0.2207
Phase: train Epoch: 3/3 Iter: 47/62 Batch time: 0.2059
Phase: train Epoch: 3/3 Iter: 48/62 Batch time: 0.1998
Phase: train Epoch: 3/3 Iter: 49/62 Batch time: 0.2088
Phase: train Epoch: 3/3 Iter: 50/62 Batch time: 0.2034
Phase: train Epoch: 3/3 Iter: 51/62 Batch time: 0.2143
Phase: train Epoch: 3/3 Iter: 52/62 Batch time: 0.2034
Phase: train Epoch: 3/3 Iter: 53/62 Batch time: 0.2053
Phase: train Epoch: 3/3 Iter: 54/62 Batch time: 0.2027
Phase: train Epoch: 3/3 Iter: 55/62 Batch time: 0.2218
Phase: train Epoch: 3/3 Iter: 56/62 Batch time: 0.2171
Phase: train Epoch: 3/3 Iter: 57/62 Batch time: 0.2048
Phase: train Epoch: 3/3 Iter: 58/62 Batch time: 0.2088
Phase: train Epoch: 3/3 Iter: 59/62 Batch time: 0.2194
Phase: train Epoch: 3/3 Iter: 60/62 Batch time: 0.2095
Phase: train Epoch: 3/3 Iter: 61/62 Batch time: 0.2142
Phase: train Epoch: 3/3 Loss: 0.5652 Acc: 0.7336
Phase: validation Epoch: 3/3 Iter: 1/39 Batch time: 0.1661
Phase: validation Epoch: 3/3 Iter: 2/39 Batch time: 0.1533
Phase: validation Epoch: 3/3 Iter: 3/39 Batch time: 0.1569
Phase: validation Epoch: 3/3 Iter: 4/39 Batch time: 0.1587
Phase: validation Epoch: 3/3 Iter: 5/39 Batch time: 0.1527
Phase: validation Epoch: 3/3 Iter: 6/39 Batch time: 0.1559
Phase: validation Epoch: 3/3 Iter: 7/39 Batch time: 0.1530
Phase: validation Epoch: 3/3 Iter: 8/39 Batch time: 0.1541
Phase: validation Epoch: 3/3 Iter: 9/39 Batch time: 0.1549
Phase: validation Epoch: 3/3 Iter: 10/39 Batch time: 0.1539
Phase: validation Epoch: 3/3 Iter: 11/39 Batch time: 0.1592
Phase: validation Epoch: 3/3 Iter: 12/39 Batch time: 0.1561
Phase: validation Epoch: 3/3 Iter: 13/39 Batch time: 0.1542
Phase: validation Epoch: 3/3 Iter: 14/39 Batch time: 0.1608
Phase: validation Epoch: 3/3 Iter: 15/39 Batch time: 0.1627
Phase: validation Epoch: 3/3 Iter: 16/39 Batch time: 0.1544
Phase: validation Epoch: 3/3 Iter: 17/39 Batch time: 0.1642
Phase: validation Epoch: 3/3 Iter: 18/39 Batch time: 0.1535
Phase: validation Epoch: 3/3 Iter: 19/39 Batch time: 0.1586
Phase: validation Epoch: 3/3 Iter: 20/39 Batch time: 0.1623
Phase: validation Epoch: 3/3 Iter: 21/39 Batch time: 0.1581
Phase: validation Epoch: 3/3 Iter: 22/39 Batch time: 0.1568
Phase: validation Epoch: 3/3 Iter: 23/39 Batch time: 0.1596
Phase: validation Epoch: 3/3 Iter: 24/39 Batch time: 0.1635
Phase: validation Epoch: 3/3 Iter: 25/39 Batch time: 0.1646
Phase: validation Epoch: 3/3 Iter: 26/39 Batch time: 0.1554
Phase: validation Epoch: 3/3 Iter: 27/39 Batch time: 0.1572
Phase: validation Epoch: 3/3 Iter: 28/39 Batch time: 0.1515
Phase: validation Epoch: 3/3 Iter: 29/39 Batch time: 0.1544
Phase: validation Epoch: 3/3 Iter: 30/39 Batch time: 0.1579
Phase: validation Epoch: 3/3 Iter: 31/39 Batch time: 0.1545
Phase: validation Epoch: 3/3 Iter: 32/39 Batch time: 0.1548
Phase: validation Epoch: 3/3 Iter: 33/39 Batch time: 0.1621
Phase: validation Epoch: 3/3 Iter: 34/39 Batch time: 0.1553
Phase: validation Epoch: 3/3 Iter: 35/39 Batch time: 0.1605
Phase: validation Epoch: 3/3 Iter: 36/39 Batch time: 0.1540
Phase: validation Epoch: 3/3 Iter: 37/39 Batch time: 0.1539
Phase: validation Epoch: 3/3 Iter: 38/39 Batch time: 0.1561
Phase: validation Epoch: 3/3 Iter: 39/39 Batch time: 0.0514
Phase: validation   Epoch: 3/3 Loss: 0.4484 Acc: 0.8497
Training completed in 1m 7s
Best test loss: 0.4484 | Best test accuracy: 0.8497
 </code>
 </pre>
 </details>

---

## 13. tutorial_quantum_metrology.html <a name="demo12"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quantum_metrology.html):

```
0: ──H──RZ(0)──H──RX(1.57)──RZ(1)──RX(-1.57)──H───────────────────────────────────────────────────────────╭X──RZ(8)──╭X──H──H─────────╭X──RZ(9)──╭X──H──────────H──╭X──RZ(10)──╭X──H──H─────────╭X──RZ(11)──╭X──H──────────H──────╭X──RZ(12)──╭X───H──H────────────────╭X──RZ(13)──╭X───H──RZ(0)──────PhaseDamp(0.2)──H───────────────RZ(14)──H───────RX(1.57)──RZ(15)────RX(-1.57)─────────────╭┤ Probs
1: ──H──RZ(2)──H──RX(1.57)──RZ(3)──RX(-1.57)──H──╭X──RZ(6)──╭X──H──H─────────╭X──RZ(7)──╭X──H──────────H──╰C─────────╰C──H──RX(1.57)──╰C─────────╰C──RX(-1.57)──H──│───────────│────────────────│───────────│─────────────────╭X──╰C──────────╰C──╭X──H──H─────────╭X──╰C──────────╰C──╭X──H──────────RZ(0)───────────PhaseDamp(0.2)──H───────RZ(16)──H─────────RX(1.57)──RZ(17)─────RX(-1.57)──├┤ Probs
2: ──H──RZ(4)──H──RX(1.57)──RZ(5)──RX(-1.57)──H──╰C─────────╰C──H──RX(1.57)──╰C─────────╰C──RX(-1.57)──H───────────────────────────────────────────────────────────╰C──────────╰C──H──RX(1.57)──╰C──────────╰C──RX(-1.57)──H──╰C──────────────────╰C──H──RX(1.57)──╰C──────────────────╰C──RX(-1.57)──RZ(0)───────────PhaseDamp(0.2)──H───────RZ(18)──H─────────RX(1.57)──RZ(19)─────RX(-1.57)──╰┤ Probs
Initialization: Cost = 3.9901
Iteration    5: Cost = 1.8267
Iteration   10: Cost = 1.7671
Iteration   15: Cost = 1.7988
Iteration   20: Cost = 1.6231
Cost for standard Ramsey sensing = 1.5543
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_quantum_metrology.html):

```
0: ──H──RZ(0.00)──H──RX(1.57)──RZ(1.00)──RX(-1.57)──H─────────────────────────────────────────────
1: ──H──RZ(2.00)──H──RX(1.57)──RZ(3.00)──RX(-1.57)──H─╭X──RZ(6.00)─╭X──H──H────────╭X──RZ(7.00)─╭X
2: ──H──RZ(4.00)──H──RX(1.57)──RZ(5.00)──RX(-1.57)──H─╰C───────────╰C──H──RX(1.57)─╰C───────────╰C
────────────────╭X──RZ(8.00)─╭X──H──H────────╭X──RZ(9.00)─╭X──H──────────H─╭X──RZ(10.00)─╭X──H
───H──────────H─╰C───────────╰C──H──RX(1.57)─╰C───────────╰C──RX(-1.57)────│─────────────│────
───RX(-1.57)──H────────────────────────────────────────────────────────────╰C────────────╰C──H
───H────────╭X──RZ(11.00)─╭X──H──────────H────╭X──RZ(12.00)─╭X──H──H──────────────╭X──RZ(13.00)─╭X
────────────│─────────────│───H────────────╭X─╰C────────────╰C─╭X──H──H────────╭X─╰C────────────╰C
───RX(1.57)─╰C────────────╰C──RX(-1.57)──H─╰C──────────────────╰C──H──RX(1.57)─╰C─────────────────
```

---

## 14. tutorial_qaoa_intro.html <a name="demo13"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_qaoa_intro.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
0: ──H──────RZ(1)──H──H──╭RZ(0.5)──H──H──────RZ(1)──H──H──╭RZ(0.5)──H──┤ ⟨Z⟩
1: ──RZ(1)──H────────────╰RZ(0.5)──H──RZ(1)──H────────────╰RZ(0.5)──H──┤ ⟨Z⟩
0: ──RX(0.5)──╭C──┤ ⟨Z⟩
1: ──H────────╰X──┤ ⟨Z⟩
0: ──RX(0.3)──╭C──RX(0.4)──╭C──RX(0.5)──╭C──┤ ⟨Z⟩
1: ──H────────╰X──H────────╰X──H────────╰X──┤ ⟨Z⟩
Cost Hamiltonian   (-0.25) [Z3]
+ (0.5) [Z0]
+ (0.5) [Z1]
+ (1.25) [Z2]
+ (0.75) [Z0 Z1]
+ (0.75) [Z0 Z2]
+ (0.75) [Z1 Z2]
+ (0.75) [Z2 Z3]
Mixer Hamiltonian   (1) [X0]
+ (1) [X1]
+ (1) [X2]
+ (1) [X3]
Optimal Parameters
[[0.59806352 0.94198485]
 [0.52797281 0.85552845]]
Optimal Parameters
[[0.45959941 0.96095271]
 [0.27029962 0.78042396]]
 </code>
 </pre>
 </details>

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_qaoa_intro.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
0: ──H──────────────MultiRZ(1.00)──H──H─╭MultiRZ(0.50)──H──H──────────────MultiRZ(1.00)──H──H
1: ──MultiRZ(1.00)──H───────────────────╰MultiRZ(0.50)──H──MultiRZ(1.00)──H──────────────────
──╭MultiRZ(0.50)──H─┤  <Z>
──╰MultiRZ(0.50)──H─┤  <Z>
0: ──RX(0.50)─╭C─┤  <Z>
1: ──H────────╰X─┤  <Z>
0: ──RX(0.30)─╭C──RX(0.40)─╭C──RX(0.50)─╭C─┤  <Z>
1: ──H────────╰X──H────────╰X──H────────╰X─┤  <Z>
Cost Hamiltonian   (-0.25) [Z3]
+ (0.5) [Z0]
+ (0.5) [Z1]
+ (1.25) [Z2]
+ (0.75) [Z0 Z1]
+ (0.75) [Z0 Z2]
+ (0.75) [Z1 Z2]
+ (0.75) [Z2 Z3]
Mixer Hamiltonian   (1) [X0]
+ (1) [X1]
+ (1) [X2]
+ (1) [X3]
Optimal Parameters
[[0.59806352 0.94198485]
 [0.52797281 0.85552845]]
Optimal Parameters
 </code>
 </pre>
 </details>

---

## 15. tutorial_backprop.html <a name="demo14"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_backprop.html):

```
-0.06518877224958124
[[-6.51887722e-02 -2.72891905e-02 -2.77555756e-17 -9.33934621e-02
  -7.61067572e-01  4.16333634e-17]]
Forward pass (best of 3): 0.017103538999981537 sec per loop
Gradient computation (best of 3): 4.833508461300016 sec per loop
6.157274039993353
Forward pass (best of 3): 0.046337660100016366 sec per loop
Backward pass (best of 3): 0.13896843400002581 sec per loop
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_backprop.html):

```
-0.06518877224958125
[[-6.51887722e-02 -2.72891905e-02 -2.83424776e-17 -9.33934621e-02
  -7.61067572e-01  4.10464615e-17]]
Forward pass (best of 3): 0.019636570000056964 sec per loop
Gradient computation (best of 3): 5.4928494678999416 sec per loop
7.069165200020507
Forward pass (best of 3): 0.0504114142000617 sec per loop
Backward pass (best of 3): 0.1567775040000015 sec per loop
```

---

## 16. tutorial_adaptive_circuits.html <a name="demo15"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_adaptive_circuits.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Excitation : [0, 1, 2, 3], Gradient: -0.012782175157665516
Excitation : [0, 1, 2, 5], Gradient: -1.1519648082658487e-19
Excitation : [0, 1, 2, 7], Gradient: -3.320369153236857e-19
Excitation : [0, 1, 2, 9], Gradient: 0.03426451170168425
Excitation : [0, 1, 3, 4], Gradient: -3.3881317890171965e-20
Excitation : [0, 1, 3, 6], Gradient: 1.4230153513872236e-19
Excitation : [0, 1, 3, 8], Gradient: -0.03426451170168414
Excitation : [0, 1, 4, 5], Gradient: -0.0235815290206774
Excitation : [0, 1, 5, 8], Gradient: 6.77626357803489e-21
Excitation : [0, 1, 6, 7], Gradient: -0.023581529020677384
Excitation : [0, 1, 7, 8], Gradient: 6.09863722023102e-20
Excitation : [0, 1, 8, 9], Gradient: -0.12362273485598785
Excitation : [0, 2], Gradient: -0.005062536239331469
Excitation : [0, 4], Gradient: 5.213244032762734e-18
Excitation : [0, 6], Gradient: -1.1009550669295385e-19
Excitation : [0, 8], Gradient: -0.0009448044625780129
Excitation : [1, 3], Gradient: 0.0049266168770000385
Excitation : [1, 5], Gradient: -6.241561796134186e-19
Excitation : [1, 7], Gradient: 1.1349630265445629e-18
Excitation : [1, 9], Gradient: 0.001453553485404355
n = 0,  E = -7.86266587 H, t = 2.38 s
n = 1,  E = -7.87094621 H, t = 2.37 s
n = 2,  E = -7.87563100 H, t = 2.37 s
n = 3,  E = -7.87829146 H, t = 2.35 s
n = 4,  E = -7.87981705 H, t = 2.64 s
n = 5,  E = -7.88070477 H, t = 2.40 s
n = 6,  E = -7.88123143 H, t = 2.39 s
n = 7,  E = -7.88155161 H, t = 2.39 s
n = 8,  E = -7.88175217 H, t = 2.39 s
n = 9,  E = -7.88188237 H, t = 2.39 s
n = 10,  E = -7.88197041 H, t = 2.37 s
n = 11,  E = -7.88203267 H, t = 2.37 s
n = 12,  E = -7.88207879 H, t = 2.37 s
n = 13,  E = -7.88211452 H, t = 2.36 s
n = 14,  E = -7.88214335 H, t = 2.67 s
n = 15,  E = -7.88216743 H, t = 2.42 s
n = 16,  E = -7.88218814 H, t = 2.41 s
n = 17,  E = -7.88220634 H, t = 2.39 s
n = 18,  E = -7.88222261 H, t = 2.39 s
n = 19,  E = -7.88223734 H, t = 2.38 s
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
Excitation : [0, 1, 2, 3], Gradient: -0.012782175157665807
Excitation : [0, 1, 2, 5], Gradient: -2.1006417091906653e-19
Excitation : [0, 1, 2, 7], Gradient: -1.8295911660692895e-19
Excitation : [0, 1, 2, 9], Gradient: 0.03426451170168446
Excitation : [0, 1, 3, 4], Gradient: -1.2874900798265363e-19
Excitation : [0, 1, 3, 6], Gradient: -1.2874900798265368e-19
Excitation : [0, 1, 3, 8], Gradient: -0.03426451170168452
Excitation : [0, 1, 4, 5], Gradient: -0.023581529020677425
Excitation : [0, 1, 5, 8], Gradient: -3.388131789017139e-20
Excitation : [0, 1, 6, 7], Gradient: -0.023581529020677446
Excitation : [0, 1, 7, 8], Gradient: -6.098637220230893e-20
Excitation : [0, 1, 8, 9], Gradient: -0.12362273485598764
Excitation : [0, 2], Gradient: -0.0050625362393311195
Excitation : [0, 4], Gradient: -4.580008099575328e-18
Excitation : [0, 6], Gradient: 3.734120838722142e-18
Excitation : [0, 8], Gradient: -0.0009448044625782528
Excitation : [1, 3], Gradient: 0.004926616876999633
Excitation : [1, 5], Gradient: 4.235009734185051e-18
Excitation : [1, 7], Gradient: 1.891891418616375e-18
Excitation : [1, 9], Gradient: 0.0014535534854045332
n = 0,  E = -7.86266587 H, t = 2.64 s
n = 1,  E = -7.87094621 H, t = 2.63 s
n = 2,  E = -7.87563100 H, t = 2.63 s
n = 3,  E = -7.87829146 H, t = 2.63 s
n = 4,  E = -7.87981705 H, t = 2.65 s
n = 5,  E = -7.88070477 H, t = 2.63 s
n = 6,  E = -7.88123143 H, t = 2.65 s
n = 7,  E = -7.88155161 H, t = 2.63 s
n = 8,  E = -7.88175217 H, t = 2.60 s
n = 9,  E = -7.88188237 H, t = 2.65 s
n = 10,  E = -7.88197041 H, t = 2.70 s
n = 11,  E = -7.88203267 H, t = 2.61 s
n = 12,  E = -7.88207879 H, t = 2.61 s
n = 13,  E = -7.88211452 H, t = 2.73 s
n = 14,  E = -7.88214335 H, t = 2.66 s
n = 15,  E = -7.88216743 H, t = 2.60 s
n = 16,  E = -7.88218814 H, t = 2.58 s
n = 17,  E = -7.88220634 H, t = 2.66 s
n = 18,  E = -7.88222261 H, t = 2.58 s
n = 19,  E = -7.88223734 H, t = 2.84 s
n = 0,  E = -7.86266587 H, t = 0.13 s
n = 1,  E = -7.87094621 H, t = 0.13 s
n = 2,  E = -7.87563100 H, t = 0.14 s
n = 3,  E = -7.87829146 H, t = 0.14 s
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

---

## 17. tutorial_jax_transformations.html <a name="demo16"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_jax_transformations.html):

```
No jit time: 0.0077 seconds
First run time: 0.0588 seconds
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_jax_transformations.html):

```
No jit time: 0.0099 seconds
First run time: 0.0777 seconds
```

---

## 18. tutorial_measurement_optimize.html <a name="demo17"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_measurement_optimize.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
   (-46.46390678868893) [I0]
+ (0.782966172595019) [Z11]
+ (0.7829661725950191) [Z10]
+ (0.8084581961720492) [Z12]
+ (0.8084581961720493) [Z13]
+ (1.2034402289145631) [Z4]
+ (1.2034402289145634) [Z5]
+ (1.30968629886154) [Z6]
+ (1.30968629886154) [Z7]
+ (1.369352563471818) [Z8]
+ (1.3693525634718182) [Z9]
+ (1.6538942226831723) [Z3]
+ (1.6538942226831725) [Z2]
+ (12.41263074211177) [Z0]
+ (12.41263074211177) [Z1]
+ (-8.19426137221021e-06) [Y10 Y12]
+ (-8.19426137221021e-06) [X10 X12]
+ (-1.854060857999677e-06) [Y5 Y7]
+ (-1.854060857999677e-06) [X5 X7]
+ (-7.76499411749896e-07) [Y3 Y5]
+ (-7.76499411749896e-07) [X3 X5]
+ (-5.929765816206068e-07) [Y4 Y6]
+ (-5.929765816206068e-07) [X4 X6]
+ (1.6021167406784645e-06) [Y2 Y4]
+ (1.6021167406784645e-06) [X2 X4]
+ (7.954413176108448e-06) [Y11 Y13]
+ (7.954413176108448e-06) [X11 X13]
+ (0.0032769719312316986) [Y1 Y3]
+ (0.0032769719312316986) [X1 X3]
+ (0.10433064780651424) [Y0 Y2]
+ (0.10433064780651424) [X0 X2]
+ (0.11270386920332214) [Z10 Z12]
+ (0.11270386920332214) [Z11 Z13]
+ (0.11383573679388652) [Z4 Z12]
+ (0.11383573679388652) [Z5 Z13]
+ (0.11952438964682663) [Z6 Z10]
+ (0.11952438964682663) [Z7 Z11]
+ (0.124899909172376) [Z4 Z10]
+ (0.124899909172376) [Z5 Z11]
+ (0.12495807739503226) [Z2 Z4]
+ (0.12495807739503226) [Z3 Z5]
+ (0.1279950249246842) [Z2 Z10]
+ (0.1279950249246842) [Z3 Z11]
+ (0.13401715261963687) [Z6 Z12]
+ (0.13401715261963687) [Z7 Z13]
+ (0.13701191674040736) [Z4 Z6]
+ (0.13701191674040736) [Z5 Z7]
+ (0.13734953064261318) [Z6 Z11]
+ (0.13734953064261318) [Z7 Z10]
+ (0.13739104762683224) [Z2 Z6]
+ (0.13739104762683224) [Z3 Z7]
+ (0.1376687264585258) [Z8 Z10]
+ (0.1376687264585258) [Z9 Z11]
+ (0.14011289865354817) [Z2 Z12]
+ (0.14011289865354817) [Z3 Z13]
+ (0.14138905291942816) [Z10 Z13]
+ (0.14138905291942816) [Z11 Z12]
+ (0.14257997712485748) [Z4 Z11]
+ (0.14257997712485748) [Z5 Z10]
+ (0.14722943218766169) [Z8 Z11]
+ (0.14722943218766169) [Z9 Z10]
+ (0.1489943057506553) [Z4 Z7]
+ (0.1489943057506553) [Z5 Z6]
+ (0.14926355147388906) [Z10 Z11]
+ (0.14960702684445298) [Z4 Z8]
+ (0.14960702684445298) [Z5 Z9]
+ (0.14973486803496922) [Z8 Z12]
+ (0.14973486803496922) [Z9 Z13]
+ (0.150714081210083) [Z2 Z8]
+ (0.150714081210083) [Z3 Z9]
+ (0.15138327161428822) [Z6 Z13]
+ (0.15138327161428822) [Z7 Z12]
+ (0.1521504070886904) [Z4 Z13]
+ (0.1521504070886904) [Z5 Z12]
+ (0.15337968243314154) [Z2 Z11]
+ (0.15337968243314154) [Z3 Z10]
+ (0.15435748657223636) [Z12 Z13]
+ (0.15569010671752462) [Z2 Z13]
+ (0.15569010671752462) [Z3 Z12]
+ (0.15676396176430996) [Z4 Z9]
+ (0.15676396176430996) [Z5 Z8]
+ (0.1575531479798565) [Z4 Z5]
+ (0.1607976453483857) [Z2 Z5]
+ (0.1607976453483857) [Z3 Z4]
+ (0.16756653265461252) [Z6 Z8]
+ (0.16756653265461252) [Z7 Z9]
+ (0.1685348656157994) [Z2 Z7]
+ (0.1685348656157994) [Z3 Z6]
+ (0.18143991440303858) [Z6 Z9]
+ (0.18143991440303858) [Z7 Z8]
+ (0.18189085790751394) [Z2 Z3]
+ (0.1869082047691257) [Z2 Z9]
+ (0.1869082047691257) [Z3 Z8]
+ (0.19299723935364219) [Z0 Z10]
+ (0.19299723935364219) [Z1 Z11]
+ (0.19392534613270157) [Z6 Z7]
+ (0.19661770890342142) [Z0 Z4]
+ (0.19661770890342142) [Z1 Z5]
+ (0.19936354537360823) [Z0 Z5]
+ (0.19936354537360823) [Z1 Z4]
+ (0.20072866460441743) [Z0 Z11]
+ (0.20072866460441743) [Z1 Z10]
+ (0.2110265984979148) [Z0 Z12]
+ (0.2110265984979148) [Z1 Z13]
+ (0.21631037498631775) [Z0 Z13]
+ (0.21631037498631775) [Z1 Z12]
+ (0.2200397733437609) [Z8 Z9]
+ (0.23671080783830442) [Z0 Z2]
+ (0.23671080783830442) [Z1 Z3]
+ (0.24164663936017156) [Z0 Z6]
+ (0.24164663936017156) [Z1 Z7]
+ (0.2485348337131421) [Z0 Z7]
+ (0.2485348337131421) [Z1 Z6]
+ (0.2512944567459171) [Z0 Z3]
+ (0.2512944567459171) [Z1 Z2]
+ (0.2723251830660566) [Z0 Z8]
+ (0.2723251830660566) [Z1 Z9]
+ (0.27883454426723386) [Z0 Z9]
+ (0.27883454426723386) [Z1 Z8]
+ (1.186176373486048) [Z0 Z1]
+ (-1.2260484989531175e-05) [Y4 Z5 Y6]
+ (-1.2260484989531175e-05) [X4 Z5 X6]
+ (-1.2260484989531174e-05) [Y5 Z6 Y7]
+ (-1.2260484989531174e-05) [X5 Z6 X7]
+ (-1.0722312158600235e-05) [Y11 Z12 Y13]
+ (-1.0722312158600235e-05) [X11 Z12 X13]
+ (-1.0722312158600231e-05) [Y10 Z11 Y12]
+ (-1.0722312158600231e-05) [X10 Z11 X12]
+ (-3.887051671763188e-06) [Y2 Z3 Y4]
+ (-3.887051671763188e-06) [X2 Z3 X4]
+ (-3.887051671763188e-06) [Y3 Z4 Y5]
+ (-3.887051671763188e-06) [X3 Z4 X5]
+ (0.1250703257977217) [Y0 Z1 Y2]
+ (0.1250703257977217) [X0 Z1 X2]
+ (0.1250703257977217) [Y1 Z2 Y3]
+ (0.1250703257977217) [X1 Z2 X3]
+ (-0.0361941235590427) [Y2 Y3 X8 X9]
+ (-0.0361941235590427) [X2 X3 Y8 Y9]
+ (-0.03583956795335342) [Y2 Y3 X4 X5]
+ (-0.03583956795335342) [X2 X3 Y4 Y5]
+ (-0.031143817988967145) [Y2 Y3 X6 X7]
+ (-0.031143817988967145) [X2 X3 Y6 Y7]
+ (-0.028685183716106042) [Y10 Y11 X12 X13]
+ (-0.028685183716106042) [X10 X11 Y12 Y13]
+ (-0.02599617759802106) [Y3 Z4 Z5 Y7]
+ (-0.02599617759802106) [X3 Z4 Z5 X7]
+ (-0.02538465750845737) [Y2 Y3 X10 X11]
+ (-0.02538465750845737) [X2 X3 Y10 Y11]
+ (-0.019028242443847227) [Y3 Y4 X11 X12]
+ (-0.019028242443847227) [X3 X4 Y11 Y12]
+ (-0.01782514099578653) [Y6 Y7 X10 X11]
+ (-0.01782514099578653) [X6 X7 Y10 Y11]
+ (-0.01768006795248149) [Y4 Y5 X10 X11]
+ (-0.01768006795248149) [X4 X5 Y10 Y11]
+ (-0.01736611899465137) [Y6 Y7 X12 X13]
+ (-0.01736611899465137) [X6 X7 Y12 Y13]
+ (-0.015577208063976432) [Y2 Y3 X12 X13]
+ (-0.015577208063976432) [X2 X3 Y12 Y13]
+ (-0.014583648907612727) [Y0 Y1 X2 X3]
+ (-0.014583648907612727) [X0 X1 Y2 Y3]
+ (-0.013873381748426061) [Y6 Y7 X8 X9]
+ (-0.013873381748426061) [X6 X7 Y8 Y9]
+ (-0.01198238901024794) [Y4 Y5 X6 X7]
+ (-0.01198238901024794) [X4 X5 Y6 Y7]
+ (-0.011285190200840931) [Y5 X6 X11 Y12]
+ (-0.011285190200840931) [X5 Y6 Y11 X12]
+ (-0.009560705729135902) [Y8 Y9 X10 X11]
+ (-0.009560705729135902) [X8 X9 Y10 Y11]
+ (-0.00812525192138104) [Y1 X2 X8 Y9]
+ (-0.00812525192138104) [Y1 Y2 Y8 Y9]
+ (-0.00812525192138104) [X1 X2 X8 X9]
+ (-0.00812525192138104) [X1 Y2 Y8 X9]
+ (-0.007731425250775275) [Y0 Y1 X10 X11]
+ (-0.007731425250775275) [X0 X1 Y10 Y11]
+ (-0.007156934919856957) [Y4 Y5 X8 X9]
+ (-0.007156934919856957) [X4 X5 Y8 Y9]
+ (-0.006888194352970547) [Y0 Y1 X6 X7]
+ (-0.006888194352970547) [X0 X1 Y6 Y7]
+ (-0.006509361201177233) [Y0 Y1 X8 X9]
+ (-0.006509361201177233) [X0 X1 Y8 Y9]
+ (-0.006087822480561841) [Y8 Y9 X12 X13]
+ (-0.006087822480561841) [X8 X9 Y12 Y13]
+ (-0.005283776488402946) [Y0 Y1 X12 X13]
+ (-0.005283776488402946) [X0 X1 Y12 Y13]
+ (-0.005143391768825075) [Y3 X4 X5 Y6]
+ (-0.005143391768825075) [X3 Y4 Y5 X6]
+ (-0.004684903388155212) [Y1 X2 X6 Y7]
+ (-0.004684903388155212) [Y1 Y2 Y6 Y7]
+ (-0.004684903388155212) [X1 X2 X6 X7]
+ (-0.004684903388155212) [X1 Y2 Y6 X7]
+ (-0.004575007626639198) [Y1 X2 X12 Y13]
+ (-0.004575007626639198) [Y1 Y2 Y12 Y13]
+ (-0.004575007626639198) [X1 X2 X12 X13]
+ (-0.004575007626639198) [X1 Y2 Y12 X13]
+ (-0.004424855449441866) [Y1 X2 X4 Y5]
+ (-0.004424855449441866) [Y1 Y2 Y4 Y5]
+ (-0.004424855449441866) [X1 X2 X4 X5]
+ (-0.004424855449441866) [X1 Y2 Y4 X5]
+ (-0.0034795118903343217) [Y2 Z3 Z5 Y6]
+ (-0.0034795118903343217) [X2 Z3 Z5 X6]
+ (-0.0034795118903343217) [Y3 Z4 Z6 Y7]
+ (-0.0034795118903343217) [X3 Z4 Z6 X7]
+ (-0.0027458364701868185) [Y0 Y1 X4 X5]
+ (-0.0027458364701868185) [X0 X1 Y4 Y5]
+ (-0.0017992194936630112) [Y1 X2 X10 Y11]
+ (-0.0017992194936630112) [Y1 Y2 Y10 Y11]
+ (-0.0017992194936630112) [X1 X2 X10 X11]
+ (-0.0017992194936630112) [X1 Y2 Y10 X11]
+ (-0.0002921986261110214) [Y7 Y8 X9 X10]
+ (-0.0002921986261110214) [X7 X8 Y9 Y10]
+ (-8.19426137221021e-06) [Z10 Y11 Z12 Y13]
+ (-8.19426137221021e-06) [Z10 X11 Z12 X13]
+ (-7.801707500355871e-06) [Y2 Z3 Y4 Z11]
+ (-7.801707500355871e-06) [X2 Z3 X4 Z11]
+ (-7.801707500355871e-06) [Y3 Z4 Y5 Z10]
+ (-7.801707500355871e-06) [X3 Z4 X5 Z10]
+ (-4.643051068429628e-06) [Y3 X4 X10 Y11]
+ (-4.643051068429628e-06) [Y3 Y4 Y10 Y11]
+ (-4.643051068429628e-06) [X3 X4 X10 X11]
+ (-4.643051068429628e-06) [X3 Y4 Y10 X11]
+ (-4.588855155676688e-06) [Y4 Z5 Y6 Z13]
+ (-4.588855155676688e-06) [X4 Z5 X6 Z13]
+ (-4.588855155676688e-06) [Y5 Z6 Y7 Z12]
+ (-4.588855155676688e-06) [X5 Z6 X7 Z12]
+ (-4.5565692180959244e-06) [Y5 X6 X12 Y13]
+ (-4.5565692180959244e-06) [Y5 Y6 Y12 Y13]
+ (-4.5565692180959244e-06) [X5 X6 X12 X13]
+ (-4.5565692180959244e-06) [X5 Y6 Y12 X13]
+ (-3.6945132944717873e-06) [Y4 X5 X11 Y12]
+ (-3.6945132944717873e-06) [Y4 Y5 Y11 Y12]
+ (-3.6945132944717873e-06) [X4 X5 X11 X12]
+ (-3.6945132944717873e-06) [X4 Y5 Y11 X12]
+ (-3.3440815565536897e-06) [Z0 Y5 Z6 Y7]
+ (-3.3440815565536897e-06) [Z0 X5 Z6 X7]
+ (-3.3440815565536897e-06) [Z1 Y4 Z5 Y6]
+ (-3.3440815565536897e-06) [Z1 X4 Z5 X6]
+ (-3.1586564319262415e-06) [Y2 Z3 Y4 Z10]
+ (-3.1586564319262415e-06) [X2 Z3 X4 Z10]
+ (-3.1586564319262415e-06) [Y3 Z4 Y5 Z11]
+ (-3.1586564319262415e-06) [X3 Z4 X5 Z11]
+ (-3.099349243682632e-06) [Z0 Y4 Z5 Y6]
+ (-3.099349243682632e-06) [Z0 X4 Z5 X6]
+ (-3.099349243682632e-06) [Z1 Y5 Z6 Y7]
+ (-3.099349243682632e-06) [Z1 X5 Z6 X7]
+ (-2.890967881779266e-06) [Z6 Y11 Z12 Y13]
+ (-2.890967881779266e-06) [Z6 X11 Z12 X13]
+ (-2.890967881779266e-06) [Z7 Y10 Z11 Y12]
+ (-2.890967881779266e-06) [Z7 X10 Z11 X12]
+ (-2.177664605196274e-06) [Z0 Y10 Z11 Y12]
+ (-2.177664605196274e-06) [Z0 X10 Z11 X12]
+ (-2.177664605196274e-06) [Z1 Y11 Z12 Y13]
+ (-2.177664605196274e-06) [Z1 X11 Z12 X13]
+ (-1.8818501832576009e-06) [Y4 Z5 Y6 Z9]
+ (-1.8818501832576009e-06) [X4 Z5 X6 Z9]
+ (-1.8818501832576009e-06) [Y5 Z6 Y7 Z8]
+ (-1.8818501832576009e-06) [X5 Z6 X7 Z8]
+ (-1.855120121631274e-06) [Z6 Y10 Z11 Y12]
+ (-1.855120121631274e-06) [Z6 X10 Z11 X12]
+ (-1.855120121631274e-06) [Z7 Y11 Z12 Y13]
+ (-1.855120121631274e-06) [Z7 X11 Z12 X13]
+ (-1.8540608579996771e-06) [Y4 Z5 Y6 Z7]
+ (-1.8540608579996771e-06) [X4 Z5 X6 Z7]
+ (-1.816303169809e-06) [Z4 Y11 Z12 Y13]
+ (-1.816303169809e-06) [Z4 X11 Z12 X13]
+ (-1.816303169809e-06) [Z5 Y10 Z11 Y12]
+ (-1.816303169809e-06) [Z5 X10 Z11 X12]
+ (-1.6923978286216784e-06) [Y4 Z5 Y6 Z10]
+ (-1.6923978286216784e-06) [X4 Z5 X6 Z10]
+ (-1.6923978286216784e-06) [Y5 Z6 Y7 Z11]
+ (-1.6923978286216784e-06) [X5 Z6 X7 Z11]
+ (-1.6148794140664079e-06) [Z0 Y11 Z12 Y13]
+ (-1.6148794140664079e-06) [Z0 X11 Z12 X13]
+ (-1.6148794140664079e-06) [Z1 Y10 Z11 Y12]
+ (-1.6148794140664079e-06) [Z1 X10 Z11 X12]
+ (-1.5973171979010596e-06) [Z8 Y10 Z11 Y12]
+ (-1.5973171979010596e-06) [Z8 X10 Z11 X12]
+ (-1.5973171979010596e-06) [Z9 Y11 Z12 Y13]
+ (-1.5973171979010596e-06) [Z9 X11 Z12 X13]
+ (-1.4548424490306174e-06) [Y3 X4 X6 Y7]
+ (-1.4548424490306174e-06) [Y3 Y4 Y6 Y7]
+ (-1.4548424490306174e-06) [X3 X4 X6 X7]
+ (-1.4548424490306174e-06) [X3 Y4 Y6 X7]
+ (-1.3980449081819972e-06) [Y4 Z5 Y6 Z8]
+ (-1.3980449081819972e-06) [X4 Z5 X6 Z8]
+ (-1.3980449081819972e-06) [Y5 Z6 Y7 Z9]
+ (-1.3980449081819972e-06) [X5 Z6 X7 Z9]
+ (-1.1954890098573842e-06) [Y2 Z3 Y4 Z7]
+ (-1.1954890098573842e-06) [X2 Z3 X4 Z7]
+ (-1.1954890098573842e-06) [Y3 Z4 Y5 Z6]
+ (-1.1954890098573842e-06) [X3 Z4 X5 Z6]
+ (-1.1908508081211731e-06) [Z0 Y3 Z4 Y5]
+ (-1.1908508081211731e-06) [Z0 X3 Z4 X5]
+ (-1.1908508081211731e-06) [Z1 Y2 Z3 Y4]
+ (-1.1908508081211731e-06) [Z1 X2 Z3 X4]
+ (-1.1708301370687092e-06) [Z2 Y5 Z6 Y7]
+ (-1.1708301370687092e-06) [Z2 X5 Z6 X7]
+ (-1.1708301370687092e-06) [Z3 Y4 Z5 Y6]
+ (-1.1708301370687092e-06) [Z3 X4 Z5 X6]
+ (-1.0632283424758415e-06) [Z2 Y10 Z11 Y12]
+ (-1.0632283424758415e-06) [Z2 X10 Z11 X12]
+ (-1.0632283424758415e-06) [Z3 Y11 Z12 Y13]
+ (-1.0632283424758415e-06) [Z3 X11 Z12 X13]
+ (-1.0358477601479918e-06) [Y6 X7 X11 Y12]
+ (-1.0358477601479918e-06) [Y6 Y7 Y11 Y12]
+ (-1.0358477601479918e-06) [X6 X7 X11 X12]
+ (-1.0358477601479918e-06) [X6 Y7 Y11 X12]
+ (-9.509249752110998e-07) [Z2 Y4 Z5 Y6]
+ (-9.509249752110998e-07) [Z2 X4 Z5 X6]
+ (-9.509249752110998e-07) [Z3 Y5 Z6 Y7]
+ (-9.509249752110998e-07) [Z3 X5 Z6 X7]
+ (-9.344557777531066e-07) [Z8 Y11 Z12 Y13]
+ (-9.344557777531066e-07) [Z8 X11 Z12 X13]
+ (-9.344557777531066e-07) [Z9 Y10 Z11 Y12]
+ (-9.344557777531066e-07) [Z9 X10 Z11 X12]
+ (-8.33774675243418e-07) [Z0 Y2 Z3 Y4]
+ (-8.33774675243418e-07) [Z0 X2 Z3 X4]
+ (-8.33774675243418e-07) [Z1 Y3 Z4 Y5]
+ (-8.33774675243418e-07) [Z1 X3 Z4 X5]
+ (-7.956895371954227e-07) [Y3 X4 X8 Y9]
+ (-7.956895371954227e-07) [Y3 Y4 Y8 Y9]
+ (-7.956895371954227e-07) [X3 X4 X8 X9]
+ (-7.956895371954227e-07) [X3 Y4 Y8 X9]
+ (-7.764994117498959e-07) [Y2 Z3 Y4 Z5]
+ (-7.764994117498959e-07) [X2 Z3 X4 Z5]
+ (-5.929765816206068e-07) [Z4 Y5 Z6 Y7]
+ (-5.929765816206068e-07) [Z4 X5 Z6 X7]
+ (-5.770052993626501e-07) [Y2 Z3 Y4 Z9]
+ (-5.770052993626501e-07) [X2 Z3 X4 Z9]
+ (-5.770052993626501e-07) [Y3 Z4 Y5 Z8]
+ (-5.770052993626501e-07) [X3 Z4 X5 Z8]
+ (-5.471647744671347e-07) [Y1 Y2 X11 X12]
+ (-5.471647744671347e-07) [X1 X2 Y11 Y12]
+ (-4.838052750756037e-07) [Y5 X6 X8 Y9]
+ (-4.838052750756037e-07) [Y5 Y6 Y8 Y9]
+ (-4.838052750756037e-07) [X5 X6 X8 X9]
+ (-4.838052750756037e-07) [X5 Y6 Y8 X9]
+ (-3.5707613287775506e-07) [Y0 X1 X3 Y4]
+ (-3.5707613287775506e-07) [Y0 Y1 Y3 Y4]
+ (-3.5707613287775506e-07) [X0 X1 X3 X4]
+ (-3.5707613287775506e-07) [X0 Y1 Y3 X4]
+ (-2.447323128710571e-07) [Y0 X1 X5 Y6]
+ (-2.447323128710571e-07) [Y0 Y1 Y5 Y6]
+ (-2.447323128710571e-07) [X0 X1 X5 X6]
+ (-2.447323128710571e-07) [X0 Y1 Y5 X6]
+ (-2.199051618576094e-07) [Y2 X3 X5 Y6]
+ (-2.199051618576094e-07) [Y2 Y3 Y5 Y6]
+ (-2.199051618576094e-07) [X2 X3 X5 X6]
+ (-2.199051618576094e-07) [X2 Y3 Y5 X6]
+ (-1.9332412770073636e-07) [Y1 X2 X3 Y4]
+ (-1.9332412770073636e-07) [X1 Y2 Y3 X4]
+ (-1.291969486193612e-07) [Y1 Z2 Z3 Y5]
+ (-1.291969486193612e-07) [X1 Z2 Z3 X5]
+ (1.7379332622952985e-07) [Y0 Z1 Z3 Y4]
+ (1.7379332622952985e-07) [X0 Z1 Z3 X4]
+ (1.7379332622952985e-07) [Y1 Z2 Z4 Y5]
+ (1.7379332622952985e-07) [X1 Z2 Z4 X5]
+ (1.9332412770073636e-07) [Y1 Y2 X3 X4]
+ (1.9332412770073636e-07) [X1 X2 Y3 Y4]
+ (2.1868423783277276e-07) [Y2 Z3 Y4 Z8]
+ (2.1868423783277276e-07) [X2 Z3 X4 Z8]
+ (2.1868423783277276e-07) [Y3 Z4 Y5 Z9]
+ (2.1868423783277276e-07) [X3 Z4 X5 Z9]
+ (2.5935343917323336e-07) [Y2 Z3 Y4 Z6]
+ (2.5935343917323336e-07) [X2 Z3 X4 Z6]
+ (2.5935343917323336e-07) [Y3 Z4 Y5 Z7]
+ (2.5935343917323336e-07) [X3 Z4 X5 Z7]
+ (3.606071867730044e-07) [Y0 Z1 Z2 Y4]
+ (3.606071867730044e-07) [X0 Z1 Z2 X4]
+ (3.606071867730044e-07) [Y1 Z3 Z4 Y5]
+ (3.606071867730044e-07) [X1 Z3 Z4 X5]
+ (5.471647744671347e-07) [Y1 X2 X11 Y12]
+ (5.471647744671347e-07) [X1 Y2 Y11 X12]
+ (5.627851911298655e-07) [Y0 X1 X11 Y12]
+ (5.627851911298655e-07) [Y0 Y1 Y11 Y12]
+ (5.627851911298655e-07) [X0 X1 X11 X12]
+ (5.627851911298655e-07) [X0 Y1 Y11 X12]
+ (6.628614201479532e-07) [Y8 X9 X11 Y12]
+ (6.628614201479532e-07) [Y8 Y9 Y11 Y12]
+ (6.628614201479532e-07) [X8 X9 X11 X12]
+ (6.628614201479532e-07) [X8 Y9 Y11 X12]
+ (1.1094407589507326e-06) [Z2 Y11 Z12 Y13]
+ (1.1094407589507326e-06) [Z2 X11 Z12 X13]
+ (1.1094407589507326e-06) [Z3 Y10 Z11 Y12]
+ (1.1094407589507326e-06) [Z3 X10 Z11 X12]
+ (1.6021167406784645e-06) [Z2 Y3 Z4 Y5]
+ (1.6021167406784645e-06) [Z2 X3 Z4 X5]
+ (1.878210124662788e-06) [Z4 Y10 Z11 Y12]
+ (1.878210124662788e-06) [Z4 X10 Z11 X12]
+ (1.878210124662788e-06) [Z5 Y11 Z12 Y13]
+ (1.878210124662788e-06) [Z5 X11 Z12 X13]
+ (2.172669101426574e-06) [Y2 X3 X11 Y12]
+ (2.172669101426574e-06) [Y2 Y3 Y11 Y12]
+ (2.172669101426574e-06) [X2 X3 X11 X12]
+ (2.172669101426574e-06) [X2 Y3 Y11 X12]
+ (3.1174479458297532e-06) [Y0 Z2 Z3 Y4]
+ (3.1174479458297532e-06) [X0 Z2 Z3 X4]
+ (3.539054184570986e-06) [Y2 Z3 Y4 Z12]
+ (3.539054184570986e-06) [X2 Z3 X4 Z12]
+ (3.539054184570986e-06) [Y3 Z4 Y5 Z13]
+ (3.539054184570986e-06) [X3 Z4 X5 Z13]
+ (4.281913884742542e-06) [Y4 Z5 Y6 Z11]
+ (4.281913884742542e-06) [X4 Z5 X6 Z11]
+ (4.281913884742542e-06) [Y5 Z6 Y7 Z10]
+ (4.281913884742542e-06) [X5 Z6 X7 Z10]
+ (5.275883122063986e-06) [Y3 X4 X12 Y13]
+ (5.275883122063986e-06) [Y3 Y4 Y12 Y13]
+ (5.275883122063986e-06) [X3 X4 X12 X13]
+ (5.275883122063986e-06) [X3 Y4 Y12 X13]
+ (5.974311713364219e-06) [Y5 X6 X10 Y11]
+ (5.974311713364219e-06) [Y5 Y6 Y10 Y11]
+ (5.974311713364219e-06) [X5 X6 X10 X11]
+ (5.974311713364219e-06) [X5 Y6 Y10 X11]
+ (7.954413176108448e-06) [Y10 Z11 Y12 Z13]
+ (7.954413176108448e-06) [X10 Z11 X12 Z13]
+ (8.814937306634974e-06) [Y2 Z3 Y4 Z13]
+ (8.814937306634974e-06) [X2 Z3 X4 Z13]
+ (8.814937306634974e-06) [Y3 Z4 Y5 Z12]
+ (8.814937306634974e-06) [X3 Z4 X5 Z12]
+ (0.0002921986261110214) [Y7 X8 X9 Y10]
+ (0.0002921986261110214) [X7 Y8 Y9 X10]
+ (0.0004956762314916422) [Y2 Z4 Z5 Y6]
+ (0.0004956762314916422) [X2 Z4 Z5 X6]
+ (0.001105903769189707) [Y0 Z1 Y2 Z5]
+ (0.001105903769189707) [X0 Z1 X2 Z5]
+ (0.001105903769189707) [Y1 Z2 Y3 Z4]
+ (0.001105903769189707) [X1 Z2 X3 Z4]
+ (0.0016638798784907542) [Y2 Z3 Z4 Y6]
+ (0.0016638798784907542) [X2 Z3 Z4 X6]
+ (0.0016638798784907542) [Y3 Z5 Z6 Y7]
+ (0.0016638798784907542) [X3 Z5 Z6 X7]
+ (0.0017560707018412557) [Y0 Z1 Y2 Z11]
+ (0.0017560707018412557) [X0 Z1 X2 Z11]
+ (0.0017560707018412557) [Y1 Z2 Y3 Z10]
+ (0.0017560707018412557) [X1 Z2 X3 Z10]
+ (0.0023262306231581005) [Y0 Z1 Y2 Z13]
+ (0.0023262306231581005) [X0 Z1 X2 Z13]
+ (0.0023262306231581005) [Y1 Z2 Y3 Z12]
+ (0.0023262306231581005) [X1 Z2 X3 Z12]
+ (0.0027458364701868185) [Y0 X1 X4 Y5]
+ (0.0027458364701868185) [X0 Y1 Y4 X5]
+ (0.002929768674751077) [Y0 Z1 Y2 Z9]
+ (0.002929768674751077) [X0 Z1 X2 Z9]
+ (0.002929768674751077) [Y1 Z2 Y3 Z8]
+ (0.002929768674751077) [X1 Z2 X3 Z8]
+ (0.0032769719312316986) [Y0 Z1 Y2 Z3]
+ (0.0032769719312316986) [X0 Z1 X2 Z3]
+ (0.0033476175306662043) [Y0 Z1 Y2 Z7]
+ (0.0033476175306662043) [X0 Z1 X2 Z7]
+ (0.0033476175306662043) [Y1 Z2 Y3 Z6]
+ (0.0033476175306662043) [X1 Z2 X3 Z6]
+ (0.003555290195504267) [Y0 Z1 Y2 Z10]
+ (0.003555290195504267) [X0 Z1 X2 Z10]
+ (0.003555290195504267) [Y1 Z2 Y3 Z11]
+ (0.003555290195504267) [X1 Z2 X3 Z11]
+ (0.005143391768825075) [Y3 Y4 X5 X6]
+ (0.005143391768825075) [X3 X4 Y5 Y6]
+ (0.005283776488402946) [Y0 X1 X12 Y13]
+ (0.005283776488402946) [X0 Y1 Y12 X13]
+ (0.0055307592186315735) [Y0 Z1 Y2 Z4]
+ (0.0055307592186315735) [X0 Z1 X2 Z4]
+ (0.0055307592186315735) [Y1 Z2 Y3 Z5]
+ (0.0055307592186315735) [X1 Z2 X3 Z5]
+ (0.006087822480561841) [Y8 X9 X12 Y13]
+ (0.006087822480561841) [X8 Y9 Y12 X13]
+ (0.006509361201177233) [Y0 X1 X8 Y9]
+ (0.006509361201177233) [X0 Y1 Y8 X9]
+ (0.006888194352970547) [Y0 X1 X6 Y7]
+ (0.006888194352970547) [X0 Y1 Y6 X7]
+ (0.006901238249797298) [Y0 Z1 Y2 Z12]
+ (0.006901238249797298) [X0 Z1 X2 Z12]
+ (0.006901238249797298) [Y1 Z2 Y3 Z13]
+ (0.006901238249797298) [X1 Z2 X3 Z13]
+ (0.007156934919856957) [Y4 X5 X8 Y9]
+ (0.007156934919856957) [X4 Y5 Y8 X9]
+ (0.007731425250775275) [Y0 X1 X10 Y11]
+ (0.007731425250775275) [X0 Y1 Y10 X11]
+ (0.008032520918821416) [Y0 Z1 Y2 Z6]
+ (0.008032520918821416) [X0 Z1 X2 Z6]
+ (0.008032520918821416) [Y1 Z2 Y3 Z7]
+ (0.008032520918821416) [X1 Z2 X3 Z7]
+ (0.009560705729135902) [Y8 X9 X10 Y11]
+ (0.009560705729135902) [X8 Y9 Y10 X11]
+ (0.011055020596132116) [Y0 Z1 Y2 Z8]
+ (0.011055020596132116) [X0 Z1 X2 Z8]
+ (0.011055020596132116) [Y1 Z2 Y3 Z9]
+ (0.011055020596132116) [X1 Z2 X3 Z9]
+ (0.011285190200840931) [Y5 Y6 X11 X12]
+ (0.011285190200840931) [X5 X6 Y11 Y12]
+ (0.011307274008848142) [Y7 Z8 Z9 Y11]
+ (0.011307274008848142) [X7 Z8 Z9 X11]
+ (0.01198238901024794) [Y4 X5 X6 Y7]
+ (0.01198238901024794) [X4 Y5 Y6 X7]
+ (0.013873381748426061) [Y6 X7 X8 Y9]
+ (0.013873381748426061) [X6 Y7 Y8 X9]
+ (0.014583648907612727) [Y0 X1 X2 Y3]
+ (0.014583648907612727) [X0 Y1 Y2 X3]
+ (0.015577208063976432) [Y2 X3 X12 Y13]
+ (0.015577208063976432) [X2 Y3 Y12 X13]
+ (0.01736611899465137) [Y6 X7 X12 Y13]
+ (0.01736611899465137) [X6 Y7 Y12 X13]
+ (0.01768006795248149) [Y4 X5 X10 Y11]
+ (0.01768006795248149) [X4 Y5 Y10 X11]
+ (0.01782514099578653) [Y6 X7 X10 Y11]
+ (0.01782514099578653) [X6 Y7 Y10 X11]
+ (0.019028242443847227) [Y3 X4 X11 Y12]
+ (0.019028242443847227) [X3 Y4 Y11 X12]
+ (0.02538465750845737) [Y2 X3 X10 Y11]
+ (0.02538465750845737) [X2 Y3 Y10 X11]
+ (0.028685183716106042) [Y10 X11 X12 Y13]
+ (0.028685183716106042) [X10 Y11 Y12 X13]
+ (0.029812424517345847) [Y6 Z7 Z8 Y10]
+ (0.029812424517345847) [X6 Z7 Z8 X10]
+ (0.029812424517345847) [Y7 Z9 Z10 Y11]
+ (0.029812424517345847) [X7 Z9 Z10 X11]
+ (0.03010462314345687) [Y6 Z7 Z9 Y10]
+ (0.03010462314345687) [X6 Z7 Z9 X10]
+ (0.03010462314345687) [Y7 Z8 Z10 Y11]
+ (0.03010462314345687) [X7 Z8 Z10 X11]
+ (0.030787505389143967) [Y6 Z8 Z9 Y10]
+ (0.030787505389143967) [X6 Z8 Z9 X10]
+ (0.031143817988967145) [Y2 X3 X6 Y7]
+ (0.031143817988967145) [X2 Y3 Y6 X7]
+ (0.03583956795335342) [Y2 X3 X4 Y5]
+ (0.03583956795335342) [X2 Y3 Y4 X5]
+ (0.0361941235590427) [Y2 X3 X8 Y9]
+ (0.0361941235590427) [X2 Y3 Y8 X9]
+ (0.10433064780651423) [Z0 Y1 Z2 Y3]
+ (0.10433064780651423) [Z0 X1 Z2 X3]
+ (-0.12133276911042258) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133276911042258) [X3 Z4 Z5 Z6 X7]
+ (-0.12133276911042254) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042254) [X2 Z3 Z4 Z5 X6]
+ (3.2020768792985595e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.2020768792985595e-06) [X0 Z1 Z2 Z3 X4]
+ (3.202076879298561e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.202076879298561e-06) [X1 Z2 Z3 Z4 X5]
+ (0.2284810656491885) [Y6 Z7 Z8 Z9 Y10]
+ (0.2284810656491885) [X6 Z7 Z8 Z9 X10]
+ (0.22848106564918855) [Y7 Z8 Z9 Z10 Y11]
+ (0.22848106564918855) [X7 Z8 Z9 Z10 X11]
+ (-0.03276765782329036) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.03276765782329036) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.03276765782329036) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.03276765782329036) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273017) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.027115036845273017) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273017) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.027115036845273017) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.025996177598021062) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.025996177598021062) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.017561202409646086) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.017561202409646086) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.017561202409646086) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.017561202409646086) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.014564531231173036) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564531231173036) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564531231173036) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564531231173036) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.012215040997613967) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.012215040997613967) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.012215040997613967) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.012215040997613967) [X4 Z5 X6 X11 Z12 X13]
+ (-0.012215040997613967) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.012215040997613967) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.012215040997613967) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.012215040997613967) [X5 Z6 X7 X10 Z11 X12]
+ (-0.011756013419819215) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.011756013419819215) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.011756013419819215) [X3 Z4 Z5 X6 X8 X9]
+ (-0.011756013419819215) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.008764827575688725) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.008764827575688725) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.008764827575688725) [X2 Z3 Z4 X5 X11 X12]
+ (-0.008764827575688725) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.008764827575688725) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.008764827575688725) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.008764827575688725) [X3 X4 X10 Z11 Z12 X13]
+ (-0.008764827575688725) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.00812525192138104) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.00812525192138104) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.007306759928832936) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306759928832936) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306759928832936) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306759928832936) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005805188989826872) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.005805188989826872) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.005805188989826872) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.005805188989826872) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.0056526209780173396) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.0056526209780173396) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.0056526209780173396) [X0 X1 X3 Z4 Z5 X6]
+ (-0.0056526209780173396) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.005143391768825075) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.005143391768825075) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.005143391768825075) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.005143391768825075) [X2 Z3 X4 X5 Z6 X7]
+ (-0.004684903388155212) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.004684903388155212) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668620318776301) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668620318776301) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.004575007626639198) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.004575007626639198) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424855449441866) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424855449441866) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.004158797381840037) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.004158797381840037) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.004158797381840037) [X3 Z4 Z5 X6 X12 X13]
+ (-0.004158797381840037) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.0034937903598900462) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.0034937903598900462) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.0034937903598900462) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.0034937903598900462) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.002779026799025519) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.002779026799025519) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.002293956611352472) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.002293956611352472) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.0017992194936630112) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.0017992194936630112) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278753941369724) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.0017278753941369724) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298507967730374) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298507967730374) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298507967730374) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298507967730374) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533856254125476) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533856254125476) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008145313270956122) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008145313270956122) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008145313270956122) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008145313270956122) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.735036880589358e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735036880589358e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735036880589358e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735036880589358e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.774817864512542e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.774817864512542e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.774817864512542e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.774817864512542e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.518362215637118e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.518362215637118e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.518362215637118e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.518362215637118e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.444344675870285e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.444344675870285e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.444344675870285e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.444344675870285e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.524373848481896e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.524373848481896e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.524373848481896e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.524373848481896e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.290028433154885e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.290028433154885e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.290028433154885e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.290028433154885e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.974311713364221e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.974311713364221e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.275883122063987e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.275883122063987e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.64305106842963e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.64305106842963e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.5565692180959244e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.5565692180959244e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.253224225531343e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.253224225531343e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.769659451942943e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.769659451942943e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.6945132944717873e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.6945132944717873e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.6102971305242485e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.6102971305242485e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.6102971305242485e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.6102971305242485e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.3131455000898384e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.3131455000898384e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.2774831954688214e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.2774831954688214e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.2774831954688214e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.2774831954688214e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.2112283483920583e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.2112283483920583e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.2112283483920583e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.2112283483920583e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.1513463111611927e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.1513463111611927e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.0882507112855786e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.0882507112855786e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.172669101426574e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.172669101426574e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.4548424490306174e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.4548424490306174e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.3304731886422563e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.3304731886422563e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2283337824822333e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2283337824822333e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.0358477601479918e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.0358477601479918e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956895371954227e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956895371954227e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733197742387853e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733197742387853e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733197742387853e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733197742387853e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.62861420147953e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.62861420147953e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.556281914566801e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.556281914566801e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.556281914566801e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.556281914566801e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.418291574600421e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.418291574600421e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.418291574600421e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.418291574600421e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.927453082832033e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927453082832033e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927453082832033e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927453082832033e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627851911298655e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627851911298655e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660624724331e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660624724331e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660624724331e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660624724331e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660624724331e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660624724331e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660624724331e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660624724331e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.838052750756037e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.838052750756037e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.5707613287775506e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.5707613287775506e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.3281393505542736e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.3281393505542736e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.086826565059291e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.086826565059291e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.086826565059291e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.086826565059291e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.447323128710571e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.447323128710571e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.3713289477466353e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.3713289477466353e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.3713289477466353e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.3713289477466353e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.199051618576094e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.199051618576094e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.9332412770073638e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.9332412770073638e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.9332412770073638e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.9332412770073638e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.839420915396627e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.839420915396627e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.839420915396627e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.839420915396627e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.5510539176794815e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.5510539176794815e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.5510539176794815e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.5510539176794815e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.3807781480387957e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.3807781480387957e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.3807781480387957e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.3807781480387957e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.3807781480387957e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.3807781480387957e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.3807781480387957e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.3807781480387957e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.3807781480387957e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.3807781480387957e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.3807781480387957e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.3807781480387957e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.291969486193612e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.291969486193612e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.1076325598857714e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.1076325598857714e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.1076325598857714e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.1076325598857714e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.1076325598857714e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.1076325598857714e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.1076325598857714e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.1076325598857714e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.057446595558198e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057446595558198e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057446595558198e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057446595558198e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649310133453136e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649310133453136e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649310133453136e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649310133453136e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.8394209153966272e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.8394209153966272e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.8394209153966272e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.8394209153966272e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.199051618576094e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.199051618576094e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.447323128710571e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.447323128710571e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.236259961091949e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.236259961091949e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.236259961091949e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.236259961091949e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.3281393505542736e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.3281393505542736e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.5707613287775506e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.5707613287775506e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.838052750756037e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.838052750756037e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.627851911298655e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627851911298655e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.62861420147953e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.62861420147953e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956895371954227e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956895371954227e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.30653665171022e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.30653665171022e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.30653665171022e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.30653665171022e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.0358477601479918e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.0358477601479918e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2283337824822333e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2283337824822333e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.239336321676951e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.239336321676951e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.239336321676951e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.239336321676951e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.3304731886422563e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.3304731886422563e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.4548424490306174e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.4548424490306174e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.172669101426574e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.172669101426574e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.0882507112855786e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.0882507112855786e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.1174479458297536e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.1174479458297536e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.1513463111611927e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.1513463111611927e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.3131455000898384e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.3131455000898384e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.3343312893863545e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.3343312893863545e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.6945132944717873e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.6945132944717873e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.183932559378109e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.183932559378109e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.5565692180959244e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.5565692180959244e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.64305106842963e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.64305106842963e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.275883122063987e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.275883122063987e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.974311713364221e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.974311713364221e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.0002921986261110214) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.0002921986261110214) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.0002921986261110214) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.0002921986261110214) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004956762314916422) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004956762314916422) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650070219499899) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650070219499899) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650070219499899) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650070219499899) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533856254125476) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533856254125476) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.0016095313817213802) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.0016095313817213802) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.0016095313817213802) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.0016095313817213802) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0016676041811440672) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.0016676041811440672) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.0016676041811440672) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.0016676041811440672) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.0017278753941369724) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.0017278753941369724) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.0017992194936630112) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.0017992194936630112) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.002293956611352472) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.002293956611352472) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.002462917007133928) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.002462917007133928) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.002462917007133928) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.002462917007133928) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.003961560792496539) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.003961560792496539) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.003961560792496539) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.003961560792496539) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.004424855449441866) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424855449441866) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004575007626639198) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.004575007626639198) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668620318776301) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668620318776301) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.004684903388155212) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.004684903388155212) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.0053248352342217045) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.0053248352342217045) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.0053248352342217045) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.0053248352342217045) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.005368659358109618) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368659358109618) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368659358109618) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368659358109618) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.00796088072592158) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.00796088072592158) [X4 Z5 X6 X10 Z11 X12]
+ (0.00796088072592158) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.00796088072592158) [X5 Z6 X7 X11 Z12 X13]
+ (0.00812525192138104) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.00812525192138104) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.008890731522694616) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.008890731522694616) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.008890731522694616) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.008890731522694616) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.010263414868158502) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.010263414868158502) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.010263414868158502) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.010263414868158502) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.010540425907671507) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.010540425907671507) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.010540425907671507) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.010540425907671507) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010960074940542663) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010960074940542663) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010960074940542663) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010960074940542663) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.01130727400884814) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.01130727400884814) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.014411099430130957) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411099430130957) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411099430130957) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411099430130957) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.01522563075722657) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.01522563075722657) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.01522563075722657) [X3 Z4 Z5 X6 X10 X11]
+ (0.01522563075722657) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.015588250102380206) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.015588250102380206) [X2 Z3 X4 X10 Z11 X12]
+ (0.015588250102380206) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.015588250102380206) [X3 Z4 X5 X11 Z12 X13]
+ (0.0182668348693756) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.0182668348693756) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.0182668348693756) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.0182668348693756) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.01902042317304) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.01902042317304) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.01902042317304) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.01902042317304) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.020175921723535547) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.020175921723535547) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.020175921723535547) [X4 Z5 Z6 X7 X11 X12]
+ (0.020175921723535547) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.020175921723535547) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.020175921723535547) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.020175921723535547) [X5 X6 X10 Z11 Z12 X13]
+ (0.020175921723535547) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.02435307767806893) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.02435307767806893) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.02435307767806893) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.02435307767806893) [X2 Z3 X4 X11 Z12 X13]
+ (0.02435307767806893) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.02435307767806893) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.02435307767806893) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.02435307767806893) [X3 Z4 X5 X10 Z11 X12]
+ (0.024389082531149617) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.024389082531149617) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.024389082531149617) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.024389082531149617) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.02510495713884454) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.02510495713884454) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.02510495713884454) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.02510495713884454) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.030787505389143967) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.030787505389143967) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.045879470781297865) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.045879470781297865) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.056007330877807744) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.056007330877807744) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.056007330877807744) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.056007330877807744) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661363) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.05608468124661363) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661363) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.05608468124661363) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.631277928399164e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631277928399164e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.63127792839916e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.63127792839916e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.595086006932464e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.595086006932464e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.5950860069324632e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.5950860069324632e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.04274327701378336) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.04274327701378336) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (0.04274327701378336) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04274327701378336) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.04764261217638306) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.04764261217638306) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.04764261217638306) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.04764261217638306) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.04171881383982172) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.04171881383982172) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.04171881383982172) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.04171881383982172) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.03956441632289326) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.03956441632289326) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.03956441632289326) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03956441632289326) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.039359168022053026) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.039359168022053026) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.039359168022053026) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.039359168022053026) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.03560837898831261) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.03560837898831261) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.029903789512624786) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.029903789512624786) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.029903789512624786) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.029903789512624786) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.028730779551905505) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.028730779551905505) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.028730779551905505) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.028730779551905505) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.025637238296026793) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.025637238296026793) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.025637238296026793) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.025637238296026793) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.024755463292890946) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.024755463292890946) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.024755463292890946) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.024755463292890946) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.024282117354692934) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.024282117354692934) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.023145130929528905) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.023145130929528905) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.022528440196013064) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.022528440196013064) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.021433810721600867) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.021433810721600867) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.021433810721600867) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.021433810721600867) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.01925750509525156) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.01925750509525156) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019028242443847227) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.019028242443847227) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.018889030304942933) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.018889030304942933) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.018889030304942933) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.018889030304942933) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.016024603689179594) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.016024603689179594) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.01522563075722657) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.01522563075722657) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.014603704729162075) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.014603704729162075) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564531231173036) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564531231173036) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.011756013419819215) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.011756013419819215) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.011285190200840931) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.011285190200840931) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.009841749246962574) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009841749246962574) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.009612634606847196) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612634606847196) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612634606847196) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612634606847196) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.008469978791023923) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.008469978791023923) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.007306759928832937) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.007306759928832937) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0059237983365613405) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.0059237983365613405) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.0056526209780173396) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.0056526209780173396) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368659358109617) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368659358109617) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.004158797381840037) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.004158797381840037) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.003356670563832875) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.003356670563832875) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.003356670563832875) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.003356670563832875) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.0032675138544235377) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.0032675138544235377) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.0032675138544235377) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.0032675138544235377) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.002779026799025519) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.002779026799025519) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.002686040977806617) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.002686040977806617) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.002686040977806617) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.002686040977806617) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.002293956611352472) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.002293956611352472) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.002293956611352472) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.002293956611352472) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.000958165583669645) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.000958165583669645) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.000958165583669645) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.000958165583669645) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.000958165583669645) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.000958165583669645) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.000958165583669645) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.000958165583669645) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.00024636437569577267) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00024636437569577267) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303551217) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.00013840177303551217) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.00013840177303551217) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.00013840177303551217) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.735036880589358e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735036880589358e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585305534157e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585305534157e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.6103585305534157e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.6103585305534157e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808795138987e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.5316808795138987e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808795138987e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5316808795138987e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.806102775090893e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.806102775090893e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.806102775090893e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.806102775090893e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.089799467430542e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.089799467430542e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.089799467430542e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.089799467430542e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.65220966939015e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.65220966939015e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.65220966939015e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.65220966939015e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851833848689e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.481851833848689e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851833848689e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.481851833848689e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.071480736423854e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.071480736423854e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.071480736423854e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.071480736423854e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.734622038667039e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.734622038667039e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.734622038667039e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.734622038667039e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.728843147166178e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.728843147166178e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.728843147166178e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.728843147166178e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.253224225531343e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.253224225531343e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.769659451942943e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.769659451942943e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.5443954292416442e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.5443954292416442e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.5443954292416442e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.5443954292416442e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.5443954292416442e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.5443954292416442e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.5443954292416442e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.5443954292416442e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.3609563202643644e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563202643644e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563202643644e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.3609563202643644e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.103215604681956e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.103215604681956e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.103215604681956e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.103215604681956e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.01112209823986e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.01112209823986e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.01112209823986e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.01112209823986e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.942946836610015e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.942946836610015e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.942946836610015e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.942946836610015e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.654117477064892e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.654117477064892e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.654117477064892e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.654117477064892e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.5224930676495055e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224930676495055e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224930676495055e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224930676495055e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224930676495055e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930676495055e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930676495055e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224930676495055e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.2283337824822333e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337824822333e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.2283337824822333e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337824822333e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.988770288287775e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988770288287775e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988770288287775e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988770288287775e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.867765103951695e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765103951695e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765103951695e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.867765103951695e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.18999097521837e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.18999097521837e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246207013363e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246207013363e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.471647744671347e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.471647744671347e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.561447179560658e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.561447179560658e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.561447179560658e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.561447179560658e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.5233896778787426e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.5233896778787426e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.4273231087271167e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.4273231087271167e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.4273231087271167e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.4273231087271167e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.3281393505542736e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.3281393505542736e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.3281393505542736e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.3281393505542736e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.0868265650592906e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.0868265650592906e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.888293595451228e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.888293595451228e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.888293595451228e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.888293595451228e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.3713289477466353e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.3713289477466353e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.839420915396627e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.839420915396627e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.057446595558196e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057446595558196e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.537178096983527e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.537178096983527e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.537178096983527e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.537178096983527e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.057446595558196e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057446595558196e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.209350644209568e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350644209568e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350644209568e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.209350644209568e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783554146182e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783554146182e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783554146182e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783554146182e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.839420915396627e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.839420915396627e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.3713289477466353e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.3713289477466353e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.0868265650592906e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.0868265650592906e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.5233896778787426e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.5233896778787426e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.471647744671347e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.471647744671347e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246207013363e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246207013363e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.18999097521837e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.18999097521837e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.3304731886422563e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.3304731886422563e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.3304731886422563e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.3304731886422563e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288532435116874e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288532435116874e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288532435116874e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288532435116874e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.6893489514728482e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.6893489514728482e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.6893489514728482e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.6893489514728482e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.7455184004128668e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.7455184004128668e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.7455184004128668e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.7455184004128668e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.7455184004128668e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.7455184004128668e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.7455184004128668e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.7455184004128668e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.2118420191223534e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420191223534e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.2118420191223534e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420191223534e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.2118420191223534e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420191223534e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.2118420191223534e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420191223534e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.3131455000898384e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.3131455000898384e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.3131455000898384e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.3131455000898384e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.3343312893863545e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.3343312893863545e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.183932559378109e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.183932559378109e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.735036880589358e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735036880589358e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00024636437569577267) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00024636437569577267) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0004458535128840928) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.0004458535128840928) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.0004458535128840928) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.0004458535128840928) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940221543005545) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005545) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005545) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005545) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005545) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005545) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940221543005545) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005545) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533856254125475) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533856254125475) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533856254125475) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533856254125475) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0010435246534907677) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435246534907677) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435246534907677) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435246534907677) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.0012803060973496797) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.0012803060973496797) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.0012803060973496797) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.0012803060973496797) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.0013038004788126982) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.0013038004788126982) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.0013038004788126982) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.0013038004788126982) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.002261966062482343) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.002261966062482343) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.002261966062482343) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.002261966062482343) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.002261966062482343) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.002261966062482343) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.002261966062482343) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.002261966062482343) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.004158797381840037) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.004158797381840037) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.004311038507914306) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.004311038507914306) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.004311038507914306) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.004311038507914306) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.004636976661182554) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.004636976661182554) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.004636976661182554) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.004636976661182554) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
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
+ (0.005262642473076855) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262642473076855) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262642473076855) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262642473076855) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368659358109617) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368659358109617) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005379937155839369) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.005379937155839369) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.005379937155839369) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.005379937155839369) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.0056526209780173396) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.0056526209780173396) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.005708495985960949) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708495985960949) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708495985960949) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708495985960949) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.0059237983365613405) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.0059237983365613405) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.007306759928832937) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.007306759928832937) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469978791023923) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.008469978791023923) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.009841749246962574) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.009841749246962574) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.011285190200840931) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.011285190200840931) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.011756013419819215) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.011756013419819215) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.014564531231173036) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564531231173036) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603704729162075) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.014603704729162075) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.01522563075722657) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.01522563075722657) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.016024603689179594) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.016024603689179594) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.019028242443847227) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.019028242443847227) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.01925750509525156) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.01925750509525156) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.045879470781297865) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.045879470781297865) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.3693708936615605) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.3693708936615605) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.3693708936615605) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.3693708936615605) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.2816425776702279) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.2816425776702279) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.2816425776702277) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.2816425776702277) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0906514420703647) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0906514420703647) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0906514420703647) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0906514420703647) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863617) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684737589863617) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863617) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684737589863617) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635021950635001) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635021950635001) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635021950635001) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635021950635001) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214016) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.06752385099214016) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214016) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.06752385099214016) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03560837898831261) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.03560837898831261) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.034903343373661855) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.034903343373661855) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.034903343373661855) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.034903343373661855) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883830006) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.024591860883830006) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883830006) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.024591860883830006) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.024282117354692934) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.024282117354692934) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.023145130929528905) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.023145130929528905) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.022528440196013064) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.022528440196013064) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019538050311314663) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.019538050311314663) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.019538050311314663) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.019538050311314663) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.01709155315589878) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.01709155315589878) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.01709155315589878) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.01709155315589878) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.016024603689179594) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.016024603689179594) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.016024603689179594) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.016024603689179594) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.01031148248983185) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.01031148248983185) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.01031148248983185) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.01031148248983185) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.009841749246962576) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962576) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.009841749246962576) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962576) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209858) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209858) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209858) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008826368514209858) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008541996625454813) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454813) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454813) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454813) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454813) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454813) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454813) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008541996625454813) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008469978791023923) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023923) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.008469978791023923) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023923) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.004668620318776301) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668620318776301) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.0038764708993369334) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.0038764708993369334) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.003804066171728542) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728542) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728542) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003804066171728542) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178878) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178878) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0033566705638328753) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.0033566705638328753) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.0032675138544235377) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.0032675138544235377) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0021413612231015724) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.0021413612231015724) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.0017278753941369724) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.0017278753941369724) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.0016407548553124245) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0016407548553124245) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.001452884321416886) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.001452884321416886) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.001452884321416886) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.001452884321416886) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870896771024436) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007870896771024436) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192743499487743) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192743499487743) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.00019400857029756353) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00019400857029756353) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303551217) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.00013840177303551217) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.141625221153441e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.141625221153441e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.141625221153441e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.141625221153441e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.071480736423854e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.071480736423854e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.1513463111611927e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.1513463111611927e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0882507112855786e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.0882507112855786e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.9885117062164124e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.9885117062164124e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8742990713396436e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.8742990713396436e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.3609563202643644e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.3609563202643644e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.300294656183618e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.300294656183618e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.146837650736712e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.146837650736712e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.146837650736712e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.146837650736712e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.352332102982809e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.352332102982809e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.352332102982809e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.352332102982809e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091637198976617e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637198976617e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637198976617e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637198976617e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637198976617e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637198976617e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637198976617e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091637198976617e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.074305985804881e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.074305985804881e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.074305985804881e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.074305985804881e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128986269246e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128986269246e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128986269246e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128986269246e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867765103951696e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.867765103951696e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560692464795558e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692464795558e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692464795558e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692464795558e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692464795558e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692464795558e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692464795558e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560692464795558e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.997018422100795e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.997018422100795e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.997018422100795e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.997018422100795e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.997018422100795e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.997018422100795e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.997018422100795e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.997018422100795e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.5682475210978735e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.5682475210978735e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.5682475210978735e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.5682475210978735e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376739308390504e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.376739308390504e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376739308390504e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.376739308390504e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376739308390504e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.376739308390504e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.376739308390504e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.376739308390504e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.888293595451228e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.888293595451228e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6863815455032067e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.6863815455032067e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035783554146187e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7035783554146187e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.209350644209565e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.209350644209565e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773243868803e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773243868803e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773243868803e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773243868803e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773243868803e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773243868803e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773243868803e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773243868803e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253793461684e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253793461684e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9742253793461684e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9742253793461684e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0474716554535194e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.0474716554535194e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.0474716554535194e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.0474716554535194e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.209350644209565e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.209350644209565e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0717282183208499e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0717282183208499e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0717282183208499e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0717282183208499e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.2004287493913429e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.2004287493913429e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.2004287493913429e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.2004287493913429e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.7035783554146187e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7035783554146187e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.3120943051833722e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.3120943051833722e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.3120943051833722e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.3120943051833722e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.6863815455032067e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815455032067e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.888293595451228e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.888293595451228e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.09225061606616e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.09225061606616e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.09225061606616e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.09225061606616e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.09225061606616e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.09225061606616e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.09225061606616e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.09225061606616e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.444597854006598e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.444597854006598e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.444597854006598e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.444597854006598e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.684915095062091e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.684915095062091e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.684915095062091e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.684915095062091e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246974425334214e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246974425334214e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246974425334214e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246974425334214e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246974425334214e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246974425334214e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246974425334214e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246974425334214e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.867765103951696e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.867765103951696e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.300294656183618e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.300294656183618e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.3609563202643644e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.3609563202643644e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.8742990713396436e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.8742990713396436e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.883676575993465e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.883676575993465e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.9473560116620843e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9473560116620843e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.9473560116620843e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9473560116620843e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117062164124e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9885117062164124e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.0882507112855786e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.0882507112855786e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.1513463111611927e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.1513463111611927e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.846201671226162e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846201671226162e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846201671226162e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846201671226162e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.071480736423854e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.071480736423854e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.10552672188612e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.10552672188612e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.10552672188612e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.10552672188612e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146496327409781e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146496327409781e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146496327409781e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146496327409781e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159350501813089e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159350501813089e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159350501813089e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159350501813089e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427988656363409e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427988656363409e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427988656363409e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427988656363409e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935867717878497e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935867717878497e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935867717878497e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935867717878497e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.2532733479276295e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.2532733479276295e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979825793225765e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979825793225765e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979825793225765e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979825793225765e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.205548411217921e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.205548411217921e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.205548411217921e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.205548411217921e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00013840177303551217) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.00013840177303551217) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.00018787053389553886) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.00018787053389553886) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.00018787053389553886) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00018787053389553886) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00019400857029756353) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00019400857029756353) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00024636437569577267) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569577267) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00024636437569577267) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569577267) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192743499487743) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192743499487743) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156734248909094) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156734248909094) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156734248909094) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156734248909094) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024436) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007870896771024436) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.0015324835230730665) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.0015324835230730665) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.0015324835230730665) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.0015324835230730665) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.0016407548553124245) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0016407548553124245) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0017278753941369724) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.0017278753941369724) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.0024464971554158843) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.0024464971554158843) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.0024464971554158843) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.0024464971554158843) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.0032675138544235377) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.0032675138544235377) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.0033566705638328753) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.0033566705638328753) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.0034841573002178878) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0034841573002178878) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0038764708993369334) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.0038764708993369334) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.004668620318776301) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668620318776301) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.0047672721882781165) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.0047672721882781165) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.0047672721882781165) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.0047672721882781165) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.0052865465382268915) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.0052865465382268915) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.0052865465382268915) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.0052865465382268915) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.00540895442241) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.00540895442241) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.00540895442241) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.00540895442241) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.0059237983365613405) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.0059237983365613405) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.0059237983365613405) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.0059237983365613405) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.010715508469796749) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010715508469796749) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010715508469796749) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010715508469796749) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.010757563953908925) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010757563953908925) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010757563953908925) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010757563953908925) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.014603704729162075) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.014603704729162075) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.014603704729162075) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.014603704729162075) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.019299560579363738) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363738) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363738) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363738) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363738) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363738) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.019299560579363738) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363738) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.05859198873386187) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.05859198873386187) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.775950527095929e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.775950527095929e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.775950527095936e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.775950527095936e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.07165035181002805) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07165035181002805) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.07165035181002807) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.07165035181002807) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.01925750509525156) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.01925750509525156) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.01031148248983185) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.01031148248983185) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008826368514209858) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209858) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.007597464029770627) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.007597464029770627) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.007597464029770627) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.007597464029770627) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311877) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311877) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311877) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311877) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311877) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311877) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311877) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311877) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676638) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005348051582676638) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005348051582676638) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676638) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.003804066171728542) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003804066171728542) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219217) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.0029841661681219217) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.0029841661681219217) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.0029841661681219217) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.0024464971554158848) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.0024464971554158848) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.0022494124470939882) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0022494124470939882) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022494124470939882) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0022494124470939882) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0021413612231015724) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.0021413612231015724) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.00186389428245875) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00186389428245875) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00186389428245875) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00186389428245875) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00186389428245875) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00186389428245875) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.00186389428245875) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00186389428245875) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0016407548553124245) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124245) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0016407548553124245) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124245) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.001222337808153829) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.001222337808153829) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.001222337808153829) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.001222337808153829) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.001222337808153829) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.001222337808153829) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.001222337808153829) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.001222337808153829) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0010283292378562652) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0010283292378562652) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0010283292378562652) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0010283292378562652) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1463061452901946e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1463061452901946e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.8742990713396436e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990713396436e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.8742990713396436e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990713396436e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.300294656183618e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.300294656183618e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.300294656183618e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.300294656183618e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.0444941297751917e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.0444941297751917e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.0444941297751917e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.0444941297751917e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.956079229722485e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.956079229722485e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.956079229722485e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.956079229722485e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105515036905538e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105515036905538e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105515036905538e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105515036905538e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661347212898048e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661347212898048e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661347212898048e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661347212898048e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341413606845e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341413606845e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.18999097521837e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.18999097521837e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.87662165805684e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.87662165805684e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.87662165805684e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.87662165805684e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246207013363e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246207013363e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.5233896778787426e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.5233896778787426e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.076732531621115e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.076732531621115e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.076732531621115e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.076732531621115e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.0134714588800663e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0134714588800663e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.9045998841450725e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.9045998841450725e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.9045998841450725e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.9045998841450725e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.666731754364091e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.666731754364091e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.666731754364091e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.666731754364091e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.850564192816946e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.850564192816946e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.656930931618818e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.656930931618818e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.656930931618818e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.656930931618818e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.850564192816946e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.850564192816946e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815455032067e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815455032067e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.6863815455032067e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815455032067e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714588800663e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.0134714588800663e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.5233896778787426e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.5233896778787426e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.670402390498885e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.670402390498885e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.670402390498885e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.670402390498885e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246207013363e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246207013363e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.18999097521837e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.18999097521837e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341413606845e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341413606845e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949476487380538e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949476487380538e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.7924939576788251e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939576788251e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939576788251e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.7924939576788251e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.883676575993465e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.883676575993465e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.9885117062164124e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117062164124e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117062164124e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117062164124e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273347927629e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253273347927629e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4017109734984578e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4017109734984578e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4017109734984578e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4017109734984578e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603692663402e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.5809603692663402e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603692663402e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.5809603692663402e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192743499487743) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487743) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192743499487743) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487743) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.0007870896771024436) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024436) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024436) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024436) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.001172634831644188) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.001172634831644188) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.001172634831644188) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.001172634831644188) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.001236647801924549) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.001236647801924549) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.001236647801924549) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.001236647801924549) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.0022009640695004533) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0022009640695004533) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0022009640695004533) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0022009640695004533) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798017) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798017) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798017) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798017) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798017) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798017) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002394972639798017) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798017) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0024464971554158848) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.0024464971554158848) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.003804066171728542) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.003804066171728542) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0038764708993369334) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.0038764708993369334) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.0038764708993369334) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.0038764708993369334) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.004220813970046471) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.004220813970046471) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.004220813970046471) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.004220813970046471) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.008826368514209858) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.008826368514209858) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.01031148248983185) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01031148248983185) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01925750509525156) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01925750509525156) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.05859198873386187) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.05859198873386187) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3987009015417888e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.3987009015417888e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.3987009015417888e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3987009015417888e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178878) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178878) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219217) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.0029841661681219217) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.00019400857029756353) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00019400857029756353) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061452901946e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061452901946e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7924939576788251e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7924939576788251e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540341413606845e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413606845e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540341413606845e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413606845e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.850564192816946e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.850564192816946e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.850564192816946e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.850564192816946e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714588800663e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0134714588800663e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714588800663e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0134714588800663e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949476487380538e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949476487380538e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.7924939576788251e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7924939576788251e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756353) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756353) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0029841661681219217) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.0029841661681219217) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.0034841573002178878) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0034841573002178878) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
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
   (-46.46390678868896) [I0]
+ (0.7829661725950202) [Z10]
+ (0.7829661725950203) [Z11]
+ (0.8084581961720481) [Z12]
+ (0.8084581961720481) [Z13]
+ (1.203440228914564) [Z4]
+ (1.2034402289145643) [Z5]
+ (1.3096862988615408) [Z6]
+ (1.3096862988615414) [Z7]
+ (1.3693525634718184) [Z8]
+ (1.3693525634718184) [Z9]
+ (1.6538942226831719) [Z3]
+ (1.653894222683172) [Z2]
+ (12.412630742111759) [Z0]
+ (12.412630742111759) [Z1]
+ (-8.194261372232783e-06) [Y10 Y12]
+ (-8.194261372232783e-06) [X10 X12]
+ (-1.854060857886043e-06) [Y5 Y7]
+ (-1.854060857886043e-06) [X5 X7]
+ (-7.764994118467231e-07) [Y3 Y5]
+ (-7.764994118467231e-07) [X3 X5]
+ (-5.929765815214663e-07) [Y4 Y6]
+ (-5.929765815214663e-07) [X4 X6]
+ (1.6021167405815614e-06) [Y2 Y4]
+ (1.6021167405815614e-06) [X2 X4]
+ (7.954413176233379e-06) [Y11 Y13]
+ (7.954413176233379e-06) [X11 X13]
+ (0.0032769719312317515) [Y1 Y3]
+ (0.0032769719312317515) [X1 X3]
+ (0.10433064780651416) [Y0 Y2]
+ (0.10433064780651416) [X0 X2]
+ (0.11270386920332219) [Z10 Z12]
+ (0.11270386920332219) [Z11 Z13]
+ (0.11383573679388657) [Z4 Z12]
+ (0.11383573679388657) [Z5 Z13]
+ (0.1195243896468268) [Z6 Z10]
+ (0.1195243896468268) [Z7 Z11]
+ (0.12489990917237612) [Z4 Z10]
+ (0.12489990917237612) [Z5 Z11]
+ (0.12495807739503222) [Z2 Z4]
+ (0.12495807739503222) [Z3 Z5]
+ (0.12799502492468415) [Z2 Z10]
+ (0.12799502492468415) [Z3 Z11]
+ (0.13401715261963693) [Z6 Z12]
+ (0.13401715261963693) [Z7 Z13]
+ (0.13701191674040755) [Z4 Z6]
+ (0.13701191674040755) [Z5 Z7]
+ (0.13734953064261338) [Z6 Z11]
+ (0.13734953064261338) [Z7 Z10]
+ (0.1373910476268322) [Z2 Z6]
+ (0.1373910476268322) [Z3 Z7]
+ (0.13766872645852585) [Z8 Z10]
+ (0.13766872645852585) [Z9 Z11]
+ (0.14011289865354803) [Z2 Z12]
+ (0.14011289865354803) [Z3 Z13]
+ (0.14138905291942805) [Z10 Z13]
+ (0.14138905291942805) [Z11 Z12]
+ (0.14257997712485762) [Z4 Z11]
+ (0.14257997712485762) [Z5 Z10]
+ (0.14722943218766177) [Z8 Z11]
+ (0.14722943218766177) [Z9 Z10]
+ (0.14899430575065556) [Z4 Z7]
+ (0.14899430575065556) [Z5 Z6]
+ (0.14926355147388903) [Z10 Z11]
+ (0.14960702684445296) [Z4 Z8]
+ (0.14960702684445296) [Z5 Z9]
+ (0.1497348680349692) [Z8 Z12]
+ (0.1497348680349692) [Z9 Z13]
+ (0.1507140812100829) [Z2 Z8]
+ (0.1507140812100829) [Z3 Z9]
+ (0.15138327161428838) [Z6 Z13]
+ (0.15138327161428838) [Z7 Z12]
+ (0.15215040708869043) [Z4 Z13]
+ (0.15215040708869043) [Z5 Z12]
+ (0.1533796824331415) [Z2 Z11]
+ (0.1533796824331415) [Z3 Z10]
+ (0.15435748657223622) [Z12 Z13]
+ (0.15569010671752448) [Z2 Z13]
+ (0.15569010671752448) [Z3 Z12]
+ (0.1567639617643099) [Z4 Z9]
+ (0.1567639617643099) [Z5 Z8]
+ (0.15755314797985673) [Z4 Z5]
+ (0.16079764534838564) [Z2 Z5]
+ (0.16079764534838564) [Z3 Z4]
+ (0.16756653265461258) [Z6 Z8]
+ (0.16756653265461258) [Z7 Z9]
+ (0.16853486561579942) [Z2 Z7]
+ (0.16853486561579942) [Z3 Z6]
+ (0.18143991440303864) [Z6 Z9]
+ (0.18143991440303864) [Z7 Z8]
+ (0.18189085790751364) [Z2 Z3]
+ (0.1869082047691255) [Z2 Z9]
+ (0.1869082047691255) [Z3 Z8]
+ (0.1929972393536422) [Z0 Z10]
+ (0.1929972393536422) [Z1 Z11]
+ (0.19392534613270176) [Z6 Z7]
+ (0.1966177089034212) [Z0 Z4]
+ (0.1966177089034212) [Z1 Z5]
+ (0.199363545373608) [Z0 Z5]
+ (0.199363545373608) [Z1 Z4]
+ (0.20072866460441752) [Z0 Z11]
+ (0.20072866460441752) [Z1 Z10]
+ (0.21102659849791472) [Z0 Z12]
+ (0.21102659849791472) [Z1 Z13]
+ (0.21631037498631767) [Z0 Z13]
+ (0.21631037498631767) [Z1 Z12]
+ (0.2200397733437608) [Z8 Z9]
+ (0.2367108078383039) [Z0 Z2]
+ (0.2367108078383039) [Z1 Z3]
+ (0.24164663936017142) [Z0 Z6]
+ (0.24164663936017142) [Z1 Z7]
+ (0.24853483371314194) [Z0 Z7]
+ (0.24853483371314194) [Z1 Z6]
+ (0.2512944567459166) [Z0 Z3]
+ (0.2512944567459166) [Z1 Z2]
+ (0.27232518306605635) [Z0 Z8]
+ (0.27232518306605635) [Z1 Z9]
+ (0.27883454426723353) [Z0 Z9]
+ (0.27883454426723353) [Z1 Z8]
+ (1.1861763734860458) [Z0 Z1]
+ (-1.2260484988421747e-05) [Y5 Z6 Y7]
+ (-1.2260484988421747e-05) [X5 Z6 X7]
+ (-1.2260484988421744e-05) [Y4 Z5 Y6]
+ (-1.2260484988421744e-05) [X4 Z5 X6]
+ (-1.0722312157405715e-05) [Y10 Z11 Y12]
+ (-1.0722312157405715e-05) [X10 Z11 X12]
+ (-1.0722312157405713e-05) [Y11 Z12 Y13]
+ (-1.0722312157405713e-05) [X11 Z12 X13]
+ (-3.887051672913066e-06) [Y3 Z4 Y5]
+ (-3.887051672913066e-06) [X3 Z4 X5]
+ (-3.887051672913063e-06) [Y2 Z3 Y4]
+ (-3.887051672913063e-06) [X2 Z3 X4]
+ (0.1250703257977228) [Y0 Z1 Y2]
+ (0.1250703257977228) [X0 Z1 X2]
+ (0.1250703257977229) [Y1 Z2 Y3]
+ (0.1250703257977229) [X1 Z2 X3]
+ (-0.03619412355904264) [Y2 Y3 X8 X9]
+ (-0.03619412355904264) [X2 X3 Y8 Y9]
+ (-0.035839567953353434) [Y2 Y3 X4 X5]
+ (-0.035839567953353434) [X2 X3 Y4 Y5]
+ (-0.03114381798896718) [Y2 Y3 X6 X7]
+ (-0.03114381798896718) [X2 X3 Y6 Y7]
+ (-0.028685183716105876) [Y10 Y11 X12 X13]
+ (-0.028685183716105876) [X10 X11 Y12 Y13]
+ (-0.02599617759802108) [Y3 Z4 Z5 Y7]
+ (-0.02599617759802108) [X3 Z4 Z5 X7]
+ (-0.025384657508457385) [Y2 Y3 X10 X11]
+ (-0.025384657508457385) [X2 X3 Y10 Y11]
+ (-0.01902824244384722) [Y3 Y4 X11 X12]
+ (-0.01902824244384722) [X3 X4 Y11 Y12]
+ (-0.017825140995786592) [Y6 Y7 X10 X11]
+ (-0.017825140995786592) [X6 X7 Y10 Y11]
+ (-0.017680067952481494) [Y4 Y5 X10 X11]
+ (-0.017680067952481494) [X4 X5 Y10 Y11]
+ (-0.017366118994651448) [Y6 Y7 X12 X13]
+ (-0.017366118994651448) [X6 X7 Y12 Y13]
+ (-0.015577208063976456) [Y2 Y3 X12 X13]
+ (-0.015577208063976456) [X2 X3 Y12 Y13]
+ (-0.01458364890761271) [Y0 Y1 X2 X3]
+ (-0.01458364890761271) [X0 X1 Y2 Y3]
+ (-0.013873381748426065) [Y6 Y7 X8 X9]
+ (-0.013873381748426065) [X6 X7 Y8 Y9]
+ (-0.011982389010248009) [Y4 Y5 X6 X7]
+ (-0.011982389010248009) [X4 X5 Y6 Y7]
+ (-0.011285190200840962) [Y5 X6 X11 Y12]
+ (-0.011285190200840962) [X5 Y6 Y11 X12]
+ (-0.009560705729135933) [Y8 Y9 X10 X11]
+ (-0.009560705729135933) [X8 X9 Y10 Y11]
+ (-0.008125251921381015) [Y1 X2 X8 Y9]
+ (-0.008125251921381015) [Y1 Y2 Y8 Y9]
+ (-0.008125251921381015) [X1 X2 X8 X9]
+ (-0.008125251921381015) [X1 Y2 Y8 X9]
+ (-0.007731425250775282) [Y0 Y1 X10 X11]
+ (-0.007731425250775282) [X0 X1 Y10 Y11]
+ (-0.007156934919856945) [Y4 Y5 X8 X9]
+ (-0.007156934919856945) [X4 X5 Y8 Y9]
+ (-0.00688819435297053) [Y0 Y1 X6 X7]
+ (-0.00688819435297053) [X0 X1 Y6 Y7]
+ (-0.00650936120117722) [Y0 Y1 X8 X9]
+ (-0.00650936120117722) [X0 X1 Y8 Y9]
+ (-0.006087822480561848) [Y8 Y9 X12 X13]
+ (-0.006087822480561848) [X8 X9 Y12 Y13]
+ (-0.005283776488402942) [Y0 Y1 X12 X13]
+ (-0.005283776488402942) [X0 X1 Y12 Y13]
+ (-0.0051433917688251726) [Y3 X4 X5 Y6]
+ (-0.0051433917688251726) [X3 Y4 Y5 X6]
+ (-0.004684903388155199) [Y1 X2 X6 Y7]
+ (-0.004684903388155199) [Y1 Y2 Y6 Y7]
+ (-0.004684903388155199) [X1 X2 X6 X7]
+ (-0.004684903388155199) [X1 Y2 Y6 X7]
+ (-0.0045750076266392) [Y1 X2 X12 Y13]
+ (-0.0045750076266392) [Y1 Y2 Y12 Y13]
+ (-0.0045750076266392) [X1 X2 X12 X13]
+ (-0.0045750076266392) [X1 Y2 Y12 X13]
+ (-0.00442485544944185) [Y1 X2 X4 Y5]
+ (-0.00442485544944185) [Y1 Y2 Y4 Y5]
+ (-0.00442485544944185) [X1 X2 X4 X5]
+ (-0.00442485544944185) [X1 Y2 Y4 X5]
+ (-0.003479511890334371) [Y2 Z3 Z5 Y6]
+ (-0.003479511890334371) [X2 Z3 Z5 X6]
+ (-0.003479511890334371) [Y3 Z4 Z6 Y7]
+ (-0.003479511890334371) [X3 Z4 Z6 X7]
+ (-0.002745836470186807) [Y0 Y1 X4 X5]
+ (-0.002745836470186807) [X0 X1 Y4 Y5]
+ (-0.0017992194936630118) [Y1 X2 X10 Y11]
+ (-0.0017992194936630118) [Y1 Y2 Y10 Y11]
+ (-0.0017992194936630118) [X1 X2 X10 X11]
+ (-0.0017992194936630118) [X1 Y2 Y10 X11]
+ (-0.00029219862611102525) [Y7 Y8 X9 X10]
+ (-0.00029219862611102525) [X7 X8 Y9 Y10]
+ (-8.194261372232783e-06) [Z10 Y11 Z12 Y13]
+ (-8.194261372232783e-06) [Z10 X11 Z12 X13]
+ (-7.801707500504788e-06) [Y2 Z3 Y4 Z11]
+ (-7.801707500504788e-06) [X2 Z3 X4 Z11]
+ (-7.801707500504788e-06) [Y3 Z4 Y5 Z10]
+ (-7.801707500504788e-06) [X3 Z4 X5 Z10]
+ (-4.6430510684694405e-06) [Y3 X4 X10 Y11]
+ (-4.6430510684694405e-06) [Y3 Y4 Y10 Y11]
+ (-4.6430510684694405e-06) [X3 X4 X10 X11]
+ (-4.6430510684694405e-06) [X3 Y4 Y10 X11]
+ (-4.58885515562827e-06) [Y4 Z5 Y6 Z13]
+ (-4.58885515562827e-06) [X4 Z5 X6 Z13]
+ (-4.58885515562827e-06) [Y5 Z6 Y7 Z12]
+ (-4.58885515562827e-06) [X5 Z6 X7 Z12]
+ (-4.556569218151564e-06) [Y5 X6 X12 Y13]
+ (-4.556569218151564e-06) [Y5 Y6 Y12 Y13]
+ (-4.556569218151564e-06) [X5 X6 X12 X13]
+ (-4.556569218151564e-06) [X5 Y6 Y12 X13]
+ (-3.6945132944473915e-06) [Y4 X5 X11 Y12]
+ (-3.6945132944473915e-06) [Y4 Y5 Y11 Y12]
+ (-3.6945132944473915e-06) [X4 X5 X11 X12]
+ (-3.6945132944473915e-06) [X4 Y5 Y11 X12]
+ (-3.344081556383889e-06) [Z0 Y5 Z6 Y7]
+ (-3.344081556383889e-06) [Z0 X5 Z6 X7]
+ (-3.344081556383889e-06) [Z1 Y4 Z5 Y6]
+ (-3.344081556383889e-06) [Z1 X4 Z5 X6]
+ (-3.158656432035346e-06) [Y2 Z3 Y4 Z10]
+ (-3.158656432035346e-06) [X2 Z3 X4 Z10]
+ (-3.158656432035346e-06) [Y3 Z4 Y5 Z11]
+ (-3.158656432035346e-06) [X3 Z4 X5 Z11]
+ (-3.099349243508838e-06) [Z0 Y4 Z5 Y6]
+ (-3.099349243508838e-06) [Z0 X4 Z5 X6]
+ (-3.099349243508838e-06) [Z1 Y5 Z6 Y7]
+ (-3.099349243508838e-06) [Z1 X5 Z6 X7]
+ (-2.890967881664903e-06) [Z6 Y11 Z12 Y13]
+ (-2.890967881664903e-06) [Z6 X11 Z12 X13]
+ (-2.890967881664903e-06) [Z7 Y10 Z11 Y12]
+ (-2.890967881664903e-06) [Z7 X10 Z11 X12]
+ (-2.1776646050156313e-06) [Z0 Y10 Z11 Y12]
+ (-2.1776646050156313e-06) [Z0 X10 Z11 X12]
+ (-2.1776646050156313e-06) [Z1 Y11 Z12 Y13]
+ (-2.1776646050156313e-06) [Z1 X11 Z12 X13]
+ (-1.8818501831424696e-06) [Y4 Z5 Y6 Z9]
+ (-1.8818501831424696e-06) [X4 Z5 X6 Z9]
+ (-1.8818501831424696e-06) [Y5 Z6 Y7 Z8]
+ (-1.8818501831424696e-06) [X5 Z6 X7 Z8]
+ (-1.8551201215178034e-06) [Z6 Y10 Z11 Y12]
+ (-1.8551201215178034e-06) [Z6 X10 Z11 X12]
+ (-1.8551201215178034e-06) [Z7 Y11 Z12 Y13]
+ (-1.8551201215178034e-06) [Z7 X11 Z12 X13]
+ (-1.8540608578860432e-06) [Y4 Z5 Y6 Z7]
+ (-1.8540608578860432e-06) [X4 Z5 X6 Z7]
+ (-1.8163031697177844e-06) [Z4 Y11 Z12 Y13]
+ (-1.8163031697177844e-06) [Z4 X11 Z12 X13]
+ (-1.8163031697177844e-06) [Z5 Y10 Z11 Y12]
+ (-1.8163031697177844e-06) [Z5 X10 Z11 X12]
+ (-1.6923978285250565e-06) [Y4 Z5 Y6 Z10]
+ (-1.6923978285250565e-06) [X4 Z5 X6 Z10]
+ (-1.6923978285250565e-06) [Y5 Z6 Y7 Z11]
+ (-1.6923978285250565e-06) [X5 Z6 X7 Z11]
+ (-1.6148794138744164e-06) [Z0 Y11 Z12 Y13]
+ (-1.6148794138744164e-06) [Z0 X11 Z12 X13]
+ (-1.6148794138744164e-06) [Z1 Y10 Z11 Y12]
+ (-1.6148794138744164e-06) [Z1 X10 Z11 X12]
+ (-1.597317197790185e-06) [Z8 Y10 Z11 Y12]
+ (-1.597317197790185e-06) [Z8 X10 Z11 X12]
+ (-1.597317197790185e-06) [Z9 Y11 Z12 Y13]
+ (-1.597317197790185e-06) [Z9 X11 Z12 X13]
+ (-1.4548424490479892e-06) [Y3 X4 X6 Y7]
+ (-1.4548424490479892e-06) [Y3 Y4 Y6 Y7]
+ (-1.4548424490479892e-06) [X3 X4 X6 X7]
+ (-1.4548424490479892e-06) [X3 Y4 Y6 X7]
+ (-1.398044908063807e-06) [Y4 Z5 Y6 Z8]
+ (-1.398044908063807e-06) [X4 Z5 X6 Z8]
+ (-1.398044908063807e-06) [Y5 Z6 Y7 Z9]
+ (-1.398044908063807e-06) [X5 Z6 X7 Z9]
+ (-1.1954890099506944e-06) [Y2 Z3 Y4 Z7]
+ (-1.1954890099506944e-06) [X2 Z3 X4 Z7]
+ (-1.1954890099506944e-06) [Y3 Z4 Y5 Z6]
+ (-1.1954890099506944e-06) [X3 Z4 X5 Z6]
+ (-1.190850808322702e-06) [Z0 Y3 Z4 Y5]
+ (-1.190850808322702e-06) [Z0 X3 Z4 X5]
+ (-1.190850808322702e-06) [Z1 Y2 Z3 Y4]
+ (-1.190850808322702e-06) [Z1 X2 Z3 X4]
+ (-1.170830136949946e-06) [Z2 Y5 Z6 Y7]
+ (-1.170830136949946e-06) [Z2 X5 Z6 X7]
+ (-1.170830136949946e-06) [Z3 Y4 Z5 Y6]
+ (-1.170830136949946e-06) [Z3 X4 Z5 X6]
+ (-1.0632283423573756e-06) [Z2 Y10 Z11 Y12]
+ (-1.0632283423573756e-06) [Z2 X10 Z11 X12]
+ (-1.0632283423573756e-06) [Z3 Y11 Z12 Y13]
+ (-1.0632283423573756e-06) [Z3 X11 Z12 X13]
+ (-1.035847760147099e-06) [Y6 X7 X11 Y12]
+ (-1.035847760147099e-06) [Y6 Y7 Y11 Y12]
+ (-1.035847760147099e-06) [X6 X7 X11 X12]
+ (-1.035847760147099e-06) [X6 Y7 Y11 X12]
+ (-9.509249751273522e-07) [Z2 Y4 Z5 Y6]
+ (-9.509249751273522e-07) [Z2 X4 Z5 X6]
+ (-9.509249751273522e-07) [Z3 Y5 Z6 Y7]
+ (-9.509249751273522e-07) [Z3 X5 Z6 X7]
+ (-9.344557776324781e-07) [Z8 Y11 Z12 Y13]
+ (-9.344557776324781e-07) [Z8 X11 Z12 X13]
+ (-9.344557776324781e-07) [Z9 Y10 Z11 Y12]
+ (-9.344557776324781e-07) [Z9 X10 Z11 X12]
+ (-8.337746754233125e-07) [Z0 Y2 Z3 Y4]
+ (-8.337746754233125e-07) [Z0 X2 Z3 X4]
+ (-8.337746754233125e-07) [Z1 Y3 Z4 Y5]
+ (-8.337746754233125e-07) [Z1 X3 Z4 X5]
+ (-7.956895372339377e-07) [Y3 X4 X8 Y9]
+ (-7.956895372339377e-07) [Y3 Y4 Y8 Y9]
+ (-7.956895372339377e-07) [X3 X4 X8 X9]
+ (-7.956895372339377e-07) [X3 Y4 Y8 X9]
+ (-7.764994118467231e-07) [Y2 Z3 Y4 Z5]
+ (-7.764994118467231e-07) [X2 Z3 X4 Z5]
+ (-5.929765815214663e-07) [Z4 Y5 Z6 Y7]
+ (-5.929765815214663e-07) [Z4 X5 Z6 X7]
+ (-5.770052994897747e-07) [Y2 Z3 Y4 Z9]
+ (-5.770052994897747e-07) [X2 Z3 X4 Z9]
+ (-5.770052994897747e-07) [Y3 Z4 Y5 Z8]
+ (-5.770052994897747e-07) [X3 Z4 X5 Z8]
+ (-5.471647744579615e-07) [Y1 Y2 X11 X12]
+ (-5.471647744579615e-07) [X1 X2 Y11 Y12]
+ (-4.838052750786624e-07) [Y5 X6 X8 Y9]
+ (-4.838052750786624e-07) [Y5 Y6 Y8 Y9]
+ (-4.838052750786624e-07) [X5 X6 X8 X9]
+ (-4.838052750786624e-07) [X5 Y6 Y8 X9]
+ (-3.570761328993895e-07) [Y0 X1 X3 Y4]
+ (-3.570761328993895e-07) [Y0 Y1 Y3 Y4]
+ (-3.570761328993895e-07) [X0 X1 X3 X4]
+ (-3.570761328993895e-07) [X0 Y1 Y3 X4]
+ (-2.447323128750515e-07) [Y0 X1 X5 Y6]
+ (-2.447323128750515e-07) [Y0 Y1 Y5 Y6]
+ (-2.447323128750515e-07) [X0 X1 X5 X6]
+ (-2.447323128750515e-07) [X0 Y1 Y5 X6]
+ (-2.1990516182259397e-07) [Y2 X3 X5 Y6]
+ (-2.1990516182259397e-07) [Y2 Y3 Y5 Y6]
+ (-2.1990516182259397e-07) [X2 X3 X5 X6]
+ (-2.1990516182259397e-07) [X2 Y3 Y5 X6]
+ (-1.9332412770585739e-07) [Y1 X2 X3 Y4]
+ (-1.9332412770585739e-07) [X1 Y2 Y3 X4]
+ (-1.2919694861825555e-07) [Y1 Z2 Z3 Y5]
+ (-1.2919694861825555e-07) [X1 Z2 Z3 X5]
+ (1.7379332623651184e-07) [Y0 Z1 Z3 Y4]
+ (1.7379332623651184e-07) [X0 Z1 Z3 X4]
+ (1.7379332623651184e-07) [Y1 Z2 Z4 Y5]
+ (1.7379332623651184e-07) [X1 Z2 Z4 X5]
+ (1.9332412770585739e-07) [Y1 Y2 X3 X4]
+ (1.9332412770585739e-07) [X1 X2 Y3 Y4]
+ (2.1868423774416295e-07) [Y2 Z3 Y4 Z8]
+ (2.1868423774416295e-07) [X2 Z3 X4 Z8]
+ (2.1868423774416295e-07) [Y3 Z4 Y5 Z9]
+ (2.1868423774416295e-07) [X3 Z4 X5 Z9]
+ (2.5935343909729453e-07) [Y2 Z3 Y4 Z6]
+ (2.5935343909729453e-07) [X2 Z3 X4 Z6]
+ (2.5935343909729453e-07) [Y3 Z4 Y5 Z7]
+ (2.5935343909729453e-07) [X3 Z4 X5 Z7]
+ (3.60607186785465e-07) [Y0 Z1 Z2 Y4]
+ (3.60607186785465e-07) [X0 Z1 Z2 X4]
+ (3.60607186785465e-07) [Y1 Z3 Z4 Y5]
+ (3.60607186785465e-07) [X1 Z3 Z4 X5]
+ (5.471647744579615e-07) [Y1 X2 X11 Y12]
+ (5.471647744579615e-07) [X1 Y2 Y11 X12]
+ (5.627851911412149e-07) [Y0 X1 X11 Y12]
+ (5.627851911412149e-07) [Y0 Y1 Y11 Y12]
+ (5.627851911412149e-07) [X0 X1 X11 X12]
+ (5.627851911412149e-07) [X0 Y1 Y11 X12]
+ (6.62861420157707e-07) [Y8 X9 X11 Y12]
+ (6.62861420157707e-07) [Y8 Y9 Y11 Y12]
+ (6.62861420157707e-07) [X8 X9 X11 X12]
+ (6.62861420157707e-07) [X8 Y9 Y11 X12]
+ (1.1094407591052695e-06) [Z2 Y11 Z12 Y13]
+ (1.1094407591052695e-06) [Z2 X11 Z12 X13]
+ (1.1094407591052695e-06) [Z3 Y10 Z11 Y12]
+ (1.1094407591052695e-06) [Z3 X10 Z11 X12]
+ (1.6021167405815614e-06) [Z2 Y3 Z4 Y5]
+ (1.6021167405815614e-06) [Z2 X3 Z4 X5]
+ (1.8782101247296075e-06) [Z4 Y10 Z11 Y12]
+ (1.8782101247296075e-06) [Z4 X10 Z11 X12]
+ (1.8782101247296075e-06) [Z5 Y11 Z12 Y13]
+ (1.8782101247296075e-06) [Z5 X11 Z12 X13]
+ (2.172669101462645e-06) [Y2 X3 X11 Y12]
+ (2.172669101462645e-06) [Y2 Y3 Y11 Y12]
+ (2.172669101462645e-06) [X2 X3 X11 X12]
+ (2.172669101462645e-06) [X2 Y3 Y11 X12]
+ (3.1174479459772335e-06) [Y0 Z2 Z3 Y4]
+ (3.1174479459772335e-06) [X0 Z2 Z3 X4]
+ (3.539054184513043e-06) [Y2 Z3 Y4 Z12]
+ (3.539054184513043e-06) [X2 Z3 X4 Z12]
+ (3.539054184513043e-06) [Y3 Z4 Y5 Z13]
+ (3.539054184513043e-06) [X3 Z4 X5 Z13]
+ (4.281913884880945e-06) [Y4 Z5 Y6 Z11]
+ (4.281913884880945e-06) [X4 Z5 X6 Z11]
+ (4.281913884880945e-06) [Y5 Z6 Y7 Z10]
+ (4.281913884880945e-06) [X5 Z6 X7 Z10]
+ (5.275883122075803e-06) [Y3 X4 X12 Y13]
+ (5.275883122075803e-06) [Y3 Y4 Y12 Y13]
+ (5.275883122075803e-06) [X3 X4 X12 X13]
+ (5.275883122075803e-06) [X3 Y4 Y12 X13]
+ (5.974311713406001e-06) [Y5 X6 X10 Y11]
+ (5.974311713406001e-06) [Y5 Y6 Y10 Y11]
+ (5.974311713406001e-06) [X5 X6 X10 X11]
+ (5.974311713406001e-06) [X5 Y6 Y10 X11]
+ (7.954413176233379e-06) [Y10 Z11 Y12 Z13]
+ (7.954413176233379e-06) [X10 Z11 X12 Z13]
+ (8.814937306588846e-06) [Y2 Z3 Y4 Z13]
+ (8.814937306588846e-06) [X2 Z3 X4 Z13]
+ (8.814937306588846e-06) [Y3 Z4 Y5 Z12]
+ (8.814937306588846e-06) [X3 Z4 X5 Z12]
+ (0.00029219862611102525) [Y7 X8 X9 Y10]
+ (0.00029219862611102525) [X7 Y8 Y9 X10]
+ (0.0004956762314915644) [Y2 Z4 Z5 Y6]
+ (0.0004956762314915644) [X2 Z4 Z5 X6]
+ (0.0011059037691897445) [Y0 Z1 Y2 Z5]
+ (0.0011059037691897445) [X0 Z1 X2 Z5]
+ (0.0011059037691897445) [Y1 Z2 Y3 Z4]
+ (0.0011059037691897445) [X1 Z2 X3 Z4]
+ (0.0016638798784908014) [Y2 Z3 Z4 Y6]
+ (0.0016638798784908014) [X2 Z3 Z4 X6]
+ (0.0016638798784908014) [Y3 Z5 Z6 Y7]
+ (0.0016638798784908014) [X3 Z5 Z6 X7]
+ (0.0017560707018413105) [Y0 Z1 Y2 Z11]
+ (0.0017560707018413105) [X0 Z1 X2 Z11]
+ (0.0017560707018413105) [Y1 Z2 Y3 Z10]
+ (0.0017560707018413105) [X1 Z2 X3 Z10]
+ (0.0023262306231581413) [Y0 Z1 Y2 Z13]
+ (0.0023262306231581413) [X0 Z1 X2 Z13]
+ (0.0023262306231581413) [Y1 Z2 Y3 Z12]
+ (0.0023262306231581413) [X1 Z2 X3 Z12]
+ (0.002745836470186807) [Y0 X1 X4 Y5]
+ (0.002745836470186807) [X0 Y1 Y4 X5]
+ (0.002929768674751137) [Y0 Z1 Y2 Z9]
+ (0.002929768674751137) [X0 Z1 X2 Z9]
+ (0.002929768674751137) [Y1 Z2 Y3 Z8]
+ (0.002929768674751137) [X1 Z2 X3 Z8]
+ (0.0032769719312317515) [Y0 Z1 Y2 Z3]
+ (0.0032769719312317515) [X0 Z1 X2 Z3]
+ (0.00334761753066624) [Y0 Z1 Y2 Z7]
+ (0.00334761753066624) [X0 Z1 X2 Z7]
+ (0.00334761753066624) [Y1 Z2 Y3 Z6]
+ (0.00334761753066624) [X1 Z2 X3 Z6]
+ (0.0035552901955043224) [Y0 Z1 Y2 Z10]
+ (0.0035552901955043224) [X0 Z1 X2 Z10]
+ (0.0035552901955043224) [Y1 Z2 Y3 Z11]
+ (0.0035552901955043224) [X1 Z2 X3 Z11]
+ (0.0051433917688251726) [Y3 Y4 X5 X6]
+ (0.0051433917688251726) [X3 X4 Y5 Y6]
+ (0.005283776488402942) [Y0 X1 X12 Y13]
+ (0.005283776488402942) [X0 Y1 Y12 X13]
+ (0.005530759218631595) [Y0 Z1 Y2 Z4]
+ (0.005530759218631595) [X0 Z1 X2 Z4]
+ (0.005530759218631595) [Y1 Z2 Y3 Z5]
+ (0.005530759218631595) [X1 Z2 X3 Z5]
+ (0.006087822480561848) [Y8 X9 X12 Y13]
+ (0.006087822480561848) [X8 Y9 Y12 X13]
+ (0.00650936120117722) [Y0 X1 X8 Y9]
+ (0.00650936120117722) [X0 Y1 Y8 X9]
+ (0.00688819435297053) [Y0 X1 X6 Y7]
+ (0.00688819435297053) [X0 Y1 Y6 X7]
+ (0.0069012382497973404) [Y0 Z1 Y2 Z12]
+ (0.0069012382497973404) [X0 Z1 X2 Z12]
+ (0.0069012382497973404) [Y1 Z2 Y3 Z13]
+ (0.0069012382497973404) [X1 Z2 X3 Z13]
+ (0.007156934919856945) [Y4 X5 X8 Y9]
+ (0.007156934919856945) [X4 Y5 Y8 X9]
+ (0.007731425250775282) [Y0 X1 X10 Y11]
+ (0.007731425250775282) [X0 Y1 Y10 X11]
+ (0.008032520918821439) [Y0 Z1 Y2 Z6]
+ (0.008032520918821439) [X0 Z1 X2 Z6]
+ (0.008032520918821439) [Y1 Z2 Y3 Z7]
+ (0.008032520918821439) [X1 Z2 X3 Z7]
+ (0.009560705729135933) [Y8 X9 X10 Y11]
+ (0.009560705729135933) [X8 Y9 Y10 X11]
+ (0.011055020596132153) [Y0 Z1 Y2 Z8]
+ (0.011055020596132153) [X0 Z1 X2 Z8]
+ (0.011055020596132153) [Y1 Z2 Y3 Z9]
+ (0.011055020596132153) [X1 Z2 X3 Z9]
+ (0.011285190200840962) [Y5 Y6 X11 X12]
+ (0.011285190200840962) [X5 X6 Y11 Y12]
+ (0.011307274008848227) [Y7 Z8 Z9 Y11]
+ (0.011307274008848227) [X7 Z8 Z9 X11]
+ (0.011982389010248009) [Y4 X5 X6 Y7]
+ (0.011982389010248009) [X4 Y5 Y6 X7]
+ (0.013873381748426065) [Y6 X7 X8 Y9]
+ (0.013873381748426065) [X6 Y7 Y8 X9]
+ (0.01458364890761271) [Y0 X1 X2 Y3]
+ (0.01458364890761271) [X0 Y1 Y2 X3]
+ (0.015577208063976456) [Y2 X3 X12 Y13]
+ (0.015577208063976456) [X2 Y3 Y12 X13]
+ (0.017366118994651448) [Y6 X7 X12 Y13]
+ (0.017366118994651448) [X6 Y7 Y12 X13]
+ (0.017680067952481494) [Y4 X5 X10 Y11]
+ (0.017680067952481494) [X4 Y5 Y10 X11]
+ (0.017825140995786592) [Y6 X7 X10 Y11]
+ (0.017825140995786592) [X6 Y7 Y10 X11]
+ (0.01902824244384722) [Y3 X4 X11 Y12]
+ (0.01902824244384722) [X3 Y4 Y11 X12]
+ (0.025384657508457385) [Y2 X3 X10 Y11]
+ (0.025384657508457385) [X2 Y3 Y10 X11]
+ (0.028685183716105876) [Y10 X11 X12 Y13]
+ (0.028685183716105876) [X10 Y11 Y12 X13]
+ (0.029812424517345858) [Y6 Z7 Z8 Y10]
+ (0.029812424517345858) [X6 Z7 Z8 X10]
+ (0.029812424517345858) [Y7 Z9 Z10 Y11]
+ (0.029812424517345858) [X7 Z9 Z10 X11]
+ (0.03010462314345688) [Y6 Z7 Z9 Y10]
+ (0.03010462314345688) [X6 Z7 Z9 X10]
+ (0.03010462314345688) [Y7 Z8 Z10 Y11]
+ (0.03010462314345688) [X7 Z8 Z10 X11]
+ (0.03078750538914396) [Y6 Z8 Z9 Y10]
+ (0.03078750538914396) [X6 Z8 Z9 X10]
+ (0.03114381798896718) [Y2 X3 X6 Y7]
+ (0.03114381798896718) [X2 Y3 Y6 X7]
+ (0.035839567953353434) [Y2 X3 X4 Y5]
+ (0.035839567953353434) [X2 Y3 Y4 X5]
+ (0.03619412355904264) [Y2 X3 X8 Y9]
+ (0.03619412355904264) [X2 Y3 Y8 X9]
+ (0.10433064780651416) [Z0 Y1 Z2 Y3]
+ (0.10433064780651416) [Z0 X1 Z2 X3]
+ (-0.12133276911042308) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042308) [X2 Z3 Z4 Z5 X6]
+ (-0.12133276911042305) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133276911042305) [X3 Z4 Z5 Z6 X7]
+ (3.202076879776275e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.202076879776275e-06) [X0 Z1 Z2 Z3 X4]
+ (3.2020768797762764e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.2020768797762764e-06) [X1 Z2 Z3 Z4 X5]
+ (0.22848106564918974) [Y6 Z7 Z8 Z9 Y10]
+ (0.22848106564918974) [X6 Z7 Z8 Z9 X10]
+ (0.22848106564918974) [Y7 Z8 Z9 Z10 Y11]
+ (0.22848106564918974) [X7 Z8 Z9 Z10 X11]
+ (-0.032767657823290476) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.032767657823290476) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.032767657823290476) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.032767657823290476) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273128) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.027115036845273128) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273128) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.027115036845273128) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.02599617759802108) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.02599617759802108) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.017561202409646155) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.017561202409646155) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.017561202409646155) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.017561202409646155) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.014564531231173041) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564531231173041) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564531231173041) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564531231173041) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.012215040997614023) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.012215040997614023) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.012215040997614023) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.012215040997614023) [X4 Z5 X6 X11 Z12 X13]
+ (-0.012215040997614023) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.012215040997614023) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.012215040997614023) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.012215040997614023) [X5 Z6 X7 X10 Z11 X12]
+ (-0.011756013419819234) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.011756013419819234) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.011756013419819234) [X3 Z4 Z5 X6 X8 X9]
+ (-0.011756013419819234) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.008764827575688727) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.008764827575688727) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.008764827575688727) [X2 Z3 Z4 X5 X11 X12]
+ (-0.008764827575688727) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.008764827575688727) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.008764827575688727) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.008764827575688727) [X3 X4 X10 Z11 Z12 X13]
+ (-0.008764827575688727) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.008125251921381015) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125251921381015) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.007306759928832999) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306759928832999) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306759928832999) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306759928832999) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005805188989826923) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.005805188989826923) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.005805188989826923) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.005805188989826923) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.0056526209780173465) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.0056526209780173465) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.0056526209780173465) [X0 X1 X3 Z4 Z5 X6]
+ (-0.0056526209780173465) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.0051433917688251726) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.0051433917688251726) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.0051433917688251726) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.0051433917688251726) [X2 Z3 X4 X5 Z6 X7]
+ (-0.004684903388155198) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.004684903388155198) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668620318776293) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668620318776293) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.0045750076266392) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.0045750076266392) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.00442485544944185) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.00442485544944185) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.004158797381840047) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.004158797381840047) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.004158797381840047) [X3 Z4 Z5 X6 X12 X13]
+ (-0.004158797381840047) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.003493790359890104) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.003493790359890104) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.003493790359890104) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.003493790359890104) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.002779026799025537) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.002779026799025537) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.0022939566113524706) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.0022939566113524706) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.0017992194936630118) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.0017992194936630118) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.001727875394136975) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.001727875394136975) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298507967730602) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298507967730602) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298507967730602) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298507967730602) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533856254125414) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533856254125414) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008145313270956707) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008145313270956707) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008145313270956707) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008145313270956707) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.735036880589619e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735036880589619e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735036880589619e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735036880589619e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.774817864523076e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.774817864523076e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.774817864523076e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.774817864523076e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.518362215617103e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.518362215617103e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.518362215617103e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.518362215617103e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.444344675866116e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.444344675866116e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.444344675866116e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.444344675866116e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.524373848566369e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.524373848566369e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.524373848566369e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.524373848566369e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.290028433140226e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.290028433140226e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.290028433140226e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.290028433140226e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.974311713406002e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.974311713406002e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.275883122075803e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.275883122075803e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.643051068469441e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.643051068469441e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.556569218151564e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.556569218151564e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.253224225530695e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.253224225530695e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.7696594519883317e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.7696594519883317e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.6945132944473915e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.6945132944473915e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.6102971305715524e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.6102971305715524e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.6102971305715524e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.6102971305715524e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.3131455001643146e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.3131455001643146e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.27748319551998e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.27748319551998e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.27748319551998e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.27748319551998e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.211228348402054e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.211228348402054e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.211228348402054e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.211228348402054e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.151346311170978e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.151346311170978e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.088250711276194e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.088250711276194e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.172669101462645e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.172669101462645e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.4548424490479892e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.4548424490479892e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.3304731886569595e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.3304731886569595e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2283337824768771e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2283337824768771e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.035847760147099e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.035847760147099e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956895372339377e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956895372339377e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733197742341386e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733197742341386e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733197742341386e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733197742341386e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.628614201577069e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.628614201577069e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.556281914522288e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.556281914522288e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.556281914522288e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.556281914522288e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.41829157460861e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.41829157460861e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.41829157460861e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.41829157460861e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.927453082743841e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927453082743841e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927453082743841e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927453082743841e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627851911412149e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627851911412149e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660624700633e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660624700633e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660624700633e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660624700633e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660624700633e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660624700633e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660624700633e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660624700633e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.838052750786624e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.838052750786624e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.570761328993895e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.570761328993895e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.328139350515722e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.328139350515722e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.086826565227153e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.086826565227153e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.086826565227153e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.086826565227153e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.447323128750515e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.447323128750515e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.3713289478559891e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.3713289478559891e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.3713289478559891e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.3713289478559891e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.1990516182259402e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.1990516182259402e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.9332412770585739e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.9332412770585739e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.9332412770585739e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.9332412770585739e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.8394209154341653e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.8394209154341653e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.8394209154341653e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.8394209154341653e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.5510539176161605e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.5510539176161605e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.5510539176161605e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.5510539176161605e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.380778148032854e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.380778148032854e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.380778148032854e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.380778148032854e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.380778148032854e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.380778148032854e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.380778148032854e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.380778148032854e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.380778148032854e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.380778148032854e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.380778148032854e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.380778148032854e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.2919694861825555e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.2919694861825555e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.1076325599253248e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.1076325599253248e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.1076325599253248e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.1076325599253248e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.1076325599253248e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.1076325599253248e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.1076325599253248e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.1076325599253248e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.057446595975443e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057446595975443e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057446595975443e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057446595975443e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.64931013407037e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.64931013407037e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.64931013407037e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.64931013407037e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.8394209154341656e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.8394209154341656e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.8394209154341656e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.8394209154341656e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.1990516182259402e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.1990516182259402e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.447323128750515e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.447323128750515e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.2362599612630265e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.2362599612630265e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.2362599612630265e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.2362599612630265e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.328139350515722e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.328139350515722e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.570761328993895e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.570761328993895e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.838052750786624e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.838052750786624e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.627851911412149e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627851911412149e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.628614201577069e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.628614201577069e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956895372339377e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956895372339377e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.306536651828381e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.306536651828381e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.306536651828381e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.306536651828381e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.035847760147099e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.035847760147099e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2283337824768771e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2283337824768771e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.2393363217055533e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393363217055533e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393363217055533e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393363217055533e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.3304731886569595e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.3304731886569595e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.4548424490479892e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.4548424490479892e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.172669101462645e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.172669101462645e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.088250711276194e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.088250711276194e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.1174479459772335e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.1174479459772335e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.151346311170978e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.151346311170978e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.3131455001643146e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.3131455001643146e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.334331289477718e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.334331289477718e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.6945132944473915e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.6945132944473915e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.1839325593713705e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.1839325593713705e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.556569218151564e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.556569218151564e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.643051068469441e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.643051068469441e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.275883122075803e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.275883122075803e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.974311713406002e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.974311713406002e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.00029219862611102525) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.00029219862611102525) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.00029219862611102525) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.00029219862611102525) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004956762314915644) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004956762314915644) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650070219499427) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650070219499427) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650070219499427) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650070219499427) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533856254125414) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533856254125414) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.0016095313817213641) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.0016095313817213641) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.0016095313817213641) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.0016095313817213641) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.001667604181144026) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.001667604181144026) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.001667604181144026) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.001667604181144026) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.001727875394136975) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.001727875394136975) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.0017992194936630118) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.0017992194936630118) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.0022939566113524706) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.0022939566113524706) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.0024629170071339057) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.0024629170071339057) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.0024629170071339057) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.0024629170071339057) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.003961560792496497) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.003961560792496497) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.003961560792496497) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.003961560792496497) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.00442485544944185) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.00442485544944185) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.0045750076266392) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.0045750076266392) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668620318776293) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668620318776293) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.004684903388155198) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.004684903388155198) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005324835234221687) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.005324835234221687) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.005324835234221687) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.005324835234221687) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.005368659358109595) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368659358109595) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368659358109595) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368659358109595) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.007960880725921568) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.007960880725921568) [X4 Z5 X6 X10 Z11 X12]
+ (0.007960880725921568) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.007960880725921568) [X5 Z6 X7 X11 Z12 X13]
+ (0.008125251921381015) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125251921381015) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.008890731522694628) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.008890731522694628) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.008890731522694628) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.008890731522694628) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.010263414868158488) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.010263414868158488) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.010263414868158488) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.010263414868158488) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.01054042590767153) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.01054042590767153) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.01054042590767153) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.01054042590767153) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010960074940542599) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010960074940542599) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010960074940542599) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010960074940542599) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307274008848227) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307274008848227) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.014411099430130908) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411099430130908) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411099430130908) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411099430130908) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.015225630757226577) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.015225630757226577) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.015225630757226577) [X3 Z4 Z5 X6 X10 X11]
+ (0.015225630757226577) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.015588250102380175) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.015588250102380175) [X2 Z3 X4 X10 Z11 X12]
+ (0.015588250102380175) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.015588250102380175) [X3 Z4 X5 X11 Z12 X13]
+ (0.018266834869375602) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.018266834869375602) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.018266834869375602) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.018266834869375602) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.019020423173039973) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.019020423173039973) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.019020423173039973) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.019020423173039973) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.020175921723535592) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.020175921723535592) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.020175921723535592) [X4 Z5 Z6 X7 X11 X12]
+ (0.020175921723535592) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.020175921723535592) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.020175921723535592) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.020175921723535592) [X5 X6 X10 Z11 Z12 X13]
+ (0.020175921723535592) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.024353077678068904) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.024353077678068904) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.024353077678068904) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.024353077678068904) [X2 Z3 X4 X11 Z12 X13]
+ (0.024353077678068904) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.024353077678068904) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.024353077678068904) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.024353077678068904) [X3 Z4 X5 X10 Z11 X12]
+ (0.024389082531149568) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.024389082531149568) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.024389082531149568) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.024389082531149568) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.025104957138844572) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.025104957138844572) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.025104957138844572) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.025104957138844572) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.03078750538914396) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.03078750538914396) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.045879470781297754) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.045879470781297754) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.05600733087780774) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.05600733087780774) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.05600733087780774) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.05600733087780774) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661364) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.05608468124661364) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661364) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.05608468124661364) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.631277928376085e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631277928376085e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.631277928376085e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631277928376085e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.5950860069669025e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.5950860069669025e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.5950860069669015e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.5950860069669015e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.04274327701378194) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.04274327701378194) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (0.04274327701378194) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04274327701378194) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.047642612176383034) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.047642612176383034) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.047642612176383034) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.047642612176383034) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.04171881383982169) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.04171881383982169) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.04171881383982169) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.04171881383982169) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.03956441632289319) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.03956441632289319) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.03956441632289319) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03956441632289319) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03935916802205305) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.03935916802205305) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.03935916802205305) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.03935916802205305) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.035608378988312574) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.035608378988312574) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.02990378951262485) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.02990378951262485) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.02990378951262485) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.02990378951262485) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.028730779551905547) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.028730779551905547) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.028730779551905547) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.028730779551905547) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.025637238296026838) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.025637238296026838) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.025637238296026838) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.025637238296026838) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.024755463292890963) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.024755463292890963) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.024755463292890963) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.024755463292890963) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.024282117354693065) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.024282117354693065) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.023145130929528954) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.023145130929528954) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.022528440196012967) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.022528440196012967) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.021433810721600836) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.021433810721600836) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.021433810721600836) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.021433810721600836) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251627) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.019257505095251627) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.01902824244384722) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.01902824244384722) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.018889030304942957) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.018889030304942957) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.018889030304942957) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.018889030304942957) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.016024603689179528) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.016024603689179528) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.015225630757226577) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225630757226577) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.014603704729162085) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.014603704729162085) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564531231173041) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564531231173041) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.011756013419819234) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.011756013419819234) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.011285190200840962) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.011285190200840962) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.009841749246962588) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009841749246962588) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.00961263460684731) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.00961263460684731) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.00961263460684731) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.00961263460684731) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.008469978791024013) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.008469978791024013) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.007306759928833) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.007306759928833) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005923798336561341) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.005923798336561341) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.0056526209780173465) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.0056526209780173465) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368659358109595) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368659358109595) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.004158797381840047) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.004158797381840047) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0033566705638328766) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.0033566705638328766) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.0033566705638328766) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.0033566705638328766) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.003267513854423541) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.003267513854423541) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.003267513854423541) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.003267513854423541) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.0027790267990255367) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.0027790267990255367) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.0026860409778066254) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.0026860409778066254) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.0026860409778066254) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.0026860409778066254) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.0022939566113524706) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524706) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.0022939566113524706) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524706) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.0009581655836696501) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.0009581655836696501) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0009581655836696501) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.0009581655836696501) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.0009581655836696501) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.0009581655836696501) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0009581655836696501) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.0009581655836696501) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.00024636437569569797) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00024636437569569797) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303551867) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.00013840177303551867) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.00013840177303551867) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.00013840177303551867) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.735036880589619e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735036880589619e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.610358530559699e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.610358530559699e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.610358530559699e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.610358530559699e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.531680879518724e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.531680879518724e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.531680879518724e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.531680879518724e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.806102775111085e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.806102775111085e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.806102775111085e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.806102775111085e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.089799467424537e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.089799467424537e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.089799467424537e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.089799467424537e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.652209669468801e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.652209669468801e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.652209669468801e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.652209669468801e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851833924357e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.481851833924357e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851833924357e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.481851833924357e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.071480736439455e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.071480736439455e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.071480736439455e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.071480736439455e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.73462203867163e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.73462203867163e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.73462203867163e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.73462203867163e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.728843147152168e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.728843147152168e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.728843147152168e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.728843147152168e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.253224225530695e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.253224225530695e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.7696594519883317e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.7696594519883317e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.544395429253453e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.544395429253453e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.544395429253453e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.544395429253453e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.544395429253453e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.544395429253453e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.544395429253453e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.544395429253453e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.3609563202723692e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563202723692e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563202723692e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.3609563202723692e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.1032156046842776e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.1032156046842776e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.1032156046842776e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.1032156046842776e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220982531523e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.0111220982531523e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220982531523e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.0111220982531523e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.942946836628897e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.942946836628897e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.942946836628897e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.942946836628897e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174771261547e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.6541174771261547e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174771261547e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.6541174771261547e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.52249306763069e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.52249306763069e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.52249306763069e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.52249306763069e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.52249306763069e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.52249306763069e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.52249306763069e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.52249306763069e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.2283337824768771e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337824768771e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.2283337824768771e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337824768771e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.988770288642334e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988770288642334e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988770288642334e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988770288642334e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.867765104097537e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104097537e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104097537e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.867765104097537e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.1899909752561e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.1899909752561e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246207007678e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246207007678e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.471647744579615e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.471647744579615e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.5614471797725973e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.5614471797725973e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.5614471797725973e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.5614471797725973e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.523389677868402e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.523389677868402e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.427323108869738e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.427323108869738e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.427323108869738e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.427323108869738e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.328139350515722e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350515722e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.328139350515722e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350515722e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.086826565227153e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.086826565227153e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.8882935950274235e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935950274235e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935950274235e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.8882935950274235e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.3713289478559894e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.3713289478559894e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.8394209154341653e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.8394209154341653e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.057446595975443e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057446595975443e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.537178095683805e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.537178095683805e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.537178095683805e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.537178095683805e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.057446595975443e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057446595975443e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.209350643112487e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350643112487e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350643112487e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.209350643112487e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783554444457e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783554444457e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783554444457e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783554444457e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.8394209154341653e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.8394209154341653e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.3713289478559894e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.3713289478559894e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.086826565227153e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.086826565227153e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.523389677868402e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.523389677868402e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.471647744579615e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.471647744579615e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246207007678e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246207007678e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.1899909752561e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.1899909752561e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.3304731886569595e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.3304731886569595e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.3304731886569595e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.3304731886569595e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288532435402877e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288532435402877e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288532435402877e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288532435402877e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.689348951482718e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.689348951482718e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.689348951482718e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.689348951482718e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.7455184003892206e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.7455184003892206e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.7455184003892206e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.7455184003892206e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.7455184003892206e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.7455184003892206e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.7455184003892206e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.7455184003892206e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.2118420191134084e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420191134084e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.2118420191134084e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420191134084e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.2118420191134084e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420191134084e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.2118420191134084e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420191134084e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.3131455001643146e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.3131455001643146e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.3131455001643146e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.3131455001643146e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.334331289477718e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.334331289477718e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.1839325593713705e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.1839325593713705e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.735036880589619e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735036880589619e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00024636437569569797) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00024636437569569797) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0004458535128840993) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.0004458535128840993) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.0004458535128840993) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.0004458535128840993) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940221543005596) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005596) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005596) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005596) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005596) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005596) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940221543005596) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005596) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533856254125414) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533856254125414) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533856254125414) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533856254125414) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0010435246534907473) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435246534907473) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435246534907473) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435246534907473) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.001280306097349653) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.001280306097349653) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.001280306097349653) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.001280306097349653) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.0013038004788126917) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.0013038004788126917) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.0013038004788126917) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.0013038004788126917) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.002261966062482341) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.002261966062482341) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.002261966062482341) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.002261966062482341) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.002261966062482341) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.002261966062482341) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.002261966062482341) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.002261966062482341) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.004158797381840047) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.004158797381840047) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.004311038507914288) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.004311038507914288) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.004311038507914288) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.004311038507914288) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.00463697666118253) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.00463697666118253) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.00463697666118253) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.00463697666118253) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.005114473831660393) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.005114473831660393) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.005114473831660393) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.005114473831660393) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.005114473831660393) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660393) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660393) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005114473831660393) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005241535382803849) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.005241535382803849) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.005241535382803849) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.005241535382803849) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.005262642473076853) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262642473076853) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262642473076853) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262642473076853) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368659358109595) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368659358109595) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005379937155839367) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.005379937155839367) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.005379937155839367) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.005379937155839367) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.0056526209780173465) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.0056526209780173465) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.005708495985960952) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708495985960952) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708495985960952) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708495985960952) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.005923798336561341) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.005923798336561341) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.007306759928833) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.007306759928833) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469978791024013) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.008469978791024013) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.009841749246962588) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.009841749246962588) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.011285190200840962) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.011285190200840962) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.011756013419819234) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.011756013419819234) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.014564531231173041) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564531231173041) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603704729162085) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.014603704729162085) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.015225630757226577) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225630757226577) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.016024603689179528) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.016024603689179528) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.01902824244384722) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.01902824244384722) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.019257505095251627) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.019257505095251627) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.045879470781297754) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.045879470781297754) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.36937089366156173) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.36937089366156173) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.36937089366156173) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.36937089366156173) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.2816425776702294) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.2816425776702294) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.28164257767022927) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.28164257767022927) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.09065144207036455) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.09065144207036455) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.09065144207036455) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.09065144207036455) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863605) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684737589863605) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863605) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684737589863605) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0763502195063499) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0763502195063499) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0763502195063499) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0763502195063499) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214005) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.06752385099214005) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214005) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.06752385099214005) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.035608378988312574) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.035608378988312574) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.034903343373661806) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.034903343373661806) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.034903343373661806) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.034903343373661806) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02459186088382997) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.02459186088382997) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02459186088382997) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.02459186088382997) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.024282117354693065) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.024282117354693065) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.023145130929528954) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.023145130929528954) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.022528440196012967) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.022528440196012967) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0195380503113147) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.0195380503113147) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.0195380503113147) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.0195380503113147) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.017091553155898845) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.017091553155898845) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.017091553155898845) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.017091553155898845) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.016024603689179528) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.016024603689179528) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.016024603689179528) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.016024603689179528) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.010311482489831832) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831832) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831832) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311482489831832) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.009841749246962588) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962588) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.009841749246962588) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962588) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209848) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209848) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209848) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008826368514209848) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.00854199662545483) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00854199662545483) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00854199662545483) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00854199662545483) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.00854199662545483) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00854199662545483) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00854199662545483) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00854199662545483) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008469978791024013) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791024013) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.008469978791024013) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791024013) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.004668620318776293) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668620318776293) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.0038764708993369304) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.0038764708993369304) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.003804066171728532) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728532) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728532) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003804066171728532) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.00348415730021788) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.00348415730021788) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0033566705638328766) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.0033566705638328766) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.003267513854423541) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.003267513854423541) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.002141361223101535) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.002141361223101535) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.001727875394136975) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.001727875394136975) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.001640754855312444) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.001640754855312444) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.001452884321416865) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.001452884321416865) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.001452884321416865) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.001452884321416865) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.000787089677102441) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.000787089677102441) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192743499487581) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192743499487581) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.00019400857029756857) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00019400857029756857) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303551867) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.00013840177303551867) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.141625221151001e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.141625221151001e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.141625221151001e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.141625221151001e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.071480736439456e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.071480736439456e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.151346311170978e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.151346311170978e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.088250711276194e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.088250711276194e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.988511706324043e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.988511706324043e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8742990713476552e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.8742990713476552e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.3609563202723692e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.3609563202723692e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.3002946562512731e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3002946562512731e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.1468376507459044e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.1468376507459044e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.1468376507459044e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.1468376507459044e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.352332103053057e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.352332103053057e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.352332103053057e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.352332103053057e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091637199046095e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637199046095e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637199046095e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637199046095e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637199046095e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637199046095e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637199046095e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091637199046095e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.074305985889887e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.074305985889887e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.074305985889887e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.074305985889887e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128986281354e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128986281354e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128986281354e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128986281354e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867765104097537e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.867765104097537e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560692464813274e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692464813274e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692464813274e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692464813274e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692464813274e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692464813274e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692464813274e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560692464813274e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.997018422066935e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.997018422066935e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.997018422066935e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.997018422066935e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.997018422066935e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.997018422066935e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.997018422066935e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.997018422066935e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.5682475211776915e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.5682475211776915e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.5682475211776915e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.5682475211776915e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376739308412951e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.376739308412951e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376739308412951e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.376739308412951e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376739308412951e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.376739308412951e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.376739308412951e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.376739308412951e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.8882935950274235e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.8882935950274235e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6863815454279124e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.6863815454279124e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035783554444457e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7035783554444457e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.209350643112487e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.209350643112487e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244367855e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773244367855e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244367855e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773244367855e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244367855e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773244367855e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773244367855e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773244367855e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253794609877e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253794609877e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9742253794609877e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9742253794609877e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0474716555718353e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.0474716555718353e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.0474716555718353e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.0474716555718353e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.209350643112487e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.209350643112487e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0717282183777156e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0717282183777156e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0717282183777156e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0717282183777156e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.2004287493962043e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.2004287493962043e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.2004287493962043e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.2004287493962043e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.7035783554444457e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7035783554444457e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.3120943052379037e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.3120943052379037e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.3120943052379037e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.3120943052379037e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.6863815454279124e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815454279124e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.8882935950274235e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.8882935950274235e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.092250616032607e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.092250616032607e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.092250616032607e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.092250616032607e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.092250616032607e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.092250616032607e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.092250616032607e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.092250616032607e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.4445978540533725e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.4445978540533725e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.4445978540533725e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.4445978540533725e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.6849150951172945e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.6849150951172945e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.6849150951172945e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.6849150951172945e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246974425385393e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246974425385393e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246974425385393e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246974425385393e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246974425385393e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246974425385393e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246974425385393e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246974425385393e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.867765104097537e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.867765104097537e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3002946562512731e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3002946562512731e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.3609563202723692e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.3609563202723692e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.8742990713476552e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.8742990713476552e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.8836765759402314e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.8836765759402314e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.947356011632672e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.947356011632672e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.947356011632672e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.947356011632672e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.988511706324043e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.988511706324043e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.088250711276194e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.088250711276194e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.151346311170978e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.151346311170978e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.846201671196146e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846201671196146e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846201671196146e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846201671196146e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.071480736439456e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.071480736439456e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.10552672188736e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.10552672188736e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.10552672188736e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.10552672188736e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146496327447419e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146496327447419e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146496327447419e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146496327447419e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159350501818385e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159350501818385e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159350501818385e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159350501818385e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427988656361176e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427988656361176e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427988656361176e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427988656361176e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.9358677179567134e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.9358677179567134e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.9358677179567134e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.9358677179567134e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273347968129e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.253273347968129e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979825793235014e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979825793235014e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979825793235014e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979825793235014e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.2055484112169667e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.2055484112169667e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.2055484112169667e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.2055484112169667e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00013840177303551867) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.00013840177303551867) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.00018787053389557893) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.00018787053389557893) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.00018787053389557893) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00018787053389557893) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00019400857029756857) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00019400857029756857) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00024636437569569797) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569569797) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00024636437569569797) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569569797) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192743499487581) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192743499487581) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.000715673424890931) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.000715673424890931) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.000715673424890931) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.000715673424890931) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.000787089677102441) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.000787089677102441) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.0015324835230731038) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.0015324835230731038) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.0015324835230731038) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.0015324835230731038) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.001640754855312444) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.001640754855312444) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.001727875394136975) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.001727875394136975) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.0024464971554158557) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.0024464971554158557) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.0024464971554158557) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.0024464971554158557) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.003267513854423541) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.003267513854423541) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.0033566705638328766) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.0033566705638328766) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.00348415730021788) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.00348415730021788) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0038764708993369304) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.0038764708993369304) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.004668620318776293) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668620318776293) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278151) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.004767272188278151) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.004767272188278151) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278151) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.005286546538226909) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.005286546538226909) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.005286546538226909) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.005286546538226909) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.005408954422410035) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.005408954422410035) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.005408954422410035) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.005408954422410035) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.005923798336561341) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561341) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.005923798336561341) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561341) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.010715508469796796) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010715508469796796) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010715508469796796) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010715508469796796) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.010757563953908967) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010757563953908967) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010757563953908967) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010757563953908967) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.014603704729162085) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.014603704729162085) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.014603704729162085) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.014603704729162085) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.019299560579363797) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363797) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363797) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363797) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363797) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363797) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.019299560579363797) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363797) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.05859198873386183) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.05859198873386183) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.775950527110124e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.775950527110124e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.7759505271101245e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.7759505271101245e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.07165035181002798) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.07165035181002798) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.07165035181002803) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07165035181002803) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.019257505095251627) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019257505095251627) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010311482489831832) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311482489831832) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008826368514209848) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209848) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.007597464029770623) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.007597464029770623) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.007597464029770623) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.007597464029770623) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311861) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311861) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311861) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311861) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311861) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311861) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311861) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311861) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0053480515826766425) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0053480515826766425) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0053480515826766425) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0053480515826766425) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.003804066171728532) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003804066171728532) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984166168121915) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.002984166168121915) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.002984166168121915) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.002984166168121915) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.0024464971554158553) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.0024464971554158553) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.002249412447093982) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.002249412447093982) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.002249412447093982) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002249412447093982) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.002141361223101535) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.002141361223101535) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.0018638942824587613) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587613) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587613) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587613) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587613) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587613) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0018638942824587613) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587613) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.001640754855312444) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.001640754855312444) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.001640754855312444) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.001640754855312444) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0012223378081538268) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538268) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538268) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538268) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538268) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538268) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538268) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0012223378081538268) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0010283292378562583) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0010283292378562583) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0010283292378562583) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0010283292378562583) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1463061453025023e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1463061453025023e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.8742990713476552e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990713476552e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.8742990713476552e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990713476552e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3002946562512731e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3002946562512731e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3002946562512731e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3002946562512731e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.0444941298233382e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.0444941298233382e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.0444941298233382e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.0444941298233382e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.956079230187672e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.956079230187672e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.956079230187672e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.956079230187672e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105515037376044e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105515037376044e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105515037376044e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105515037376044e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661347213282257e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661347213282257e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661347213282257e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661347213282257e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341413605817e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341413605817e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.1899909752561e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.1899909752561e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876621658469794e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876621658469794e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876621658469794e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876621658469794e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246207007678e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246207007678e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.523389677868402e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.523389677868402e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0767325320262473e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.0767325320262473e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0767325320262473e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.0767325320262473e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.013471458903223e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.013471458903223e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.904599884627566e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.904599884627566e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.904599884627566e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.904599884627566e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.6667317548727086e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.6667317548727086e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.6667317548727086e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6667317548727086e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641928116295e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.8505641928116295e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309311932195e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6569309311932195e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309311932195e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6569309311932195e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.8505641928116295e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.8505641928116295e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.686381545427913e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.686381545427913e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.686381545427913e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.686381545427913e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458903223e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.013471458903223e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.523389677868402e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.523389677868402e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.6704023900964426e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.6704023900964426e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.6704023900964426e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.6704023900964426e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246207007678e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246207007678e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.1899909752561e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.1899909752561e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341413605817e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341413605817e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949476486928373e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949476486928373e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.7924939576903458e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939576903458e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939576903458e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.7924939576903458e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.8836765759402314e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.8836765759402314e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.988511706324043e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.988511706324043e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.988511706324043e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.988511706324043e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273347968129e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253273347968129e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4017109735032516e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4017109735032516e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4017109735032516e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4017109735032516e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603692722863e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.5809603692722863e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603692722863e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.5809603692722863e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192743499487581) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487581) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192743499487581) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487581) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.000787089677102441) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.000787089677102441) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.000787089677102441) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.000787089677102441) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0011726348316441915) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0011726348316441915) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0011726348316441915) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0011726348316441915) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012366478019245851) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.0012366478019245851) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.0012366478019245851) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.0012366478019245851) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.00220096406950045) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.00220096406950045) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.00220096406950045) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00220096406950045) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798019) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798019) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798019) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798019) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798019) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798019) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002394972639798019) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798019) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0024464971554158553) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.0024464971554158553) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.003804066171728532) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.003804066171728532) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00387647089933693) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.00387647089933693) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.00387647089933693) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.00387647089933693) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.0042208139700465) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.0042208139700465) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.0042208139700465) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.0042208139700465) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.008826368514209848) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.008826368514209848) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.010311482489831832) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311482489831832) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257505095251627) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019257505095251627) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.05859198873386183) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.05859198873386183) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.398700901612191e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.398700901612191e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.398700901612191e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.398700901612191e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00348415730021788) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00348415730021788) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219147) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.0029841661681219147) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.00019400857029756857) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00019400857029756857) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453025023e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453025023e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7924939576903458e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7924939576903458e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540341413605817e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413605817e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540341413605817e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413605817e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.8505641928116295e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641928116295e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641928116295e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641928116295e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458903223e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471458903223e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458903223e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471458903223e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949476486928373e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949476486928373e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.7924939576903458e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7924939576903458e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756857) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756857) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0029841661681219147) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.0029841661681219147) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.00348415730021788) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00348415730021788) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
 </code>
 </pre>
 </details>

---

## 19. tutorial_kernel_based_training.html <a name="demo18"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_kernel_based_training.html):

```
step 20 , loss 0.43849890579633266
step 30 , loss 0.6458829274590642
step 40 , loss 0.5540116701446128
step 50 , loss 0.41322391458182645
step 60 , loss 0.5209433003814099
step 70 , loss 0.469419342316038
step 80 , loss 0.4858145744021137
step 90 , loss 0.4196234621534023
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_kernel_based_training.html):

```
step 20 , loss 0.4384989057963324
step 30 , loss 0.645882927459064
step 40 , loss 0.5540116701446127
step 50 , loss 0.4132239145818265
step 60 , loss 0.5209433003814097
step 70 , loss 0.4694193423160377
step 80 , loss 0.48581457440211384
step 90 , loss 0.41962346215340246
```

---

## 20. tutorial_qnn_module_tf.html <a name="demo19"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_qnn_module_tf.html):

```
30/30 - 9s - loss: 0.4997 - accuracy: 0.5000 - val_loss: 0.5081 - val_accuracy: 0.4400
30/30 - 10s - loss: 0.4673 - accuracy: 0.6200 - val_loss: 0.4488 - val_accuracy: 0.6200
30/30 - 9s - loss: 0.3230 - accuracy: 0.8267 - val_loss: 0.2562 - val_accuracy: 0.8400
30/30 - 9s - loss: 0.2124 - accuracy: 0.8867 - val_loss: 0.1997 - val_accuracy: 0.8400
30/30 - 9s - loss: 0.1800 - accuracy: 0.8933 - val_loss: 0.1841 - val_accuracy: 0.8400
30/30 - 9s - loss: 0.1593 - accuracy: 0.8667 - val_loss: 0.2177 - val_accuracy: 0.8400
30/30 - 17s - loss: 0.5189 - accuracy: 0.4000 - val_loss: 0.4945 - val_accuracy: 0.5400
30/30 - 17s - loss: 0.4822 - accuracy: 0.6200 - val_loss: 0.4412 - val_accuracy: 0.7200
30/30 - 18s - loss: 0.3850 - accuracy: 0.7133 - val_loss: 0.2898 - val_accuracy: 0.7800
30/30 - 18s - loss: 0.2720 - accuracy: 0.7867 - val_loss: 0.2185 - val_accuracy: 0.8200
30/30 - 17s - loss: 0.2056 - accuracy: 0.8400 - val_loss: 0.1893 - val_accuracy: 0.8400
30/30 - 16s - loss: 0.1753 - accuracy: 0.8400 - val_loss: 0.1856 - val_accuracy: 0.8400
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_qnn_module_tf.html):

```
30/30 - 13s - loss: 0.4997 - accuracy: 0.5000 - val_loss: 0.5081 - val_accuracy: 0.4400 - 13s/epoch - 425ms/step
30/30 - 13s - loss: 0.4673 - accuracy: 0.6200 - val_loss: 0.4488 - val_accuracy: 0.6200 - 13s/epoch - 421ms/step
30/30 - 13s - loss: 0.3230 - accuracy: 0.8267 - val_loss: 0.2562 - val_accuracy: 0.8400 - 13s/epoch - 425ms/step
30/30 - 13s - loss: 0.2124 - accuracy: 0.8867 - val_loss: 0.1997 - val_accuracy: 0.8400 - 13s/epoch - 425ms/step
30/30 - 13s - loss: 0.1800 - accuracy: 0.8933 - val_loss: 0.1841 - val_accuracy: 0.8400 - 13s/epoch - 420ms/step
30/30 - 12s - loss: 0.1593 - accuracy: 0.8667 - val_loss: 0.2177 - val_accuracy: 0.8400 - 12s/epoch - 413ms/step
30/30 - 25s - loss: 0.5189 - accuracy: 0.4000 - val_loss: 0.4945 - val_accuracy: 0.5400 - 25s/epoch - 830ms/step
30/30 - 25s - loss: 0.4822 - accuracy: 0.6200 - val_loss: 0.4412 - val_accuracy: 0.7200 - 25s/epoch - 832ms/step
30/30 - 25s - loss: 0.3850 - accuracy: 0.7133 - val_loss: 0.2898 - val_accuracy: 0.7800 - 25s/epoch - 834ms/step
30/30 - 25s - loss: 0.2720 - accuracy: 0.7867 - val_loss: 0.2185 - val_accuracy: 0.8200 - 25s/epoch - 848ms/step
30/30 - 25s - loss: 0.2056 - accuracy: 0.8400 - val_loss: 0.1893 - val_accuracy: 0.8400 - 25s/epoch - 843ms/step
30/30 - 25s - loss: 0.1753 - accuracy: 0.8400 - val_loss: 0.1856 - val_accuracy: 0.8400 - 25s/epoch - 844ms/step
```

---

## 21. tutorial_quantum_chemistry.html <a name="demo20"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quantum_chemistry.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Qubit Hamiltonian of the water molecule
(-46.46390678868893+0j) [] +
(-0.01458364890761252+0j) [X0 X1 Y2 Y3] +
(-3.570761329316954e-07+0j) [X0 X1 Y2 Z3 Z4 Y5] +
(-0.005652620978017329+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7] +
(-0.008826368514209829+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.7924939578497242e-06+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.5707613293169544e-07+0j) [X0 X1 X3 X4] +
(-0.005652620978017329+0j) [X0 X1 X3 Z4 Z5 X6] +
(-0.008826368514209829+0j) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.7924939578497242e-06+0j) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.0027458364701868033+0j) [X0 X1 Y4 Y5] +
(-2.447323128922431e-07+0j) [X0 X1 Y4 Z5 Z6 Y7] +
(-7.86776510495246e-07+0j) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0038040661717285355+0j) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.447323128922431e-07+0j) [X0 X1 X5 X6] +
(-7.86776510495246e-07+0j) [X0 X1 X5 Z6 Z7 Z8 Z9 X10] +
(-0.003804066171728536+0j) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.006888194352970585+0j) [X0 X1 Y6 Y7] +
(-7.735036880592557e-05+0j) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11] +
(1.703578355465942e-07+0j) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.735036880592557e-05+0j) [X0 X1 X7 Z8 Z9 X10] +
(1.703578355465942e-07+0j) [X0 X1 X7 Z8 Z9 Z10 Z11 X12] +
(-0.006509361201177234+0j) [X0 X1 Y8 Y9] +
(-0.007731425250775297+0j) [X0 X1 Y10 Y11] +
(5.627851911837115e-07+0j) [X0 X1 Y10 Z11 Z12 Y13] +
(5.627851911837115e-07+0j) [X0 X1 X11 X12] +
(-0.005283776488402961+0j) [X0 X1 Y12 Y13] +
(0.01458364890761252+0j) [X0 Y1 Y2 X3] +
(3.570761329316954e-07+0j) [X0 Y1 Y2 Z3 Z4 X5] +
(0.005652620978017329+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7] +
(0.008826368514209829+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-1.7924939578497242e-06+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.5707613293169544e-07+0j) [X0 Y1 Y3 X4] +
(-0.005652620978017329+0j) [X0 Y1 Y3 Z4 Z5 X6] +
(-0.008826368514209829+0j) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.7924939578497242e-06+0j) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0027458364701868033+0j) [X0 Y1 Y4 X5] +
(2.447323128922431e-07+0j) [X0 Y1 Y4 Z5 Z6 X7] +
(7.86776510495246e-07+0j) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0038040661717285355+0j) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.447323128922431e-07+0j) [X0 Y1 Y5 X6] +
(-7.86776510495246e-07+0j) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.003804066171728536+0j) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.006888194352970585+0j) [X0 Y1 Y6 X7] +
(7.735036880592557e-05+0j) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11] +
(-1.703578355465942e-07+0j) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.735036880592557e-05+0j) [X0 Y1 Y7 Z8 Z9 X10] +
(1.703578355465942e-07+0j) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12] +
(0.006509361201177234+0j) [X0 Y1 Y8 X9] +
(0.007731425250775297+0j) [X0 Y1 Y10 X11] +
(-5.627851911837115e-07+0j) [X0 Y1 Y10 Z11 Z12 X13] +
(5.627851911837115e-07+0j) [X0 Y1 Y11 X12] +
(0.005283776488402961+0j) [X0 Y1 Y12 X13] +
(0.12507032579771682+0j) [X0 Z1 X2] +
(-1.9332412771451438e-07+0j) [X0 Z1 X2 X3 Z4 X5] +
(-0.0022939566113524545+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 X7] +
(-0.0016407548553123874+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(3.0134714592101927e-07+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.9332412771451438e-07+0j) [X0 Z1 X2 Y3 Z4 Y5] +
(-0.0022939566113524545+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7] +
(-0.0016407548553123874+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(3.0134714592101927e-07+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0032769719312315095+0j) [X0 Z1 X2 Z3] +
(-1.5510539177035902e-07+0j) [X0 Z1 X2 X4 Z5 X6] +
(-1.1468376508721707e-06+0j) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.007597464029770566+0j) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.3807781481303515e-07+0j) [X0 Z1 X2 Y4 Z5 Y6] +
(-7.90012898723309e-07+0j) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0053480515826765706+0j) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005530759218631444+0j) [X0 Z1 X2 Z4] +
(-1.3807781481303515e-07+0j) [X0 Z1 X2 X5 Z6 X7] +
(-3.376739308778351e-07+0j) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0018638942824587006+0j) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.3807781481303515e-07+0j) [X0 Z1 X2 Y5 Z6 Y7] +
(-3.376739308778351e-07+0j) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0018638942824587006+0j) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0011059037691896003+0j) [X0 Z1 X2 Z5] +
(0.005708495985960871+0j) [X0 Z1 X2 X6 Z7 Z8 Z9 X10] +
(-8.352332104000157e-07+0j) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
1.9742253796755265e-08j [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005262642473076793+0j) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10] +
(-8.074305986801008e-07+0j) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.008032520918821295+0j) [X0 Z1 X2 Z6] +
(0.0005940221543005044+0j) [X0 Z1 X2 X7 Z8 Z9 Z10 X11] +
(-8.379773245074612e-08+0j) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005940221543005044+0j) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11] +
(-8.379773245074612e-08+0j) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0033476175306660924+0j) [X0 Z1 X2 Z7] +
(0.011055020596131957+0j) [X0 Z1 X2 Z8] +
(0.002929768674750943+0j) [X0 Z1 X2 Z9] +
(-6.41829157513796e-07+0j) [X0 Z1 X2 X10 Z11 X12] +
(-6.556281915171837e-07+0j) [X0 Z1 X2 Y10 Z11 Y12] +
(0.0035552901955041654+0j) [X0 Z1 X2 Z10] +
(-1.107632560018249e-07+0j) [X0 Z1 X2 X11 Z12 X13] +
(-1.107632560018249e-07+0j) [X0 Z1 X2 Y11 Z12 Y13] +
(0.0017560707018411516+0j) [X0 Z1 X2 Z11] +
(0.0069012382497971834+0j) [X0 Z1 X2 Z12] +
(0.0023262306231579934+0j) [X0 Z1 X2 Z13] +
(-3.5682475214886193e-07+0j) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0022494124470939956+0j) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.047471655639631e-08+0j) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0004458535128840776+0j) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10] +
(-1.974225379610358e-08+0j) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441844+0j) [X0 Z1 Z2 X3 Y4 Y5] +
(-4.52338967845474e-07+0j) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0034841573002178704+0j) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-8.091637199943361e-07+0j) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10] +
(-0.005733569747311866+0j) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004684903388155203+0j) [X0 Z1 Z2 X3 Y6 Y7] +
(0.004668620318776289+0j) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11] +
(-7.189990976091752e-07+0j) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005114473831660366+0j) [X0 Z1 Z2 X3 X7 Z8 Z9 X10] +
(-7.560692465694491e-07+0j) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381015+0j) [X0 Z1 Z2 X3 Y8 Y9] +
(-0.001799219493663014+0j) [X0 Z1 Z2 X3 Y10 Y11] +
(-5.471647745154932e-07+0j) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13] +
(-5.287660625118364e-07+0j) [X0 Z1 Z2 X3 X11 X12] +
(-0.00457500762663919+0j) [X0 Z1 Z2 X3 Y12 Y13] +
(0.004424855449441844+0j) [X0 Z1 Z2 Y3 Y4 X5] +
(4.52338967845474e-07+0j) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0034841573002178704+0j) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.091637199943361e-07+0j) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.005733569747311866+0j) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.004684903388155203+0j) [X0 Z1 Z2 Y3 Y6 X7] +
(-0.004668620318776289+0j) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11] +
(7.189990976091752e-07+0j) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005114473831660366+0j) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10] +
(-7.560692465694491e-07+0j) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12] +
(0.008125251921381015+0j) [X0 Z1 Z2 Y3 Y8 X9] +
(0.001799219493663014+0j) [X0 Z1 Z2 Y3 Y10 X11] +
(5.471647745154932e-07+0j) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13] +
(-5.287660625118364e-07+0j) [X0 Z1 Z2 Y3 Y11 X12] +
(0.00457500762663919+0j) [X0 Z1 Z2 Y3 Y12 X13] +
(3.2020768796847134e-06+0j) [X0 Z1 Z2 Z3 X4] +
(0.0008533856254125567+0j) [X0 Z1 Z2 Z3 X4 X5 Z6 X7] +
(0.0007870896771024556+0j) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0008533856254125567+0j) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7] +
(0.0007870896771024556+0j) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.2919694865715258e-07+0j) [X0 Z1 Z2 Z3 X4 Z5] +
(4.444597854444543e-07+0j) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10] +
(0.0011726348316441796+0j) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.684915095611484e-07+0j) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10] +
(0.0022009640695004533+0j) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.839420915573406e-07+0j) [X0 Z1 Z2 Z3 X4 Z6] +
(4.092250616481608e-07+0j) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11] +
(0.002394972639798026+0j) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.092250616481608e-07+0j) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11] +
(0.002394972639798026+0j) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.2362599613934214e-07+0j) [X0 Z1 Z2 Z3 X4 Z8] +
(8.649310133292417e-08+0j) [X0 Z1 Z2 Z3 X4 Z9] +
(0.0013038004788126938+0j) [X0 Z1 Z2 Z3 X4 X10 Z11 X12] +
(0.003989841456619303+0j) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12] +
(-6.733197743250526e-07+0j) [X0 Z1 Z2 Z3 X4 Z10] +
(0.0022619660624823602+0j) [X0 Z1 Z2 Z3 X4 X11 Z12 X13] +
(0.0022619660624823602+0j) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13] +
(-5.927453083636187e-07+0j) [X0 Z1 Z2 Z3 X4 Z11] +
(1.2393363217829389e-06+0j) [X0 Z1 Z2 Z3 X4 Z12] +
(9.30653665268027e-07+0j) [X0 Z1 Z2 Z3 X4 Z13] +
(-0.0010283292378562734+0j) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.0026860409778066085+0j) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12] +
(-1.8394209155734062e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7] +
(-0.00019400857029757307+0j) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0012223378081538466+0j) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-2.3713289480641795e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9] +
(8.057446596143386e-08+0j) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11] +
(0.0017278753941369436+0j) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13] +
(-0.000958165583669666+0j) [X0 Z1 Z2 Z3 Z4 X5 X11 X12] +
(-3.086826565149119e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13] +
(1.8394209155734062e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7] +
(0.00019400857029757307+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0012223378081538466+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(2.3713289480641795e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9] +
(-8.057446596143386e-08+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11] +
(-0.0017278753941369436+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13] +
(-0.000958165583669666+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12] +
(3.086826565149119e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13] +
(0.042743277013782374+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6] +
(0.0005192743499487784+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(-1.8505641930257292e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005192743499487784+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(-1.8505641930257292e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0027790267990256004+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7] +
(0.004636976661182546+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8] +
(0.0012803060973496407+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9] +
(2.312094305421145e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12] +
(1.071728218553196e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12] +
(0.005379937155839342+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10] +
(7.246974426187344e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13] +
(7.246974426187344e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13] +
(0.00524153538280387+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11] +
(0.004311038507914298+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12] +
(0.0010435246534907373+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13] +
(1.2004287494859707e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12] +
(-0.003356670563832906+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9] +
(-0.00013840177303547186+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11] +
(-6.175246207634148e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(-4.997018422736259e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12] +
(-0.003267513854423561+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13] +
(0.003356670563832906+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9] +
(0.00013840177303547186+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11] +
(6.175246207634148e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(-4.997018422736259e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12] +
(0.003267513854423561+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13] +
(0.0038764708993369733+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(-7.540341414466275e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.0038764708993369733+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(-7.540341414466275e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(0.07165035181002433+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0021413612231016457+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(0.004220813970046442+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(0.0012366478019244917+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(-0.0029841661681219503+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(0.0029841661681219503+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-1.3987009014503855e-05+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(8.949476488676168e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-6.876621658535399e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(-7.661347213519868e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(0.0015324835230729771+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10] +
(-2.904599884093951e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(0.00540895442240995+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10] +
(-1.0444941298560226e-06+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(0.004767272188278078+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10] +
(-8.105515037511588e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(0.005286546538226857+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10] +
(-9.956079230537316e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(0.001609531381721358+0j) [X0 Z1 Z2 Z3 Z4 X6] +
(-7.14162522116003e-05+0j) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10] +
(-2.6667317545432573e-07+0j) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0024629170071339144+0j) [X0 Z1 Z2 Z3 Z5 X6] +
(0.0007156734248908554+0j) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10] +
(-3.0767325319160605e-07+0j) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.6060718679074047e-07+0j) [X0 Z1 Z2 X4] +
(0.0039615607924965+0j) [X0 Z1 Z2 Z4 Z5 X6] +
(0.00018787053389542632+0j) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.6569309320508123e-07+0j) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.7379332623569244e-07+0j) [X0 Z1 Z3 X4] +
(0.0016676041811440462+0j) [X0 Z1 Z3 Z4 Z5 X6] +
(-0.001452884321416961+0j) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(4.670402391261005e-07+0j) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.10433064780651333+0j) [X0 X2] +
(3.117447946206898e-06+0j) [X0 Z2 Z3 X4] +
(0.045879470781298046+0j) [X0 Z2 Z3 Z4 Z5 X6] +
(0.058591988733861886+0j) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-1.1463061453953919e-05+0j) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.01458364890761252+0j) [Y0 X1 X2 Y3] +
(3.570761329316954e-07+0j) [Y0 X1 X2 Z3 Z4 Y5] +
(0.005652620978017329+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7] +
(0.008826368514209829+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.7924939578497242e-06+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.5707613293169544e-07+0j) [Y0 X1 X3 Y4] +
(-0.005652620978017329+0j) [Y0 X1 X3 Z4 Z5 Y6] +
(-0.008826368514209829+0j) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.7924939578497242e-06+0j) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0027458364701868033+0j) [Y0 X1 X4 Y5] +
(2.447323128922431e-07+0j) [Y0 X1 X4 Z5 Z6 Y7] +
(7.86776510495246e-07+0j) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.0038040661717285355+0j) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.447323128922431e-07+0j) [Y0 X1 X5 Y6] +
(-7.86776510495246e-07+0j) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.003804066171728536+0j) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.006888194352970585+0j) [Y0 X1 X6 Y7] +
(7.735036880592557e-05+0j) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11] +
(-1.703578355465942e-07+0j) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.735036880592557e-05+0j) [Y0 X1 X7 Z8 Z9 Y10] +
(1.703578355465942e-07+0j) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12] +
(0.006509361201177234+0j) [Y0 X1 X8 Y9] +
(0.007731425250775297+0j) [Y0 X1 X10 Y11] +
(-5.627851911837115e-07+0j) [Y0 X1 X10 Z11 Z12 Y13] +
(5.627851911837115e-07+0j) [Y0 X1 X11 Y12] +
(0.005283776488402961+0j) [Y0 X1 X12 Y13] +
(-0.01458364890761252+0j) [Y0 Y1 X2 X3] +
(-3.570761329316954e-07+0j) [Y0 Y1 X2 Z3 Z4 X5] +
(-0.005652620978017329+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7] +
(-0.008826368514209829+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.7924939578497242e-06+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.5707613293169544e-07+0j) [Y0 Y1 Y3 Y4] +
(-0.005652620978017329+0j) [Y0 Y1 Y3 Z4 Z5 Y6] +
(-0.008826368514209829+0j) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.7924939578497242e-06+0j) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.0027458364701868033+0j) [Y0 Y1 X4 X5] +
(-2.447323128922431e-07+0j) [Y0 Y1 X4 Z5 Z6 X7] +
(-7.86776510495246e-07+0j) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0038040661717285355+0j) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.447323128922431e-07+0j) [Y0 Y1 Y5 Y6] +
(-7.86776510495246e-07+0j) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.003804066171728536+0j) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.006888194352970585+0j) [Y0 Y1 X6 X7] +
(-7.735036880592557e-05+0j) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11] +
(1.703578355465942e-07+0j) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.735036880592557e-05+0j) [Y0 Y1 Y7 Z8 Z9 Y10] +
(1.703578355465942e-07+0j) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.006509361201177234+0j) [Y0 Y1 X8 X9] +
(-0.007731425250775297+0j) [Y0 Y1 X10 X11] +
(5.627851911837115e-07+0j) [Y0 Y1 X10 Z11 Z12 X13] +
(5.627851911837115e-07+0j) [Y0 Y1 Y11 Y12] +
(-0.005283776488402961+0j) [Y0 Y1 X12 X13] +
(-3.5682475214886193e-07+0j) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0022494124470939956+0j) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0004458535128840776+0j) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10] +
(-1.974225379610358e-08+0j) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.047471655639631e-08+0j) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.12507032579771682+0j) [Y0 Z1 Y2] +
(-1.9332412771451438e-07+0j) [Y0 Z1 Y2 X3 Z4 X5] +
(-0.0022939566113524545+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7] +
(-0.0016407548553123874+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(3.0134714592101927e-07+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.9332412771451438e-07+0j) [Y0 Z1 Y2 Y3 Z4 Y5] +
(-0.0022939566113524545+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7] +
(-0.0016407548553123874+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(3.0134714592101927e-07+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0032769719312315095+0j) [Y0 Z1 Y2 Z3] +
(-1.3807781481303515e-07+0j) [Y0 Z1 Y2 X4 Z5 X6] +
(-7.90012898723309e-07+0j) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0053480515826765706+0j) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.5510539177035902e-07+0j) [Y0 Z1 Y2 Y4 Z5 Y6] +
(-1.1468376508721707e-06+0j) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.007597464029770566+0j) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005530759218631444+0j) [Y0 Z1 Y2 Z4] +
(-1.3807781481303515e-07+0j) [Y0 Z1 Y2 X5 Z6 X7] +
(-3.376739308778351e-07+0j) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0018638942824587006+0j) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.3807781481303515e-07+0j) [Y0 Z1 Y2 Y5 Z6 Y7] +
(-3.376739308778351e-07+0j) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0018638942824587006+0j) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0011059037691896003+0j) [Y0 Z1 Y2 Z5] +
(0.005262642473076793+0j) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10] +
(-8.074305986801008e-07+0j) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.005708495985960871+0j) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10] +
-1.9742253796755265e-08j [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-8.352332104000157e-07+0j) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.008032520918821295+0j) [Y0 Z1 Y2 Z6] +
(0.0005940221543005044+0j) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11] +
(-8.379773245074612e-08+0j) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005940221543005044+0j) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11] +
(-8.379773245074612e-08+0j) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0033476175306660924+0j) [Y0 Z1 Y2 Z7] +
(0.011055020596131957+0j) [Y0 Z1 Y2 Z8] +
(0.002929768674750943+0j) [Y0 Z1 Y2 Z9] +
(-6.556281915171837e-07+0j) [Y0 Z1 Y2 X10 Z11 X12] +
(-6.41829157513796e-07+0j) [Y0 Z1 Y2 Y10 Z11 Y12] +
(0.0035552901955041654+0j) [Y0 Z1 Y2 Z10] +
(-1.107632560018249e-07+0j) [Y0 Z1 Y2 X11 Z12 X13] +
(-1.107632560018249e-07+0j) [Y0 Z1 Y2 Y11 Z12 Y13] +
(0.0017560707018411516+0j) [Y0 Z1 Y2 Z11] +
(0.0069012382497971834+0j) [Y0 Z1 Y2 Z12] +
(0.0023262306231579934+0j) [Y0 Z1 Y2 Z13] +
(0.004424855449441844+0j) [Y0 Z1 Z2 X3 X4 Y5] +
(4.52338967845474e-07+0j) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.0034841573002178704+0j) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-8.091637199943361e-07+0j) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.005733569747311866+0j) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.004684903388155203+0j) [Y0 Z1 Z2 X3 X6 Y7] +
(-0.004668620318776289+0j) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11] +
(7.189990976091752e-07+0j) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005114473831660366+0j) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10] +
(-7.560692465694491e-07+0j) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12] +
(0.008125251921381015+0j) [Y0 Z1 Z2 X3 X8 Y9] +
(0.001799219493663014+0j) [Y0 Z1 Z2 X3 X10 Y11] +
(5.471647745154932e-07+0j) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13] +
(-5.287660625118364e-07+0j) [Y0 Z1 Z2 X3 X11 Y12] +
(0.00457500762663919+0j) [Y0 Z1 Z2 X3 X12 Y13] +
(-0.004424855449441844+0j) [Y0 Z1 Z2 Y3 X4 X5] +
(-4.52338967845474e-07+0j) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0034841573002178704+0j) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.091637199943361e-07+0j) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.005733569747311866+0j) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004684903388155203+0j) [Y0 Z1 Z2 Y3 X6 X7] +
(0.004668620318776289+0j) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11] +
(-7.189990976091752e-07+0j) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005114473831660366+0j) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10] +
(-7.560692465694491e-07+0j) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381015+0j) [Y0 Z1 Z2 Y3 X8 X9] +
(-0.001799219493663014+0j) [Y0 Z1 Z2 Y3 X10 X11] +
(-5.471647745154932e-07+0j) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13] +
(-5.287660625118364e-07+0j) [Y0 Z1 Z2 Y3 Y11 Y12] +
(-0.00457500762663919+0j) [Y0 Z1 Z2 Y3 X12 X13] +
(-0.0010283292378562734+0j) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.0026860409778066085+0j) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12] +
(3.2020768796847134e-06+0j) [Y0 Z1 Z2 Z3 Y4] +
(0.0008533856254125567+0j) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7] +
(0.0007870896771024556+0j) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0008533856254125567+0j) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7] +
(0.0007870896771024556+0j) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.2919694865715258e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z5] +
(4.684915095611484e-07+0j) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10] +
(0.0022009640695004533+0j) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.444597854444543e-07+0j) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10] +
(0.0011726348316441796+0j) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.839420915573406e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z6] +
(4.092250616481608e-07+0j) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11] +
(0.002394972639798026+0j) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.092250616481608e-07+0j) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11] +
(0.002394972639798026+0j) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.2362599613934214e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z8] +
(8.649310133292417e-08+0j) [Y0 Z1 Z2 Z3 Y4 Z9] +
(0.003989841456619303+0j) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12] +
(0.0013038004788126938+0j) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12] +
(-6.733197743250526e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z10] +
(0.0022619660624823602+0j) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13] +
(0.0022619660624823602+0j) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13] +
(-5.927453083636187e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z11] +
(1.2393363217829389e-06+0j) [Y0 Z1 Z2 Z3 Y4 Z12] +
(9.30653665268027e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z13] +
(1.8394209155734062e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7] +
(0.00019400857029757307+0j) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0012223378081538466+0j) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(2.3713289480641795e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9] +
(-8.057446596143386e-08+0j) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11] +
(-0.0017278753941369436+0j) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13] +
(-0.000958165583669666+0j) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12] +
(3.086826565149119e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13] +
(-1.8394209155734062e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7] +
(-0.00019400857029757307+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0012223378081538466+0j) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-2.3713289480641795e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9] +
(8.057446596143386e-08+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11] +
(0.0017278753941369436+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13] +
(-0.000958165583669666+0j) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12] +
(-3.086826565149119e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13] +
(1.2004287494859707e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12] +
(0.042743277013782374+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6] +
(0.0005192743499487784+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(-1.8505641930257292e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005192743499487784+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(-1.8505641930257292e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0027790267990256004+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7] +
(0.004636976661182546+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8] +
(0.0012803060973496407+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9] +
(1.071728218553196e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12] +
(2.312094305421145e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12] +
(0.005379937155839342+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10] +
(7.246974426187344e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13] +
(7.246974426187344e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13] +
(0.00524153538280387+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11] +
(0.004311038507914298+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12] +
(0.0010435246534907373+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13] +
(0.003356670563832906+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9] +
(0.00013840177303547186+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11] +
(6.175246207634148e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(-4.997018422736259e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12] +
(0.003267513854423561+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13] +
(-0.003356670563832906+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9] +
(-0.00013840177303547186+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11] +
(-6.175246207634148e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(-4.997018422736259e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12] +
(-0.003267513854423561+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13] +
(0.0038764708993369733+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(-7.540341414466275e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.0038764708993369733+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(-7.540341414466275e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(0.07165035181002433+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0021413612231016457+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(0.004220813970046442+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(0.0012366478019244917+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(0.0029841661681219503+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-0.0029841661681219503+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-1.3987009014503855e-05+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(8.949476488676168e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-6.876621658535399e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(-7.661347213519868e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(0.0015324835230729771+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10] +
(-2.904599884093951e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(0.00540895442240995+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10] +
(-1.0444941298560226e-06+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(0.004767272188278078+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10] +
(-8.105515037511588e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(0.005286546538226857+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10] +
(-9.956079230537316e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.001609531381721358+0j) [Y0 Z1 Z2 Z3 Z4 Y6] +
(-7.14162522116003e-05+0j) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10] +
(-2.6667317545432573e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0024629170071339144+0j) [Y0 Z1 Z2 Z3 Z5 Y6] +
(0.0007156734248908554+0j) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10] +
(-3.0767325319160605e-07+0j) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(3.6060718679074047e-07+0j) [Y0 Z1 Z2 Y4] +
(0.0039615607924965+0j) [Y0 Z1 Z2 Z4 Z5 Y6] +
(0.00018787053389542632+0j) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.6569309320508123e-07+0j) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.7379332623569244e-07+0j) [Y0 Z1 Z3 Y4] +
(0.0016676041811440462+0j) [Y0 Z1 Z3 Z4 Z5 Y6] +
(-0.001452884321416961+0j) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(4.670402391261005e-07+0j) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.10433064780651333+0j) [Y0 Y2] +
(3.117447946206898e-06+0j) [Y0 Z2 Z3 Y4] +
(0.045879470781298046+0j) [Y0 Z2 Z3 Z4 Z5 Y6] +
(0.058591988733861886+0j) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-1.1463061453953919e-05+0j) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(12.412630742111755+0j) [Z0] +
(0.10433064780651331+0j) [Z0 X1 Z2 X3] +
(3.1174479462068976e-06+0j) [Z0 X1 Z2 Z3 Z4 X5] +
(0.045879470781298046+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7] +
(0.058591988733861886+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-1.1463061453953919e-05+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.10433064780651331+0j) [Z0 Y1 Z2 Y3] +
(3.1174479462068976e-06+0j) [Z0 Y1 Z2 Z3 Z4 Y5] +
(0.045879470781298046+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7] +
(0.058591988733861886+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.1463061453953919e-05+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.1861763734860473+0j) [Z0 Z1] +
(-8.33774675611155e-07+0j) [Z0 X2 Z3 X4] +
(-0.027115036845273083+0j) [Z0 X2 Z3 Z4 Z5 X6] +
(-0.06752385099214048+0j) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.4017109736576185e-05+0j) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-8.33774675611155e-07+0j) [Z0 Y2 Z3 Y4] +
(-0.027115036845273083+0j) [Z0 Y2 Z3 Z4 Z5 Y6] +
(-0.06752385099214048+0j) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.4017109736576185e-05+0j) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.23671080783830328+0j) [Z0 Z2] +
(-1.1908508085428505e-06+0j) [Z0 X3 Z4 X5] +
(-0.032767657823290414+0j) [Z0 X3 Z4 Z5 Z6 X7] +
(-0.07635021950635029+0j) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.580960369442591e-05+0j) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.1908508085428505e-06+0j) [Z0 Y3 Z4 Y5] +
(-0.032767657823290414+0j) [Z0 Y3 Z4 Z5 Z6 Y7] +
(-0.07635021950635029+0j) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.580960369442591e-05+0j) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.2512944567459158+0j) [Z0 Z3] +
(-3.0993492437639624e-06+0j) [Z0 X4 Z5 X6] +
(-1.531680879689693e-05+0j) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0868473758986361+0j) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-3.0993492437639624e-06+0j) [Z0 Y4 Z5 Y6] +
(-1.531680879689693e-05+0j) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0868473758986361+0j) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.19661770890342095+0j) [Z0 Z4] +
(-3.344081556656205e-06+0j) [Z0 X5 Z6 X7] +
(-1.610358530739218e-05+0j) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.09065144207036463+0j) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.344081556656205e-06+0j) [Z0 Y5 Z6 Y7] +
(-1.610358530739218e-05+0j) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.09065144207036463+0j) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.19936354537360776+0j) [Z0 Z5] +
(0.05608468124661322+0j) [Z0 X6 Z7 Z8 Z9 X10] +
(-6.652209670273944e-06+0j) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.05608468124661322+0j) [Z0 Y6 Z7 Z8 Z9 Y10] +
(-6.652209670273944e-06+0j) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.241646639360172+0j) [Z0 Z6] +
(0.05600733087780729+0j) [Z0 X7 Z8 Z9 Z10 X11] +
(-6.48185183472735e-06+0j) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.05600733087780729+0j) [Z0 Y7 Z8 Z9 Z10 Y11] +
(-6.48185183472735e-06+0j) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.2485348337131426+0j) [Z0 Z7] +
(0.2723251830660566+0j) [Z0 Z8] +
(0.27883454426723386+0j) [Z0 Z9] +
(-2.177664605269155e-06+0j) [Z0 X10 Z11 X12] +
(-2.177664605269155e-06+0j) [Z0 Y10 Z11 Y12] +
(0.19299723935364227+0j) [Z0 Z10] +
(-1.6148794140854437e-06+0j) [Z0 X11 Z12 X13] +
(-1.6148794140854437e-06+0j) [Z0 Y11 Z12 Y13] +
(0.20072866460441757+0j) [Z0 Z11] +
(0.211026598497915+0j) [Z0 Z12] +
(0.21631037498631794+0j) [Z0 Z13] +
(1.9332412771451438e-07+0j) [X1 X2 Y3 Y4] +
(0.0022939566113524545+0j) [X1 X2 Y3 Z4 Z5 Y6] +
(0.0016407548553123872+0j) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-3.0134714592101927e-07+0j) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004424855449441844+0j) [X1 X2 X4 X5] +
(-8.091637199943361e-07+0j) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.005733569747311866+0j) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-4.52338967845474e-07+0j) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.0034841573002178704+0j) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004684903388155202+0j) [X1 X2 X6 X7] +
(0.005114473831660366+0j) [X1 X2 X6 Z7 Z8 Z9 Z10 X11] +
(-7.560692465694491e-07+0j) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.004668620318776289+0j) [X1 X2 Y7 Z8 Z9 Y10] +
(-7.189990976091752e-07+0j) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381015+0j) [X1 X2 X8 X9] +
(-0.001799219493663014+0j) [X1 X2 X10 X11] +
(-5.287660625118364e-07+0j) [X1 X2 X10 Z11 Z12 X13] +
(-5.471647745154932e-07+0j) [X1 X2 Y11 Y12] +
(-0.00457500762663919+0j) [X1 X2 X12 X13] +
(-1.9332412771451438e-07+0j) [X1 Y2 Y3 X4] +
(-0.0022939566113524545+0j) [X1 Y2 Y3 Z4 Z5 X6] +
(-0.0016407548553123872+0j) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(3.0134714592101927e-07+0j) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441844+0j) [X1 Y2 Y4 X5] +
(-8.091637199943361e-07+0j) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.005733569747311866+0j) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.52338967845474e-07+0j) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10] +
(0.0034841573002178704+0j) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004684903388155202+0j) [X1 Y2 Y6 X7] +
(0.005114473831660366+0j) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11] +
(-7.560692465694491e-07+0j) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.004668620318776289+0j) [X1 Y2 Y7 Z8 Z9 X10] +
(7.189990976091752e-07+0j) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381015+0j) [X1 Y2 Y8 X9] +
(-0.001799219493663014+0j) [X1 Y2 Y10 X11] +
(-5.287660625118364e-07+0j) [X1 Y2 Y10 Z11 Z12 X13] +
(5.471647745154932e-07+0j) [X1 Y2 Y11 X12] +
(-0.00457500762663919+0j) [X1 Y2 Y12 X13] +
(0.12507032579771682+0j) [X1 Z2 X3] +
(-1.3807781481303515e-07+0j) [X1 Z2 X3 X4 Z5 X6] +
(-3.376739308778351e-07+0j) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0018638942824587006+0j) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.3807781481303515e-07+0j) [X1 Z2 X3 Y4 Z5 Y6] +
(-3.376739308778351e-07+0j) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0018638942824587006+0j) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0011059037691896003+0j) [X1 Z2 X3 Z4] +
(-1.5510539177035902e-07+0j) [X1 Z2 X3 X5 Z6 X7] +
(-1.1468376508721707e-06+0j) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.007597464029770566+0j) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.3807781481303515e-07+0j) [X1 Z2 X3 Y5 Z6 Y7] +
(-7.90012898723309e-07+0j) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0053480515826765706+0j) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005530759218631444+0j) [X1 Z2 X3 Z5] +
(0.0005940221543005044+0j) [X1 Z2 X3 X6 Z7 Z8 Z9 X10] +
(-8.379773245074612e-08+0j) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0005940221543005044+0j) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10] +
(-8.379773245074612e-08+0j) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0033476175306660924+0j) [X1 Z2 X3 Z6] +
(0.005708495985960871+0j) [X1 Z2 X3 X7 Z8 Z9 Z10 X11] +
(-8.352332104000157e-07+0j) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
1.9742253796755265e-08j [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005262642473076793+0j) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11] +
(-8.074305986801008e-07+0j) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.008032520918821295+0j) [X1 Z2 X3 Z7] +
(0.002929768674750943+0j) [X1 Z2 X3 Z8] +
(0.011055020596131957+0j) [X1 Z2 X3 Z9] +
(-1.107632560018249e-07+0j) [X1 Z2 X3 X10 Z11 X12] +
(-1.107632560018249e-07+0j) [X1 Z2 X3 Y10 Z11 Y12] +
(0.0017560707018411516+0j) [X1 Z2 X3 Z10] +
(-6.41829157513796e-07+0j) [X1 Z2 X3 X11 Z12 X13] +
(-6.556281915171837e-07+0j) [X1 Z2 X3 Y11 Z12 Y13] +
(0.0035552901955041654+0j) [X1 Z2 X3 Z11] +
(0.0023262306231579934+0j) [X1 Z2 X3 Z12] +
(0.0069012382497971834+0j) [X1 Z2 X3 Z13] +
(-3.5682475214886193e-07+0j) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0022494124470939956+0j) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.047471655639631e-08+0j) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0004458535128840776+0j) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11] +
(-1.974225379610358e-08+0j) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0008533856254125567+0j) [X1 Z2 Z3 X4 Y5 Y6] +
(-0.0007870896771024556+0j) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10] +
(-1.8394209155734062e-07+0j) [X1 Z2 Z3 X4 X6 X7] +
(-0.0012223378081538466+0j) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.00019400857029757307+0j) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12] +
(-2.3713289480641795e-07+0j) [X1 Z2 Z3 X4 X8 X9] +
(8.057446596143386e-08+0j) [X1 Z2 Z3 X4 X10 X11] +
(-0.000958165583669666+0j) [X1 Z2 Z3 X4 X10 Z11 Z12 X13] +
(0.0017278753941369436+0j) [X1 Z2 Z3 X4 Y11 Y12] +
(-3.086826565149119e-07+0j) [X1 Z2 Z3 X4 X12 X13] +
(0.0008533856254125567+0j) [X1 Z2 Z3 Y4 Y5 X6] +
(0.0007870896771024556+0j) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10] +
(-1.8394209155734062e-07+0j) [X1 Z2 Z3 Y4 Y6 X7] +
(-0.0012223378081538466+0j) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.00019400857029757307+0j) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12] +
(-2.3713289480641795e-07+0j) [X1 Z2 Z3 Y4 Y8 X9] +
(8.057446596143386e-08+0j) [X1 Z2 Z3 Y4 Y10 X11] +
(-0.000958165583669666+0j) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13] +
(-0.0017278753941369436+0j) [X1 Z2 Z3 Y4 Y11 X12] +
(-3.086826565149119e-07+0j) [X1 Z2 Z3 Y4 Y12 X13] +
(3.2020768796847125e-06+0j) [X1 Z2 Z3 Z4 X5] +
(4.092250616481608e-07+0j) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10] +
(0.002394972639798026+0j) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.092250616481608e-07+0j) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10] +
(0.002394972639798026+0j) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.444597854444543e-07+0j) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11] +
(0.0011726348316441796+0j) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.684915095611484e-07+0j) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11] +
(0.0022009640695004533+0j) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.839420915573406e-07+0j) [X1 Z2 Z3 Z4 X5 Z7] +
(8.649310133292417e-08+0j) [X1 Z2 Z3 Z4 X5 Z8] +
(3.2362599613934214e-07+0j) [X1 Z2 Z3 Z4 X5 Z9] +
(0.0022619660624823602+0j) [X1 Z2 Z3 Z4 X5 X10 Z11 X12] +
(0.0022619660624823602+0j) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12] +
(-5.927453083636187e-07+0j) [X1 Z2 Z3 Z4 X5 Z10] +
(0.0013038004788126938+0j) [X1 Z2 Z3 Z4 X5 X11 Z12 X13] +
(0.003989841456619303+0j) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13] +
(-6.733197743250526e-07+0j) [X1 Z2 Z3 Z4 X5 Z11] +
(9.30653665268027e-07+0j) [X1 Z2 Z3 Z4 X5 Z12] +
(1.2393363217829389e-06+0j) [X1 Z2 Z3 Z4 X5 Z13] +
(-0.0010283292378562734+0j) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0026860409778066085+0j) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13] +
(-0.0005192743499487783+0j) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10] +
(1.8505641930257292e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.003356670563832906+0j) [X1 Z2 Z3 Z4 Z5 X6 X8 X9] +
(-0.00013840177303547186+0j) [X1 Z2 Z3 Z4 Z5 X6 X10 X11] +
(-4.997018422736259e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13] +
(-6.175246207634148e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12] +
(-0.0032675138544235607+0j) [X1 Z2 Z3 Z4 Z5 X6 X12 X13] +
(0.0005192743499487783+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10] +
(-1.8505641930257292e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.003356670563832906+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9] +
(-0.00013840177303547186+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11] +
(-4.997018422736259e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13] +
(6.175246207634148e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12] +
(-0.0032675138544235607+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13] +
(0.04274327701378236+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7] +
(0.0012803060973496407+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8] +
(0.004636976661182546+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9] +
(7.246974426187344e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12] +
(7.246974426187344e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12] +
(0.00524153538280387+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10] +
(2.312094305421145e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13] +
(1.071728218553196e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13] +
(0.005379937155839342+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11] +
(0.0010435246534907373+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12] +
(0.004311038507914298+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13] +
(1.2004287494859707e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13] +
(-0.003876470899336973+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10] +
(7.540341414466275e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(0.003876470899336973+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10] +
(-7.540341414466275e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(-0.0029841661681219507+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-0.0029841661681219507+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(0.07165035181002441+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0012366478019244917+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(0.004220813970046442+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(-1.3987009014503855e-05+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(8.949476488676168e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(-7.661347213519868e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(-0.0021413612231016453+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11] +
(-6.876621658535399e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(0.00540895442240995+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11] +
(-1.0444941298560226e-06+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(0.0015324835230729771+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11] +
(-2.904599884093951e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(0.005286546538226857+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11] +
(-9.956079230537316e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0027790267990256004+0j) [X1 Z2 Z3 Z4 Z5 X7] +
(0.004767272188278078+0j) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11] +
(-8.105515037511588e-07+0j) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0024629170071339144+0j) [X1 Z2 Z3 Z4 Z6 X7] +
(0.0007156734248908554+0j) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11] +
(-3.0767325319160605e-07+0j) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2919694865715258e-07+0j) [X1 Z2 Z3 X5] +
(0.001609531381721358+0j) [X1 Z2 Z3 Z5 Z6 X7] +
(-7.14162522116003e-05+0j) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-2.6667317545432573e-07+0j) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.7379332623569244e-07+0j) [X1 Z2 Z4 X5] +
(0.0016676041811440462+0j) [X1 Z2 Z4 Z5 Z6 X7] +
(-0.001452884321416961+0j) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(4.670402391261005e-07+0j) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0032769719312315095+0j) [X1 X3] +
(3.6060718679074047e-07+0j) [X1 Z3 Z4 X5] +
(0.0039615607924965+0j) [X1 Z3 Z4 Z5 Z6 X7] +
(0.00018787053389542632+0j) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.6569309320508123e-07+0j) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.9332412771451438e-07+0j) [Y1 X2 X3 Y4] +
(-0.0022939566113524545+0j) [Y1 X2 X3 Z4 Z5 Y6] +
(-0.0016407548553123872+0j) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(3.0134714592101927e-07+0j) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004424855449441844+0j) [Y1 X2 X4 Y5] +
(-8.091637199943361e-07+0j) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.005733569747311866+0j) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(4.52338967845474e-07+0j) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10] +
(0.0034841573002178704+0j) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004684903388155202+0j) [Y1 X2 X6 Y7] +
(0.005114473831660366+0j) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11] +
(-7.560692465694491e-07+0j) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.004668620318776289+0j) [Y1 X2 X7 Z8 Z9 Y10] +
(7.189990976091752e-07+0j) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381015+0j) [Y1 X2 X8 Y9] +
(-0.001799219493663014+0j) [Y1 X2 X10 Y11] +
(-5.287660625118364e-07+0j) [Y1 X2 X10 Z11 Z12 Y13] +
(5.471647745154932e-07+0j) [Y1 X2 X11 Y12] +
(-0.00457500762663919+0j) [Y1 X2 X12 Y13] +
(1.9332412771451438e-07+0j) [Y1 Y2 X3 X4] +
(0.0022939566113524545+0j) [Y1 Y2 X3 Z4 Z5 X6] +
(0.0016407548553123872+0j) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-3.0134714592101927e-07+0j) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441844+0j) [Y1 Y2 Y4 Y5] +
(-8.091637199943361e-07+0j) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.005733569747311866+0j) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-4.52338967845474e-07+0j) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10] +
(-0.0034841573002178704+0j) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004684903388155202+0j) [Y1 Y2 Y6 Y7] +
(0.005114473831660366+0j) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11] +
(-7.560692465694491e-07+0j) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.004668620318776289+0j) [Y1 Y2 X7 Z8 Z9 X10] +
(-7.189990976091752e-07+0j) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381015+0j) [Y1 Y2 Y8 Y9] +
(-0.001799219493663014+0j) [Y1 Y2 Y10 Y11] +
(-5.287660625118364e-07+0j) [Y1 Y2 Y10 Z11 Z12 Y13] +
(-5.471647745154932e-07+0j) [Y1 Y2 X11 X12] +
(-0.00457500762663919+0j) [Y1 Y2 Y12 Y13] +
(-3.5682475214886193e-07+0j) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0022494124470939956+0j) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0004458535128840776+0j) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11] +
(-1.974225379610358e-08+0j) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.047471655639631e-08+0j) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.12507032579771682+0j) [Y1 Z2 Y3] +
(-1.3807781481303515e-07+0j) [Y1 Z2 Y3 X4 Z5 X6] +
(-3.376739308778351e-07+0j) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0018638942824587006+0j) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.3807781481303515e-07+0j) [Y1 Z2 Y3 Y4 Z5 Y6] +
(-3.376739308778351e-07+0j) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0018638942824587006+0j) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0011059037691896003+0j) [Y1 Z2 Y3 Z4] +
(-1.3807781481303515e-07+0j) [Y1 Z2 Y3 X5 Z6 X7] +
(-7.90012898723309e-07+0j) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0053480515826765706+0j) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.5510539177035902e-07+0j) [Y1 Z2 Y3 Y5 Z6 Y7] +
(-1.1468376508721707e-06+0j) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.007597464029770566+0j) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005530759218631444+0j) [Y1 Z2 Y3 Z5] +
(0.0005940221543005044+0j) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10] +
(-8.379773245074612e-08+0j) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0005940221543005044+0j) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10] +
(-8.379773245074612e-08+0j) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0033476175306660924+0j) [Y1 Z2 Y3 Z6] +
(0.005262642473076793+0j) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11] +
(-8.074305986801008e-07+0j) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005708495985960871+0j) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11] +
-1.9742253796755265e-08j [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.352332104000157e-07+0j) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.008032520918821295+0j) [Y1 Z2 Y3 Z7] +
(0.002929768674750943+0j) [Y1 Z2 Y3 Z8] +
(0.011055020596131957+0j) [Y1 Z2 Y3 Z9] +
(-1.107632560018249e-07+0j) [Y1 Z2 Y3 X10 Z11 X12] +
(-1.107632560018249e-07+0j) [Y1 Z2 Y3 Y10 Z11 Y12] +
(0.0017560707018411516+0j) [Y1 Z2 Y3 Z10] +
(-6.556281915171837e-07+0j) [Y1 Z2 Y3 X11 Z12 X13] +
(-6.41829157513796e-07+0j) [Y1 Z2 Y3 Y11 Z12 Y13] +
(0.0035552901955041654+0j) [Y1 Z2 Y3 Z11] +
(0.0023262306231579934+0j) [Y1 Z2 Y3 Z12] +
(0.0069012382497971834+0j) [Y1 Z2 Y3 Z13] +
(0.0008533856254125567+0j) [Y1 Z2 Z3 X4 X5 Y6] +
(0.0007870896771024556+0j) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10] +
(-1.8394209155734062e-07+0j) [Y1 Z2 Z3 X4 X6 Y7] +
(-0.0012223378081538466+0j) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.00019400857029757307+0j) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12] +
(-2.3713289480641795e-07+0j) [Y1 Z2 Z3 X4 X8 Y9] +
(8.057446596143386e-08+0j) [Y1 Z2 Z3 X4 X10 Y11] +
(-0.000958165583669666+0j) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13] +
(-0.0017278753941369436+0j) [Y1 Z2 Z3 X4 X11 Y12] +
(-3.086826565149119e-07+0j) [Y1 Z2 Z3 X4 X12 Y13] +
(-0.0008533856254125567+0j) [Y1 Z2 Z3 Y4 X5 X6] +
(-0.0007870896771024556+0j) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10] +
(-1.8394209155734062e-07+0j) [Y1 Z2 Z3 Y4 Y6 Y7] +
(-0.0012223378081538466+0j) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.00019400857029757307+0j) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12] +
(-2.3713289480641795e-07+0j) [Y1 Z2 Z3 Y4 Y8 Y9] +
(8.057446596143386e-08+0j) [Y1 Z2 Z3 Y4 Y10 Y11] +
(-0.000958165583669666+0j) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13] +
(0.0017278753941369436+0j) [Y1 Z2 Z3 Y4 X11 X12] +
(-3.086826565149119e-07+0j) [Y1 Z2 Z3 Y4 Y12 Y13] +
(-0.0010283292378562734+0j) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0026860409778066085+0j) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13] +
(3.2020768796847125e-06+0j) [Y1 Z2 Z3 Z4 Y5] +
(4.092250616481608e-07+0j) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10] +
(0.002394972639798026+0j) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.092250616481608e-07+0j) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10] +
(0.002394972639798026+0j) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.684915095611484e-07+0j) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11] +
(0.0022009640695004533+0j) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.444597854444543e-07+0j) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11] +
(0.0011726348316441796+0j) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.839420915573406e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z7] +
(8.649310133292417e-08+0j) [Y1 Z2 Z3 Z4 Y5 Z8] +
(3.2362599613934214e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z9] +
(0.0022619660624823602+0j) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12] +
(0.0022619660624823602+0j) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12] +
(-5.927453083636187e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z10] +
(0.003989841456619303+0j) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13] +
(0.0013038004788126938+0j) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13] +
(-6.733197743250526e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z11] +
(9.30653665268027e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z12] +
(1.2393363217829389e-06+0j) [Y1 Z2 Z3 Z4 Y5 Z13] +
(0.0005192743499487783+0j) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10] +
(-1.8505641930257292e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.003356670563832906+0j) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9] +
(-0.00013840177303547186+0j) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11] +
(-4.997018422736259e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13] +
(6.175246207634148e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12] +
(-0.0032675138544235607+0j) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13] +
(-0.0005192743499487783+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10] +
(1.8505641930257292e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-0.003356670563832906+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9] +
(-0.00013840177303547186+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11] +
(-4.997018422736259e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13] +
(-6.175246207634148e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12] +
(-0.0032675138544235607+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13] +
(1.2004287494859707e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13] +
(0.04274327701378236+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7] +
(0.0012803060973496407+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8] +
(0.004636976661182546+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9] +
(7.246974426187344e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12] +
(7.246974426187344e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12] +
(0.00524153538280387+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10] +
(1.071728218553196e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13] +
(2.312094305421145e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13] +
(0.005379937155839342+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11] +
(0.0010435246534907373+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12] +
(0.004311038507914298+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13] +
(0.003876470899336973+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10] +
(-7.540341414466275e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-0.003876470899336973+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10] +
(7.540341414466275e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-0.0029841661681219507+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(-0.0029841661681219507+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(0.07165035181002441+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.0012366478019244917+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(0.004220813970046442+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(-1.3987009014503855e-05+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(8.949476488676168e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(-7.661347213519868e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(-0.0021413612231016453+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11] +
(-6.876621658535399e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(0.00540895442240995+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11] +
(-1.0444941298560226e-06+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(0.0015324835230729771+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11] +
(-2.904599884093951e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(0.005286546538226857+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11] +
(-9.956079230537316e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0027790267990256004+0j) [Y1 Z2 Z3 Z4 Z5 Y7] +
(0.004767272188278078+0j) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11] +
(-8.105515037511588e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0024629170071339144+0j) [Y1 Z2 Z3 Z4 Z6 Y7] +
(0.0007156734248908554+0j) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11] +
(-3.0767325319160605e-07+0j) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.2919694865715258e-07+0j) [Y1 Z2 Z3 Y5] +
(0.001609531381721358+0j) [Y1 Z2 Z3 Z5 Z6 Y7] +
(-7.14162522116003e-05+0j) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-2.6667317545432573e-07+0j) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.7379332623569244e-07+0j) [Y1 Z2 Z4 Y5] +
(0.0016676041811440462+0j) [Y1 Z2 Z4 Z5 Z6 Y7] +
(-0.001452884321416961+0j) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(4.670402391261005e-07+0j) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0032769719312315095+0j) [Y1 Y3] +
(3.6060718679074047e-07+0j) [Y1 Z3 Z4 Y5] +
(0.0039615607924965+0j) [Y1 Z3 Z4 Z5 Z6 Y7] +
(0.00018787053389542632+0j) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.6569309320508123e-07+0j) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(12.412630742111755+0j) [Z1] +
(-1.1908508085428505e-06+0j) [Z1 X2 Z3 X4] +
(-0.032767657823290414+0j) [Z1 X2 Z3 Z4 Z5 X6] +
(-0.07635021950635029+0j) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.580960369442591e-05+0j) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.1908508085428505e-06+0j) [Z1 Y2 Z3 Y4] +
(-0.032767657823290414+0j) [Z1 Y2 Z3 Z4 Z5 Y6] +
(-0.07635021950635029+0j) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.580960369442591e-05+0j) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.2512944567459158+0j) [Z1 Z2] +
(-8.33774675611155e-07+0j) [Z1 X3 Z4 X5] +
(-0.027115036845273083+0j) [Z1 X3 Z4 Z5 Z6 X7] +
(-0.06752385099214048+0j) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.4017109736576185e-05+0j) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.33774675611155e-07+0j) [Z1 Y3 Z4 Y5] +
(-0.027115036845273083+0j) [Z1 Y3 Z4 Z5 Z6 Y7] +
(-0.06752385099214048+0j) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.4017109736576185e-05+0j) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.23671080783830328+0j) [Z1 Z3] +
(-3.344081556656205e-06+0j) [Z1 X4 Z5 X6] +
(-1.610358530739218e-05+0j) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.09065144207036463+0j) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-3.344081556656205e-06+0j) [Z1 Y4 Z5 Y6] +
(-1.610358530739218e-05+0j) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.09065144207036463+0j) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.19936354537360776+0j) [Z1 Z4] +
(-3.0993492437639624e-06+0j) [Z1 X5 Z6 X7] +
(-1.531680879689693e-05+0j) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0868473758986361+0j) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.0993492437639624e-06+0j) [Z1 Y5 Z6 Y7] +
(-1.531680879689693e-05+0j) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0868473758986361+0j) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.19661770890342095+0j) [Z1 Z5] +
(0.05600733087780729+0j) [Z1 X6 Z7 Z8 Z9 X10] +
(-6.48185183472735e-06+0j) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.05600733087780729+0j) [Z1 Y6 Z7 Z8 Z9 Y10] +
(-6.48185183472735e-06+0j) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.2485348337131426+0j) [Z1 Z6] +
(0.05608468124661322+0j) [Z1 X7 Z8 Z9 Z10 X11] +
(-6.652209670273944e-06+0j) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.05608468124661322+0j) [Z1 Y7 Z8 Z9 Z10 Y11] +
(-6.652209670273944e-06+0j) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.241646639360172+0j) [Z1 Z7] +
(0.27883454426723386+0j) [Z1 Z8] +
(0.2723251830660566+0j) [Z1 Z9] +
(-1.6148794140854437e-06+0j) [Z1 X10 Z11 X12] +
(-1.6148794140854437e-06+0j) [Z1 Y10 Z11 Y12] +
(0.20072866460441757+0j) [Z1 Z10] +
(-2.177664605269155e-06+0j) [Z1 X11 Z12 X13] +
(-2.177664605269155e-06+0j) [Z1 Y11 Z12 Y13] +
(0.19299723935364227+0j) [Z1 Z11] +
(0.21631037498631794+0j) [Z1 Z12] +
(0.211026598497915+0j) [Z1 Z13] +
(-0.03583956795335346+0j) [X2 X3 Y4 Y5] +
(-2.1990516193966083e-07+0j) [X2 X3 Y4 Z5 Z6 Y7] +
(-2.3609563206114886e-06+0j) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.010311482489831634+0j) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.1990516193966083e-07+0j) [X2 X3 X5 X6] +
(-2.3609563206114886e-06+0j) [X2 X3 X5 Z6 Z7 Z8 Z9 X10] +
(-0.010311482489831634+0j) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.031143817988967006+0j) [X2 X3 Y6 Y7] +
(0.0053686593581093985+0j) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11] +
(9.20935064316753e-08+0j) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0053686593581093985+0j) [X2 X3 X7 Z8 Z9 X10] +
(9.20935064316753e-08+0j) [X2 X3 X7 Z8 Z9 Z10 Z11 X12] +
(-0.036194123559042446+0j) [X2 X3 Y8 Y9] +
(-0.025384657508457482+0j) [X2 X3 Y10 Y11] +
(2.1726691017466324e-06+0j) [X2 X3 Y10 Z11 Z12 Y13] +
(2.1726691017466324e-06+0j) [X2 X3 X11 X12] +
(-0.01557720806397644+0j) [X2 X3 Y12 Y13] +
(0.03583956795335346+0j) [X2 Y3 Y4 X5] +
(2.1990516193966083e-07+0j) [X2 Y3 Y4 Z5 Z6 X7] +
(2.3609563206114886e-06+0j) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.010311482489831634+0j) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.1990516193966083e-07+0j) [X2 Y3 Y5 X6] +
(-2.3609563206114886e-06+0j) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.010311482489831634+0j) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.031143817988967006+0j) [X2 Y3 Y6 X7] +
(-0.0053686593581093985+0j) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11] +
(-9.20935064316753e-08+0j) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0053686593581093985+0j) [X2 Y3 Y7 Z8 Z9 X10] +
(9.20935064316753e-08+0j) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12] +
(0.036194123559042446+0j) [X2 Y3 Y8 X9] +
(0.025384657508457482+0j) [X2 Y3 Y10 X11] +
(-2.1726691017466324e-06+0j) [X2 Y3 Y10 Z11 Z12 X13] +
(2.1726691017466324e-06+0j) [X2 Y3 Y11 X12] +
(0.01557720806397644+0j) [X2 Y3 Y12 X13] +
(-3.887051674447983e-06+0j) [X2 Z3 X4] +
(-0.005143391768825049+0j) [X2 Z3 X4 X5 Z6 X7] +
(-0.009841749246962583+0j) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.988511706512941e-06+0j) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.005143391768825049+0j) [X2 Z3 X4 Y5 Z6 Y7] +
(-0.009841749246962583+0j) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.988511706512941e-06+0j) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.764994120667342e-07+0j) [X2 Z3 X4 Z5] +
(1.6893489516427878e-06+0j) [X2 Z3 X4 X6 Z7 Z8 Z9 X10] +
(0.0107575639539089+0j) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.537178095785608e-08+0j) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10] +
(4.205548411219569e-05+0j) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.593534390545774e-07+0j) [X2 Z3 X4 Z6] +
(3.211842019379424e-06+0j) [X2 Z3 X4 X7 Z8 Z9 Z10 X11] +
(0.019299560579363755+0j) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.211842019379424e-06+0j) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11] +
(0.019299560579363755+0j) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.1954890100550876e-06+0j) [X2 Z3 X4 Z7] +
(2.1868423771003565e-07+0j) [X2 Z3 X4 Z8] +
(-5.770052996073175e-07+0j) [X2 Z3 X4 Z9] +
(0.015588250102380201+0j) [X2 Z3 X4 X10 Z11 X12] +
(0.005324835234221652+0j) [X2 Z3 X4 Y10 Z11 Y12] +
(-3.15865643239749e-06+0j) [X2 Z3 X4 Z10] +
(0.024353077678069053+0j) [X2 Z3 X4 X11 Z12 X13] +
(0.024353077678069053+0j) [X2 Z3 X4 Y11 Z12 Y13] +
(-7.801707501317178e-06+0j) [X2 Z3 X4 Z11] +
(3.5390541847828075e-06+0j) [X2 Z3 X4 Z12] +
(8.814937307474002e-06+0j) [X2 Z3 X4 Z13] +
(1.6288532436995536e-06+0j) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10] +
(0.0107155084697967+0j) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.010263414868158549+0j) [X2 Z3 Y4 Y10 Z11 X12] +
(-1.454842449109665e-06+0j) [X2 Z3 Z4 X5 Y6 Y7] +
(-3.1513463114361887e-06+0j) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11] +
(-0.01925750509525156+0j) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.522493067736636e-06+0j) [X2 Z3 Z4 X5 X7 Z8 Z9 X10] +
(-0.008541996625454858+0j) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-7.956895373173531e-07+0j) [X2 Z3 Z4 X5 Y8 Y9] +
(-4.6430510689196885e-06+0j) [X2 Z3 Z4 X5 Y10 Y11] +
(-0.019028242443847404+0j) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13] +
(-0.008764827575688859+0j) [X2 Z3 Z4 X5 X11 X12] +
(5.2758831226911934e-06+0j) [X2 Z3 Z4 X5 Y12 Y13] +
(1.454842449109665e-06+0j) [X2 Z3 Z4 Y5 Y6 X7] +
(3.1513463114361887e-06+0j) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11] +
(0.01925750509525156+0j) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.522493067736636e-06+0j) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10] +
(-0.008541996625454858+0j) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(7.956895373173531e-07+0j) [X2 Z3 Z4 Y5 Y8 X9] +
(4.6430510689196885e-06+0j) [X2 Z3 Z4 Y5 Y10 X11] +
(0.019028242443847404+0j) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13] +
(-0.008764827575688859+0j) [X2 Z3 Z4 Y5 Y11 X12] +
(-5.2758831226911934e-06+0j) [X2 Z3 Z4 Y5 Y12 X13] +
(-0.12133276911042422+0j) [X2 Z3 Z4 Z5 X6] +
(-0.008469978791023788+0j) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(2.686381546503742e-07+0j) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.008469978791023788+0j) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(2.686381546503742e-07+0j) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.02599617759802131+0j) [X2 Z3 Z4 Z5 X6 Z7] +
(-0.005805188989826881+0j) [X2 Z3 Z4 Z5 X6 Z8] +
(-0.01756120240964616+0j) [X2 Z3 Z4 Z5 X6 Z9] +
(-7.988770288869174e-07+0j) [X2 Z3 Z4 Z5 X6 X10 Z11 X12] +
(-3.427323108883532e-07+0j) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12] +
(-0.0008145313270957019+0j) [X2 Z3 Z4 Z5 X6 Z10] +
(2.7455184007945525e-06+0j) [X2 Z3 Z4 Z5 X6 X11 Z12 X13] +
(2.7455184007945525e-06+0j) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13] +
(0.01441109943013092+0j) [X2 Z3 Z4 Z5 X6 Z11] +
(0.0006650070219498812+0j) [X2 Z3 Z4 Z5 X6 Z12] +
(-0.0034937903598901456+0j) [X2 Z3 Z4 Z5 X6 Z13] +
(-4.561447179985641e-07+0j) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12] +
(-0.011756013419819276+0j) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9] +
(0.01522563075722662+0j) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11] +
(-3.0882507116829057e-06+0j) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(-3.54439542968147e-06+0j) [X2 Z3 Z4 Z5 Z6 X7 X11 X12] +
(-0.004158797381840026+0j) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13] +
(0.011756013419819276+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9] +
(-0.01522563075722662+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11] +
(3.0882507116829057e-06+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(-3.54439542968147e-06+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12] +
(0.004158797381840026+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13] +
(0.014603704729162184+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(-2.874299071684717e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.014603704729162184+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(-2.874299071684717e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(-0.2816425776702328+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-1.3002946563348209e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(-1.3002946563348209e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(-0.02428211735469306+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(-0.019538050311314833+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(-0.01709155315589892+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(0.0024464971554159116+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(-0.0024464971554159116+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(5.7759505278920545e-05+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(2.8836765763667134e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(5.146496328001374e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(3.8462016716665535e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(-0.039359168022053276+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10] +
(7.979825794174398e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(-0.02475546329289109+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10] +
(5.105526722489681e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(-0.021433810721601113+0j) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10] +
(5.15935050238745e-06+0j) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(-0.029903789512624904+0j) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10] +
(5.4279886570378245e-06+0j) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0016638798784907843+0j) [X2 Z3 Z4 X6] +
(-0.01888903030494292+0j) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10] +
(2.9473560120308617e-06+0j) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.0034795118903342658+0j) [X2 Z3 Z5 X6] +
(-0.0287307795519055+0j) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10] +
(5.935867718543803e-06+0j) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.6021167406863997e-06+0j) [X2 X4] +
(0.0004956762314917624+0j) [X2 Z4 Z5 X6] +
(-0.03560837898831255+0j) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(7.253273348785306e-06+0j) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.03583956795335346+0j) [Y2 X3 X4 Y5] +
(2.1990516193966083e-07+0j) [Y2 X3 X4 Z5 Z6 Y7] +
(2.3609563206114886e-06+0j) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.010311482489831634+0j) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.1990516193966083e-07+0j) [Y2 X3 X5 Y6] +
(-2.3609563206114886e-06+0j) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.010311482489831634+0j) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.031143817988967006+0j) [Y2 X3 X6 Y7] +
(-0.0053686593581093985+0j) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11] +
(-9.20935064316753e-08+0j) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0053686593581093985+0j) [Y2 X3 X7 Z8 Z9 Y10] +
(9.20935064316753e-08+0j) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12] +
(0.036194123559042446+0j) [Y2 X3 X8 Y9] +
(0.025384657508457482+0j) [Y2 X3 X10 Y11] +
(-2.1726691017466324e-06+0j) [Y2 X3 X10 Z11 Z12 Y13] +
(2.1726691017466324e-06+0j) [Y2 X3 X11 Y12] +
(0.01557720806397644+0j) [Y2 X3 X12 Y13] +
(-0.03583956795335346+0j) [Y2 Y3 X4 X5] +
(-2.1990516193966083e-07+0j) [Y2 Y3 X4 Z5 Z6 X7] +
(-2.3609563206114886e-06+0j) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.010311482489831634+0j) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.1990516193966083e-07+0j) [Y2 Y3 Y5 Y6] +
(-2.3609563206114886e-06+0j) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.010311482489831634+0j) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.031143817988967006+0j) [Y2 Y3 X6 X7] +
(0.0053686593581093985+0j) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11] +
(9.20935064316753e-08+0j) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0053686593581093985+0j) [Y2 Y3 Y7 Z8 Z9 Y10] +
(9.20935064316753e-08+0j) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.036194123559042446+0j) [Y2 Y3 X8 X9] +
(-0.025384657508457482+0j) [Y2 Y3 X10 X11] +
(2.1726691017466324e-06+0j) [Y2 Y3 X10 Z11 Z12 X13] +
(2.1726691017466324e-06+0j) [Y2 Y3 Y11 Y12] +
(-0.01557720806397644+0j) [Y2 Y3 X12 X13] +
(1.6288532436995536e-06+0j) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10] +
(0.0107155084697967+0j) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.010263414868158549+0j) [Y2 Z3 X4 X10 Z11 Y12] +
(-3.887051674447983e-06+0j) [Y2 Z3 Y4] +
(-0.005143391768825049+0j) [Y2 Z3 Y4 X5 Z6 X7] +
(-0.009841749246962583+0j) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.988511706512941e-06+0j) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.005143391768825049+0j) [Y2 Z3 Y4 Y5 Z6 Y7] +
(-0.009841749246962583+0j) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.988511706512941e-06+0j) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.764994120667342e-07+0j) [Y2 Z3 Y4 Z5] +
(4.537178095785608e-08+0j) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10] +
(4.205548411219569e-05+0j) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.6893489516427878e-06+0j) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10] +
(0.0107575639539089+0j) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.593534390545774e-07+0j) [Y2 Z3 Y4 Z6] +
(3.211842019379424e-06+0j) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11] +
(0.019299560579363755+0j) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.211842019379424e-06+0j) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11] +
(0.019299560579363755+0j) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.1954890100550876e-06+0j) [Y2 Z3 Y4 Z7] +
(2.1868423771003565e-07+0j) [Y2 Z3 Y4 Z8] +
(-5.770052996073175e-07+0j) [Y2 Z3 Y4 Z9] +
(0.005324835234221652+0j) [Y2 Z3 Y4 X10 Z11 X12] +
(0.015588250102380201+0j) [Y2 Z3 Y4 Y10 Z11 Y12] +
(-3.15865643239749e-06+0j) [Y2 Z3 Y4 Z10] +
(0.024353077678069053+0j) [Y2 Z3 Y4 X11 Z12 X13] +
(0.024353077678069053+0j) [Y2 Z3 Y4 Y11 Z12 Y13] +
(-7.801707501317178e-06+0j) [Y2 Z3 Y4 Z11] +
(3.5390541847828075e-06+0j) [Y2 Z3 Y4 Z12] +
(8.814937307474002e-06+0j) [Y2 Z3 Y4 Z13] +
(1.454842449109665e-06+0j) [Y2 Z3 Z4 X5 X6 Y7] +
(3.1513463114361887e-06+0j) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11] +
(0.01925750509525156+0j) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.522493067736636e-06+0j) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10] +
(-0.008541996625454858+0j) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(7.956895373173531e-07+0j) [Y2 Z3 Z4 X5 X8 Y9] +
(4.6430510689196885e-06+0j) [Y2 Z3 Z4 X5 X10 Y11] +
(0.019028242443847404+0j) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13] +
(-0.008764827575688859+0j) [Y2 Z3 Z4 X5 X11 Y12] +
(-5.2758831226911934e-06+0j) [Y2 Z3 Z4 X5 X12 Y13] +
(-1.454842449109665e-06+0j) [Y2 Z3 Z4 Y5 X6 X7] +
(-3.1513463114361887e-06+0j) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11] +
(-0.01925750509525156+0j) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.522493067736636e-06+0j) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10] +
(-0.008541996625454858+0j) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895373173531e-07+0j) [Y2 Z3 Z4 Y5 X8 X9] +
(-4.6430510689196885e-06+0j) [Y2 Z3 Z4 Y5 X10 X11] +
(-0.019028242443847404+0j) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13] +
(-0.008764827575688859+0j) [Y2 Z3 Z4 Y5 Y11 Y12] +
(5.2758831226911934e-06+0j) [Y2 Z3 Z4 Y5 X12 X13] +
(-4.561447179985641e-07+0j) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12] +
(-0.12133276911042422+0j) [Y2 Z3 Z4 Z5 Y6] +
(-0.008469978791023788+0j) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(2.686381546503742e-07+0j) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.008469978791023788+0j) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(2.686381546503742e-07+0j) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.02599617759802131+0j) [Y2 Z3 Z4 Z5 Y6 Z7] +
(-0.005805188989826881+0j) [Y2 Z3 Z4 Z5 Y6 Z8] +
(-0.01756120240964616+0j) [Y2 Z3 Z4 Z5 Y6 Z9] +
(-3.427323108883532e-07+0j) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12] +
(-7.988770288869174e-07+0j) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12] +
(-0.0008145313270957019+0j) [Y2 Z3 Z4 Z5 Y6 Z10] +
(2.7455184007945525e-06+0j) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13] +
(2.7455184007945525e-06+0j) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13] +
(0.01441109943013092+0j) [Y2 Z3 Z4 Z5 Y6 Z11] +
(0.0006650070219498812+0j) [Y2 Z3 Z4 Z5 Y6 Z12] +
(-0.0034937903598901456+0j) [Y2 Z3 Z4 Z5 Y6 Z13] +
(0.011756013419819276+0j) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9] +
(-0.01522563075722662+0j) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11] +
(3.0882507116829057e-06+0j) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(-3.54439542968147e-06+0j) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12] +
(0.004158797381840026+0j) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13] +
(-0.011756013419819276+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9] +
(0.01522563075722662+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11] +
(-3.0882507116829057e-06+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(-3.54439542968147e-06+0j) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12] +
(-0.004158797381840026+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13] +
(0.014603704729162184+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(-2.874299071684717e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.014603704729162184+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(-2.874299071684717e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(-0.2816425776702328+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-1.3002946563348209e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(-1.3002946563348209e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(-0.02428211735469306+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(-0.019538050311314833+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(-0.01709155315589892+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(-0.0024464971554159116+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(0.0024464971554159116+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(5.7759505278920545e-05+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.8836765763667134e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(5.146496328001374e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(3.8462016716665535e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(-0.039359168022053276+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10] +
(7.979825794174398e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(-0.02475546329289109+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10] +
(5.105526722489681e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(-0.021433810721601113+0j) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10] +
(5.15935050238745e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(-0.029903789512624904+0j) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10] +
(5.4279886570378245e-06+0j) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0016638798784907843+0j) [Y2 Z3 Z4 Y6] +
(-0.01888903030494292+0j) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10] +
(2.9473560120308617e-06+0j) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.0034795118903342658+0j) [Y2 Z3 Z5 Y6] +
(-0.0287307795519055+0j) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10] +
(5.935867718543803e-06+0j) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.6021167406863997e-06+0j) [Y2 Y4] +
(0.0004956762314917624+0j) [Y2 Z4 Z5 Y6] +
(-0.03560837898831255+0j) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(7.253273348785306e-06+0j) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.6538942226831685+0j) [Z2] +
(1.6021167406863997e-06+0j) [Z2 X3 Z4 X5] +
(0.0004956762314917624+0j) [Z2 X3 Z4 Z5 Z6 X7] +
(-0.03560837898831255+0j) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(7.253273348785306e-06+0j) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.6021167406863997e-06+0j) [Z2 Y3 Z4 Y5] +
(0.0004956762314917624+0j) [Z2 Y3 Z4 Z5 Z6 Y7] +
(-0.03560837898831255+0j) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(7.253273348785306e-06+0j) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.18189085790751286+0j) [Z2 Z3] +
(-9.509249751833101e-07+0j) [Z2 X4 Z5 X6] +
(-4.7288431476609985e-06+0j) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.024591860883829912+0j) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-9.509249751833101e-07+0j) [Z2 Y4 Z5 Y6] +
(-4.7288431476609985e-06+0j) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.024591860883829912+0j) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.1249580773950317+0j) [Z2 Z4] +
(-1.170830137122971e-06+0j) [Z2 X5 Z6 X7] +
(-7.0897994682724884e-06+0j) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.03490334337366154+0j) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.170830137122971e-06+0j) [Z2 Y5 Z6 Y7] +
(-7.0897994682724884e-06+0j) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.03490334337366154+0j) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.16079764534838517+0j) [Z2 Z5] +
(0.0190204231730399+0j) [Z2 X6 Z7 Z8 Z9 X10] +
(-2.1032156049474973e-06+0j) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0190204231730399+0j) [Z2 Y6 Z7 Z8 Z9 Y10] +
(-2.1032156049474973e-06+0j) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.13739104762683194+0j) [Z2 Z6] +
(0.024389082531149298+0j) [Z2 X7 Z8 Z9 Z10 X11] +
(-2.0111220985158223e-06+0j) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.024389082531149298+0j) [Z2 Y7 Z8 Z9 Z10 Y11] +
(-2.0111220985158223e-06+0j) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.16853486561579895+0j) [Z2 Z7] +
(0.1507140812100826+0j) [Z2 Z8] +
(0.18690820476912504+0j) [Z2 Z9] +
(-1.0632283424805205e-06+0j) [Z2 X10 Z11 X12] +
(-1.0632283424805205e-06+0j) [Z2 Y10 Z11 Y12] +
(0.12799502492468384+0j) [Z2 Z10] +
(1.1094407592661122e-06+0j) [Z2 X11 Z12 X13] +
(1.1094407592661122e-06+0j) [Z2 Y11 Z12 Y13] +
(0.15337968243314132+0j) [Z2 Z11] +
(0.14011289865354787+0j) [Z2 Z12] +
(0.1556901067175243+0j) [Z2 Z13] +
(0.00514339176882505+0j) [X3 X4 Y5 Y6] +
(0.009841749246962583+0j) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10] +
(-2.988511706512941e-06+0j) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.454842449109665e-06+0j) [X3 X4 X6 X7] +
(-1.522493067736636e-06+0j) [X3 X4 X6 Z7 Z8 Z9 Z10 X11] +
(-0.008541996625454858+0j) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.1513463114361887e-06+0j) [X3 X4 Y7 Z8 Z9 Y10] +
(-0.01925750509525156+0j) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895373173532e-07+0j) [X3 X4 X8 X9] +
(-4.6430510689196885e-06+0j) [X3 X4 X10 X11] +
(-0.008764827575688859+0j) [X3 X4 X10 Z11 Z12 X13] +
(-0.019028242443847404+0j) [X3 X4 Y11 Y12] +
(5.275883122691193e-06+0j) [X3 X4 X12 X13] +
(-0.00514339176882505+0j) [X3 Y4 Y5 X6] +
(-0.009841749246962583+0j) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10] +
(2.988511706512941e-06+0j) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.454842449109665e-06+0j) [X3 Y4 Y6 X7] +
(-1.522493067736636e-06+0j) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11] +
(-0.008541996625454858+0j) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.1513463114361887e-06+0j) [X3 Y4 Y7 Z8 Z9 X10] +
(0.01925750509525156+0j) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12] +
(-7.956895373173532e-07+0j) [X3 Y4 Y8 X9] +
(-4.6430510689196885e-06+0j) [X3 Y4 Y10 X11] +
(-0.008764827575688859+0j) [X3 Y4 Y10 Z11 Z12 X13] +
(0.019028242443847404+0j) [X3 Y4 Y11 X12] +
(5.275883122691193e-06+0j) [X3 Y4 Y12 X13] +
(-3.887051674447985e-06+0j) [X3 Z4 X5] +
(3.211842019379424e-06+0j) [X3 Z4 X5 X6 Z7 Z8 Z9 X10] +
(0.019299560579363755+0j) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.211842019379424e-06+0j) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10] +
(0.019299560579363755+0j) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.1954890100550876e-06+0j) [X3 Z4 X5 Z6] +
(1.6893489516427878e-06+0j) [X3 Z4 X5 X7 Z8 Z9 Z10 X11] +
(0.0107575639539089+0j) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.537178095785608e-08+0j) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11] +
(4.205548411219569e-05+0j) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.593534390545774e-07+0j) [X3 Z4 X5 Z7] +
(-5.770052996073175e-07+0j) [X3 Z4 X5 Z8] +
(2.1868423771003565e-07+0j) [X3 Z4 X5 Z9] +
(0.024353077678069053+0j) [X3 Z4 X5 X10 Z11 X12] +
(0.024353077678069053+0j) [X3 Z4 X5 Y10 Z11 Y12] +
(-7.801707501317178e-06+0j) [X3 Z4 X5 Z10] +
(0.015588250102380201+0j) [X3 Z4 X5 X11 Z12 X13] +
(0.005324835234221652+0j) [X3 Z4 X5 Y11 Z12 Y13] +
(-3.15865643239749e-06+0j) [X3 Z4 X5 Z11] +
(8.814937307474002e-06+0j) [X3 Z4 X5 Z12] +
(3.5390541847828075e-06+0j) [X3 Z4 X5 Z13] +
(1.6288532436995536e-06+0j) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11] +
(0.0107155084697967+0j) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.010263414868158549+0j) [X3 Z4 Y5 Y11 Z12 X13] +
(0.008469978791023788+0j) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10] +
(-2.686381546503742e-07+0j) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.011756013419819274+0j) [X3 Z4 Z5 X6 X8 X9] +
(0.01522563075722662+0j) [X3 Z4 Z5 X6 X10 X11] +
(-3.54439542968147e-06+0j) [X3 Z4 Z5 X6 X10 Z11 Z12 X13] +
(-3.0882507116829057e-06+0j) [X3 Z4 Z5 X6 Y11 Y12] +
(-0.004158797381840026+0j) [X3 Z4 Z5 X6 X12 X13] +
(-0.008469978791023788+0j) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10] +
(2.686381546503742e-07+0j) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.011756013419819274+0j) [X3 Z4 Z5 Y6 Y8 X9] +
(0.01522563075722662+0j) [X3 Z4 Z5 Y6 Y10 X11] +
(-3.54439542968147e-06+0j) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13] +
(3.0882507116829057e-06+0j) [X3 Z4 Z5 Y6 Y11 X12] +
(-0.004158797381840026+0j) [X3 Z4 Z5 Y6 Y12 X13] +
(-0.12133276911042419+0j) [X3 Z4 Z5 Z6 X7] +
(-0.01756120240964616+0j) [X3 Z4 Z5 Z6 X7 Z8] +
(-0.005805188989826881+0j) [X3 Z4 Z5 Z6 X7 Z9] +
(2.7455184007945525e-06+0j) [X3 Z4 Z5 Z6 X7 X10 Z11 X12] +
(2.7455184007945525e-06+0j) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12] +
(0.01441109943013092+0j) [X3 Z4 Z5 Z6 X7 Z10] +
(-7.988770288869174e-07+0j) [X3 Z4 Z5 Z6 X7 X11 Z12 X13] +
(-3.427323108883532e-07+0j) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13] +
(-0.0008145313270957019+0j) [X3 Z4 Z5 Z6 X7 Z11] +
(-0.0034937903598901456+0j) [X3 Z4 Z5 Z6 X7 Z12] +
(0.0006650070219498812+0j) [X3 Z4 Z5 Z6 X7 Z13] +
(-4.561447179985641e-07+0j) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13] +
(-0.014603704729162184+0j) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10] +
(2.874299071684717e-06+0j) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(0.014603704729162184+0j) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10] +
(-2.874299071684717e-06+0j) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(1.3002946563348209e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12] +
(0.0024464971554159116+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-1.3002946563348209e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12] +
(0.0024464971554159116+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(-0.28164257767023293+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.01709155315589892+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(-0.019538050311314833+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(5.7759505278920525e-05+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(2.8836765763667134e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(3.8462016716665535e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(-0.024282117354693062+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11] +
(5.146496328001374e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(-0.02475546329289109+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11] +
(5.105526722489681e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(-0.039359168022053276+0j) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11] +
(7.979825794174398e-06+0j) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(-0.029903789512624904+0j) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11] +
(5.4279886570378245e-06+0j) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.025996177598021312+0j) [X3 Z4 Z5 X7] +
(-0.021433810721601113+0j) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11] +
(5.15935050238745e-06+0j) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0034795118903342658+0j) [X3 Z4 Z6 X7] +
(-0.0287307795519055+0j) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11] +
(5.935867718543803e-06+0j) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.764994120667341e-07+0j) [X3 X5] +
(0.0016638798784907843+0j) [X3 Z5 Z6 X7] +
(-0.01888903030494292+0j) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.9473560120308617e-06+0j) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.00514339176882505+0j) [Y3 X4 X5 Y6] +
(-0.009841749246962583+0j) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10] +
(2.988511706512941e-06+0j) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.454842449109665e-06+0j) [Y3 X4 X6 Y7] +
(-1.522493067736636e-06+0j) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11] +
(-0.008541996625454858+0j) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.1513463114361887e-06+0j) [Y3 X4 X7 Z8 Z9 Y10] +
(0.01925750509525156+0j) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895373173532e-07+0j) [Y3 X4 X8 Y9] +
(-4.6430510689196885e-06+0j) [Y3 X4 X10 Y11] +
(-0.008764827575688859+0j) [Y3 X4 X10 Z11 Z12 Y13] +
(0.019028242443847404+0j) [Y3 X4 X11 Y12] +
(5.275883122691193e-06+0j) [Y3 X4 X12 Y13] +
(0.00514339176882505+0j) [Y3 Y4 X5 X6] +
(0.009841749246962583+0j) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10] +
(-2.988511706512941e-06+0j) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.454842449109665e-06+0j) [Y3 Y4 Y6 Y7] +
(-1.522493067736636e-06+0j) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11] +
(-0.008541996625454858+0j) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.1513463114361887e-06+0j) [Y3 Y4 X7 Z8 Z9 X10] +
(-0.01925750509525156+0j) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12] +
(-7.956895373173532e-07+0j) [Y3 Y4 Y8 Y9] +
(-4.6430510689196885e-06+0j) [Y3 Y4 Y10 Y11] +
(-0.008764827575688859+0j) [Y3 Y4 Y10 Z11 Z12 Y13] +
(-0.019028242443847404+0j) [Y3 Y4 X11 X12] +
(5.275883122691193e-06+0j) [Y3 Y4 Y12 Y13] +
(1.6288532436995536e-06+0j) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11] +
(0.0107155084697967+0j) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.010263414868158549+0j) [Y3 Z4 X5 X11 Z12 Y13] +
(-3.887051674447985e-06+0j) [Y3 Z4 Y5] +
(3.211842019379424e-06+0j) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10] +
(0.019299560579363755+0j) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.211842019379424e-06+0j) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10] +
(0.019299560579363755+0j) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.1954890100550876e-06+0j) [Y3 Z4 Y5 Z6] +
(4.537178095785608e-08+0j) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11] +
(4.205548411219569e-05+0j) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.6893489516427878e-06+0j) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11] +
(0.0107575639539089+0j) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.593534390545774e-07+0j) [Y3 Z4 Y5 Z7] +
(-5.770052996073175e-07+0j) [Y3 Z4 Y5 Z8] +
(2.1868423771003565e-07+0j) [Y3 Z4 Y5 Z9] +
(0.024353077678069053+0j) [Y3 Z4 Y5 X10 Z11 X12] +
(0.024353077678069053+0j) [Y3 Z4 Y5 Y10 Z11 Y12] +
(-7.801707501317178e-06+0j) [Y3 Z4 Y5 Z10] +
(0.005324835234221652+0j) [Y3 Z4 Y5 X11 Z12 X13] +
(0.015588250102380201+0j) [Y3 Z4 Y5 Y11 Z12 Y13] +
(-3.15865643239749e-06+0j) [Y3 Z4 Y5 Z11] +
(8.814937307474002e-06+0j) [Y3 Z4 Y5 Z12] +
(3.5390541847828075e-06+0j) [Y3 Z4 Y5 Z13] +
(-0.008469978791023788+0j) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10] +
(2.686381546503742e-07+0j) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.011756013419819274+0j) [Y3 Z4 Z5 X6 X8 Y9] +
(0.01522563075722662+0j) [Y3 Z4 Z5 X6 X10 Y11] +
(-3.54439542968147e-06+0j) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13] +
(3.0882507116829057e-06+0j) [Y3 Z4 Z5 X6 X11 Y12] +
(-0.004158797381840026+0j) [Y3 Z4 Z5 X6 X12 Y13] +
(0.008469978791023788+0j) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10] +
(-2.686381546503742e-07+0j) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-0.011756013419819274+0j) [Y3 Z4 Z5 Y6 Y8 Y9] +
(0.01522563075722662+0j) [Y3 Z4 Z5 Y6 Y10 Y11] +
(-3.54439542968147e-06+0j) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13] +
(-3.0882507116829057e-06+0j) [Y3 Z4 Z5 Y6 X11 X12] +
(-0.004158797381840026+0j) [Y3 Z4 Z5 Y6 Y12 Y13] +
(-4.561447179985641e-07+0j) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13] +
(-0.12133276911042419+0j) [Y3 Z4 Z5 Z6 Y7] +
(-0.01756120240964616+0j) [Y3 Z4 Z5 Z6 Y7 Z8] +
(-0.005805188989826881+0j) [Y3 Z4 Z5 Z6 Y7 Z9] +
(2.7455184007945525e-06+0j) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12] +
(2.7455184007945525e-06+0j) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12] +
(0.01441109943013092+0j) [Y3 Z4 Z5 Z6 Y7 Z10] +
(-3.427323108883532e-07+0j) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13] +
(-7.988770288869174e-07+0j) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13] +
(-0.0008145313270957019+0j) [Y3 Z4 Z5 Z6 Y7 Z11] +
(-0.0034937903598901456+0j) [Y3 Z4 Z5 Z6 Y7 Z12] +
(0.0006650070219498812+0j) [Y3 Z4 Z5 Z6 Y7 Z13] +
(0.014603704729162184+0j) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10] +
(-2.874299071684717e-06+0j) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-0.014603704729162184+0j) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10] +
(2.874299071684717e-06+0j) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-1.3002946563348209e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12] +
(0.0024464971554159116+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(1.3002946563348209e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12] +
(0.0024464971554159116+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(-0.28164257767023293+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.01709155315589892+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(-0.019538050311314833+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(5.7759505278920525e-05+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.8836765763667134e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(3.8462016716665535e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(-0.024282117354693062+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11] +
(5.146496328001374e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(-0.02475546329289109+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11] +
(5.105526722489681e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(-0.039359168022053276+0j) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11] +
(7.979825794174398e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(-0.029903789512624904+0j) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11] +
(5.4279886570378245e-06+0j) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.025996177598021312+0j) [Y3 Z4 Z5 Y7] +
(-0.021433810721601113+0j) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11] +
(5.15935050238745e-06+0j) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0034795118903342658+0j) [Y3 Z4 Z6 Y7] +
(-0.0287307795519055+0j) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11] +
(5.935867718543803e-06+0j) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.764994120667341e-07+0j) [Y3 Y5] +
(0.0016638798784907843+0j) [Y3 Z5 Z6 Y7] +
(-0.01888903030494292+0j) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.9473560120308617e-06+0j) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.6538942226831688+0j) [Z3] +
(-1.170830137122971e-06+0j) [Z3 X4 Z5 X6] +
(-7.0897994682724884e-06+0j) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.03490334337366154+0j) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.170830137122971e-06+0j) [Z3 Y4 Z5 Y6] +
(-7.0897994682724884e-06+0j) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.03490334337366154+0j) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.16079764534838517+0j) [Z3 Z4] +
(-9.509249751833101e-07+0j) [Z3 X5 Z6 X7] +
(-4.7288431476609985e-06+0j) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.024591860883829912+0j) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-9.509249751833101e-07+0j) [Z3 Y5 Z6 Y7] +
(-4.7288431476609985e-06+0j) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.024591860883829912+0j) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.1249580773950317+0j) [Z3 Z5] +
(0.024389082531149298+0j) [Z3 X6 Z7 Z8 Z9 X10] +
(-2.0111220985158223e-06+0j) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.024389082531149298+0j) [Z3 Y6 Z7 Z8 Z9 Y10] +
(-2.0111220985158223e-06+0j) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.16853486561579895+0j) [Z3 Z6] +
(0.0190204231730399+0j) [Z3 X7 Z8 Z9 Z10 X11] +
(-2.1032156049474973e-06+0j) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0190204231730399+0j) [Z3 Y7 Z8 Z9 Z10 Y11] +
(-2.1032156049474973e-06+0j) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.13739104762683194+0j) [Z3 Z7] +
(0.18690820476912504+0j) [Z3 Z8] +
(0.1507140812100826+0j) [Z3 Z9] +
(1.1094407592661122e-06+0j) [Z3 X10 Z11 X12] +
(1.1094407592661122e-06+0j) [Z3 Y10 Z11 Y12] +
(0.15337968243314132+0j) [Z3 Z10] +
(-1.0632283424805205e-06+0j) [Z3 X11 Z12 X13] +
(-1.0632283424805205e-06+0j) [Z3 Y11 Z12 Y13] +
(0.12799502492468384+0j) [Z3 Z11] +
(0.1556901067175243+0j) [Z3 Z12] +
(0.14011289865354787+0j) [Z3 Z13] +
(-0.011982389010247878+0j) [X4 X5 Y6 Y7] +
(-0.007306759928832907+0j) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11] +
(-2.8882935950670954e-07+0j) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.007306759928832907+0j) [X4 X5 X7 Z8 Z9 X10] +
(-2.8882935950670954e-07+0j) [X4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-0.007156934919856931+0j) [X4 X5 Y8 Y9] +
(-0.017680067952481473+0j) [X4 X5 Y10 Y11] +
(-3.6945132949090782e-06+0j) [X4 X5 Y10 Z11 Z12 Y13] +
(-3.6945132949090782e-06+0j) [X4 X5 X11 X12] +
(-0.038314670294803836+0j) [X4 X5 Y12 Y13] +
(0.011982389010247878+0j) [X4 Y5 Y6 X7] +
(0.007306759928832907+0j) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11] +
(2.8882935950670954e-07+0j) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.007306759928832907+0j) [X4 Y5 Y7 Z8 Z9 X10] +
(-2.8882935950670954e-07+0j) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(0.007156934919856931+0j) [X4 Y5 Y8 X9] +
(0.017680067952481473+0j) [X4 Y5 Y10 X11] +
(3.6945132949090782e-06+0j) [X4 Y5 Y10 Z11 Z12 X13] +
(-3.6945132949090782e-06+0j) [X4 Y5 Y11 X12] +
(0.038314670294803836+0j) [X4 Y5 Y12 X13] +
(-1.2260484989636889e-05+0j) [X4 Z5 X6] +
(-1.228333782514429e-06+0j) [X4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(0.0002463643756959099+0j) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.228333782514429e-06+0j) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(0.0002463643756959099+0j) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.8540608580768723e-06+0j) [X4 Z5 X6 Z7] +
(-1.3980449081888264e-06+0j) [X4 Z5 X6 Z8] +
(-1.8818501833075115e-06+0j) [X4 Z5 X6 Z9] +
(0.007960880725921535+0j) [X4 Z5 X6 X10 Z11 X12] +
(-0.0009298507967730194+0j) [X4 Z5 X6 Y10 Z11 Y12] +
(-1.6923978287297623e-06+0j) [X4 Z5 X6 Z10] +
(-0.012215040997613813+0j) [X4 Z5 X6 X11 Z12 X13] +
(-0.012215040997613813+0j) [X4 Z5 X6 Y11 Z12 Y13] +
(4.281913885243384e-06+0j) [X4 Z5 X6 Z11] +
(-4.588855156092119e-06+0j) [X4 Z5 X6 Z13] +
(0.008890731522694553+0j) [X4 Z5 Y6 Y10 Z11 X12] +
(-4.838052751186853e-07+0j) [X4 Z5 Z6 X7 Y8 Y9] +
(5.974311713973145e-06+0j) [X4 Z5 Z6 X7 Y10 Y11] +
(0.011285190200840794+0j) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(0.020175921723535346+0j) [X4 Z5 Z6 X7 X11 X12] +
(-4.556569218597985e-06+0j) [X4 Z5 Z6 X7 Y12 Y13] +
(4.838052751186853e-07+0j) [X4 Z5 Z6 Y7 Y8 X9] +
(-5.974311713973145e-06+0j) [X4 Z5 Z6 Y7 Y10 X11] +
(-0.011285190200840794+0j) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(0.020175921723535346+0j) [X4 Z5 Z6 Y7 Y11 X12] +
(4.556569218597985e-06+0j) [X4 Z5 Z6 Y7 Y12 X13] +
(1.3304731887982762e-06+0j) [X4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(0.005923798336561341+0j) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(1.3304731887982762e-06+0j) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(0.005923798336561341+0j) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(-6.631277929145478e-05+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.016024603689179483+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(-0.016024603689179483+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(3.3343312897080042e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(-4.734622039243275e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(-9.806102776118823e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(-5.0714807368755475e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(5.0714807368755475e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-0.36937089366156256+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.023145130929529058+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-0.009612634606847317+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(-0.0256372382960268+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(-8.774817865477397e-06+0j) [X4 Z5 Z6 Z7 Z8 X10] +
(-0.047642612176383076+0j) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(-7.44434467667912e-06+0j) [X4 Z5 Z6 Z7 Z9 X10] +
(-0.04171881383982173+0j) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(-6.290028433919158e-06+0j) [X4 Z5 Z6 Z8 Z9 X10] +
(-0.039564416322893495+0j) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(-7.518362216433588e-06+0j) [X4 Z5 Z7 Z8 Z9 X10] +
(-0.03931805194719759+0j) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(-5.929765815789581e-07+0j) [X4 X6] +
(-4.253224225945801e-06+0j) [X4 Z6 Z7 Z8 Z9 X10] +
(-0.022528440196012953+0j) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.011982389010247878+0j) [Y4 X5 X6 Y7] +
(0.007306759928832907+0j) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11] +
(2.8882935950670954e-07+0j) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.007306759928832907+0j) [Y4 X5 X7 Z8 Z9 Y10] +
(-2.8882935950670954e-07+0j) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(0.007156934919856931+0j) [Y4 X5 X8 Y9] +
(0.017680067952481473+0j) [Y4 X5 X10 Y11] +
(3.6945132949090782e-06+0j) [Y4 X5 X10 Z11 Z12 Y13] +
(-3.6945132949090782e-06+0j) [Y4 X5 X11 Y12] +
(0.038314670294803836+0j) [Y4 X5 X12 Y13] +
(-0.011982389010247878+0j) [Y4 Y5 X6 X7] +
(-0.007306759928832907+0j) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11] +
(-2.8882935950670954e-07+0j) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.007306759928832907+0j) [Y4 Y5 Y7 Z8 Z9 Y10] +
(-2.8882935950670954e-07+0j) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.007156934919856931+0j) [Y4 Y5 X8 X9] +
(-0.017680067952481473+0j) [Y4 Y5 X10 X11] +
(-3.6945132949090782e-06+0j) [Y4 Y5 X10 Z11 Z12 X13] +
(-3.6945132949090782e-06+0j) [Y4 Y5 Y11 Y12] +
(-0.038314670294803836+0j) [Y4 Y5 X12 X13] +
(0.008890731522694553+0j) [Y4 Z5 X6 X10 Z11 Y12] +
(-1.2260484989636889e-05+0j) [Y4 Z5 Y6] +
(-1.228333782514429e-06+0j) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(0.0002463643756959099+0j) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.228333782514429e-06+0j) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(0.0002463643756959099+0j) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.8540608580768723e-06+0j) [Y4 Z5 Y6 Z7] +
(-1.3980449081888264e-06+0j) [Y4 Z5 Y6 Z8] +
(-1.8818501833075115e-06+0j) [Y4 Z5 Y6 Z9] +
(-0.0009298507967730194+0j) [Y4 Z5 Y6 X10 Z11 X12] +
(0.007960880725921535+0j) [Y4 Z5 Y6 Y10 Z11 Y12] +
(-1.6923978287297623e-06+0j) [Y4 Z5 Y6 Z10] +
(-0.012215040997613813+0j) [Y4 Z5 Y6 X11 Z12 X13] +
(-0.012215040997613813+0j) [Y4 Z5 Y6 Y11 Z12 Y13] +
(4.281913885243384e-06+0j) [Y4 Z5 Y6 Z11] +
(-4.588855156092119e-06+0j) [Y4 Z5 Y6 Z13] +
(4.838052751186853e-07+0j) [Y4 Z5 Z6 X7 X8 Y9] +
(-5.974311713973145e-06+0j) [Y4 Z5 Z6 X7 X10 Y11] +
(-0.011285190200840794+0j) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(0.020175921723535346+0j) [Y4 Z5 Z6 X7 X11 Y12] +
(4.556569218597985e-06+0j) [Y4 Z5 Z6 X7 X12 Y13] +
(-4.838052751186853e-07+0j) [Y4 Z5 Z6 Y7 X8 X9] +
(5.974311713973145e-06+0j) [Y4 Z5 Z6 Y7 X10 X11] +
(0.011285190200840794+0j) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(0.020175921723535346+0j) [Y4 Z5 Z6 Y7 Y11 Y12] +
(-4.556569218597985e-06+0j) [Y4 Z5 Z6 Y7 X12 X13] +
(1.3304731887982762e-06+0j) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(0.005923798336561341+0j) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(1.3304731887982762e-06+0j) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(0.005923798336561341+0j) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(-6.631277929145478e-05+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.016024603689179483+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(-0.016024603689179483+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(3.3343312897080042e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(-4.734622039243275e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(-9.806102776118823e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(5.0714807368755475e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-5.0714807368755475e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-0.36937089366156256+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.023145130929529058+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-0.009612634606847317+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(-0.0256372382960268+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(-8.774817865477397e-06+0j) [Y4 Z5 Z6 Z7 Z8 Y10] +
(-0.047642612176383076+0j) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(-7.44434467667912e-06+0j) [Y4 Z5 Z6 Z7 Z9 Y10] +
(-0.04171881383982173+0j) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(-6.290028433919158e-06+0j) [Y4 Z5 Z6 Z8 Z9 Y10] +
(-0.039564416322893495+0j) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(-7.518362216433588e-06+0j) [Y4 Z5 Z7 Z8 Z9 Y10] +
(-0.03931805194719759+0j) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(-5.929765815789581e-07+0j) [Y4 Y6] +
(-4.253224225945801e-06+0j) [Y4 Z6 Z7 Z8 Z9 Y10] +
(-0.022528440196012953+0j) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.2034402289145645+0j) [Z4] +
(-5.929765815789581e-07+0j) [Z4 X5 Z6 X7] +
(-4.253224225945801e-06+0j) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.022528440196012953+0j) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-5.929765815789581e-07+0j) [Z4 Y5 Z6 Y7] +
(-4.253224225945801e-06+0j) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.022528440196012953+0j) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.15755314797985623+0j) [Z4 Z5] +
(0.018266834869375474+0j) [Z4 X6 Z7 Z8 Z9 X10] +
(-1.6541174774018358e-06+0j) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.018266834869375474+0j) [Z4 Y6 Z7 Z8 Z9 Y10] +
(-1.6541174774018358e-06+0j) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.13701191674040733+0j) [Z4 Z6] +
(0.010960074940542566+0j) [Z4 X7 Z8 Z9 Z10 X11] +
(-1.942946836908545e-06+0j) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.010960074940542566+0j) [Z4 Y7 Z8 Z9 Z10 Y11] +
(-1.942946836908545e-06+0j) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.1489943057506552+0j) [Z4 Z7] +
(0.14960702684445276+0j) [Z4 Z8] +
(0.1567639617643097+0j) [Z4 Z9] +
(1.8782101248974045e-06+0j) [Z4 X10 Z11 X12] +
(1.8782101248974045e-06+0j) [Z4 Y10 Z11 Y12] +
(0.12489990917237583+0j) [Z4 Z10] +
(-1.8163031700116744e-06+0j) [Z4 X11 Z12 X13] +
(-1.8163031700116744e-06+0j) [Z4 Y11 Z12 Y13] +
(0.14257997712485732+0j) [Z4 Z11] +
(0.11383573679388642+0j) [Z4 Z12] +
(0.15215040708869026+0j) [Z4 Z13] +
(1.228333782514429e-06+0j) [X5 X6 Y7 Z8 Z9 Y10] +
(-0.0002463643756959099+0j) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-4.838052751186853e-07+0j) [X5 X6 X8 X9] +
(5.974311713973146e-06+0j) [X5 X6 X10 X11] +
(0.020175921723535346+0j) [X5 X6 X10 Z11 Z12 X13] +
(0.011285190200840794+0j) [X5 X6 Y11 Y12] +
(-4.556569218597985e-06+0j) [X5 X6 X12 X13] +
(-1.228333782514429e-06+0j) [X5 Y6 Y7 Z8 Z9 X10] +
(0.0002463643756959099+0j) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-4.838052751186853e-07+0j) [X5 Y6 Y8 X9] +
(5.974311713973146e-06+0j) [X5 Y6 Y10 X11] +
(0.020175921723535346+0j) [X5 Y6 Y10 Z11 Z12 X13] +
(-0.011285190200840794+0j) [X5 Y6 Y11 X12] +
(-4.556569218597985e-06+0j) [X5 Y6 Y12 X13] +
(-1.2260484989636892e-05+0j) [X5 Z6 X7] +
(-1.8818501833075115e-06+0j) [X5 Z6 X7 Z8] +
(-1.3980449081888264e-06+0j) [X5 Z6 X7 Z9] +
(-0.012215040997613813+0j) [X5 Z6 X7 X10 Z11 X12] +
(-0.012215040997613813+0j) [X5 Z6 X7 Y10 Z11 Y12] +
(4.281913885243384e-06+0j) [X5 Z6 X7 Z10] +
(0.007960880725921535+0j) [X5 Z6 X7 X11 Z12 X13] +
(-0.0009298507967730194+0j) [X5 Z6 X7 Y11 Z12 Y13] +
(-1.6923978287297623e-06+0j) [X5 Z6 X7 Z11] +
(-4.588855156092119e-06+0j) [X5 Z6 X7 Z12] +
(0.008890731522694553+0j) [X5 Z6 Y7 Y11 Z12 X13] +
(-1.3304731887982762e-06+0j) [X5 Z6 Z7 X8 Y9 Y10] +
(-0.005923798336561341+0j) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(1.3304731887982762e-06+0j) [X5 Z6 Z7 Y8 Y9 X10] +
(0.005923798336561341+0j) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(0.016024603689179483+0j) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12] +
(-5.0714807368755475e-06+0j) [X5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-0.016024603689179483+0j) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12] +
(-5.0714807368755475e-06+0j) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(-6.63127792914548e-05+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-9.806102776118823e-06+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(-4.734622039243275e-06+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(-0.36937089366156256+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.023145130929529058+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(-0.0256372382960268+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(3.3343312897080042e-06+0j) [X5 Z6 Z7 Z8 Z9 X11] +
(-0.009612634606847317+0j) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(-7.44434467667912e-06+0j) [X5 Z6 Z7 Z8 Z10 X11] +
(-0.04171881383982173+0j) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(-8.774817865477397e-06+0j) [X5 Z6 Z7 Z9 Z10 X11] +
(-0.047642612176383076+0j) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(-7.518362216433588e-06+0j) [X5 Z6 Z8 Z9 Z10 X11] +
(-0.03931805194719759+0j) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.8540608580768723e-06+0j) [X5 X7] +
(-6.290028433919158e-06+0j) [X5 Z7 Z8 Z9 Z10 X11] +
(-0.039564416322893495+0j) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.228333782514429e-06+0j) [Y5 X6 X7 Z8 Z9 Y10] +
(0.0002463643756959099+0j) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-4.838052751186853e-07+0j) [Y5 X6 X8 Y9] +
(5.974311713973146e-06+0j) [Y5 X6 X10 Y11] +
(0.020175921723535346+0j) [Y5 X6 X10 Z11 Z12 Y13] +
(-0.011285190200840794+0j) [Y5 X6 X11 Y12] +
(-4.556569218597985e-06+0j) [Y5 X6 X12 Y13] +
(1.228333782514429e-06+0j) [Y5 Y6 X7 Z8 Z9 X10] +
(-0.0002463643756959099+0j) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-4.838052751186853e-07+0j) [Y5 Y6 Y8 Y9] +
(5.974311713973146e-06+0j) [Y5 Y6 Y10 Y11] +
(0.020175921723535346+0j) [Y5 Y6 Y10 Z11 Z12 Y13] +
(0.011285190200840794+0j) [Y5 Y6 X11 X12] +
(-4.556569218597985e-06+0j) [Y5 Y6 Y12 Y13] +
(0.008890731522694553+0j) [Y5 Z6 X7 X11 Z12 Y13] +
(-1.2260484989636892e-05+0j) [Y5 Z6 Y7] +
(-1.8818501833075115e-06+0j) [Y5 Z6 Y7 Z8] +
(-1.3980449081888264e-06+0j) [Y5 Z6 Y7 Z9] +
(-0.012215040997613813+0j) [Y5 Z6 Y7 X10 Z11 X12] +
(-0.012215040997613813+0j) [Y5 Z6 Y7 Y10 Z11 Y12] +
(4.281913885243384e-06+0j) [Y5 Z6 Y7 Z10] +
(-0.0009298507967730194+0j) [Y5 Z6 Y7 X11 Z12 X13] +
(0.007960880725921535+0j) [Y5 Z6 Y7 Y11 Z12 Y13] +
(-1.6923978287297623e-06+0j) [Y5 Z6 Y7 Z11] +
(-4.588855156092119e-06+0j) [Y5 Z6 Y7 Z12] +
(1.3304731887982762e-06+0j) [Y5 Z6 Z7 X8 X9 Y10] +
(0.005923798336561341+0j) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-1.3304731887982762e-06+0j) [Y5 Z6 Z7 Y8 X9 X10] +
(-0.005923798336561341+0j) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-0.016024603689179483+0j) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12] +
(-5.0714807368755475e-06+0j) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(0.016024603689179483+0j) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12] +
(-5.0714807368755475e-06+0j) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(-6.63127792914548e-05+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-9.806102776118823e-06+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(-4.734622039243275e-06+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(-0.36937089366156256+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.023145130929529058+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(-0.0256372382960268+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(3.3343312897080042e-06+0j) [Y5 Z6 Z7 Z8 Z9 Y11] +
(-0.009612634606847317+0j) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(-7.44434467667912e-06+0j) [Y5 Z6 Z7 Z8 Z10 Y11] +
(-0.04171881383982173+0j) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(-8.774817865477397e-06+0j) [Y5 Z6 Z7 Z9 Z10 Y11] +
(-0.047642612176383076+0j) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(-7.518362216433588e-06+0j) [Y5 Z6 Z8 Z9 Z10 Y11] +
(-0.03931805194719759+0j) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.8540608580768723e-06+0j) [Y5 Y7] +
(-6.290028433919158e-06+0j) [Y5 Z7 Z8 Z9 Z10 Y11] +
(-0.039564416322893495+0j) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.2034402289145645+0j) [Z5] +
(0.010960074940542566+0j) [Z5 X6 Z7 Z8 Z9 X10] +
(-1.942946836908545e-06+0j) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.010960074940542566+0j) [Z5 Y6 Z7 Z8 Z9 Y10] +
(-1.942946836908545e-06+0j) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.1489943057506552+0j) [Z5 Z6] +
(0.018266834869375474+0j) [Z5 X7 Z8 Z9 Z10 X11] +
(-1.6541174774018358e-06+0j) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.018266834869375474+0j) [Z5 Y7 Z8 Z9 Z10 Y11] +
(-1.6541174774018358e-06+0j) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.13701191674040733+0j) [Z5 Z7] +
(0.1567639617643097+0j) [Z5 Z8] +
(0.14960702684445276+0j) [Z5 Z9] +
(-1.8163031700116744e-06+0j) [Z5 X10 Z11 X12] +
(-1.8163031700116744e-06+0j) [Z5 Y10 Z11 Y12] +
(0.14257997712485732+0j) [Z5 Z10] +
(1.8782101248974045e-06+0j) [Z5 X11 Z12 X13] +
(1.8782101248974045e-06+0j) [Z5 Y11 Z12 Y13] +
(0.12489990917237583+0j) [Z5 Z11] +
(0.15215040708869026+0j) [Z5 Z12] +
(0.11383573679388642+0j) [Z5 Z13] +
(-0.013873381748426162+0j) [X6 X7 Y8 Y9] +
(-0.017825140995786266+0j) [X6 X7 Y10 Y11] +
(-1.0358477601594679e-06+0j) [X6 X7 Y10 Z11 Z12 Y13] +
(-1.0358477601594679e-06+0j) [X6 X7 X11 X12] +
(-0.01736611899465135+0j) [X6 X7 Y12 Y13] +
(0.013873381748426162+0j) [X6 Y7 Y8 X9] +
(0.017825140995786266+0j) [X6 Y7 Y10 X11] +
(1.0358477601594679e-06+0j) [X6 Y7 Y10 Z11 Z12 X13] +
(-1.0358477601594679e-06+0j) [X6 Y7 Y11 X12] +
(0.01736611899465135+0j) [X6 Y7 Y12 X13] +
(0.00029219862611113063+0j) [X6 Z7 X8 X9 Z10 X11] +
(-3.3281393508779926e-07+0j) [X6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.00029219862611113063+0j) [X6 Z7 X8 Y9 Z10 Y11] +
(-3.3281393508779926e-07+0j) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(0.2284810656491873+0j) [X6 Z7 Z8 Z9 X10] +
(3.313145500397252e-06+0j) [X6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(3.313145500397252e-06+0j) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(0.011307274008848242+0j) [X6 Z7 Z8 Z9 X10 Z11] +
(0.025104957138844503+0j) [X6 Z7 Z8 Z9 X10 Z12] +
(0.010540425907671536+0j) [X6 Z7 Z8 Z9 X10 Z13] +
(-0.014564531231172965+0j) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(0.014564531231172965+0j) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-2.5950860072900985e-05+0j) [X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.1839325597747225e-06+0j) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-6.524373849199108e-06+0j) [X6 Z7 Z8 Z9 Z10 X12] +
(-3.2112283488018562e-06+0j) [X6 Z7 Z8 Z9 Z11 X12] +
(0.029812424517345608+0j) [X6 Z7 Z8 X10] +
(-3.2774831959329014e-06+0j) [X6 Z7 Z8 Z10 Z11 X12] +
(0.030104623143456737+0j) [X6 Z7 Z9 X10] +
(-3.6102971310207005e-06+0j) [X6 Z7 Z9 Z10 Z11 X12] +
(0.030787505389143863+0j) [X6 Z8 Z9 X10] +
(-3.7696594524056707e-06+0j) [X6 Z8 Z9 Z10 Z11 X12] +
(0.013873381748426162+0j) [Y6 X7 X8 Y9] +
(0.017825140995786266+0j) [Y6 X7 X10 Y11] +
(1.0358477601594679e-06+0j) [Y6 X7 X10 Z11 Z12 Y13] +
(-1.0358477601594679e-06+0j) [Y6 X7 X11 Y12] +
(0.01736611899465135+0j) [Y6 X7 X12 Y13] +
(-0.013873381748426162+0j) [Y6 Y7 X8 X9] +
(-0.017825140995786266+0j) [Y6 Y7 X10 X11] +
(-1.0358477601594679e-06+0j) [Y6 Y7 X10 Z11 Z12 X13] +
(-1.0358477601594679e-06+0j) [Y6 Y7 Y11 Y12] +
(-0.01736611899465135+0j) [Y6 Y7 X12 X13] +
(0.00029219862611113063+0j) [Y6 Z7 Y8 X9 Z10 X11] +
(-3.3281393508779926e-07+0j) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.00029219862611113063+0j) [Y6 Z7 Y8 Y9 Z10 Y11] +
(-3.3281393508779926e-07+0j) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(0.2284810656491873+0j) [Y6 Z7 Z8 Z9 Y10] +
(3.313145500397252e-06+0j) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(3.313145500397252e-06+0j) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(0.011307274008848242+0j) [Y6 Z7 Z8 Z9 Y10 Z11] +
(0.025104957138844503+0j) [Y6 Z7 Z8 Z9 Y10 Z12] +
(0.010540425907671536+0j) [Y6 Z7 Z8 Z9 Y10 Z13] +
(0.014564531231172965+0j) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-0.014564531231172965+0j) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-2.5950860072900985e-05+0j) [Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.1839325597747225e-06+0j) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-6.524373849199108e-06+0j) [Y6 Z7 Z8 Z9 Z10 Y12] +
(-3.2112283488018562e-06+0j) [Y6 Z7 Z8 Z9 Z11 Y12] +
(0.029812424517345608+0j) [Y6 Z7 Z8 Y10] +
(-3.2774831959329014e-06+0j) [Y6 Z7 Z8 Z10 Z11 Y12] +
(0.030104623143456737+0j) [Y6 Z7 Z9 Y10] +
(-3.6102971310207005e-06+0j) [Y6 Z7 Z9 Z10 Z11 Y12] +
(0.030787505389143863+0j) [Y6 Z8 Z9 Y10] +
(-3.7696594524056707e-06+0j) [Y6 Z8 Z9 Z10 Z11 Y12] +
(1.309686298861547+0j) [Z6] +
(0.03078750538914386+0j) [Z6 X7 Z8 Z9 Z10 X11] +
(-3.7696594524056707e-06+0j) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.03078750538914386+0j) [Z6 Y7 Z8 Z9 Z10 Y11] +
(-3.7696594524056707e-06+0j) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.19392534613270232+0j) [Z6 Z7] +
(0.16756653265461274+0j) [Z6 Z8] +
(0.18143991440303892+0j) [Z6 Z9] +
(-1.8551201217487526e-06+0j) [Z6 X10 Z11 X12] +
(-1.8551201217487526e-06+0j) [Z6 Y10 Z11 Y12] +
(0.11952438964682678+0j) [Z6 Z10] +
(-2.8909678819082207e-06+0j) [Z6 X11 Z12 X13] +
(-2.8909678819082207e-06+0j) [Z6 Y11 Z12 Y13] +
(0.13734953064261304+0j) [Z6 Z11] +
(0.1340171526196371+0j) [Z6 Z12] +
(0.15138327161428844+0j) [Z6 Z13] +
(-0.00029219862611113063+0j) [X7 X8 Y9 Y10] +
(3.3281393508779926e-07+0j) [X7 X8 Y9 Z10 Z11 Y12] +
(0.00029219862611113063+0j) [X7 Y8 Y9 X10] +
(-3.3281393508779926e-07+0j) [X7 Y8 Y9 Z10 Z11 X12] +
(-3.313145500397252e-06+0j) [X7 Z8 Z9 X10 Y11 Y12] +
(-0.014564531231172965+0j) [X7 Z8 Z9 X10 X12 X13] +
(3.313145500397252e-06+0j) [X7 Z8 Z9 Y10 Y11 X12] +
(-0.014564531231172965+0j) [X7 Z8 Z9 Y10 Y12 X13] +
(0.22848106564918733+0j) [X7 Z8 Z9 Z10 X11] +
(0.010540425907671536+0j) [X7 Z8 Z9 Z10 X11 Z12] +
(0.025104957138844503+0j) [X7 Z8 Z9 Z10 X11 Z13] +
(-2.595086007290098e-05+0j) [X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.1839325597747225e-06+0j) [X7 Z8 Z9 Z10 Z11 X13] +
(-3.2112283488018562e-06+0j) [X7 Z8 Z9 Z10 Z12 X13] +
(0.01130727400884824+0j) [X7 Z8 Z9 X11] +
(-6.524373849199108e-06+0j) [X7 Z8 Z9 Z11 Z12 X13] +
(0.030104623143456737+0j) [X7 Z8 Z10 X11] +
(-3.6102971310207005e-06+0j) [X7 Z8 Z10 Z11 Z12 X13] +
(0.029812424517345608+0j) [X7 Z9 Z10 X11] +
(-3.2774831959329014e-06+0j) [X7 Z9 Z10 Z11 Z12 X13] +
(0.00029219862611113063+0j) [Y7 X8 X9 Y10] +
(-3.3281393508779926e-07+0j) [Y7 X8 X9 Z10 Z11 Y12] +
(-0.00029219862611113063+0j) [Y7 Y8 X9 X10] +
(3.3281393508779926e-07+0j) [Y7 Y8 X9 Z10 Z11 X12] +
(3.313145500397252e-06+0j) [Y7 Z8 Z9 X10 X11 Y12] +
(-0.014564531231172965+0j) [Y7 Z8 Z9 X10 X12 Y13] +
(-3.313145500397252e-06+0j) [Y7 Z8 Z9 Y10 X11 X12] +
(-0.014564531231172965+0j) [Y7 Z8 Z9 Y10 Y12 Y13] +
(0.22848106564918733+0j) [Y7 Z8 Z9 Z10 Y11] +
(0.010540425907671536+0j) [Y7 Z8 Z9 Z10 Y11 Z12] +
(0.025104957138844503+0j) [Y7 Z8 Z9 Z10 Y11 Z13] +
(-2.595086007290098e-05+0j) [Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(4.1839325597747225e-06+0j) [Y7 Z8 Z9 Z10 Z11 Y13] +
(-3.2112283488018562e-06+0j) [Y7 Z8 Z9 Z10 Z12 Y13] +
(0.01130727400884824+0j) [Y7 Z8 Z9 Y11] +
(-6.524373849199108e-06+0j) [Y7 Z8 Z9 Z11 Z12 Y13] +
(0.030104623143456737+0j) [Y7 Z8 Z10 Y11] +
(-3.6102971310207005e-06+0j) [Y7 Z8 Z10 Z11 Z12 Y13] +
(0.029812424517345608+0j) [Y7 Z9 Z10 Y11] +
(-3.2774831959329014e-06+0j) [Y7 Z9 Z10 Z11 Z12 Y13] +
(1.3096862988615472+0j) [Z7] +
(0.18143991440303892+0j) [Z7 Z8] +
(0.16756653265461274+0j) [Z7 Z9] +
(-2.8909678819082207e-06+0j) [Z7 X10 Z11 X12] +
(-2.8909678819082207e-06+0j) [Z7 Y10 Z11 Y12] +
(0.13734953064261304+0j) [Z7 Z10] +
(-1.8551201217487526e-06+0j) [Z7 X11 Z12 X13] +
(-1.8551201217487526e-06+0j) [Z7 Y11 Z12 Y13] +
(0.11952438964682678+0j) [Z7 Z11] +
(0.15138327161428844+0j) [Z7 Z12] +
(0.1340171526196371+0j) [Z7 Z13] +
(-0.009560705729136013+0j) [X8 X9 Y10 Y11] +
(6.628614202428987e-07+0j) [X8 X9 Y10 Z11 Z12 Y13] +
(6.628614202428987e-07+0j) [X8 X9 X11 X12] +
(-0.006087822480561861+0j) [X8 X9 Y12 Y13] +
(0.009560705729136013+0j) [X8 Y9 Y10 X11] +
(-6.628614202428987e-07+0j) [X8 Y9 Y10 Z11 Z12 X13] +
(6.628614202428987e-07+0j) [X8 Y9 Y11 X12] +
(0.006087822480561861+0j) [X8 Y9 Y12 X13] +
(0.009560705729136013+0j) [Y8 X9 X10 Y11] +
(-6.628614202428987e-07+0j) [Y8 X9 X10 Z11 Z12 Y13] +
(6.628614202428987e-07+0j) [Y8 X9 X11 Y12] +
(0.006087822480561861+0j) [Y8 X9 X12 Y13] +
(-0.009560705729136013+0j) [Y8 Y9 X10 X11] +
(6.628614202428987e-07+0j) [Y8 Y9 X10 Z11 Z12 X13] +
(6.628614202428987e-07+0j) [Y8 Y9 Y11 Y12] +
(-0.006087822480561861+0j) [Y8 Y9 X12 X13] +
(1.3693525634718189+0j) [Z8] +
(0.2200397733437609+0j) [Z8 Z9] +
(-1.5973171979897842e-06+0j) [Z8 X10 Z11 X12] +
(-1.5973171979897842e-06+0j) [Z8 Y10 Z11 Y12] +
(0.13766872645852574+0j) [Z8 Z10] +
(-9.344557777468855e-07+0j) [Z8 X11 Z12 X13] +
(-9.344557777468855e-07+0j) [Z8 Y11 Z12 Y13] +
(0.14722943218766174+0j) [Z8 Z11] +
(0.14973486803496927+0j) [Z8 Z12] +
(0.15582269051553113+0j) [Z8 Z13] +
(1.3693525634718193+0j) [Z9] +
(-9.344557777468855e-07+0j) [Z9 X10 Z11 X12] +
(-9.344557777468855e-07+0j) [Z9 Y10 Z11 Y12] +
(0.14722943218766174+0j) [Z9 Z10] +
(-1.5973171979897842e-06+0j) [Z9 X11 Z12 X13] +
(-1.5973171979897842e-06+0j) [Z9 Y11 Z12 Y13] +
(0.13766872645852574+0j) [Z9 Z11] +
(0.15582269051553113+0j) [Z9 Z12] +
(0.14973486803496927+0j) [Z9 Z13] +
(-0.028685183716105924+0j) [X10 X11 Y12 Y13] +
(0.028685183716105924+0j) [X10 Y11 Y12 X13] +
(-1.072231215814083e-05+0j) [X10 Z11 X12] +
(7.954413177014746e-06+0j) [X10 Z11 X12 Z13] +
(-8.19426137295896e-06+0j) [X10 X12] +
(0.028685183716105924+0j) [Y10 X11 X12 Y13] +
(-0.028685183716105924+0j) [Y10 Y11 X12 X13] +
(-1.072231215814083e-05+0j) [Y10 Z11 Y12] +
(7.954413177014746e-06+0j) [Y10 Z11 Y12 Z13] +
(-8.19426137295896e-06+0j) [Y10 Y12] +
(0.782966172595021+0j) [Z10] +
(-8.19426137295896e-06+0j) [Z10 X11 Z12 X13] +
(-8.19426137295896e-06+0j) [Z10 Y11 Z12 Y13] +
(0.14926355147388895+0j) [Z10 Z11] +
(0.11270386920332211+0j) [Z10 Z12] +
(0.14138905291942802+0j) [Z10 Z13] +
(-1.0722312158140835e-05+0j) [X11 Z12 X13] +
(7.954413177014746e-06+0j) [X11 X13] +
(-1.0722312158140835e-05+0j) [Y11 Z12 Y13] +
(7.954413177014746e-06+0j) [Y11 Y13] +
(0.7829661725950211+0j) [Z11] +
(0.14138905291942802+0j) [Z11 Z12] +
(0.11270386920332211+0j) [Z11 Z13] +
(0.8084581961720476+0j) [Z12] +
(0.1543574865722363+0j) [Z12 Z13] +
(0.8084581961720477+0j) [Z13]
Number of qubits: 14
Qubit Hamiltonian
  (-46.46390678868893) [I0]
+ (0.7829661725950193) [Z10]
+ (0.7829661725950197) [Z11]
+ (0.8084581961720502) [Z12]
+ (0.8084581961720504) [Z13]
+ (1.2034402289145623) [Z4]
+ (1.2034402289145623) [Z5]
+ (1.3096862988615434) [Z6]
+ (1.3096862988615436) [Z7]
+ (1.3693525634718176) [Z8]
+ (1.369352563471818) [Z9]
+ (1.6538942226831705) [Z2]
+ (1.6538942226831708) [Z3]
+ (12.412630742111748) [Z0]
+ (12.412630742111748) [Z1]
+ (-8.194261372474607e-06) [Y10 Y12]
+ (-8.194261372474607e-06) [X10 X12]
+ (-1.8540608580256239e-06) [Y5 Y7]
+ (-1.8540608580256239e-06) [X5 X7]
+ (-7.764994120230911e-07) [Y3 Y5]
+ (-7.764994120230911e-07) [X3 X5]
+ (-5.929765815665e-07) [Y4 Y6]
+ (-5.929765815665e-07) [X4 X6]
+ (1.6021167407100203e-06) [Y2 Y4]
+ (1.6021167407100203e-06) [X2 X4]
+ (7.95441317667538e-06) [Y11 Y13]
+ (7.95441317667538e-06) [X11 X13]
+ (0.003276971931231539) [Y1 Y3]
+ (0.003276971931231539) [X1 X3]
+ (0.10433064780651347) [Y0 Y2]
+ (0.10433064780651347) [X0 X2]
+ (0.11270386920332248) [Z10 Z12]
+ (0.11270386920332248) [Z11 Z13]
+ (0.11383573679388669) [Z4 Z12]
+ (0.11383573679388669) [Z5 Z13]
+ (0.11952438964682692) [Z6 Z10]
+ (0.11952438964682692) [Z7 Z11]
+ (0.12489990917237605) [Z4 Z10]
+ (0.12489990917237605) [Z5 Z11]
+ (0.12495807739503206) [Z2 Z4]
+ (0.12495807739503206) [Z3 Z5]
+ (0.12799502492468426) [Z2 Z10]
+ (0.12799502492468426) [Z3 Z11]
+ (0.13401715261963726) [Z6 Z12]
+ (0.13401715261963726) [Z7 Z13]
+ (0.13701191674040739) [Z4 Z6]
+ (0.13701191674040739) [Z5 Z7]
+ (0.13734953064261324) [Z6 Z11]
+ (0.13734953064261324) [Z7 Z10]
+ (0.13739104762683227) [Z2 Z6]
+ (0.13739104762683227) [Z3 Z7]
+ (0.13766872645852588) [Z8 Z10]
+ (0.13766872645852588) [Z9 Z11]
+ (0.14011289865354845) [Z2 Z12]
+ (0.14011289865354845) [Z3 Z13]
+ (0.14138905291942838) [Z10 Z13]
+ (0.14138905291942838) [Z11 Z12]
+ (0.1425799771248576) [Z4 Z11]
+ (0.1425799771248576) [Z5 Z10]
+ (0.14722943218766188) [Z8 Z11]
+ (0.14722943218766188) [Z9 Z10]
+ (0.14899430575065528) [Z4 Z7]
+ (0.14899430575065528) [Z5 Z6]
+ (0.1492635514738893) [Z10 Z11]
+ (0.14960702684445285) [Z4 Z8]
+ (0.14960702684445285) [Z5 Z9]
+ (0.14973486803496955) [Z8 Z12]
+ (0.14973486803496955) [Z9 Z13]
+ (0.15071408121008295) [Z2 Z8]
+ (0.15071408121008295) [Z3 Z9]
+ (0.15138327161428872) [Z6 Z13]
+ (0.15138327161428872) [Z7 Z12]
+ (0.1521504070886906) [Z4 Z13]
+ (0.1521504070886906) [Z5 Z12]
+ (0.15337968243314185) [Z2 Z11]
+ (0.15337968243314185) [Z3 Z10]
+ (0.1543574865722368) [Z12 Z13]
+ (0.15569010671752492) [Z2 Z13]
+ (0.15569010671752492) [Z3 Z12]
+ (0.15582269051553144) [Z8 Z13]
+ (0.15582269051553144) [Z9 Z12]
+ (0.15676396176430976) [Z4 Z9]
+ (0.15676396176430976) [Z5 Z8]
+ (0.15755314797985645) [Z4 Z5]
+ (0.1607976453483856) [Z2 Z5]
+ (0.1607976453483856) [Z3 Z4]
+ (0.16756653265461274) [Z6 Z8]
+ (0.16756653265461274) [Z7 Z9]
+ (0.16853486561579925) [Z2 Z7]
+ (0.16853486561579925) [Z3 Z6]
+ (0.18143991440303892) [Z6 Z9]
+ (0.18143991440303892) [Z7 Z8]
+ (0.18189085790751372) [Z2 Z3]
+ (0.18690820476912545) [Z2 Z9]
+ (0.18690820476912545) [Z3 Z8]
+ (0.1929972393536425) [Z0 Z10]
+ (0.1929972393536425) [Z1 Z11]
+ (0.19392534613270235) [Z6 Z7]
+ (0.1966177089034209) [Z0 Z4]
+ (0.1966177089034209) [Z1 Z5]
+ (0.19936354537360768) [Z0 Z5]
+ (0.19936354537360768) [Z1 Z4]
+ (0.2007286646044178) [Z0 Z11]
+ (0.2007286646044178) [Z1 Z10]
+ (0.2110265984979154) [Z0 Z12]
+ (0.2110265984979154) [Z1 Z13]
+ (0.2163103749863184) [Z0 Z13]
+ (0.2163103749863184) [Z1 Z12]
+ (0.2200397733437609) [Z8 Z9]
+ (0.23671080783830378) [Z0 Z2]
+ (0.23671080783830378) [Z1 Z3]
+ (0.24164663936017192) [Z0 Z6]
+ (0.24164663936017192) [Z1 Z7]
+ (0.24853483371314253) [Z0 Z7]
+ (0.24853483371314253) [Z1 Z6]
+ (0.25129445674591633) [Z0 Z3]
+ (0.25129445674591633) [Z1 Z2]
+ (0.27232518306605646) [Z0 Z8]
+ (0.27232518306605646) [Z1 Z9]
+ (0.27883454426723375) [Z0 Z9]
+ (0.27883454426723375) [Z1 Z8]
+ (1.186176373486047) [Z0 Z1]
+ (-1.2260484989204488e-05) [Y4 Z5 Y6]
+ (-1.2260484989204488e-05) [X4 Z5 X6]
+ (-1.2260484989204485e-05) [Y5 Z6 Y7]
+ (-1.2260484989204485e-05) [X5 Z6 X7]
+ (-1.0722312156943916e-05) [Y11 Z12 Y13]
+ (-1.0722312156943916e-05) [X11 Z12 X13]
+ (-1.0722312156943914e-05) [Y10 Z11 Y12]
+ (-1.0722312156943914e-05) [X10 Z11 X12]
+ (-3.887051674646459e-06) [Y2 Z3 Y4]
+ (-3.887051674646459e-06) [X2 Z3 X4]
+ (-3.887051674646458e-06) [Y3 Z4 Y5]
+ (-3.887051674646458e-06) [X3 Z4 X5]
+ (0.12507032579771765) [Y0 Z1 Y2]
+ (0.12507032579771765) [X0 Z1 X2]
+ (0.12507032579771774) [Y1 Z2 Y3]
+ (0.12507032579771774) [X1 Z2 X3]
+ (-0.03831467029480393) [Y4 Y5 X12 X13]
+ (-0.03831467029480393) [X4 X5 Y12 Y13]
+ (-0.03619412355904253) [Y2 Y3 X8 X9]
+ (-0.03619412355904253) [X2 X3 Y8 Y9]
+ (-0.03583956795335353) [Y2 Y3 X4 X5]
+ (-0.03583956795335353) [X2 X3 Y4 Y5]
+ (-0.031143817988966985) [Y2 Y3 X6 X7]
+ (-0.031143817988966985) [X2 X3 Y6 Y7]
+ (-0.028685183716105907) [Y10 Y11 X12 X13]
+ (-0.028685183716105907) [X10 X11 Y12 Y13]
+ (-0.025996177598021336) [Y3 Z4 Z5 Y7]
+ (-0.025996177598021336) [X3 Z4 Z5 X7]
+ (-0.02538465750845759) [Y2 Y3 X10 X11]
+ (-0.02538465750845759) [X2 X3 Y10 Y11]
+ (-0.019028242443847435) [Y3 Y4 X11 X12]
+ (-0.019028242443847435) [X3 X4 Y11 Y12]
+ (-0.01782514099578633) [Y6 Y7 X10 X11]
+ (-0.01782514099578633) [X6 X7 Y10 Y11]
+ (-0.017680067952481542) [Y4 Y5 X10 X11]
+ (-0.017680067952481542) [X4 X5 Y10 Y11]
+ (-0.017366118994651434) [Y6 Y7 X12 X13]
+ (-0.017366118994651434) [X6 X7 Y12 Y13]
+ (-0.015577208063976503) [Y2 Y3 X12 X13]
+ (-0.015577208063976503) [X2 X3 Y12 Y13]
+ (-0.01458364890761257) [Y0 Y1 X2 X3]
+ (-0.01458364890761257) [X0 X1 Y2 Y3]
+ (-0.013873381748426184) [Y6 Y7 X8 X9]
+ (-0.013873381748426184) [X6 X7 Y8 Y9]
+ (-0.01198238901024791) [Y4 Y5 X6 X7]
+ (-0.01198238901024791) [X4 X5 Y6 Y7]
+ (-0.01128519020084086) [Y5 X6 X11 Y12]
+ (-0.01128519020084086) [X5 Y6 Y11 X12]
+ (-0.009560705729135997) [Y8 Y9 X10 X11]
+ (-0.009560705729135997) [X8 X9 Y10 Y11]
+ (-0.008125251921381018) [Y1 X2 X8 Y9]
+ (-0.008125251921381018) [Y1 Y2 Y8 Y9]
+ (-0.008125251921381018) [X1 X2 X8 X9]
+ (-0.008125251921381018) [X1 Y2 Y8 X9]
+ (-0.0077314252507753025) [Y0 Y1 X10 X11]
+ (-0.0077314252507753025) [X0 X1 Y10 Y11]
+ (-0.007156934919856924) [Y4 Y5 X8 X9]
+ (-0.007156934919856924) [X4 X5 Y8 Y9]
+ (-0.006888194352970603) [Y0 Y1 X6 X7]
+ (-0.006888194352970603) [X0 X1 Y6 Y7]
+ (-0.006509361201177233) [Y0 Y1 X8 X9]
+ (-0.006509361201177233) [X0 X1 Y8 Y9]
+ (-0.006087822480561887) [Y8 Y9 X12 X13]
+ (-0.006087822480561887) [X8 X9 Y12 Y13]
+ (-0.005283776488402974) [Y0 Y1 X12 X13]
+ (-0.005283776488402974) [X0 X1 Y12 Y13]
+ (-0.005143391768825089) [Y3 X4 X5 Y6]
+ (-0.005143391768825089) [X3 Y4 Y5 X6]
+ (-0.00468490338815518) [Y1 X2 X6 Y7]
+ (-0.00468490338815518) [Y1 Y2 Y6 Y7]
+ (-0.00468490338815518) [X1 X2 X6 X7]
+ (-0.00468490338815518) [X1 Y2 Y6 X7]
+ (-0.004575007626639211) [Y1 X2 X12 Y13]
+ (-0.004575007626639211) [Y1 Y2 Y12 Y13]
+ (-0.004575007626639211) [X1 X2 X12 X13]
+ (-0.004575007626639211) [X1 Y2 Y12 X13]
+ (-0.004424855449441841) [Y1 X2 X4 Y5]
+ (-0.004424855449441841) [Y1 Y2 Y4 Y5]
+ (-0.004424855449441841) [X1 X2 X4 X5]
+ (-0.004424855449441841) [X1 Y2 Y4 X5]
+ (-0.0034795118903342714) [Y2 Z3 Z5 Y6]
+ (-0.0034795118903342714) [X2 Z3 Z5 X6]
+ (-0.0034795118903342714) [Y3 Z4 Z6 Y7]
+ (-0.0034795118903342714) [X3 Z4 Z6 X7]
+ (-0.002745836470186797) [Y0 Y1 X4 X5]
+ (-0.002745836470186797) [X0 X1 Y4 Y5]
+ (-0.0017992194936630214) [Y1 X2 X10 Y11]
+ (-0.0017992194936630214) [Y1 Y2 Y10 Y11]
+ (-0.0017992194936630214) [X1 X2 X10 X11]
+ (-0.0017992194936630214) [X1 Y2 Y10 X11]
+ (-0.00029219862611113036) [Y7 Y8 X9 X10]
+ (-0.00029219862611113036) [X7 X8 Y9 Y10]
+ (-8.194261372474607e-06) [Z10 Y11 Z12 Y13]
+ (-8.194261372474607e-06) [Z10 X11 Z12 X13]
+ (-7.801707500852634e-06) [Y2 Z3 Y4 Z11]
+ (-7.801707500852634e-06) [X2 Z3 X4 Z11]
+ (-7.801707500852634e-06) [Y3 Z4 Y5 Z10]
+ (-7.801707500852634e-06) [X3 Z4 X5 Z10]
+ (-4.643051068650268e-06) [Y3 X4 X10 Y11]
+ (-4.643051068650268e-06) [Y3 Y4 Y10 Y11]
+ (-4.643051068650268e-06) [X3 X4 X10 X11]
+ (-4.643051068650268e-06) [X3 Y4 Y10 X11]
+ (-4.588855155826711e-06) [Y4 Z5 Y6 Z13]
+ (-4.588855155826711e-06) [X4 Z5 X6 Z13]
+ (-4.588855155826711e-06) [Y5 Z6 Y7 Z12]
+ (-4.588855155826711e-06) [X5 Z6 X7 Z12]
+ (-4.556569218311546e-06) [Y5 X6 X12 Y13]
+ (-4.556569218311546e-06) [Y5 Y6 Y12 Y13]
+ (-4.556569218311546e-06) [X5 X6 X12 X13]
+ (-4.556569218311546e-06) [X5 Y6 Y12 X13]
+ (-3.6945132946254306e-06) [Y4 X5 X11 Y12]
+ (-3.6945132946254306e-06) [Y4 Y5 Y11 Y12]
+ (-3.6945132946254306e-06) [X4 X5 X11 X12]
+ (-3.6945132946254306e-06) [X4 Y5 Y11 X12]
+ (-3.344081556603231e-06) [Z0 Y5 Z6 Y7]
+ (-3.344081556603231e-06) [Z0 X5 Z6 X7]
+ (-3.344081556603231e-06) [Z1 Y4 Z5 Y6]
+ (-3.344081556603231e-06) [Z1 X4 Z5 X6]
+ (-3.1586564322023666e-06) [Y2 Z3 Y4 Z10]
+ (-3.1586564322023666e-06) [X2 Z3 X4 Z10]
+ (-3.1586564322023666e-06) [Y3 Z4 Y5 Z11]
+ (-3.1586564322023666e-06) [X3 Z4 X5 Z11]
+ (-3.0993492437065717e-06) [Z0 Y4 Z5 Y6]
+ (-3.0993492437065717e-06) [Z0 X4 Z5 X6]
+ (-3.0993492437065717e-06) [Z1 Y5 Z6 Y7]
+ (-3.0993492437065717e-06) [Z1 X5 Z6 X7]
+ (-2.89096788170617e-06) [Z6 Y11 Z12 Y13]
+ (-2.89096788170617e-06) [Z6 X11 Z12 X13]
+ (-2.89096788170617e-06) [Z7 Y10 Z11 Y12]
+ (-2.89096788170617e-06) [Z7 X10 Z11 X12]
+ (-2.1776646049188574e-06) [Z0 Y10 Z11 Y12]
+ (-2.1776646049188574e-06) [Z0 X10 Z11 X12]
+ (-2.1776646049188574e-06) [Z1 Y11 Z12 Y13]
+ (-2.1776646049188574e-06) [Z1 X11 Z12 X13]
+ (-1.8818501832644462e-06) [Y4 Z5 Y6 Z9]
+ (-1.8818501832644462e-06) [X4 Z5 X6 Z9]
+ (-1.8818501832644462e-06) [Y5 Z6 Y7 Z8]
+ (-1.8818501832644462e-06) [X5 Z6 X7 Z8]
+ (-1.8551201215083797e-06) [Z6 Y10 Z11 Y12]
+ (-1.8551201215083797e-06) [Z6 X10 Z11 X12]
+ (-1.8551201215083797e-06) [Z7 Y11 Z12 Y13]
+ (-1.8551201215083797e-06) [Z7 X11 Z12 X13]
+ (-1.8540608580256237e-06) [Y4 Z5 Y6 Z7]
+ (-1.8540608580256237e-06) [X4 Z5 X6 Z7]
+ (-1.8163031697666817e-06) [Z4 Y11 Z12 Y13]
+ (-1.8163031697666817e-06) [Z4 X11 Z12 X13]
+ (-1.8163031697666817e-06) [Z5 Y10 Z11 Y12]
+ (-1.8163031697666817e-06) [Z5 X10 Z11 X12]
+ (-1.692397828603289e-06) [Y4 Z5 Y6 Z10]
+ (-1.692397828603289e-06) [X4 Z5 X6 Z10]
+ (-1.692397828603289e-06) [Y5 Z6 Y7 Z11]
+ (-1.692397828603289e-06) [X5 Z6 X7 Z11]
+ (-1.6148794137382728e-06) [Z0 Y11 Z12 Y13]
+ (-1.6148794137382728e-06) [Z0 X11 Z12 X13]
+ (-1.6148794137382728e-06) [Z1 Y10 Z11 Y12]
+ (-1.6148794137382728e-06) [Z1 X10 Z11 X12]
+ (-1.5973171977785969e-06) [Z8 Y10 Z11 Y12]
+ (-1.5973171977785969e-06) [Z8 X10 Z11 X12]
+ (-1.5973171977785969e-06) [Z9 Y11 Z12 Y13]
+ (-1.5973171977785969e-06) [Z9 X11 Z12 X13]
+ (-1.454842449147877e-06) [Y3 X4 X6 Y7]
+ (-1.454842449147877e-06) [Y3 Y4 Y6 Y7]
+ (-1.454842449147877e-06) [X3 X4 X6 X7]
+ (-1.454842449147877e-06) [X3 Y4 Y6 X7]
+ (-1.3980449081377011e-06) [Y4 Z5 Y6 Z8]
+ (-1.3980449081377011e-06) [X4 Z5 X6 Z8]
+ (-1.3980449081377011e-06) [Y5 Z6 Y7 Z9]
+ (-1.3980449081377011e-06) [X5 Z6 X7 Z9]
+ (-1.195489010088138e-06) [Y2 Z3 Y4 Z7]
+ (-1.195489010088138e-06) [X2 Z3 X4 Z7]
+ (-1.195489010088138e-06) [Y3 Z4 Y5 Z6]
+ (-1.195489010088138e-06) [X3 Z4 X5 Z6]
+ (-1.190850808626359e-06) [Z0 Y3 Z4 Y5]
+ (-1.190850808626359e-06) [Z0 X3 Z4 X5]
+ (-1.190850808626359e-06) [Z1 Y2 Z3 Y4]
+ (-1.190850808626359e-06) [Z1 X2 Z3 X4]
+ (-1.170830137055109e-06) [Z2 Y5 Z6 Y7]
+ (-1.170830137055109e-06) [Z2 X5 Z6 X7]
+ (-1.170830137055109e-06) [Z3 Y4 Z5 Y6]
+ (-1.170830137055109e-06) [Z3 X4 Z5 X6]
+ (-1.0632283422648561e-06) [Z2 Y10 Z11 Y12]
+ (-1.0632283422648561e-06) [Z2 X10 Z11 X12]
+ (-1.0632283422648561e-06) [Z3 Y11 Z12 Y13]
+ (-1.0632283422648561e-06) [Z3 X11 Z12 X13]
+ (-1.0358477601977902e-06) [Y6 X7 X11 Y12]
+ (-1.0358477601977902e-06) [Y6 Y7 Y11 Y12]
+ (-1.0358477601977902e-06) [X6 X7 X11 X12]
+ (-1.0358477601977902e-06) [X6 Y7 Y11 X12]
+ (-9.509249751527842e-07) [Z2 Y4 Z5 Y6]
+ (-9.509249751527842e-07) [Z2 X4 Z5 X6]
+ (-9.509249751527842e-07) [Z3 Y5 Z6 Y7]
+ (-9.509249751527842e-07) [Z3 X5 Z6 X7]
+ (-9.34455777563054e-07) [Z8 Y11 Z12 Y13]
+ (-9.34455777563054e-07) [Z8 X11 Z12 X13]
+ (-9.34455777563054e-07) [Z9 Y10 Z11 Y12]
+ (-9.34455777563054e-07) [Z9 X10 Z11 X12]
+ (-8.33774675679169e-07) [Z0 Y2 Z3 Y4]
+ (-8.33774675679169e-07) [Z0 X2 Z3 X4]
+ (-8.33774675679169e-07) [Z1 Y3 Z4 Y5]
+ (-8.33774675679169e-07) [Z1 X3 Z4 X5]
+ (-7.95689537351061e-07) [Y3 X4 X8 Y9]
+ (-7.95689537351061e-07) [Y3 Y4 Y8 Y9]
+ (-7.95689537351061e-07) [X3 X4 X8 X9]
+ (-7.95689537351061e-07) [X3 Y4 Y8 X9]
+ (-7.764994120230911e-07) [Y2 Z3 Y4 Z5]
+ (-7.764994120230911e-07) [X2 Z3 X4 Z5]
+ (-5.929765815665e-07) [Z4 Y5 Z6 Y7]
+ (-5.929765815665e-07) [Z4 X5 Z6 X7]
+ (-5.770052996414511e-07) [Y2 Z3 Y4 Z9]
+ (-5.770052996414511e-07) [X2 Z3 X4 Z9]
+ (-5.770052996414511e-07) [Y3 Z4 Y5 Z8]
+ (-5.770052996414511e-07) [X3 Z4 X5 Z8]
+ (-5.471647744839624e-07) [Y1 Y2 X11 X12]
+ (-5.471647744839624e-07) [X1 X2 Y11 Y12]
+ (-4.83805275126745e-07) [Y5 X6 X8 Y9]
+ (-4.83805275126745e-07) [Y5 Y6 Y8 Y9]
+ (-4.83805275126745e-07) [X5 X6 X8 X9]
+ (-4.83805275126745e-07) [X5 Y6 Y8 X9]
+ (-3.5707613294719016e-07) [Y0 X1 X3 Y4]
+ (-3.5707613294719016e-07) [Y0 Y1 Y3 Y4]
+ (-3.5707613294719016e-07) [X0 X1 X3 X4]
+ (-3.5707613294719016e-07) [X0 Y1 Y3 X4]
+ (-2.4473231289665856e-07) [Y0 X1 X5 Y6]
+ (-2.4473231289665856e-07) [Y0 Y1 Y5 Y6]
+ (-2.4473231289665856e-07) [X0 X1 X5 X6]
+ (-2.4473231289665856e-07) [X0 Y1 Y5 X6]
+ (-2.199051619023248e-07) [Y2 X3 X5 Y6]
+ (-2.199051619023248e-07) [Y2 Y3 Y5 Y6]
+ (-2.199051619023248e-07) [X2 X3 X5 X6]
+ (-2.199051619023248e-07) [X2 Y3 Y5 X6]
+ (-1.9332412772431913e-07) [Y1 X2 X3 Y4]
+ (-1.9332412772431913e-07) [X1 Y2 Y3 X4]
+ (-1.2919694865365825e-07) [Y1 Z2 Z3 Y5]
+ (-1.2919694865365825e-07) [X1 Z2 Z3 X5]
+ (1.7379332624209236e-07) [Y0 Z1 Z3 Y4]
+ (1.7379332624209236e-07) [X0 Z1 Z3 X4]
+ (1.7379332624209236e-07) [Y1 Z2 Z4 Y5]
+ (1.7379332624209236e-07) [X1 Z2 Z4 X5]
+ (1.9332412772431913e-07) [Y1 Y2 X3 X4]
+ (1.9332412772431913e-07) [X1 X2 Y3 Y4]
+ (2.1868423770960994e-07) [Y2 Z3 Y4 Z8]
+ (2.1868423770960994e-07) [X2 Z3 X4 Z8]
+ (2.1868423770960994e-07) [Y3 Z4 Y5 Z9]
+ (2.1868423770960994e-07) [X3 Z4 X5 Z9]
+ (2.593534390597391e-07) [Y2 Z3 Y4 Z6]
+ (2.593534390597391e-07) [X2 Z3 X4 Z6]
+ (2.593534390597391e-07) [Y3 Z4 Y5 Z7]
+ (2.593534390597391e-07) [X3 Z4 X5 Z7]
+ (3.6060718680565465e-07) [Y0 Z1 Z2 Y4]
+ (3.6060718680565465e-07) [X0 Z1 Z2 X4]
+ (3.6060718680565465e-07) [Y1 Z3 Z4 Y5]
+ (3.6060718680565465e-07) [X1 Z3 Z4 X5]
+ (5.471647744839624e-07) [Y1 X2 X11 Y12]
+ (5.471647744839624e-07) [X1 Y2 Y11 X12]
+ (5.627851911805849e-07) [Y0 X1 X11 Y12]
+ (5.627851911805849e-07) [Y0 Y1 Y11 Y12]
+ (5.627851911805849e-07) [X0 X1 X11 X12]
+ (5.627851911805849e-07) [X0 Y1 Y11 X12]
+ (6.628614202155426e-07) [Y8 X9 X11 Y12]
+ (6.628614202155426e-07) [Y8 Y9 Y11 Y12]
+ (6.628614202155426e-07) [X8 X9 X11 X12]
+ (6.628614202155426e-07) [X8 Y9 Y11 X12]
+ (1.1094407594140565e-06) [Z2 Y11 Z12 Y13]
+ (1.1094407594140565e-06) [Z2 X11 Z12 X13]
+ (1.1094407594140565e-06) [Z3 Y10 Z11 Y12]
+ (1.1094407594140565e-06) [Z3 X10 Z11 X12]
+ (1.6021167407100205e-06) [Z2 Y3 Z4 Y5]
+ (1.6021167407100205e-06) [Z2 X3 Z4 X5]
+ (1.878210124858749e-06) [Z4 Y10 Z11 Y12]
+ (1.878210124858749e-06) [Z4 X10 Z11 X12]
+ (1.878210124858749e-06) [Z5 Y11 Z12 Y13]
+ (1.878210124858749e-06) [Z5 X11 Z12 X13]
+ (2.1726691016789126e-06) [Y2 X3 X11 Y12]
+ (2.1726691016789126e-06) [Y2 Y3 Y11 Y12]
+ (2.1726691016789126e-06) [X2 X3 X11 X12]
+ (2.1726691016789126e-06) [X2 Y3 Y11 X12]
+ (3.1174479463418126e-06) [Y0 Z2 Z3 Y4]
+ (3.1174479463418126e-06) [X0 Z2 Z3 X4]
+ (3.5390541845836345e-06) [Y2 Z3 Y4 Z12]
+ (3.5390541845836345e-06) [X2 Z3 X4 Z12]
+ (3.5390541845836345e-06) [Y3 Z4 Y5 Z13]
+ (3.5390541845836345e-06) [X3 Z4 X5 Z13]
+ (4.2819138850843745e-06) [Y4 Z5 Y6 Z11]
+ (4.2819138850843745e-06) [X4 Z5 X6 Z11]
+ (4.2819138850843745e-06) [Y5 Z6 Y7 Z10]
+ (4.2819138850843745e-06) [X5 Z6 X7 Z10]
+ (5.27588312241988e-06) [Y3 X4 X12 Y13]
+ (5.27588312241988e-06) [Y3 Y4 Y12 Y13]
+ (5.27588312241988e-06) [X3 X4 X12 X13]
+ (5.27588312241988e-06) [X3 Y4 Y12 X13]
+ (5.974311713687664e-06) [Y5 X6 X10 Y11]
+ (5.974311713687664e-06) [Y5 Y6 Y10 Y11]
+ (5.974311713687664e-06) [X5 X6 X10 X11]
+ (5.974311713687664e-06) [X5 Y6 Y10 X11]
+ (7.95441317667538e-06) [Y10 Z11 Y12 Z13]
+ (7.95441317667538e-06) [X10 Z11 X12 Z13]
+ (8.814937307003513e-06) [Y2 Z3 Y4 Z13]
+ (8.814937307003513e-06) [X2 Z3 X4 Z13]
+ (8.814937307003513e-06) [Y3 Z4 Y5 Z12]
+ (8.814937307003513e-06) [X3 Z4 X5 Z12]
+ (0.00029219862611113036) [Y7 X8 X9 Y10]
+ (0.00029219862611113036) [X7 Y8 Y9 X10]
+ (0.0004956762314918012) [Y2 Z4 Z5 Y6]
+ (0.0004956762314918012) [X2 Z4 Z5 X6]
+ (0.0011059037691896066) [Y0 Z1 Y2 Z5]
+ (0.0011059037691896066) [X0 Z1 X2 Z5]
+ (0.0011059037691896066) [Y1 Z2 Y3 Z4]
+ (0.0011059037691896066) [X1 Z2 X3 Z4]
+ (0.0016638798784908177) [Y2 Z3 Z4 Y6]
+ (0.0016638798784908177) [X2 Z3 Z4 X6]
+ (0.0016638798784908177) [Y3 Z5 Z6 Y7]
+ (0.0016638798784908177) [X3 Z5 Z6 X7]
+ (0.0017560707018411702) [Y0 Z1 Y2 Z11]
+ (0.0017560707018411702) [X0 Z1 X2 Z11]
+ (0.0017560707018411702) [Y1 Z2 Y3 Z10]
+ (0.0017560707018411702) [X1 Z2 X3 Z10]
+ (0.0023262306231580164) [Y0 Z1 Y2 Z13]
+ (0.0023262306231580164) [X0 Z1 X2 Z13]
+ (0.0023262306231580164) [Y1 Z2 Y3 Z12]
+ (0.0023262306231580164) [X1 Z2 X3 Z12]
+ (0.002745836470186797) [Y0 X1 X4 Y5]
+ (0.002745836470186797) [X0 Y1 Y4 X5]
+ (0.002929768674750955) [Y0 Z1 Y2 Z9]
+ (0.002929768674750955) [X0 Z1 X2 Z9]
+ (0.002929768674750955) [Y1 Z2 Y3 Z8]
+ (0.002929768674750955) [X1 Z2 X3 Z8]
+ (0.003276971931231539) [Y0 Z1 Y2 Z3]
+ (0.003276971931231539) [X0 Z1 X2 Z3]
+ (0.0033476175306661263) [Y0 Z1 Y2 Z7]
+ (0.0033476175306661263) [X0 Z1 X2 Z7]
+ (0.0033476175306661263) [Y1 Z2 Y3 Z6]
+ (0.0033476175306661263) [X1 Z2 X3 Z6]
+ (0.0035552901955041914) [Y0 Z1 Y2 Z10]
+ (0.0035552901955041914) [X0 Z1 X2 Z10]
+ (0.0035552901955041914) [Y1 Z2 Y3 Z11]
+ (0.0035552901955041914) [X1 Z2 X3 Z11]
+ (0.005143391768825089) [Y3 Y4 X5 X6]
+ (0.005143391768825089) [X3 X4 Y5 Y6]
+ (0.005283776488402974) [Y0 X1 X12 Y13]
+ (0.005283776488402974) [X0 Y1 Y12 X13]
+ (0.005530759218631447) [Y0 Z1 Y2 Z4]
+ (0.005530759218631447) [X0 Z1 X2 Z4]
+ (0.005530759218631447) [Y1 Z2 Y3 Z5]
+ (0.005530759218631447) [X1 Z2 X3 Z5]
+ (0.006087822480561887) [Y8 X9 X12 Y13]
+ (0.006087822480561887) [X8 Y9 Y12 X13]
+ (0.006509361201177233) [Y0 X1 X8 Y9]
+ (0.006509361201177233) [X0 Y1 Y8 X9]
+ (0.006888194352970603) [Y0 X1 X6 Y7]
+ (0.006888194352970603) [X0 Y1 Y6 X7]
+ (0.006901238249797227) [Y0 Z1 Y2 Z12]
+ (0.006901238249797227) [X0 Z1 X2 Z12]
+ (0.006901238249797227) [Y1 Z2 Y3 Z13]
+ (0.006901238249797227) [X1 Z2 X3 Z13]
+ (0.007156934919856924) [Y4 X5 X8 Y9]
+ (0.007156934919856924) [X4 Y5 Y8 X9]
+ (0.0077314252507753025) [Y0 X1 X10 Y11]
+ (0.0077314252507753025) [X0 Y1 Y10 X11]
+ (0.008032520918821307) [Y0 Z1 Y2 Z6]
+ (0.008032520918821307) [X0 Z1 X2 Z6]
+ (0.008032520918821307) [Y1 Z2 Y3 Z7]
+ (0.008032520918821307) [X1 Z2 X3 Z7]
+ (0.009560705729135997) [Y8 X9 X10 Y11]
+ (0.009560705729135997) [X8 Y9 Y10 X11]
+ (0.011055020596131974) [Y0 Z1 Y2 Z8]
+ (0.011055020596131974) [X0 Z1 X2 Z8]
+ (0.011055020596131974) [Y1 Z2 Y3 Z9]
+ (0.011055020596131974) [X1 Z2 X3 Z9]
+ (0.01128519020084086) [Y5 Y6 X11 X12]
+ (0.01128519020084086) [X5 X6 Y11 Y12]
+ (0.011307274008848137) [Y7 Z8 Z9 Y11]
+ (0.011307274008848137) [X7 Z8 Z9 X11]
+ (0.01198238901024791) [Y4 X5 X6 Y7]
+ (0.01198238901024791) [X4 Y5 Y6 X7]
+ (0.013873381748426184) [Y6 X7 X8 Y9]
+ (0.013873381748426184) [X6 Y7 Y8 X9]
+ (0.01458364890761257) [Y0 X1 X2 Y3]
+ (0.01458364890761257) [X0 Y1 Y2 X3]
+ (0.015577208063976503) [Y2 X3 X12 Y13]
+ (0.015577208063976503) [X2 Y3 Y12 X13]
+ (0.017366118994651434) [Y6 X7 X12 Y13]
+ (0.017366118994651434) [X6 Y7 Y12 X13]
+ (0.017680067952481542) [Y4 X5 X10 Y11]
+ (0.017680067952481542) [X4 Y5 Y10 X11]
+ (0.01782514099578633) [Y6 X7 X10 Y11]
+ (0.01782514099578633) [X6 Y7 Y10 X11]
+ (0.019028242443847435) [Y3 X4 X11 Y12]
+ (0.019028242443847435) [X3 Y4 Y11 X12]
+ (0.02538465750845759) [Y2 X3 X10 Y11]
+ (0.02538465750845759) [X2 Y3 Y10 X11]
+ (0.028685183716105907) [Y10 X11 X12 Y13]
+ (0.028685183716105907) [X10 Y11 Y12 X13]
+ (0.029812424517345573) [Y6 Z7 Z8 Y10]
+ (0.029812424517345573) [X6 Z7 Z8 X10]
+ (0.029812424517345573) [Y7 Z9 Z10 Y11]
+ (0.029812424517345573) [X7 Z9 Z10 X11]
+ (0.030104623143456702) [Y6 Z7 Z9 Y10]
+ (0.030104623143456702) [X6 Z7 Z9 X10]
+ (0.030104623143456702) [Y7 Z8 Z10 Y11]
+ (0.030104623143456702) [X7 Z8 Z10 X11]
+ (0.030787505389143835) [Y6 Z8 Z9 Y10]
+ (0.030787505389143835) [X6 Z8 Z9 X10]
+ (0.031143817988966985) [Y2 X3 X6 Y7]
+ (0.031143817988966985) [X2 Y3 Y6 X7]
+ (0.03583956795335353) [Y2 X3 X4 Y5]
+ (0.03583956795335353) [X2 Y3 Y4 X5]
+ (0.03619412355904253) [Y2 X3 X8 Y9]
+ (0.03619412355904253) [X2 Y3 Y8 X9]
+ (0.03831467029480393) [Y4 X5 X12 Y13]
+ (0.03831467029480393) [X4 Y5 Y12 X13]
+ (0.10433064780651347) [Z0 Y1 Z2 Y3]
+ (0.10433064780651347) [Z0 X1 Z2 X3]
+ (-0.12133276911042495) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042495) [X2 Z3 Z4 Z5 X6]
+ (-0.12133276911042495) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133276911042495) [X3 Z4 Z5 Z6 X7]
+ (3.2020768796858984e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.2020768796858984e-06) [X0 Z1 Z2 Z3 X4]
+ (3.2020768796858984e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.2020768796858984e-06) [X1 Z2 Z3 Z4 X5]
+ (0.22848106564918702) [Y6 Z7 Z8 Z9 Y10]
+ (0.22848106564918702) [X6 Z7 Z8 Z9 X10]
+ (0.2284810656491871) [Y7 Z8 Z9 Z10 Y11]
+ (0.2284810656491871) [X7 Z8 Z9 Z10 X11]
+ (-0.032767657823290525) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.032767657823290525) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.032767657823290525) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.032767657823290525) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.02711503684527316) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.02711503684527316) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.02711503684527316) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.02711503684527316) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.025996177598021336) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.025996177598021336) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.017561202409646193) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.017561202409646193) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.017561202409646193) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.017561202409646193) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.01456453123117298) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.01456453123117298) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.01456453123117298) [X7 Z8 Z9 X10 X12 X13]
+ (-0.01456453123117298) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.01221504099761386) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.01221504099761386) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.01221504099761386) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.01221504099761386) [X4 Z5 X6 X11 Z12 X13]
+ (-0.01221504099761386) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.01221504099761386) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.01221504099761386) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.01221504099761386) [X5 Z6 X7 X10 Z11 X12]
+ (-0.01175601341981933) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.01175601341981933) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.01175601341981933) [X3 Z4 Z5 X6 X8 X9]
+ (-0.01175601341981933) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.008764827575688854) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.008764827575688854) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.008764827575688854) [X2 Z3 Z4 X5 X11 X12]
+ (-0.008764827575688854) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.008764827575688854) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.008764827575688854) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.008764827575688854) [X3 X4 X10 Z11 Z12 X13]
+ (-0.008764827575688854) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.008125251921381018) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125251921381018) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.0073067599288329605) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.0073067599288329605) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.0073067599288329605) [X4 X5 X7 Z8 Z9 X10]
+ (-0.0073067599288329605) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005805188989826862) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.005805188989826862) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.005805188989826862) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.005805188989826862) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.005652620978017365) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.005652620978017365) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.005652620978017365) [X0 X1 X3 Z4 Z5 X6]
+ (-0.005652620978017365) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.005143391768825089) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.005143391768825089) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.005143391768825089) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.005143391768825089) [X2 Z3 X4 X5 Z6 X7]
+ (-0.0046849033881551806) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.0046849033881551806) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.0046686203187763075) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.0046686203187763075) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.004575007626639211) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.004575007626639211) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424855449441841) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424855449441841) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.004158797381840078) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.004158797381840078) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.004158797381840078) [X3 Z4 Z5 X6 X12 X13]
+ (-0.004158797381840078) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.0034937903598902115) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.0034937903598902115) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.0034937903598902115) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.0034937903598902115) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.0027790267990255705) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.0027790267990255705) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.0022939566113524667) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.0022939566113524667) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.0017992194936630214) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.0017992194936630214) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278753941369534) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.0017278753941369534) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298507967730014) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298507967730014) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298507967730014) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298507967730014) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533856254125578) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533856254125578) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008145313270956933) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008145313270956933) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008145313270956933) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008145313270956933) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.735036880594073e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735036880594073e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735036880594073e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735036880594073e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.774817864944868e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.774817864944868e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.774817864944868e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.774817864944868e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.518362215974511e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.518362215974511e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.518362215974511e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.518362215974511e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.4443446762002164e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.4443446762002164e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.4443446762002164e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.4443446762002164e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.5243738488606e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.5243738488606e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.5243738488606e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.5243738488606e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.290028433427619e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.290028433427619e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.290028433427619e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.290028433427619e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.974311713687663e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.974311713687663e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.27588312241988e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.27588312241988e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.643051068650268e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.643051068650268e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.556569218311546e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.556569218311546e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.253224225737478e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.253224225737478e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.7696594520888855e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.7696594520888855e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.6945132946254306e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.6945132946254306e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.6102971306974236e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.6102971306974236e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.6102971306974236e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.6102971306974236e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.3131455003353657e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.3131455003353657e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.2774831956102556e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.2774831956102556e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.2774831956102556e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.2774831956102556e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.211228348525234e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.211228348525234e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.211228348525234e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.211228348525234e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.1513463112438483e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.1513463112438483e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.0882507114488603e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.0882507114488603e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.1726691016789126e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.1726691016789126e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.454842449147877e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.454842449147877e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.330473188744651e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.330473188744651e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2283337825468933e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2283337825468933e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.0358477601977902e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.0358477601977902e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.95689537351061e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.95689537351061e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733197742638716e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733197742638716e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733197742638716e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733197742638716e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.628614202155426e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.628614202155426e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.556281914816208e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.556281914816208e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.556281914816208e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.556281914816208e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.418291574878964e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.418291574878964e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.418291574878964e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.418291574878964e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.927453083092801e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927453083092801e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927453083092801e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927453083092801e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627851911805849e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627851911805849e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660624925055e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660624925055e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660624925055e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660624925055e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660624925055e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660624925055e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660624925055e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660624925055e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.83805275126745e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.83805275126745e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.5707613294719016e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.5707613294719016e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.328139350871679e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.328139350871679e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.0868265653946637e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.0868265653946637e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.0868265653946637e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.0868265653946637e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.4473231289665856e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.4473231289665856e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.3713289481609035e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.3713289481609035e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.3713289481609035e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.3713289481609035e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.199051619023248e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.199051619023248e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.9332412772431913e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.9332412772431913e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.9332412772431913e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.9332412772431913e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.8394209156366094e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.8394209156366094e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.8394209156366094e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.8394209156366094e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.5510539176743308e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.5510539176743308e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.5510539176743308e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.5510539176743308e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.380778148156451e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.380778148156451e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.380778148156451e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.380778148156451e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.380778148156451e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.380778148156451e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.380778148156451e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.380778148156451e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.380778148156451e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.380778148156451e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.380778148156451e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.380778148156451e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.2919694865365825e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.2919694865365825e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.1076325599652466e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.1076325599652466e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.1076325599652466e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.1076325599652466e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.1076325599652466e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.1076325599652466e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.1076325599652466e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.1076325599652466e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.057446595459161e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057446595459161e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057446595459161e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057446595459161e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649310134074021e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649310134074021e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649310134074021e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649310134074021e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.8394209156366094e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.8394209156366094e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.8394209156366094e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.8394209156366094e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.199051619023248e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.199051619023248e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.4473231289665856e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.4473231289665856e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.2362599615683056e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.2362599615683056e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.2362599615683056e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.2362599615683056e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.328139350871679e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.328139350871679e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.5707613294719016e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.5707613294719016e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.83805275126745e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.83805275126745e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.627851911805849e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627851911805849e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.628614202155426e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.628614202155426e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.95689537351061e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.95689537351061e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.30653665226687e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.30653665226687e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.30653665226687e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.30653665226687e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.0358477601977902e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.0358477601977902e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2283337825468933e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2283337825468933e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.2393363217661532e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393363217661532e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393363217661532e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393363217661532e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.330473188744651e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.330473188744651e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.454842449147877e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.454842449147877e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.1726691016789126e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.1726691016789126e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.0882507114488603e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.0882507114488603e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.1174479463418126e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.1174479463418126e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.1513463112438483e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.1513463112438483e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.3131455003353657e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.3131455003353657e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.3343312895760437e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.3343312895760437e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.6945132946254306e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.6945132946254306e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.1839325596075275e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.1839325596075275e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.556569218311546e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.556569218311546e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.643051068650268e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.643051068650268e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.27588312241988e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.27588312241988e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.974311713687663e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.974311713687663e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.0002921986261111303) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.0002921986261111303) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.0002921986261111303) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.0002921986261111303) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004956762314918012) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004956762314918012) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650070219498668) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650070219498668) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650070219498668) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650070219498668) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533856254125578) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533856254125578) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.0016095313817213763) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.0016095313817213763) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.0016095313817213763) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.0016095313817213763) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0016676041811440868) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.0016676041811440868) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.0016676041811440868) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.0016676041811440868) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.0017278753941369534) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.0017278753941369534) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.0017992194936630214) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.0017992194936630214) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.0022939566113524667) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.0022939566113524667) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.0024629170071339343) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.0024629170071339343) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.0024629170071339343) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.0024629170071339343) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.003961560792496554) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.003961560792496554) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.003961560792496554) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.003961560792496554) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.004424855449441841) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424855449441841) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004575007626639211) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.004575007626639211) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.0046686203187763075) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.0046686203187763075) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.0046849033881551806) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.0046849033881551806) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005324835234221642) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.005324835234221642) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.005324835234221642) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.005324835234221642) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.0053686593581094115) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.0053686593581094115) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.0053686593581094115) [X2 X3 X7 Z8 Z9 X10]
+ (0.0053686593581094115) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.00796088072592155) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.00796088072592155) [X4 Z5 X6 X10 Z11 X12]
+ (0.00796088072592155) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.00796088072592155) [X5 Z6 X7 X11 Z12 X13]
+ (0.008125251921381018) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125251921381018) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.008890731522694552) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.008890731522694552) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.008890731522694552) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.008890731522694552) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.01026341486815858) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.01026341486815858) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.01026341486815858) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.01026341486815858) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.01054042590767152) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.01054042590767152) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.01054042590767152) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.01054042590767152) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010960074940542443) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010960074940542443) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010960074940542443) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010960074940542443) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307274008848137) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307274008848137) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.0144110994301309) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.0144110994301309) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.0144110994301309) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.0144110994301309) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.015225630757226593) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.015225630757226593) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.015225630757226593) [X3 Z4 Z5 X6 X10 X11]
+ (0.015225630757226593) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.015588250102380222) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.015588250102380222) [X2 Z3 X4 X10 Z11 X12]
+ (0.015588250102380222) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.015588250102380222) [X3 Z4 X5 X11 Z12 X13]
+ (0.018266834869375404) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.018266834869375404) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.018266834869375404) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.018266834869375404) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.01902042317303985) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.01902042317303985) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.01902042317303985) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.01902042317303985) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.020175921723535415) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.020175921723535415) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.020175921723535415) [X4 Z5 Z6 X7 X11 X12]
+ (0.020175921723535415) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.020175921723535415) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.020175921723535415) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.020175921723535415) [X5 X6 X10 Z11 Z12 X13]
+ (0.020175921723535415) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.024353077678069074) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.024353077678069074) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.024353077678069074) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.024353077678069074) [X2 Z3 X4 X11 Z12 X13]
+ (0.024353077678069074) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.024353077678069074) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.024353077678069074) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.024353077678069074) [X3 Z4 X5 X10 Z11 X12]
+ (0.024389082531149263) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.024389082531149263) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.024389082531149263) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.024389082531149263) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.025104957138844503) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.025104957138844503) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.025104957138844503) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.025104957138844503) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.03078750538914384) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.03078750538914384) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.04587947078129831) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.04587947078129831) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.056007330877807224) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.056007330877807224) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.056007330877807224) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.056007330877807224) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.056084681246613165) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.056084681246613165) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.056084681246613165) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.056084681246613165) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.631277928664189e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631277928664189e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.631277928664187e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631277928664187e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.595086007046773e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.595086007046773e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.5950860070467727e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.5950860070467727e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0427432770137838) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.0427432770137838) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (0.042743277013783824) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.042743277013783824) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.04764261217638315) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.04764261217638315) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.04764261217638315) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.04764261217638315) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.0417188138398218) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.0417188138398218) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.0417188138398218) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.0417188138398218) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.03956441632289352) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.03956441632289352) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.03956441632289352) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03956441632289352) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03935916802205328) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.03935916802205328) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.03935916802205328) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.03935916802205328) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.03931805194719763) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03931805194719763) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.03931805194719763) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03931805194719763) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.035608378988312706) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.035608378988312706) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.029903789512624946) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.029903789512624946) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.029903789512624946) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.029903789512624946) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.028730779551905568) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.028730779551905568) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.028730779551905568) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.028730779551905568) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.025637238296026904) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.025637238296026904) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.025637238296026904) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.025637238296026904) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.024755463292891113) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.024755463292891113) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.024755463292891113) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.024755463292891113) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.02428211735469314) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.02428211735469314) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.023145130929529217) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.023145130929529217) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.022528440196012932) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.022528440196012932) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02143381072160106) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.02143381072160106) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.02143381072160106) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.02143381072160106) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251617) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.019257505095251617) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019028242443847435) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.019028242443847435) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.018889030304942905) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.018889030304942905) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.018889030304942905) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.018889030304942905) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.01602460368917953) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.01602460368917953) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.015225630757226593) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225630757226593) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.014603704729162172) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.014603704729162172) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.01456453123117298) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.01456453123117298) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.011756013419819331) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.011756013419819331) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.01128519020084086) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.01128519020084086) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.009841749246962664) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009841749246962664) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.009612634606847375) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612634606847375) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612634606847375) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612634606847375) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.008469978791023885) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.008469978791023885) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.0073067599288329605) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.0073067599288329605) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.00592379833656135) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.00592379833656135) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.005652620978017365) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.005652620978017365) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.0053686593581094115) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0053686593581094115) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.004158797381840078) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.004158797381840078) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.003356670563832917) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.003356670563832917) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.003356670563832917) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.003356670563832917) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.0032675138544235767) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.0032675138544235767) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.0032675138544235767) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.0032675138544235767) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.0027790267990255705) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.0027790267990255705) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.002686040977806611) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.002686040977806611) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.002686040977806611) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.002686040977806611) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.0022939566113524667) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524667) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.0022939566113524667) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524667) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.0009581655836696571) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.0009581655836696571) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0009581655836696571) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.0009581655836696571) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.0009581655836696571) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.0009581655836696571) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0009581655836696571) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.0009581655836696571) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.0002463643756958867) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0002463643756958867) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0001384017730354948) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.0001384017730354948) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.0001384017730354948) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.0001384017730354948) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.735036880594073e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735036880594073e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585306416434e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585306416434e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.6103585306416434e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.6103585306416434e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808795953226e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.5316808795953226e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808795953226e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5316808795953226e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.806102775548001e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.806102775548001e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.806102775548001e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.806102775548001e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.089799467872635e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.089799467872635e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.089799467872635e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.089799467872635e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.652209669658607e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.652209669658607e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.652209669658607e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.652209669658607e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851834095528e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.481851834095528e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851834095528e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.481851834095528e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.071480736608926e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.071480736608926e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.071480736608926e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.071480736608926e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.734622038939074e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.734622038939074e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.734622038939074e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.734622038939074e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.728843147398368e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.728843147398368e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.728843147398368e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.728843147398368e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.253224225737478e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.253224225737478e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.7696594520888855e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.7696594520888855e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.5443954294745084e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.5443954294745084e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.5443954294745084e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.5443954294745084e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.5443954294745084e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.5443954294745084e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.5443954294745084e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.5443954294745084e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.3609563204742663e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563204742663e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563204742663e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.3609563204742663e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.103215604735264e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.103215604735264e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.103215604735264e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.103215604735264e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.011122098241627e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.011122098241627e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.011122098241627e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.011122098241627e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.942946836752961e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.942946836752961e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.942946836752961e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.942946836752961e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174771951568e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.6541174771951568e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174771951568e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.6541174771951568e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.5224930676413337e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224930676413337e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224930676413337e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224930676413337e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224930676413337e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930676413337e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930676413337e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224930676413337e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.2283337825468933e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337825468933e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.2283337825468933e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337825468933e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.988770289183155e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988770289183155e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988770289183155e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988770289183155e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.867765104632099e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104632099e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104632099e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.867765104632099e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.189990975603595e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.189990975603595e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246207334028e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246207334028e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.471647744839624e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.471647744839624e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.5614471802564823e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.5614471802564823e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.5614471802564823e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.5614471802564823e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.5233896780048626e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.5233896780048626e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.427323108926673e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.427323108926673e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.427323108926673e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.427323108926673e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.328139350871679e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350871679e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.328139350871679e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350871679e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.0868265653946637e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.0868265653946637e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.8882935955780405e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935955780405e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935955780405e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.8882935955780405e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.3713289481609033e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.3713289481609033e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.8394209156366094e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.8394209156366094e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.057446595459161e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057446595459161e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.5371780954480096e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.5371780954480096e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.5371780954480096e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.5371780954480096e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.057446595459161e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057446595459161e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.209350649363685e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350649363685e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350649363685e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.209350649363685e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783556307958e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783556307958e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783556307958e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783556307958e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.8394209156366094e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.8394209156366094e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.3713289481609033e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.3713289481609033e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.0868265653946637e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.0868265653946637e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.5233896780048626e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.5233896780048626e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.471647744839624e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.471647744839624e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246207334028e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246207334028e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189990975603595e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.189990975603595e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.330473188744651e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.330473188744651e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.330473188744651e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.330473188744651e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288532436025145e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288532436025145e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288532436025145e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288532436025145e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.6893489515418418e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.6893489515418418e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.6893489515418418e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.6893489515418418e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.7455184005561932e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.7455184005561932e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.7455184005561932e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.7455184005561932e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.7455184005561932e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.7455184005561932e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.7455184005561932e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.7455184005561932e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.2118420191831755e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420191831755e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.2118420191831755e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420191831755e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.2118420191831755e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420191831755e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.2118420191831755e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420191831755e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.3131455003353653e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.3131455003353653e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.3131455003353653e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.3131455003353653e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.3343312895760433e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.3343312895760433e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.1839325596075275e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.1839325596075275e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.735036880594073e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735036880594073e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.0002463643756958867) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.0002463643756958867) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00044585351288406504) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.00044585351288406504) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.00044585351288406504) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.00044585351288406504) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940221543005155) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005155) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005155) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005155) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005155) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005155) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940221543005155) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005155) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533856254125578) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533856254125578) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533856254125578) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533856254125578) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0010435246534907612) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435246534907612) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435246534907612) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435246534907612) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.0012803060973496775) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.0012803060973496775) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.0012803060973496775) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.0012803060973496775) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.0013038004788126958) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.0013038004788126958) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.0013038004788126958) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.0013038004788126958) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.002261966062482353) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.002261966062482353) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.002261966062482353) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.002261966062482353) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.002261966062482353) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.002261966062482353) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.002261966062482353) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.002261966062482353) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.003989841456619307) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.003989841456619307) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.003989841456619307) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.003989841456619307) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.004158797381840078) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.004158797381840078) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.004311038507914338) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.004311038507914338) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.004311038507914338) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.004311038507914338) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.004636976661182594) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.004636976661182594) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.004636976661182594) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.004636976661182594) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.005114473831660374) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.005114473831660374) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.005114473831660374) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.005114473831660374) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.005114473831660374) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660374) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660374) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005114473831660374) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00524153538280389) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.00524153538280389) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.00524153538280389) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.00524153538280389) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.005262642473076823) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262642473076823) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262642473076823) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262642473076823) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.0053686593581094115) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.0053686593581094115) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005379937155839385) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.005379937155839385) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.005379937155839385) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.005379937155839385) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.005652620978017365) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.005652620978017365) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.005708495985960889) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708495985960889) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708495985960889) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708495985960889) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.00592379833656135) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.00592379833656135) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.0073067599288329605) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.0073067599288329605) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469978791023885) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.008469978791023885) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.009841749246962664) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.009841749246962664) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.01128519020084086) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.01128519020084086) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.011756013419819331) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.011756013419819331) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.01456453123117298) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.01456453123117298) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603704729162172) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.014603704729162172) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.015225630757226593) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225630757226593) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.01602460368917953) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.01602460368917953) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.019028242443847435) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.019028242443847435) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.019257505095251617) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.019257505095251617) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.04587947078129831) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04587947078129831) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.36937089366156334) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.36937089366156334) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.36937089366156334) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.36937089366156334) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.28164257767023154) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.28164257767023154) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.28164257767023154) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.28164257767023154) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.09065144207036471) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.09065144207036471) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.09065144207036471) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.09065144207036471) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863618) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684737589863618) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863618) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684737589863618) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635021950635038) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635021950635038) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635021950635038) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635021950635038) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214055) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.06752385099214055) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214055) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.06752385099214055) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.035608378988312706) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.035608378988312706) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.034903343373661674) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.034903343373661674) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.034903343373661674) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.034903343373661674) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02459186088382998) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.02459186088382998) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02459186088382998) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.02459186088382998) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.02428211735469314) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.02428211735469314) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.023145130929529217) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.023145130929529217) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.022528440196012932) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.022528440196012932) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019538050311314892) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.019538050311314892) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.019538050311314892) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.019538050311314892) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.017091553155899008) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.017091553155899008) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.017091553155899008) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.017091553155899008) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.01602460368917953) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.01602460368917953) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.01602460368917953) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.01602460368917953) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.010311482489831691) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831691) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831691) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311482489831691) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.009841749246962666) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962666) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.009841749246962666) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962666) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209842) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209842) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209842) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008826368514209842) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008541996625454882) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454882) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454882) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454882) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454882) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454882) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454882) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008541996625454882) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008469978791023885) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023885) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.008469978791023885) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023885) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.0046686203187763075) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0046686203187763075) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.0038764708993369638) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.0038764708993369638) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.003804066171728536) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728536) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728536) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003804066171728536) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178765) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178765) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003356670563832917) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.003356670563832917) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.0032675138544235767) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.0032675138544235767) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.002141361223101622) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.002141361223101622) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.0017278753941369534) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.0017278753941369534) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.0016407548553123855) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0016407548553123855) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214169417) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0014528843214169417) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214169417) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0014528843214169417) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870896771024483) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007870896771024483) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192743499487663) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192743499487663) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.00019400857029755887) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00019400857029755887) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0001384017730354948) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.0001384017730354948) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.14162522115909e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.14162522115909e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.14162522115909e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.14162522115909e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.071480736608925e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.071480736608925e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.1513463112438483e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.1513463112438483e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0882507114488603e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.0882507114488603e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.9885117065294364e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.9885117065294364e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.874299071538304e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.874299071538304e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.3609563204742663e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.3609563204742663e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.3002946563098448e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3002946563098448e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.1468376507892282e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.1468376507892282e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.1468376507892282e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.1468376507892282e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.352332103407341e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.352332103407341e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.352332103407341e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.352332103407341e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091637199287883e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637199287883e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637199287883e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637199287883e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637199287883e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637199287883e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637199287883e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091637199287883e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.074305986224726e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.074305986224726e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.074305986224726e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.074305986224726e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128986609258e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128986609258e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128986609258e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128986609258e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867765104632099e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.867765104632099e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560692465184438e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692465184438e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692465184438e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692465184438e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692465184438e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692465184438e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692465184438e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560692465184438e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.997018422275467e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.997018422275467e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.997018422275467e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.997018422275467e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.997018422275467e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.997018422275467e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.997018422275467e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.997018422275467e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.56824752128302e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.56824752128302e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.56824752128302e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.56824752128302e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393086043967e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393086043967e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393086043967e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393086043967e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393086043967e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393086043967e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.3767393086043967e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393086043967e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.8882935955780405e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.8882935955780405e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6863815451650034e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.6863815451650034e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035783556307958e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7035783556307958e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.209350649363684e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.209350649363684e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244220184e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773244220184e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244220184e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773244220184e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244220184e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773244220184e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773244220184e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773244220184e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253794777827e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253794777827e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9742253794777827e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9742253794777827e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0474716555420013e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.0474716555420013e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.0474716555420013e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.0474716555420013e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.209350649363684e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.209350649363684e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0717282183921088e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0717282183921088e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0717282183921088e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0717282183921088e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.2004287494671857e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.2004287494671857e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.2004287494671857e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.2004287494671857e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.7035783556307958e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7035783556307958e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.3120943053599002e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.3120943053599002e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.3120943053599002e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.3120943053599002e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.6863815451650034e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815451650034e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.8882935955780405e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.8882935955780405e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.092250616227479e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.092250616227479e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.092250616227479e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.092250616227479e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.092250616227479e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.092250616227479e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.092250616227479e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.092250616227479e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.444597854342669e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.444597854342669e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.444597854342669e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.444597854342669e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.6849150954147767e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.6849150954147767e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.6849150954147767e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.6849150954147767e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246974425726136e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246974425726136e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246974425726136e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246974425726136e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246974425726136e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246974425726136e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246974425726136e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246974425726136e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.867765104632099e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.867765104632099e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3002946563098448e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3002946563098448e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.3609563204742663e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.3609563204742663e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.874299071538304e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.874299071538304e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.883676576243825e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.883676576243825e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.9473560118220867e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9473560118220867e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.9473560118220867e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9473560118220867e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117065294364e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9885117065294364e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.0882507114488603e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.0882507114488603e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.1513463112438483e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.1513463112438483e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.846201671467053e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846201671467053e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846201671467053e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846201671467053e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.071480736608925e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.071480736608925e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.105526722273649e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.105526722273649e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.105526722273649e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.105526722273649e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.1464963277768975e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.1464963277768975e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.1464963277768975e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.1464963277768975e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159350502223255e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159350502223255e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159350502223255e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159350502223255e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427988656739754e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427988656739754e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427988656739754e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427988656739754e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935867718351523e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935867718351523e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935867718351523e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935867718351523e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348466434e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.253273348466434e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979825793811952e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979825793811952e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979825793811952e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979825793811952e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.2055484112213035e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.2055484112213035e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.2055484112213035e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.2055484112213035e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0001384017730354948) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.0001384017730354948) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.0001878705338954441) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0001878705338954441) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0001878705338954441) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0001878705338954441) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00019400857029755887) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00019400857029755887) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0002463643756958867) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0002463643756958867) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0002463643756958867) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0002463643756958867) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192743499487663) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192743499487663) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156734248908572) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156734248908572) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156734248908572) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156734248908572) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024483) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007870896771024483) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.0015324835230730112) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.0015324835230730112) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.0015324835230730112) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.0015324835230730112) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.0016407548553123855) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0016407548553123855) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0017278753941369534) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.0017278753941369534) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.0024464971554158843) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.0024464971554158843) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.0024464971554158843) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.0024464971554158843) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.0032675138544235767) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.0032675138544235767) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.003356670563832917) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.003356670563832917) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.0034841573002178765) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0034841573002178765) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0038764708993369638) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.0038764708993369638) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.0046686203187763075) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.0046686203187763075) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278123) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.004767272188278123) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.004767272188278123) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278123) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.005286546538226889) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.005286546538226889) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.005286546538226889) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.005286546538226889) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.005408954422409976) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.005408954422409976) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.005408954422409976) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.005408954422409976) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.00592379833656135) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.00592379833656135) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.00592379833656135) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.00592379833656135) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.010715508469796735) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010715508469796735) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010715508469796735) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010715508469796735) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.010757563953908946) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010757563953908946) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010757563953908946) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010757563953908946) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.014603704729162172) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.014603704729162172) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.014603704729162172) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.014603704729162172) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.019299560579363828) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363828) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363828) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363828) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363828) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363828) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.019299560579363828) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363828) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0585919887338619) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0585919887338619) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.775950527652497e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.775950527652497e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.7759505276525e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.7759505276525e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.07165035181002549) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07165035181002549) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.07165035181002556) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.07165035181002556) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251617) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019257505095251617) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010311482489831691) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311482489831691) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008826368514209842) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209842) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.007597464029770588) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.007597464029770588) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.007597464029770588) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.007597464029770588) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311875) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311875) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311875) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311875) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311875) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311875) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311875) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311875) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676588) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005348051582676588) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005348051582676588) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676588) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.003804066171728536) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003804066171728536) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219503) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.0029841661681219503) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.0029841661681219503) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.0029841661681219503) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.0024464971554158848) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.0024464971554158848) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.0022494124470939995) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0022494124470939995) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022494124470939995) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0022494124470939995) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.002141361223101622) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.002141361223101622) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.0018638942824587116) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587116) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587116) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587116) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587116) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587116) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0018638942824587116) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587116) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0016407548553123857) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553123857) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0016407548553123857) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553123857) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0012223378081538503) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538503) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538503) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538503) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538503) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538503) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538503) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0012223378081538503) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0010283292378562917) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0010283292378562917) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0010283292378562917) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0010283292378562917) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1463061453667715e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1463061453667715e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.874299071538304e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.874299071538304e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.874299071538304e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.874299071538304e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3002946563098448e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3002946563098448e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3002946563098448e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3002946563098448e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.0444941298661891e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.0444941298661891e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.0444941298661891e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.0444941298661891e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.956079230574537e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.956079230574537e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.956079230574537e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.956079230574537e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105515037580785e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105515037580785e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105515037580785e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105515037580785e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661347213524373e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661347213524373e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661347213524373e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661347213524373e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341414061602e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341414061602e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189990975603595e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.189990975603595e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.87662165857547e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.87662165857547e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.87662165857547e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.87662165857547e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246207334028e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246207334028e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.5233896780048626e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.5233896780048626e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.076732532301885e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.076732532301885e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.076732532301885e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.076732532301885e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.0134714590578633e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0134714590578633e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.904599884600289e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.904599884600289e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.904599884600289e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.904599884600289e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.6667317550829396e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.6667317550829396e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.6667317550829396e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6667317550829396e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641929937532e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.8505641929937532e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309315628543e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6569309315628543e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309315628543e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6569309315628543e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.8505641929937532e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.8505641929937532e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815451650034e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815451650034e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.6863815451650034e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815451650034e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714590578633e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.0134714590578633e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.5233896780048626e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.5233896780048626e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.670402390620718e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.670402390620718e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.670402390620718e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.670402390620718e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246207334028e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246207334028e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189990975603595e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.189990975603595e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341414061602e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341414061602e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949476487660031e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949476487660031e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.7924939577927263e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939577927263e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939577927263e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.7924939577927263e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.8836765762438254e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.8836765762438254e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.9885117065294364e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117065294364e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117065294364e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117065294364e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348466434e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253273348466434e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4017109735963575e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4017109735963575e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4017109735963575e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4017109735963575e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603693756298e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.5809603693756298e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603693756298e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.5809603693756298e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192743499487663) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487663) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192743499487663) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487663) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.0007870896771024483) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024483) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024483) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024483) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00117263483164418) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.00117263483164418) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.00117263483164418) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00117263483164418) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012366478019245153) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.0012366478019245153) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.0012366478019245153) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.0012366478019245153) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.0022009640695004715) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0022009640695004715) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0022009640695004715) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0022009640695004715) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00239497263979803) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00239497263979803) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00239497263979803) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00239497263979803) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00239497263979803) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.00239497263979803) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.00239497263979803) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.00239497263979803) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0024464971554158848) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.0024464971554158848) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.003804066171728536) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.003804066171728536) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0038764708993369638) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.0038764708993369638) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.0038764708993369638) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.0038764708993369638) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.004220813970046465) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.004220813970046465) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.004220813970046465) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.004220813970046465) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.008826368514209842) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.008826368514209842) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.010311482489831691) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311482489831691) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257505095251617) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019257505095251617) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.05859198873386189) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.05859198873386189) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3987009017009292e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.3987009017009292e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.3987009017009292e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3987009017009292e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178765) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178765) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219503) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.0029841661681219503) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.00019400857029755887) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00019400857029755887) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453667715e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453667715e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7924939577927263e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7924939577927263e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540341414061602e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341414061602e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540341414061602e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341414061602e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.8505641929937532e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641929937532e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641929937532e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641929937532e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714590578633e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0134714590578633e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714590578633e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0134714590578633e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949476487660032e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949476487660032e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.7924939577927263e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7924939577927263e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029755887) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029755887) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0029841661681219503) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.0029841661681219503) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.0034841573002178765) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0034841573002178765) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
List of core orbitals: [0, 1, 2]
List of active orbitals: [3, 4, 5, 6]
Number of qubits: 8
Number of qubits required to perform quantum simulations: 8
Hamiltonian of the water molecule
  (-73.13873231352537) [I0]
+ (-0.18066792656583636) [Z6]
+ (-0.18066792656583636) [Z7]
+ (-0.1596143250181016) [Z5]
+ (-0.15961432501810158) [Z4]
+ (0.17419956155055485) [Z2]
+ (0.17419956155055494) [Z3]
+ (0.22757269005453334) [Z0]
+ (0.2275726900545334) [Z1]
+ (-8.194261372082751e-06) [Y4 Y6]
+ (-8.194261372082751e-06) [X4 X6]
+ (7.954413176383619e-06) [Y5 Y7]
+ (7.954413176383619e-06) [X5 X7]
+ (0.11270386920332197) [Z4 Z6]
+ (0.11270386920332197) [Z5 Z7]
+ (0.11952438964682674) [Z0 Z4]
+ (0.11952438964682674) [Z1 Z5]
+ (0.1340171526196371) [Z0 Z6]
+ (0.1340171526196371) [Z1 Z7]
+ (0.13734953064261315) [Z0 Z5]
+ (0.13734953064261315) [Z1 Z4]
+ (0.1376687264585256) [Z2 Z4]
+ (0.1376687264585256) [Z3 Z5]
+ (0.141389052919428) [Z4 Z7]
+ (0.141389052919428) [Z5 Z6]
+ (0.14722943218766152) [Z2 Z5]
+ (0.14722943218766152) [Z3 Z4]
+ (0.1492635514738888) [Z4 Z5]
+ (0.14973486803496922) [Z2 Z6]
+ (0.14973486803496922) [Z3 Z7]
+ (0.15138327161428844) [Z0 Z7]
+ (0.15138327161428844) [Z1 Z6]
+ (0.15435748657223627) [Z6 Z7]
+ (0.15582269051553105) [Z2 Z7]
+ (0.15582269051553105) [Z3 Z6]
+ (0.16756653265461285) [Z0 Z2]
+ (0.16756653265461285) [Z1 Z3]
+ (0.18143991440303897) [Z0 Z3]
+ (0.18143991440303897) [Z1 Z2]
+ (0.1939253461327024) [Z0 Z1]
+ (0.2200397733437609) [Z2 Z3]
+ (-7.037887510685426e-06) [Y4 Z5 Y6]
+ (-7.037887510685426e-06) [X4 Z5 X6]
+ (-7.037887510685425e-06) [Y5 Z6 Y7]
+ (-7.037887510685425e-06) [X5 Z6 X7]
+ (-0.028685183716106035) [Y4 Y5 X6 X7]
+ (-0.028685183716106035) [X4 X5 Y6 Y7]
+ (-0.01782514099578641) [Y0 Y1 X4 X5]
+ (-0.01782514099578641) [X0 X1 Y4 Y5]
+ (-0.017366118994651333) [Y0 Y1 X6 X7]
+ (-0.017366118994651333) [X0 X1 Y6 Y7]
+ (-0.0138733817484261) [Y0 Y1 X2 X3]
+ (-0.0138733817484261) [X0 X1 Y2 Y3]
+ (-0.009560705729135933) [Y2 Y3 X4 X5]
+ (-0.009560705729135933) [X2 X3 Y4 Y5]
+ (-0.006087822480561847) [Y2 Y3 X6 X7]
+ (-0.006087822480561847) [X2 X3 Y6 Y7]
+ (-0.0002921986261110599) [Y1 Y2 X3 X4]
+ (-0.0002921986261110599) [X1 X2 Y3 Y4]
+ (-8.194261372082751e-06) [Z4 Y5 Z6 Y7]
+ (-8.194261372082751e-06) [Z4 X5 Z6 X7]
+ (-2.8909678815373494e-06) [Z0 Y5 Z6 Y7]
+ (-2.8909678815373494e-06) [Z0 X5 Z6 X7]
+ (-2.8909678815373494e-06) [Z1 Y4 Z5 Y6]
+ (-2.8909678815373494e-06) [Z1 X4 Z5 X6]
+ (-1.8551201214471913e-06) [Z0 Y4 Z5 Y6]
+ (-1.8551201214471913e-06) [Z0 X4 Z5 X6]
+ (-1.8551201214471913e-06) [Z1 Y5 Z6 Y7]
+ (-1.8551201214471913e-06) [Z1 X5 Z6 X7]
+ (-1.597317197697585e-06) [Z2 Y4 Z5 Y6]
+ (-1.597317197697585e-06) [Z2 X4 Z5 X6]
+ (-1.597317197697585e-06) [Z3 Y5 Z6 Y7]
+ (-1.597317197697585e-06) [Z3 X5 Z6 X7]
+ (-1.0358477600901583e-06) [Y0 X1 X5 Y6]
+ (-1.0358477600901583e-06) [Y0 Y1 Y5 Y6]
+ (-1.0358477600901583e-06) [X0 X1 X5 X6]
+ (-1.0358477600901583e-06) [X0 Y1 Y5 X6]
+ (-9.344557775241119e-07) [Z2 Y5 Z6 Y7]
+ (-9.344557775241119e-07) [Z2 X5 Z6 X7]
+ (-9.344557775241119e-07) [Z3 Y4 Z5 Y6]
+ (-9.344557775241119e-07) [Z3 X4 Z5 X6]
+ (6.62861420173473e-07) [Y2 X3 X5 Y6]
+ (6.62861420173473e-07) [Y2 Y3 Y5 Y6]
+ (6.62861420173473e-07) [X2 X3 X5 X6]
+ (6.62861420173473e-07) [X2 Y3 Y5 X6]
+ (7.954413176383617e-06) [Y4 Z5 Y6 Z7]
+ (7.954413176383617e-06) [X4 Z5 X6 Z7]
+ (0.0002921986261110599) [Y1 X2 X3 Y4]
+ (0.0002921986261110599) [X1 Y2 Y3 X4]
+ (0.006087822480561847) [Y2 X3 X6 Y7]
+ (0.006087822480561847) [X2 Y3 Y6 X7]
+ (0.009560705729135933) [Y2 X3 X4 Y5]
+ (0.009560705729135933) [X2 Y3 Y4 X5]
+ (0.011307274008848126) [Y1 Z2 Z3 Y5]
+ (0.011307274008848126) [X1 Z2 Z3 X5]
+ (0.0138733817484261) [Y0 X1 X2 Y3]
+ (0.0138733817484261) [X0 Y1 Y2 X3]
+ (0.017366118994651333) [Y0 X1 X6 Y7]
+ (0.017366118994651333) [X0 Y1 Y6 X7]
+ (0.01782514099578641) [Y0 X1 X4 Y5]
+ (0.01782514099578641) [X0 Y1 Y4 X5]
+ (0.028685183716106035) [Y4 X5 X6 Y7]
+ (0.028685183716106035) [X4 Y5 Y6 X7]
+ (0.029812424517345712) [Y0 Z1 Z2 Y4]
+ (0.029812424517345712) [X0 Z1 Z2 X4]
+ (0.029812424517345712) [Y1 Z3 Z4 Y5]
+ (0.029812424517345712) [X1 Z3 Z4 X5]
+ (0.03010462314345677) [Y0 Z1 Z3 Y4]
+ (0.03010462314345677) [X0 Z1 Z3 X4]
+ (0.03010462314345677) [Y1 Z2 Z4 Y5]
+ (0.03010462314345677) [X1 Z2 Z4 X5]
+ (0.0307875053891439) [Y0 Z2 Z3 Y4]
+ (0.0307875053891439) [X0 Z2 Z3 X4]
+ (0.04375263801065892) [Y0 Z1 Z2 Z3 Y4]
+ (0.04375263801065892) [X0 Z1 Z2 Z3 X4]
+ (0.04375263801065892) [Y1 Z2 Z3 Z4 Y5]
+ (0.04375263801065892) [X1 Z2 Z3 Z4 X5]
+ (-0.01456453123117301) [Y1 Z2 Z3 X4 X6 Y7]
+ (-0.01456453123117301) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-0.01456453123117301) [X1 Z2 Z3 X4 X6 X7]
+ (-0.01456453123117301) [X1 Z2 Z3 Y4 Y6 X7]
+ (-6.524373848556706e-06) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (-6.524373848556706e-06) [X0 Z1 Z2 Z3 Z4 X6]
+ (-6.524373848556706e-06) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (-6.524373848556706e-06) [X1 Z2 Z3 Z5 Z6 X7]
+ (-3.769659452021961e-06) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (-3.769659452021961e-06) [X0 Z2 Z3 Z4 Z5 X6]
+ (-3.6102971306194906e-06) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (-3.6102971306194906e-06) [X0 Z1 Z3 Z4 Z5 X6]
+ (-3.6102971306194906e-06) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (-3.6102971306194906e-06) [X1 Z2 Z4 Z5 Z6 X7]
+ (-3.313145500135761e-06) [Y1 Z2 Z3 Y4 X5 X6]
+ (-3.313145500135761e-06) [X1 Z2 Z3 X4 Y5 Y6]
+ (-3.277483195566787e-06) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (-3.277483195566787e-06) [X0 Z1 Z2 Z4 Z5 X6]
+ (-3.277483195566787e-06) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (-3.277483195566787e-06) [X1 Z3 Z4 Z5 Z6 X7]
+ (-3.2112283484209455e-06) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (-3.2112283484209455e-06) [X0 Z1 Z2 Z3 Z5 X6]
+ (-3.2112283484209455e-06) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (-3.2112283484209455e-06) [X1 Z2 Z3 Z4 Z6 X7]
+ (-1.0358477600901583e-06) [Y0 Y1 X4 Z5 Z6 X7]
+ (-1.0358477600901583e-06) [X0 X1 Y4 Z5 Z6 Y7]
+ (-6.62861420173473e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (-6.62861420173473e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (-3.3281393505270383e-07) [Y1 X2 X3 Z4 Z5 Y6]
+ (-3.3281393505270383e-07) [X1 Y2 Y3 Z4 Z5 X6]
+ (3.3281393505270383e-07) [Y1 Y2 X3 Z4 Z5 X6]
+ (3.3281393505270383e-07) [X1 X2 Y3 Z4 Z5 Y6]
+ (6.62861420173473e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (6.62861420173473e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (1.0358477600901583e-06) [Y0 X1 X4 Z5 Z6 Y7]
+ (1.0358477600901583e-06) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.313145500135761e-06) [Y1 Z2 Z3 X4 X5 Y6]
+ (3.313145500135761e-06) [X1 Z2 Z3 Y4 Y5 X6]
+ (4.183932559391285e-06) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (4.183932559391285e-06) [X1 Z2 Z3 Z4 Z5 X7]
+ (0.00029219862611105994) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (0.00029219862611105994) [Y0 Z1 Y2 X3 Z4 X5]
+ (0.00029219862611105994) [X0 Z1 X2 Y3 Z4 Y5]
+ (0.00029219862611105994) [X0 Z1 X2 X3 Z4 X5]
+ (0.010540425907671448) [Y0 Z1 Z2 Z3 Y4 Z7]
+ (0.010540425907671448) [X0 Z1 Z2 Z3 X4 Z7]
+ (0.010540425907671448) [Y1 Z2 Z3 Z4 Y5 Z6]
+ (0.010540425907671448) [X1 Z2 Z3 Z4 X5 Z6]
+ (0.011307274008848126) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (0.011307274008848126) [X0 Z1 Z2 Z3 X4 Z5]
+ (0.02510495713884446) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (0.02510495713884446) [X0 Z1 Z2 Z3 X4 Z6]
+ (0.02510495713884446) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (0.02510495713884446) [X1 Z2 Z3 Z4 X5 Z7]
+ (0.0307875053891439) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (0.0307875053891439) [Z0 X1 Z2 Z3 Z4 X5]
+ (-5.105396549879134e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-5.105396549879134e-06) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-5.10539654987913e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (-5.10539654987913e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (-0.01456453123117301) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-0.01456453123117301) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-3.769659452021961e-06) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-3.769659452021961e-06) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-3.3281393505270383e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-3.3281393505270383e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-3.3281393505270383e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-3.3281393505270383e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (3.3131455001357616e-06) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (3.3131455001357616e-06) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (3.3131455001357616e-06) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (3.3131455001357616e-06) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (4.183932559391285e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (4.183932559391285e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (0.01456453123117301) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (0.01456453123117301) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
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
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pyscf/lib/misc.py:46: H5pyDeprecationWarning: Using default_file_mode other than 'r' is deprecated. Pass the mode to h5py.File() instead.
  h5py.get_config().default_file_mode = 'a'
Qubit Hamiltonian of the water molecule
(-46.463906788688966+0j) [] +
(-0.014583648907612644+0j) [X0 X1 Y2 Y3] +
(-3.570761329422948e-07+0j) [X0 X1 Y2 Z3 Z4 Y5] +
(-0.005652620978017359+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7] +
(-0.008826368514209856+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.7924939577204805e-06+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.570761329422948e-07+0j) [X0 X1 X3 X4] +
(-0.005652620978017359+0j) [X0 X1 X3 Z4 Z5 X6] +
(-0.008826368514209856+0j) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.7924939577204805e-06+0j) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.002745836470186815+0j) [X0 X1 Y4 Y5] +
(-2.447323129032567e-07+0j) [X0 X1 Y4 Z5 Z6 Y7] +
(-7.867765104221476e-07+0j) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.003804066171728548+0j) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.447323129032567e-07+0j) [X0 X1 X5 X6] +
(-7.867765104221476e-07+0j) [X0 X1 X5 Z6 Z7 Z8 Z9 X10] +
(-0.003804066171728548+0j) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.006888194352970589+0j) [X0 X1 Y6 Y7] +
(-7.735036880591239e-05+0j) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11] +
(1.7035783557295562e-07+0j) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.735036880591237e-05+0j) [X0 X1 X7 Z8 Z9 X10] +
(1.7035783557295562e-07+0j) [X0 X1 X7 Z8 Z9 Z10 Z11 X12] +
(-0.006509361201177252+0j) [X0 X1 Y8 Y9] +
(-0.007731425250775305+0j) [X0 X1 Y10 Y11] +
(5.627851911548589e-07+0j) [X0 X1 Y10 Z11 Z12 Y13] +
(5.627851911548589e-07+0j) [X0 X1 X11 X12] +
(-0.005283776488402971+0j) [X0 X1 Y12 Y13] +
(0.014583648907612644+0j) [X0 Y1 Y2 X3] +
(3.570761329422948e-07+0j) [X0 Y1 Y2 Z3 Z4 X5] +
(0.005652620978017359+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7] +
(0.008826368514209856+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-1.7924939577204805e-06+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.570761329422948e-07+0j) [X0 Y1 Y3 X4] +
(-0.005652620978017359+0j) [X0 Y1 Y3 Z4 Z5 X6] +
(-0.008826368514209856+0j) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.7924939577204805e-06+0j) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.002745836470186815+0j) [X0 Y1 Y4 X5] +
(2.447323129032567e-07+0j) [X0 Y1 Y4 Z5 Z6 X7] +
(7.867765104221476e-07+0j) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.003804066171728548+0j) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.447323129032567e-07+0j) [X0 Y1 Y5 X6] +
(-7.867765104221476e-07+0j) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.003804066171728548+0j) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.006888194352970589+0j) [X0 Y1 Y6 X7] +
(7.735036880591239e-05+0j) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11] +
(-1.7035783557295562e-07+0j) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.735036880591237e-05+0j) [X0 Y1 Y7 Z8 Z9 X10] +
(1.7035783557295562e-07+0j) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12] +
(0.006509361201177252+0j) [X0 Y1 Y8 X9] +
(0.007731425250775305+0j) [X0 Y1 Y10 X11] +
(-5.627851911548589e-07+0j) [X0 Y1 Y10 Z11 Z12 X13] +
(5.627851911548589e-07+0j) [X0 Y1 Y11 X12] +
(0.005283776488402971+0j) [X0 Y1 Y12 X13] +
(0.1250703257977186+0j) [X0 Z1 X2] +
(-1.9332412771370562e-07+0j) [X0 Z1 X2 X3 Z4 X5] +
(-0.002293956611352467+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 X7] +
(-0.0016407548553123929+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(3.013471458984604e-07+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.9332412771370562e-07+0j) [X0 Z1 X2 Y3 Z4 Y5] +
(-0.002293956611352467+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7] +
(-0.0016407548553123929+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(3.013471458984604e-07+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.003276971931231557+0j) [X0 Z1 X2 Z3] +
(-1.5510539177754705e-07+0j) [X0 Z1 X2 X4 Z5 X6] +
(-1.1468376507194344e-06+0j) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.007597464029770599+0j) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.3807781481927127e-07+0j) [X0 Z1 X2 Y4 Z5 Y6] +
(-7.900128986023348e-07+0j) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0053480515826766+0j) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005530759218631476+0j) [X0 Z1 X2 Z4] +
(-1.3807781481927127e-07+0j) [X0 Z1 X2 X5 Z6 X7] +
(-3.3767393084402523e-07+0j) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.001863894282458711+0j) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.3807781481927127e-07+0j) [X0 Z1 X2 Y5 Z6 Y7] +
(-3.3767393084402523e-07+0j) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.001863894282458711+0j) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.001105903769189611+0j) [X0 Z1 X2 Z5] +
(0.005708495985960916+0j) [X0 Z1 X2 X6 Z7 Z8 Z9 X10] +
(-8.352332102713157e-07+0j) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
1.9742253789775635e-08j [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005262642473076829+0j) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10] +
(-8.074305985628489e-07+0j) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.00803252091882131+0j) [X0 Z1 X2 Z6] +
(0.0005940221543005191+0j) [X0 Z1 X2 X7 Z8 Z9 Z10 X11] +
(-8.379773242469647e-08+0j) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005940221543005191+0j) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11] +
(-8.379773242469647e-08+0j) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0033476175306661007+0j) [X0 Z1 X2 Z7] +
(0.011055020596132+0j) [X0 Z1 X2 Z8] +
(0.0029297686747509536+0j) [X0 Z1 X2 Z9] +
(-6.418291574617451e-07+0j) [X0 Z1 X2 X10 Z11 X12] +
(-6.556281914577381e-07+0j) [X0 Z1 X2 Y10 Z11 Y12] +
(0.0035552901955041953+0j) [X0 Z1 X2 Z10] +
(-1.1076325598560301e-07+0j) [X0 Z1 X2 X11 Z12 X13] +
(-1.1076325598560301e-07+0j) [X0 Z1 X2 Y11 Z12 Y13] +
(0.0017560707018411665+0j) [X0 Z1 X2 Z11] +
(0.006901238249797219+0j) [X0 Z1 X2 Z12] +
(0.0023262306231580094+0j) [X0 Z1 X2 Z13] +
(-3.568247521170997e-07+0j) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0022494124470939995+0j) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.0474716553009873e-08+0j) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0004458535128840877+0j) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10] +
(-1.974225378900023e-08+0j) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441864+0j) [X0 Z1 Z2 X3 Y4 Y5] +
(-4.523389677583095e-07+0j) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.00348415730021789+0j) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-8.091637198754091e-07+0j) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10] +
(-0.005733569747311889+0j) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.00468490338815521+0j) [X0 Z1 Z2 X3 Y6 Y7] +
(0.00466862031877631+0j) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11] +
(-7.189990975197696e-07+0j) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005114473831660396+0j) [X0 Z1 Z2 X3 X7 Z8 Z9 X10] +
(-7.560692464650021e-07+0j) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381046+0j) [X0 Z1 Z2 X3 Y8 Y9] +
(-0.0017992194936630292+0j) [X0 Z1 Z2 X3 Y10 Y11] +
(-5.471647744712641e-07+0j) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13] +
(-5.287660624770131e-07+0j) [X0 Z1 Z2 X3 X11 X12] +
(-0.004575007626639211+0j) [X0 Z1 Z2 X3 Y12 Y13] +
(0.004424855449441864+0j) [X0 Z1 Z2 Y3 Y4 X5] +
(4.523389677583095e-07+0j) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.00348415730021789+0j) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.091637198754091e-07+0j) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.005733569747311889+0j) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.00468490338815521+0j) [X0 Z1 Z2 Y3 Y6 X7] +
(-0.00466862031877631+0j) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11] +
(7.189990975197696e-07+0j) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005114473831660396+0j) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10] +
(-7.560692464650021e-07+0j) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12] +
(0.008125251921381046+0j) [X0 Z1 Z2 Y3 Y8 X9] +
(0.0017992194936630292+0j) [X0 Z1 Z2 Y3 Y10 X11] +
(5.471647744712641e-07+0j) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13] +
(-5.287660624770131e-07+0j) [X0 Z1 Z2 Y3 Y11 X12] +
(0.004575007626639211+0j) [X0 Z1 Z2 Y3 Y12 X13] +
(3.20207688109673e-06+0j) [X0 Z1 Z2 Z3 X4] +
(0.0008533856254125535+0j) [X0 Z1 Z2 Z3 X4 X5 Z6 X7] +
(0.0007870896771024511+0j) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0008533856254125535+0j) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7] +
(0.0007870896771024511+0j) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.2919694860689314e-07+0j) [X0 Z1 Z2 Z3 X4 Z5] +
(4.4445978543531407e-07+0j) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10] +
(0.0011726348316441913+0j) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.684915095225614e-07+0j) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10] +
(0.0022009640695004593+0j) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.8394209156333915e-07+0j) [X0 Z1 Z2 Z3 X4 Z6] +
(4.092250616193178e-07+0j) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11] +
(0.0023949726397980288+0j) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.092250616193178e-07+0j) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11] +
(0.0023949726397980288+0j) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.2362599617135675e-07+0j) [X0 Z1 Z2 Z3 X4 Z8] +
(8.649310137048479e-08+0j) [X0 Z1 Z2 Z3 X4 Z9] +
(0.0013038004788127012+0j) [X0 Z1 Z2 Z3 X4 X10 Z11 X12] +
(0.003989841456619312+0j) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12] +
(-6.733197742038225e-07+0j) [X0 Z1 Z2 Z3 X4 Z10] +
(0.0022619660624823524+0j) [X0 Z1 Z2 Z3 X4 X11 Z12 X13] +
(0.0022619660624823524+0j) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13] +
(-5.927453082507015e-07+0j) [X0 Z1 Z2 Z3 X4 Z11] +
(1.239336321749729e-06+0j) [X0 Z1 Z2 Z3 X4 Z12] +
(9.306536652189402e-07+0j) [X0 Z1 Z2 Z3 X4 Z13] +
(-0.0010283292378562685+0j) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.0026860409778066098+0j) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12] +
(-1.8394209156333912e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7] +
(-0.00019400857029756911+0j) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0012223378081538377+0j) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-2.3713289480087198e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9] +
(8.057446595312111e-08+0j) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11] +
(0.0017278753941369586+0j) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13] +
(-0.0009581655836696515+0j) [X0 Z1 Z2 Z3 Z4 X5 X11 X12] +
(-3.0868265653078873e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13] +
(1.8394209156333912e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7] +
(0.00019400857029756911+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0012223378081538377+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(2.3713289480087198e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9] +
(-8.057446595312111e-08+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11] +
(-0.0017278753941369586+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13] +
(-0.0009581655836696515+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12] +
(3.0868265653078873e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13] +
(0.04274327701378291+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6] +
(0.0005192743499487672+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(-1.850564193067663e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005192743499487672+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(-1.850564193067663e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.002779026799025573+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7] +
(0.004636976661182559+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8] +
(0.001280306097349656+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9] +
(2.312094305180259e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12] +
(1.0717282181370591e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12] +
(0.0053799371558393635+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10] +
(7.246974425350932e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13] +
(7.246974425350932e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13] +
(0.005241535382803868+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11] +
(0.004311038507914311+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12] +
(0.0010435246534907436+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13] +
(1.2004287494357673e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12] +
(-0.003356670563832903+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9] +
(-0.00013840177303549577+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11] +
(-6.175246207213874e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(-4.997018422118328e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12] +
(-0.003267513854423567+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13] +
(0.003356670563832903+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9] +
(0.00013840177303549577+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11] +
(6.175246207213874e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(-4.997018422118328e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12] +
(0.003267513854423567+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13] +
(0.0038764708993369603+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(-7.540341413821506e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.0038764708993369603+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(-7.540341413821506e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(0.07165035181002619+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0021413612231016444+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(0.00422081397004644+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(0.0012366478019244951+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(-0.002984166168121945+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(0.002984166168121945+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-1.3987009015719039e-05+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(8.949476487734458e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-6.876621658058509e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(-7.661347213024096e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(0.001532483523072997+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10] +
(-2.904599884143673e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(0.005408954422409958+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10] +
(-1.0444941297965177e-06+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(0.004767272188278084+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10] +
(-8.105515036867528e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(0.005286546538226852+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10] +
(-9.956079229935192e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0016095313817213672+0j) [X0 Z1 Z2 Z3 Z4 X6] +
(-7.141625221160109e-05+0j) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10] +
(-2.6667317544310593e-07+0j) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0024629170071339204+0j) [X0 Z1 Z2 Z3 Z5 X6] +
(0.00071567342489085+0j) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10] +
(-3.076732531703681e-07+0j) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.6060718681366187e-07+0j) [X0 Z1 Z2 X4] +
(0.00396156079249652+0j) [X0 Z1 Z2 Z4 Z5 X6] +
(0.00018787053389545029+0j) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.6569309316031466e-07+0j) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.7379332625337298e-07+0j) [X0 Z1 Z3 X4] +
(0.001667604181144053+0j) [X0 Z1 Z3 Z4 Z5 X6] +
(-0.0014528843214169427+0j) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(4.670402390587751e-07+0j) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.10433064780651397+0j) [X0 X2] +
(3.1174479463360316e-06+0j) [X0 Z2 Z3 X4] +
(0.04587947078129816+0j) [X0 Z2 Z3 Z4 Z5 X6] +
(0.058591988733861976+0j) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-1.1463061453202262e-05+0j) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.014583648907612644+0j) [Y0 X1 X2 Y3] +
(3.570761329422948e-07+0j) [Y0 X1 X2 Z3 Z4 Y5] +
(0.005652620978017359+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7] +
(0.008826368514209856+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.7924939577204805e-06+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.570761329422948e-07+0j) [Y0 X1 X3 Y4] +
(-0.005652620978017359+0j) [Y0 X1 X3 Z4 Z5 Y6] +
(-0.008826368514209856+0j) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.7924939577204805e-06+0j) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.002745836470186815+0j) [Y0 X1 X4 Y5] +
(2.447323129032567e-07+0j) [Y0 X1 X4 Z5 Z6 Y7] +
(7.867765104221476e-07+0j) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.003804066171728548+0j) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.447323129032567e-07+0j) [Y0 X1 X5 Y6] +
(-7.867765104221476e-07+0j) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.003804066171728548+0j) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.006888194352970589+0j) [Y0 X1 X6 Y7] +
(7.735036880591239e-05+0j) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11] +
(-1.7035783557295562e-07+0j) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.735036880591237e-05+0j) [Y0 X1 X7 Z8 Z9 Y10] +
(1.7035783557295562e-07+0j) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12] +
(0.006509361201177252+0j) [Y0 X1 X8 Y9] +
(0.007731425250775305+0j) [Y0 X1 X10 Y11] +
(-5.627851911548589e-07+0j) [Y0 X1 X10 Z11 Z12 Y13] +
(5.627851911548589e-07+0j) [Y0 X1 X11 Y12] +
(0.005283776488402971+0j) [Y0 X1 X12 Y13] +
(-0.014583648907612644+0j) [Y0 Y1 X2 X3] +
(-3.570761329422948e-07+0j) [Y0 Y1 X2 Z3 Z4 X5] +
(-0.005652620978017359+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7] +
(-0.008826368514209856+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.7924939577204805e-06+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.570761329422948e-07+0j) [Y0 Y1 Y3 Y4] +
(-0.005652620978017359+0j) [Y0 Y1 Y3 Z4 Z5 Y6] +
(-0.008826368514209856+0j) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.7924939577204805e-06+0j) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.002745836470186815+0j) [Y0 Y1 X4 X5] +
(-2.447323129032567e-07+0j) [Y0 Y1 X4 Z5 Z6 X7] +
(-7.867765104221476e-07+0j) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.003804066171728548+0j) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.447323129032567e-07+0j) [Y0 Y1 Y5 Y6] +
(-7.867765104221476e-07+0j) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.003804066171728548+0j) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.006888194352970589+0j) [Y0 Y1 X6 X7] +
(-7.735036880591239e-05+0j) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11] +
(1.7035783557295562e-07+0j) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.735036880591237e-05+0j) [Y0 Y1 Y7 Z8 Z9 Y10] +
(1.7035783557295562e-07+0j) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.006509361201177252+0j) [Y0 Y1 X8 X9] +
(-0.007731425250775305+0j) [Y0 Y1 X10 X11] +
(5.627851911548589e-07+0j) [Y0 Y1 X10 Z11 Z12 X13] +
(5.627851911548589e-07+0j) [Y0 Y1 Y11 Y12] +
(-0.005283776488402971+0j) [Y0 Y1 X12 X13] +
(-3.568247521170997e-07+0j) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0022494124470939995+0j) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0004458535128840877+0j) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10] +
(-1.974225378900023e-08+0j) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.0474716553009873e-08+0j) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.1250703257977186+0j) [Y0 Z1 Y2] +
(-1.9332412771370562e-07+0j) [Y0 Z1 Y2 X3 Z4 X5] +
(-0.002293956611352467+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7] +
(-0.0016407548553123929+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(3.013471458984604e-07+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.9332412771370562e-07+0j) [Y0 Z1 Y2 Y3 Z4 Y5] +
(-0.002293956611352467+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7] +
(-0.0016407548553123929+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(3.013471458984604e-07+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.003276971931231557+0j) [Y0 Z1 Y2 Z3] +
(-1.3807781481927127e-07+0j) [Y0 Z1 Y2 X4 Z5 X6] +
(-7.900128986023348e-07+0j) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0053480515826766+0j) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.5510539177754705e-07+0j) [Y0 Z1 Y2 Y4 Z5 Y6] +
(-1.1468376507194344e-06+0j) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.007597464029770599+0j) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005530759218631476+0j) [Y0 Z1 Y2 Z4] +
(-1.3807781481927127e-07+0j) [Y0 Z1 Y2 X5 Z6 X7] +
(-3.3767393084402523e-07+0j) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.001863894282458711+0j) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.3807781481927127e-07+0j) [Y0 Z1 Y2 Y5 Z6 Y7] +
(-3.3767393084402523e-07+0j) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.001863894282458711+0j) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.001105903769189611+0j) [Y0 Z1 Y2 Z5] +
(0.005262642473076829+0j) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10] +
(-8.074305985628489e-07+0j) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.005708495985960916+0j) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10] +
-1.9742253789775635e-08j [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-8.352332102713157e-07+0j) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.00803252091882131+0j) [Y0 Z1 Y2 Z6] +
(0.0005940221543005191+0j) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11] +
(-8.379773242469647e-08+0j) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005940221543005191+0j) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11] +
(-8.379773242469647e-08+0j) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0033476175306661007+0j) [Y0 Z1 Y2 Z7] +
(0.011055020596132+0j) [Y0 Z1 Y2 Z8] +
(0.0029297686747509536+0j) [Y0 Z1 Y2 Z9] +
(-6.556281914577381e-07+0j) [Y0 Z1 Y2 X10 Z11 X12] +
(-6.418291574617451e-07+0j) [Y0 Z1 Y2 Y10 Z11 Y12] +
(0.0035552901955041953+0j) [Y0 Z1 Y2 Z10] +
(-1.1076325598560301e-07+0j) [Y0 Z1 Y2 X11 Z12 X13] +
(-1.1076325598560301e-07+0j) [Y0 Z1 Y2 Y11 Z12 Y13] +
(0.0017560707018411665+0j) [Y0 Z1 Y2 Z11] +
(0.006901238249797219+0j) [Y0 Z1 Y2 Z12] +
(0.0023262306231580094+0j) [Y0 Z1 Y2 Z13] +
(0.004424855449441864+0j) [Y0 Z1 Z2 X3 X4 Y5] +
(4.523389677583095e-07+0j) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.00348415730021789+0j) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-8.091637198754091e-07+0j) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.005733569747311889+0j) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.00468490338815521+0j) [Y0 Z1 Z2 X3 X6 Y7] +
(-0.00466862031877631+0j) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11] +
(7.189990975197696e-07+0j) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005114473831660396+0j) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10] +
(-7.560692464650021e-07+0j) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12] +
(0.008125251921381046+0j) [Y0 Z1 Z2 X3 X8 Y9] +
(0.0017992194936630292+0j) [Y0 Z1 Z2 X3 X10 Y11] +
(5.471647744712641e-07+0j) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13] +
(-5.287660624770131e-07+0j) [Y0 Z1 Z2 X3 X11 Y12] +
(0.004575007626639211+0j) [Y0 Z1 Z2 X3 X12 Y13] +
(-0.004424855449441864+0j) [Y0 Z1 Z2 Y3 X4 X5] +
(-4.523389677583095e-07+0j) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.00348415730021789+0j) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.091637198754091e-07+0j) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.005733569747311889+0j) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.00468490338815521+0j) [Y0 Z1 Z2 Y3 X6 X7] +
(0.00466862031877631+0j) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11] +
(-7.189990975197696e-07+0j) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005114473831660396+0j) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10] +
(-7.560692464650021e-07+0j) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381046+0j) [Y0 Z1 Z2 Y3 X8 X9] +
(-0.0017992194936630292+0j) [Y0 Z1 Z2 Y3 X10 X11] +
(-5.471647744712641e-07+0j) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13] +
(-5.287660624770131e-07+0j) [Y0 Z1 Z2 Y3 Y11 Y12] +
(-0.004575007626639211+0j) [Y0 Z1 Z2 Y3 X12 X13] +
(-0.0010283292378562685+0j) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.0026860409778066098+0j) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12] +
(3.20207688109673e-06+0j) [Y0 Z1 Z2 Z3 Y4] +
(0.0008533856254125535+0j) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7] +
(0.0007870896771024511+0j) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0008533856254125535+0j) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7] +
(0.0007870896771024511+0j) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.2919694860689314e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z5] +
(4.684915095225614e-07+0j) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10] +
(0.0022009640695004593+0j) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.4445978543531407e-07+0j) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10] +
(0.0011726348316441913+0j) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.8394209156333915e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z6] +
(4.092250616193178e-07+0j) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11] +
(0.0023949726397980288+0j) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.092250616193178e-07+0j) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11] +
(0.0023949726397980288+0j) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.2362599617135675e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z8] +
(8.649310137048479e-08+0j) [Y0 Z1 Z2 Z3 Y4 Z9] +
(0.003989841456619312+0j) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12] +
(0.0013038004788127012+0j) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12] +
(-6.733197742038225e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z10] +
(0.0022619660624823524+0j) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13] +
(0.0022619660624823524+0j) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13] +
(-5.927453082507015e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z11] +
(1.239336321749729e-06+0j) [Y0 Z1 Z2 Z3 Y4 Z12] +
(9.306536652189402e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z13] +
(1.8394209156333912e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7] +
(0.00019400857029756911+0j) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0012223378081538377+0j) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(2.3713289480087198e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9] +
(-8.057446595312111e-08+0j) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11] +
(-0.0017278753941369586+0j) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13] +
(-0.0009581655836696515+0j) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12] +
(3.0868265653078873e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13] +
(-1.8394209156333912e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7] +
(-0.00019400857029756911+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0012223378081538377+0j) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-2.3713289480087198e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9] +
(8.057446595312111e-08+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11] +
(0.0017278753941369586+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13] +
(-0.0009581655836696515+0j) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12] +
(-3.0868265653078873e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13] +
(1.2004287494357673e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12] +
(0.04274327701378291+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6] +
(0.0005192743499487672+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(-1.850564193067663e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005192743499487672+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(-1.850564193067663e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.002779026799025573+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7] +
(0.004636976661182559+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8] +
(0.001280306097349656+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9] +
(1.0717282181370591e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12] +
(2.312094305180259e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12] +
(0.0053799371558393635+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10] +
(7.246974425350932e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13] +
(7.246974425350932e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13] +
(0.005241535382803868+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11] +
(0.004311038507914311+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12] +
(0.0010435246534907436+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13] +
(0.003356670563832903+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9] +
(0.00013840177303549577+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11] +
(6.175246207213874e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(-4.997018422118328e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12] +
(0.003267513854423567+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13] +
(-0.003356670563832903+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9] +
(-0.00013840177303549577+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11] +
(-6.175246207213874e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(-4.997018422118328e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12] +
(-0.003267513854423567+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13] +
(0.0038764708993369603+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(-7.540341413821506e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.0038764708993369603+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(-7.540341413821506e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(0.07165035181002619+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0021413612231016444+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(0.00422081397004644+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(0.0012366478019244951+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(0.002984166168121945+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-0.002984166168121945+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-1.3987009015719039e-05+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(8.949476487734458e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-6.876621658058509e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(-7.661347213024096e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(0.001532483523072997+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10] +
(-2.904599884143673e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(0.005408954422409958+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10] +
(-1.0444941297965177e-06+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(0.004767272188278084+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10] +
(-8.105515036867528e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(0.005286546538226852+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10] +
(-9.956079229935192e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0016095313817213672+0j) [Y0 Z1 Z2 Z3 Z4 Y6] +
(-7.141625221160109e-05+0j) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10] +
(-2.6667317544310593e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0024629170071339204+0j) [Y0 Z1 Z2 Z3 Z5 Y6] +
(0.00071567342489085+0j) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10] +
(-3.076732531703681e-07+0j) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(3.6060718681366187e-07+0j) [Y0 Z1 Z2 Y4] +
(0.00396156079249652+0j) [Y0 Z1 Z2 Z4 Z5 Y6] +
(0.00018787053389545029+0j) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.6569309316031466e-07+0j) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.7379332625337298e-07+0j) [Y0 Z1 Z3 Y4] +
(0.001667604181144053+0j) [Y0 Z1 Z3 Z4 Z5 Y6] +
(-0.0014528843214169427+0j) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(4.670402390587751e-07+0j) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.10433064780651397+0j) [Y0 Y2] +
(3.1174479463360316e-06+0j) [Y0 Z2 Z3 Y4] +
(0.04587947078129816+0j) [Y0 Z2 Z3 Z4 Z5 Y6] +
(0.058591988733861976+0j) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-1.1463061453202262e-05+0j) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(12.412630742111777+0j) [Z0] +
(0.10433064780651397+0j) [Z0 X1 Z2 X3] +
(3.1174479463360316e-06+0j) [Z0 X1 Z2 Z3 Z4 X5] +
(0.04587947078129816+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7] +
(0.058591988733861976+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-1.1463061453202262e-05+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.10433064780651397+0j) [Z0 Y1 Z2 Y3] +
(3.1174479463360316e-06+0j) [Z0 Y1 Z2 Z3 Z4 Y5] +
(0.04587947078129816+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7] +
(0.058591988733861976+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.1463061453202262e-05+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.1861763734860518+0j) [Z0 Z1] +
(-8.337746756037756e-07+0j) [Z0 X2 Z3 X4] +
(-0.0271150368452732+0j) [Z0 X2 Z3 Z4 Z5 X6] +
(-0.06752385099214041+0j) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.4017109735241801e-05+0j) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-8.337746756037756e-07+0j) [Z0 Y2 Z3 Y4] +
(-0.0271150368452732+0j) [Z0 Y2 Z3 Z4 Z5 Y6] +
(-0.06752385099214041+0j) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.4017109735241801e-05+0j) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.2367108078383043+0j) [Z0 Z2] +
(-1.1908508085460702e-06+0j) [Z0 X3 Z4 X5] +
(-0.03276765782329056+0j) [Z0 X3 Z4 Z5 Z6 X7] +
(-0.07635021950635026+0j) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.5809603692962282e-05+0j) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.1908508085460702e-06+0j) [Z0 Y3 Z4 Y5] +
(-0.03276765782329056+0j) [Z0 Y3 Z4 Z5 Z6 Y7] +
(-0.07635021950635026+0j) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.5809603692962282e-05+0j) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.251294456745917+0j) [Z0 Z3] +
(-3.099349243847675e-06+0j) [Z0 X4 Z5 X6] +
(-1.5316808795056106e-05+0j) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.08684737589863631+0j) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-3.099349243847675e-06+0j) [Z0 Y4 Z5 Y6] +
(-1.5316808795056106e-05+0j) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.08684737589863631+0j) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.19661770890342162+0j) [Z0 Z4] +
(-3.3440815567509314e-06+0j) [Z0 X5 Z6 X7] +
(-1.6103585305478253e-05+0j) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.09065144207036487+0j) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.3440815567509314e-06+0j) [Z0 Y5 Z6 Y7] +
(-1.6103585305478253e-05+0j) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.09065144207036487+0j) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.19936354537360845+0j) [Z0 Z5] +
(0.05608468124661356+0j) [Z0 X6 Z7 Z8 Z9 X10] +
(-6.652209669013128e-06+0j) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.05608468124661356+0j) [Z0 Y6 Z7 Z8 Z9 Y10] +
(-6.652209669013128e-06+0j) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.24164663936017228+0j) [Z0 Z6] +
(0.056007330877807654+0j) [Z0 X7 Z8 Z9 Z10 X11] +
(-6.481851833440172e-06+0j) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.056007330877807654+0j) [Z0 Y7 Z8 Z9 Z10 Y11] +
(-6.481851833440172e-06+0j) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.24853483371314286+0j) [Z0 Z7] +
(0.272325183066057+0j) [Z0 Z8] +
(0.27883454426723425+0j) [Z0 Z9] +
(-2.177664605015797e-06+0j) [Z0 X10 Z11 X12] +
(-2.177664605015797e-06+0j) [Z0 Y10 Z11 Y12] +
(0.1929972393536426+0j) [Z0 Z10] +
(-1.6148794138609382e-06+0j) [Z0 X11 Z12 X13] +
(-1.6148794138609382e-06+0j) [Z0 Y11 Z12 Y13] +
(0.2007286646044179+0j) [Z0 Z11] +
(0.21102659849791544+0j) [Z0 Z12] +
(0.2163103749863184+0j) [Z0 Z13] +
(1.9332412771370562e-07+0j) [X1 X2 Y3 Y4] +
(0.002293956611352467+0j) [X1 X2 Y3 Z4 Z5 Y6] +
(0.0016407548553123929+0j) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-3.013471458984604e-07+0j) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004424855449441864+0j) [X1 X2 X4 X5] +
(-8.091637198754091e-07+0j) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.005733569747311889+0j) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-4.523389677583095e-07+0j) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.00348415730021789+0j) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.00468490338815521+0j) [X1 X2 X6 X7] +
(0.005114473831660396+0j) [X1 X2 X6 Z7 Z8 Z9 Z10 X11] +
(-7.560692464650021e-07+0j) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.00466862031877631+0j) [X1 X2 Y7 Z8 Z9 Y10] +
(-7.189990975197696e-07+0j) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381046+0j) [X1 X2 X8 X9] +
(-0.0017992194936630292+0j) [X1 X2 X10 X11] +
(-5.287660624770131e-07+0j) [X1 X2 X10 Z11 Z12 X13] +
(-5.471647744712641e-07+0j) [X1 X2 Y11 Y12] +
(-0.004575007626639211+0j) [X1 X2 X12 X13] +
(-1.9332412771370562e-07+0j) [X1 Y2 Y3 X4] +
(-0.002293956611352467+0j) [X1 Y2 Y3 Z4 Z5 X6] +
(-0.0016407548553123929+0j) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(3.013471458984604e-07+0j) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441864+0j) [X1 Y2 Y4 X5] +
(-8.091637198754091e-07+0j) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.005733569747311889+0j) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.523389677583095e-07+0j) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10] +
(0.00348415730021789+0j) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.00468490338815521+0j) [X1 Y2 Y6 X7] +
(0.005114473831660396+0j) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11] +
(-7.560692464650021e-07+0j) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.00466862031877631+0j) [X1 Y2 Y7 Z8 Z9 X10] +
(7.189990975197696e-07+0j) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381046+0j) [X1 Y2 Y8 X9] +
(-0.0017992194936630292+0j) [X1 Y2 Y10 X11] +
(-5.287660624770131e-07+0j) [X1 Y2 Y10 Z11 Z12 X13] +
(5.471647744712641e-07+0j) [X1 Y2 Y11 X12] +
(-0.004575007626639211+0j) [X1 Y2 Y12 X13] +
(0.12507032579771865+0j) [X1 Z2 X3] +
(-1.3807781481927127e-07+0j) [X1 Z2 X3 X4 Z5 X6] +
(-3.3767393084402523e-07+0j) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.001863894282458711+0j) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.3807781481927127e-07+0j) [X1 Z2 X3 Y4 Z5 Y6] +
(-3.3767393084402523e-07+0j) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.001863894282458711+0j) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.001105903769189611+0j) [X1 Z2 X3 Z4] +
(-1.5510539177754705e-07+0j) [X1 Z2 X3 X5 Z6 X7] +
(-1.1468376507194344e-06+0j) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.007597464029770599+0j) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.3807781481927127e-07+0j) [X1 Z2 X3 Y5 Z6 Y7] +
(-7.900128986023348e-07+0j) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0053480515826766+0j) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005530759218631476+0j) [X1 Z2 X3 Z5] +
(0.0005940221543005191+0j) [X1 Z2 X3 X6 Z7 Z8 Z9 X10] +
(-8.379773242469647e-08+0j) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0005940221543005191+0j) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10] +
(-8.379773242469647e-08+0j) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0033476175306661007+0j) [X1 Z2 X3 Z6] +
(0.005708495985960916+0j) [X1 Z2 X3 X7 Z8 Z9 Z10 X11] +
(-8.352332102713157e-07+0j) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
1.9742253789775635e-08j [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005262642473076829+0j) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11] +
(-8.074305985628489e-07+0j) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.00803252091882131+0j) [X1 Z2 X3 Z7] +
(0.0029297686747509536+0j) [X1 Z2 X3 Z8] +
(0.011055020596132+0j) [X1 Z2 X3 Z9] +
(-1.1076325598560301e-07+0j) [X1 Z2 X3 X10 Z11 X12] +
(-1.1076325598560301e-07+0j) [X1 Z2 X3 Y10 Z11 Y12] +
(0.0017560707018411665+0j) [X1 Z2 X3 Z10] +
(-6.418291574617451e-07+0j) [X1 Z2 X3 X11 Z12 X13] +
(-6.556281914577381e-07+0j) [X1 Z2 X3 Y11 Z12 Y13] +
(0.0035552901955041953+0j) [X1 Z2 X3 Z11] +
(0.0023262306231580094+0j) [X1 Z2 X3 Z12] +
(0.006901238249797219+0j) [X1 Z2 X3 Z13] +
(-3.568247521170997e-07+0j) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0022494124470939995+0j) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.0474716553009873e-08+0j) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0004458535128840877+0j) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11] +
(-1.974225378900023e-08+0j) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0008533856254125534+0j) [X1 Z2 Z3 X4 Y5 Y6] +
(-0.0007870896771024511+0j) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10] +
(-1.8394209156333912e-07+0j) [X1 Z2 Z3 X4 X6 X7] +
(-0.0012223378081538377+0j) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.00019400857029756911+0j) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12] +
(-2.3713289480087198e-07+0j) [X1 Z2 Z3 X4 X8 X9] +
(8.057446595312111e-08+0j) [X1 Z2 Z3 X4 X10 X11] +
(-0.0009581655836696515+0j) [X1 Z2 Z3 X4 X10 Z11 Z12 X13] +
(0.0017278753941369586+0j) [X1 Z2 Z3 X4 Y11 Y12] +
(-3.0868265653078873e-07+0j) [X1 Z2 Z3 X4 X12 X13] +
(0.0008533856254125534+0j) [X1 Z2 Z3 Y4 Y5 X6] +
(0.0007870896771024511+0j) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10] +
(-1.8394209156333912e-07+0j) [X1 Z2 Z3 Y4 Y6 X7] +
(-0.0012223378081538377+0j) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.00019400857029756911+0j) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12] +
(-2.3713289480087198e-07+0j) [X1 Z2 Z3 Y4 Y8 X9] +
(8.057446595312111e-08+0j) [X1 Z2 Z3 Y4 Y10 X11] +
(-0.0009581655836696515+0j) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13] +
(-0.0017278753941369586+0j) [X1 Z2 Z3 Y4 Y11 X12] +
(-3.0868265653078873e-07+0j) [X1 Z2 Z3 Y4 Y12 X13] +
(3.202076881096729e-06+0j) [X1 Z2 Z3 Z4 X5] +
(4.092250616193178e-07+0j) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10] +
(0.0023949726397980288+0j) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.092250616193178e-07+0j) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10] +
(0.0023949726397980288+0j) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.4445978543531407e-07+0j) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11] +
(0.0011726348316441913+0j) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.684915095225614e-07+0j) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11] +
(0.0022009640695004593+0j) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.8394209156333915e-07+0j) [X1 Z2 Z3 Z4 X5 Z7] +
(8.649310137048479e-08+0j) [X1 Z2 Z3 Z4 X5 Z8] +
(3.2362599617135675e-07+0j) [X1 Z2 Z3 Z4 X5 Z9] +
(0.0022619660624823524+0j) [X1 Z2 Z3 Z4 X5 X10 Z11 X12] +
(0.0022619660624823524+0j) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12] +
(-5.927453082507015e-07+0j) [X1 Z2 Z3 Z4 X5 Z10] +
(0.0013038004788127012+0j) [X1 Z2 Z3 Z4 X5 X11 Z12 X13] +
(0.003989841456619312+0j) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13] +
(-6.733197742038225e-07+0j) [X1 Z2 Z3 Z4 X5 Z11] +
(9.306536652189402e-07+0j) [X1 Z2 Z3 Z4 X5 Z12] +
(1.239336321749729e-06+0j) [X1 Z2 Z3 Z4 X5 Z13] +
(-0.0010283292378562685+0j) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0026860409778066098+0j) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13] +
(-0.0005192743499487672+0j) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10] +
(1.850564193067663e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.003356670563832903+0j) [X1 Z2 Z3 Z4 Z5 X6 X8 X9] +
(-0.00013840177303549577+0j) [X1 Z2 Z3 Z4 Z5 X6 X10 X11] +
(-4.997018422118328e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13] +
(-6.175246207213874e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12] +
(-0.003267513854423567+0j) [X1 Z2 Z3 Z4 Z5 X6 X12 X13] +
(0.0005192743499487672+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10] +
(-1.850564193067663e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.003356670563832903+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9] +
(-0.00013840177303549577+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11] +
(-4.997018422118328e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13] +
(6.175246207213874e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12] +
(-0.003267513854423567+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13] +
(0.04274327701378291+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7] +
(0.001280306097349656+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8] +
(0.004636976661182559+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9] +
(7.246974425350932e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12] +
(7.246974425350932e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12] +
(0.005241535382803868+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10] +
(2.312094305180259e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13] +
(1.0717282181370591e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13] +
(0.0053799371558393635+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11] +
(0.0010435246534907436+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12] +
(0.004311038507914311+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13] +
(1.2004287494357673e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13] +
(-0.0038764708993369603+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10] +
(7.540341413821506e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(0.0038764708993369603+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10] +
(-7.540341413821506e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(-0.002984166168121945+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-0.002984166168121945+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(0.07165035181002621+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0012366478019244951+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(0.00422081397004644+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(-1.3987009015719039e-05+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(8.949476487734458e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(-7.661347213024096e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(-0.0021413612231016444+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11] +
(-6.876621658058509e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(0.005408954422409958+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11] +
(-1.0444941297965177e-06+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(0.001532483523072997+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11] +
(-2.904599884143673e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(0.005286546538226852+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11] +
(-9.956079229935192e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0027790267990255727+0j) [X1 Z2 Z3 Z4 Z5 X7] +
(0.004767272188278084+0j) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11] +
(-8.105515036867528e-07+0j) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0024629170071339204+0j) [X1 Z2 Z3 Z4 Z6 X7] +
(0.00071567342489085+0j) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11] +
(-3.076732531703681e-07+0j) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2919694860689314e-07+0j) [X1 Z2 Z3 X5] +
(0.0016095313817213672+0j) [X1 Z2 Z3 Z5 Z6 X7] +
(-7.141625221160109e-05+0j) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-2.6667317544310593e-07+0j) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.7379332625337298e-07+0j) [X1 Z2 Z4 X5] +
(0.001667604181144053+0j) [X1 Z2 Z4 Z5 Z6 X7] +
(-0.0014528843214169427+0j) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(4.670402390587751e-07+0j) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.003276971931231557+0j) [X1 X3] +
(3.6060718681366187e-07+0j) [X1 Z3 Z4 X5] +
(0.00396156079249652+0j) [X1 Z3 Z4 Z5 Z6 X7] +
(0.00018787053389545029+0j) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.6569309316031466e-07+0j) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.9332412771370562e-07+0j) [Y1 X2 X3 Y4] +
(-0.002293956611352467+0j) [Y1 X2 X3 Z4 Z5 Y6] +
(-0.0016407548553123929+0j) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(3.013471458984604e-07+0j) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004424855449441864+0j) [Y1 X2 X4 Y5] +
(-8.091637198754091e-07+0j) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.005733569747311889+0j) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(4.523389677583095e-07+0j) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10] +
(0.00348415730021789+0j) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.00468490338815521+0j) [Y1 X2 X6 Y7] +
(0.005114473831660396+0j) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11] +
(-7.560692464650021e-07+0j) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.00466862031877631+0j) [Y1 X2 X7 Z8 Z9 Y10] +
(7.189990975197696e-07+0j) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381046+0j) [Y1 X2 X8 Y9] +
(-0.0017992194936630292+0j) [Y1 X2 X10 Y11] +
(-5.287660624770131e-07+0j) [Y1 X2 X10 Z11 Z12 Y13] +
(5.471647744712641e-07+0j) [Y1 X2 X11 Y12] +
(-0.004575007626639211+0j) [Y1 X2 X12 Y13] +
(1.9332412771370562e-07+0j) [Y1 Y2 X3 X4] +
(0.002293956611352467+0j) [Y1 Y2 X3 Z4 Z5 X6] +
(0.0016407548553123929+0j) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-3.013471458984604e-07+0j) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441864+0j) [Y1 Y2 Y4 Y5] +
(-8.091637198754091e-07+0j) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.005733569747311889+0j) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-4.523389677583095e-07+0j) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10] +
(-0.00348415730021789+0j) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.00468490338815521+0j) [Y1 Y2 Y6 Y7] +
(0.005114473831660396+0j) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11] +
(-7.560692464650021e-07+0j) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.00466862031877631+0j) [Y1 Y2 X7 Z8 Z9 X10] +
(-7.189990975197696e-07+0j) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381046+0j) [Y1 Y2 Y8 Y9] +
(-0.0017992194936630292+0j) [Y1 Y2 Y10 Y11] +
(-5.287660624770131e-07+0j) [Y1 Y2 Y10 Z11 Z12 Y13] +
(-5.471647744712641e-07+0j) [Y1 Y2 X11 X12] +
(-0.004575007626639211+0j) [Y1 Y2 Y12 Y13] +
(-3.568247521170997e-07+0j) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0022494124470939995+0j) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0004458535128840877+0j) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11] +
(-1.974225378900023e-08+0j) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.0474716553009873e-08+0j) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.12507032579771865+0j) [Y1 Z2 Y3] +
(-1.3807781481927127e-07+0j) [Y1 Z2 Y3 X4 Z5 X6] +
(-3.3767393084402523e-07+0j) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.001863894282458711+0j) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.3807781481927127e-07+0j) [Y1 Z2 Y3 Y4 Z5 Y6] +
(-3.3767393084402523e-07+0j) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.001863894282458711+0j) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.001105903769189611+0j) [Y1 Z2 Y3 Z4] +
(-1.3807781481927127e-07+0j) [Y1 Z2 Y3 X5 Z6 X7] +
(-7.900128986023348e-07+0j) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0053480515826766+0j) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.5510539177754705e-07+0j) [Y1 Z2 Y3 Y5 Z6 Y7] +
(-1.1468376507194344e-06+0j) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.007597464029770599+0j) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005530759218631476+0j) [Y1 Z2 Y3 Z5] +
(0.0005940221543005191+0j) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10] +
(-8.379773242469647e-08+0j) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0005940221543005191+0j) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10] +
(-8.379773242469647e-08+0j) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0033476175306661007+0j) [Y1 Z2 Y3 Z6] +
(0.005262642473076829+0j) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11] +
(-8.074305985628489e-07+0j) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005708495985960916+0j) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11] +
-1.9742253789775635e-08j [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.352332102713157e-07+0j) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.00803252091882131+0j) [Y1 Z2 Y3 Z7] +
(0.0029297686747509536+0j) [Y1 Z2 Y3 Z8] +
(0.011055020596132+0j) [Y1 Z2 Y3 Z9] +
(-1.1076325598560301e-07+0j) [Y1 Z2 Y3 X10 Z11 X12] +
(-1.1076325598560301e-07+0j) [Y1 Z2 Y3 Y10 Z11 Y12] +
(0.0017560707018411665+0j) [Y1 Z2 Y3 Z10] +
(-6.556281914577381e-07+0j) [Y1 Z2 Y3 X11 Z12 X13] +
(-6.418291574617451e-07+0j) [Y1 Z2 Y3 Y11 Z12 Y13] +
(0.0035552901955041953+0j) [Y1 Z2 Y3 Z11] +
(0.0023262306231580094+0j) [Y1 Z2 Y3 Z12] +
(0.006901238249797219+0j) [Y1 Z2 Y3 Z13] +
(0.0008533856254125534+0j) [Y1 Z2 Z3 X4 X5 Y6] +
(0.0007870896771024511+0j) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10] +
(-1.8394209156333912e-07+0j) [Y1 Z2 Z3 X4 X6 Y7] +
(-0.0012223378081538377+0j) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.00019400857029756911+0j) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12] +
(-2.3713289480087198e-07+0j) [Y1 Z2 Z3 X4 X8 Y9] +
(8.057446595312111e-08+0j) [Y1 Z2 Z3 X4 X10 Y11] +
(-0.0009581655836696515+0j) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13] +
(-0.0017278753941369586+0j) [Y1 Z2 Z3 X4 X11 Y12] +
(-3.0868265653078873e-07+0j) [Y1 Z2 Z3 X4 X12 Y13] +
(-0.0008533856254125534+0j) [Y1 Z2 Z3 Y4 X5 X6] +
(-0.0007870896771024511+0j) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10] +
(-1.8394209156333912e-07+0j) [Y1 Z2 Z3 Y4 Y6 Y7] +
(-0.0012223378081538377+0j) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.00019400857029756911+0j) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12] +
(-2.3713289480087198e-07+0j) [Y1 Z2 Z3 Y4 Y8 Y9] +
(8.057446595312111e-08+0j) [Y1 Z2 Z3 Y4 Y10 Y11] +
(-0.0009581655836696515+0j) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13] +
(0.0017278753941369586+0j) [Y1 Z2 Z3 Y4 X11 X12] +
(-3.0868265653078873e-07+0j) [Y1 Z2 Z3 Y4 Y12 Y13] +
(-0.0010283292378562685+0j) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0026860409778066098+0j) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13] +
(3.202076881096729e-06+0j) [Y1 Z2 Z3 Z4 Y5] +
(4.092250616193178e-07+0j) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10] +
(0.0023949726397980288+0j) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.092250616193178e-07+0j) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10] +
(0.0023949726397980288+0j) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.684915095225614e-07+0j) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11] +
(0.0022009640695004593+0j) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.4445978543531407e-07+0j) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11] +
(0.0011726348316441913+0j) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.8394209156333915e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z7] +
(8.649310137048479e-08+0j) [Y1 Z2 Z3 Z4 Y5 Z8] +
(3.2362599617135675e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z9] +
(0.0022619660624823524+0j) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12] +
(0.0022619660624823524+0j) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12] +
(-5.927453082507015e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z10] +
(0.003989841456619312+0j) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13] +
(0.0013038004788127012+0j) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13] +
(-6.733197742038225e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z11] +
(9.306536652189402e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z12] +
(1.239336321749729e-06+0j) [Y1 Z2 Z3 Z4 Y5 Z13] +
(0.0005192743499487672+0j) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10] +
(-1.850564193067663e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.003356670563832903+0j) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9] +
(-0.00013840177303549577+0j) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11] +
(-4.997018422118328e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13] +
(6.175246207213874e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12] +
(-0.003267513854423567+0j) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13] +
(-0.0005192743499487672+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10] +
(1.850564193067663e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-0.003356670563832903+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9] +
(-0.00013840177303549577+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11] +
(-4.997018422118328e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13] +
(-6.175246207213874e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12] +
(-0.003267513854423567+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13] +
(1.2004287494357673e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13] +
(0.04274327701378291+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7] +
(0.001280306097349656+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8] +
(0.004636976661182559+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9] +
(7.246974425350932e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12] +
(7.246974425350932e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12] +
(0.005241535382803868+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10] +
(1.0717282181370591e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13] +
(2.312094305180259e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13] +
(0.0053799371558393635+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11] +
(0.0010435246534907436+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12] +
(0.004311038507914311+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13] +
(0.0038764708993369603+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10] +
(-7.540341413821506e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-0.0038764708993369603+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10] +
(7.540341413821506e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-0.002984166168121945+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(-0.002984166168121945+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(0.07165035181002621+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.0012366478019244951+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(0.00422081397004644+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(-1.3987009015719039e-05+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(8.949476487734458e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(-7.661347213024096e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(-0.0021413612231016444+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11] +
(-6.876621658058509e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(0.005408954422409958+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11] +
(-1.0444941297965177e-06+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(0.001532483523072997+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11] +
(-2.904599884143673e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(0.005286546538226852+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11] +
(-9.956079229935192e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0027790267990255727+0j) [Y1 Z2 Z3 Z4 Z5 Y7] +
(0.004767272188278084+0j) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11] +
(-8.105515036867528e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0024629170071339204+0j) [Y1 Z2 Z3 Z4 Z6 Y7] +
(0.00071567342489085+0j) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11] +
(-3.076732531703681e-07+0j) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.2919694860689314e-07+0j) [Y1 Z2 Z3 Y5] +
(0.0016095313817213672+0j) [Y1 Z2 Z3 Z5 Z6 Y7] +
(-7.141625221160109e-05+0j) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-2.6667317544310593e-07+0j) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.7379332625337298e-07+0j) [Y1 Z2 Z4 Y5] +
(0.001667604181144053+0j) [Y1 Z2 Z4 Z5 Z6 Y7] +
(-0.0014528843214169427+0j) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(4.670402390587751e-07+0j) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.003276971931231557+0j) [Y1 Y3] +
(3.6060718681366187e-07+0j) [Y1 Z3 Z4 Y5] +
(0.00396156079249652+0j) [Y1 Z3 Z4 Z5 Z6 Y7] +
(0.00018787053389545029+0j) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.6569309316031466e-07+0j) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(12.412630742111777+0j) [Z1] +
(-1.1908508085460702e-06+0j) [Z1 X2 Z3 X4] +
(-0.03276765782329056+0j) [Z1 X2 Z3 Z4 Z5 X6] +
(-0.07635021950635026+0j) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.5809603692962282e-05+0j) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.1908508085460702e-06+0j) [Z1 Y2 Z3 Y4] +
(-0.03276765782329056+0j) [Z1 Y2 Z3 Z4 Z5 Y6] +
(-0.07635021950635026+0j) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.5809603692962282e-05+0j) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.251294456745917+0j) [Z1 Z2] +
(-8.337746756037756e-07+0j) [Z1 X3 Z4 X5] +
(-0.0271150368452732+0j) [Z1 X3 Z4 Z5 Z6 X7] +
(-0.06752385099214041+0j) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.4017109735241801e-05+0j) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.337746756037756e-07+0j) [Z1 Y3 Z4 Y5] +
(-0.0271150368452732+0j) [Z1 Y3 Z4 Z5 Z6 Y7] +
(-0.06752385099214041+0j) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.4017109735241801e-05+0j) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.2367108078383043+0j) [Z1 Z3] +
(-3.3440815567509314e-06+0j) [Z1 X4 Z5 X6] +
(-1.6103585305478253e-05+0j) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.09065144207036487+0j) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-3.3440815567509314e-06+0j) [Z1 Y4 Z5 Y6] +
(-1.6103585305478253e-05+0j) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.09065144207036487+0j) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.19936354537360845+0j) [Z1 Z4] +
(-3.099349243847675e-06+0j) [Z1 X5 Z6 X7] +
(-1.5316808795056106e-05+0j) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.08684737589863631+0j) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.099349243847675e-06+0j) [Z1 Y5 Z6 Y7] +
(-1.5316808795056106e-05+0j) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.08684737589863631+0j) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.19661770890342162+0j) [Z1 Z5] +
(0.056007330877807654+0j) [Z1 X6 Z7 Z8 Z9 X10] +
(-6.481851833440172e-06+0j) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.056007330877807654+0j) [Z1 Y6 Z7 Z8 Z9 Y10] +
(-6.481851833440172e-06+0j) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.24853483371314286+0j) [Z1 Z6] +
(0.05608468124661356+0j) [Z1 X7 Z8 Z9 Z10 X11] +
(-6.652209669013128e-06+0j) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.05608468124661356+0j) [Z1 Y7 Z8 Z9 Z10 Y11] +
(-6.652209669013128e-06+0j) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.24164663936017228+0j) [Z1 Z7] +
(0.27883454426723425+0j) [Z1 Z8] +
(0.272325183066057+0j) [Z1 Z9] +
(-1.6148794138609382e-06+0j) [Z1 X10 Z11 X12] +
(-1.6148794138609382e-06+0j) [Z1 Y10 Z11 Y12] +
(0.2007286646044179+0j) [Z1 Z10] +
(-2.177664605015797e-06+0j) [Z1 X11 Z12 X13] +
(-2.177664605015797e-06+0j) [Z1 Y11 Z12 Y13] +
(0.1929972393536426+0j) [Z1 Z11] +
(0.2163103749863184+0j) [Z1 Z12] +
(0.21102659849791544+0j) [Z1 Z13] +
(-0.035839567953353496+0j) [X2 X3 Y4 Y5] +
(-2.19905161840579e-07+0j) [X2 X3 Y4 Z5 Z6 Y7] +
(-2.3609563202907338e-06+0j) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.010311482489831754+0j) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.1990516184057902e-07+0j) [X2 X3 X5 X6] +
(-2.3609563202907338e-06+0j) [X2 X3 X5 Z6 Z7 Z8 Z9 X10] +
(-0.010311482489831755+0j) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.0311438179889671+0j) [X2 X3 Y6 Y7] +
(0.0053686593581095+0j) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11] +
(9.20935064824041e-08+0j) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0053686593581095+0j) [X2 X3 X7 Z8 Z9 X10] +
(9.20935064824041e-08+0j) [X2 X3 X7 Z8 Z9 Z10 Z11 X12] +
(-0.03619412355904258+0j) [X2 X3 Y8 Y9] +
(-0.02538465750845745+0j) [X2 X3 Y10 Y11] +
(2.1726691015061793e-06+0j) [X2 X3 Y10 Z11 Z12 Y13] +
(2.1726691015061793e-06+0j) [X2 X3 X11 X12] +
(-0.015577208063976469+0j) [X2 X3 Y12 Y13] +
(0.035839567953353496+0j) [X2 Y3 Y4 X5] +
(2.19905161840579e-07+0j) [X2 Y3 Y4 Z5 Z6 X7] +
(2.3609563202907338e-06+0j) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.010311482489831754+0j) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.1990516184057902e-07+0j) [X2 Y3 Y5 X6] +
(-2.3609563202907338e-06+0j) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.010311482489831755+0j) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0311438179889671+0j) [X2 Y3 Y6 X7] +
(-0.0053686593581095+0j) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11] +
(-9.20935064824041e-08+0j) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0053686593581095+0j) [X2 Y3 Y7 Z8 Z9 X10] +
(9.20935064824041e-08+0j) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12] +
(0.03619412355904258+0j) [X2 Y3 Y8 X9] +
(0.02538465750845745+0j) [X2 Y3 Y10 X11] +
(-2.1726691015061793e-06+0j) [X2 Y3 Y10 Z11 Z12 X13] +
(2.1726691015061793e-06+0j) [X2 Y3 Y11 X12] +
(0.015577208063976469+0j) [X2 Y3 Y12 X13] +
(-3.887051673510447e-06+0j) [X2 Z3 X4] +
(-0.005143391768825118+0j) [X2 Z3 X4 X5 Z6 X7] +
(-0.009841749246962614+0j) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.988511706578377e-06+0j) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.005143391768825118+0j) [X2 Z3 X4 Y5 Z6 Y7] +
(-0.009841749246962614+0j) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.988511706578377e-06+0j) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.764994117020342e-07+0j) [X2 Z3 X4 Z5] +
(1.6893489514962132e-06+0j) [X2 Z3 X4 X6 Z7 Z8 Z9 X10] +
(0.010757563953908946+0j) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.537178096713223e-08+0j) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10] +
(4.2055484112186146e-05+0j) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.5935343905802883e-07+0j) [X2 Z3 X4 Z6] +
(3.2118420191711192e-06+0j) [X2 Z3 X4 X7 Z8 Z9 Z10 X11] +
(0.019299560579363797+0j) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.2118420191711192e-06+0j) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11] +
(0.019299560579363797+0j) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.1954890101854388e-06+0j) [X2 Z3 X4 Z7] +
(2.186842377040007e-07+0j) [X2 Z3 X4 Z8] +
(-5.770052995914854e-07+0j) [X2 Z3 X4 Z9] +
(0.015588250102380198+0j) [X2 Z3 X4 X10 Z11 X12] +
(0.005324835234221674+0j) [X2 Z3 X4 Y10 Z11 Y12] +
(-3.158656432078617e-06+0j) [X2 Z3 X4 Z10] +
(0.02435307767806899+0j) [X2 Z3 X4 X11 Z12 X13] +
(0.02435307767806899+0j) [X2 Z3 X4 Y11 Z12 Y13] +
(-7.801707500512476e-06+0j) [X2 Z3 X4 Z11] +
(3.539054184582881e-06+0j) [X2 Z3 X4 Z12] +
(8.814937306748825e-06+0j) [X2 Z3 X4 Z13] +
(1.6288532435397403e-06+0j) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10] +
(0.010715508469796756+0j) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.010263414868158524+0j) [X2 Z3 Y4 Y10 Z11 X12] +
(-1.454842449243468e-06+0j) [X2 Z3 Z4 X5 Y6 Y7] +
(-3.151346311214646e-06+0j) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11] +
(-0.019257505095251613+0j) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.5224930676749059e-06+0j) [X2 Z3 Z4 X5 X7 Z8 Z9 X10] +
(-0.008541996625454854+0j) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-7.956895372954862e-07+0j) [X2 Z3 Z4 X5 Y8 Y9] +
(-4.6430510684338575e-06+0j) [X2 Z3 Z4 X5 Y10 Y11] +
(-0.019028242443847317+0j) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13] +
(-0.008764827575688795+0j) [X2 Z3 Z4 X5 X11 X12] +
(5.275883122165944e-06+0j) [X2 Z3 Z4 X5 Y12 Y13] +
(1.454842449243468e-06+0j) [X2 Z3 Z4 Y5 Y6 X7] +
(3.151346311214646e-06+0j) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11] +
(0.019257505095251613+0j) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.5224930676749059e-06+0j) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10] +
(-0.008541996625454854+0j) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(7.956895372954862e-07+0j) [X2 Z3 Z4 Y5 Y8 X9] +
(4.6430510684338575e-06+0j) [X2 Z3 Z4 Y5 Y10 X11] +
(0.019028242443847317+0j) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13] +
(-0.008764827575688795+0j) [X2 Z3 Z4 Y5 Y11 X12] +
(-5.275883122165944e-06+0j) [X2 Z3 Z4 Y5 Y12 X13] +
(-0.12133276911042368+0j) [X2 Z3 Z4 Z5 X6] +
(-0.008469978791023921+0j) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(2.6863815430103157e-07+0j) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.008469978791023921+0j) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(2.6863815430103157e-07+0j) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.025996177598021218+0j) [X2 Z3 Z4 Z5 X6 Z7] +
(-0.005805188989826909+0j) [X2 Z3 Z4 Z5 X6 Z8] +
(-0.01756120240964618+0j) [X2 Z3 Z4 Z5 X6 Z9] +
(-7.988770289489799e-07+0j) [X2 Z3 Z4 Z5 X6 X10 Z11 X12] +
(-3.4273231090971276e-07+0j) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12] +
(-0.0008145313270956898+0j) [X2 Z3 Z4 Z5 X6 Z10] +
(2.7455184003009737e-06+0j) [X2 Z3 Z4 Z5 X6 X11 Z12 X13] +
(2.7455184003009737e-06+0j) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13] +
(0.014411099430130893+0j) [X2 Z3 Z4 Z5 X6 Z11] +
(0.0006650070219498999+0j) [X2 Z3 Z4 Z5 X6 Z12] +
(-0.0034937903598901464+0j) [X2 Z3 Z4 Z5 X6 Z13] +
(-4.56144718039267e-07+0j) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12] +
(-0.011756013419819272+0j) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9] +
(0.015225630757226582+0j) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11] +
(-3.088250711210686e-06+0j) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(-3.5443954292499532e-06+0j) [X2 Z3 Z4 Z5 Z6 X7 X11 X12] +
(-0.004158797381840046+0j) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13] +
(0.011756013419819272+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9] +
(-0.015225630757226582+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11] +
(3.088250711210686e-06+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(-3.5443954292499532e-06+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12] +
(0.004158797381840046+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13] +
(0.014603704729162127+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(-2.8742990714188077e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.014603704729162127+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(-2.8742990714188077e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(-0.2816425776702301+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-1.3002946562939022e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(-1.3002946562939022e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(-0.024282117354693045+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(-0.019538050311314767+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(-0.017091553155898876+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(0.002446497155415891+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(-0.002446497155415891+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(5.775950527268799e-05+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(2.883676576039133e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(5.146496327547127e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(3.846201671253224e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(-0.03935916802205312+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10] +
(7.979825793387279e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(-0.02475546329289099+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10] +
(5.1055267219684716e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(-0.02143381072160093+0j) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10] +
(5.159350502009226e-06+0j) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(-0.02990378951262485+0j) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10] +
(5.427988656310258e-06+0j) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0016638798784907945+0j) [X2 Z3 Z4 X6] +
(-0.018889030304942923+0j) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10] +
(2.9473560116005596e-06+0j) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.003479511890334323+0j) [X2 Z3 Z5 X6] +
(-0.028730779551905533+0j) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10] +
(5.935867718178937e-06+0j) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.6021167405610082e-06+0j) [X2 X4] +
(0.0004956762314916675+0j) [X2 Z4 Z5 X6] +
(-0.03560837898831257+0j) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(7.253273348068956e-06+0j) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.035839567953353496+0j) [Y2 X3 X4 Y5] +
(2.19905161840579e-07+0j) [Y2 X3 X4 Z5 Z6 Y7] +
(2.3609563202907338e-06+0j) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.010311482489831754+0j) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.1990516184057902e-07+0j) [Y2 X3 X5 Y6] +
(-2.3609563202907338e-06+0j) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.010311482489831755+0j) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0311438179889671+0j) [Y2 X3 X6 Y7] +
(-0.0053686593581095+0j) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11] +
(-9.20935064824041e-08+0j) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0053686593581095+0j) [Y2 X3 X7 Z8 Z9 Y10] +
(9.20935064824041e-08+0j) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12] +
(0.03619412355904258+0j) [Y2 X3 X8 Y9] +
(0.02538465750845745+0j) [Y2 X3 X10 Y11] +
(-2.1726691015061793e-06+0j) [Y2 X3 X10 Z11 Z12 Y13] +
(2.1726691015061793e-06+0j) [Y2 X3 X11 Y12] +
(0.015577208063976469+0j) [Y2 X3 X12 Y13] +
(-0.035839567953353496+0j) [Y2 Y3 X4 X5] +
(-2.19905161840579e-07+0j) [Y2 Y3 X4 Z5 Z6 X7] +
(-2.3609563202907338e-06+0j) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.010311482489831754+0j) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.1990516184057902e-07+0j) [Y2 Y3 Y5 Y6] +
(-2.3609563202907338e-06+0j) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.010311482489831755+0j) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.0311438179889671+0j) [Y2 Y3 X6 X7] +
(0.0053686593581095+0j) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11] +
(9.20935064824041e-08+0j) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0053686593581095+0j) [Y2 Y3 Y7 Z8 Z9 Y10] +
(9.20935064824041e-08+0j) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.03619412355904258+0j) [Y2 Y3 X8 X9] +
(-0.02538465750845745+0j) [Y2 Y3 X10 X11] +
(2.1726691015061793e-06+0j) [Y2 Y3 X10 Z11 Z12 X13] +
(2.1726691015061793e-06+0j) [Y2 Y3 Y11 Y12] +
(-0.015577208063976469+0j) [Y2 Y3 X12 X13] +
(1.6288532435397403e-06+0j) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10] +
(0.010715508469796756+0j) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.010263414868158524+0j) [Y2 Z3 X4 X10 Z11 Y12] +
(-3.887051673510447e-06+0j) [Y2 Z3 Y4] +
(-0.005143391768825118+0j) [Y2 Z3 Y4 X5 Z6 X7] +
(-0.009841749246962614+0j) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.988511706578377e-06+0j) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.005143391768825118+0j) [Y2 Z3 Y4 Y5 Z6 Y7] +
(-0.009841749246962614+0j) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.988511706578377e-06+0j) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.764994117020342e-07+0j) [Y2 Z3 Y4 Z5] +
(4.537178096713223e-08+0j) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10] +
(4.2055484112186146e-05+0j) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.6893489514962132e-06+0j) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10] +
(0.010757563953908946+0j) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.5935343905802883e-07+0j) [Y2 Z3 Y4 Z6] +
(3.2118420191711192e-06+0j) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11] +
(0.019299560579363797+0j) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.2118420191711192e-06+0j) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11] +
(0.019299560579363797+0j) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.1954890101854388e-06+0j) [Y2 Z3 Y4 Z7] +
(2.186842377040007e-07+0j) [Y2 Z3 Y4 Z8] +
(-5.770052995914854e-07+0j) [Y2 Z3 Y4 Z9] +
(0.005324835234221674+0j) [Y2 Z3 Y4 X10 Z11 X12] +
(0.015588250102380198+0j) [Y2 Z3 Y4 Y10 Z11 Y12] +
(-3.158656432078617e-06+0j) [Y2 Z3 Y4 Z10] +
(0.02435307767806899+0j) [Y2 Z3 Y4 X11 Z12 X13] +
(0.02435307767806899+0j) [Y2 Z3 Y4 Y11 Z12 Y13] +
(-7.801707500512476e-06+0j) [Y2 Z3 Y4 Z11] +
(3.539054184582881e-06+0j) [Y2 Z3 Y4 Z12] +
(8.814937306748825e-06+0j) [Y2 Z3 Y4 Z13] +
(1.454842449243468e-06+0j) [Y2 Z3 Z4 X5 X6 Y7] +
(3.151346311214646e-06+0j) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11] +
(0.019257505095251613+0j) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.5224930676749059e-06+0j) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10] +
(-0.008541996625454854+0j) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(7.956895372954862e-07+0j) [Y2 Z3 Z4 X5 X8 Y9] +
(4.6430510684338575e-06+0j) [Y2 Z3 Z4 X5 X10 Y11] +
(0.019028242443847317+0j) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13] +
(-0.008764827575688795+0j) [Y2 Z3 Z4 X5 X11 Y12] +
(-5.275883122165944e-06+0j) [Y2 Z3 Z4 X5 X12 Y13] +
(-1.454842449243468e-06+0j) [Y2 Z3 Z4 Y5 X6 X7] +
(-3.151346311214646e-06+0j) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11] +
(-0.019257505095251613+0j) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.5224930676749059e-06+0j) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10] +
(-0.008541996625454854+0j) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895372954862e-07+0j) [Y2 Z3 Z4 Y5 X8 X9] +
(-4.6430510684338575e-06+0j) [Y2 Z3 Z4 Y5 X10 X11] +
(-0.019028242443847317+0j) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13] +
(-0.008764827575688795+0j) [Y2 Z3 Z4 Y5 Y11 Y12] +
(5.275883122165944e-06+0j) [Y2 Z3 Z4 Y5 X12 X13] +
(-4.56144718039267e-07+0j) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12] +
(-0.12133276911042368+0j) [Y2 Z3 Z4 Z5 Y6] +
(-0.008469978791023921+0j) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(2.6863815430103157e-07+0j) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.008469978791023921+0j) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(2.6863815430103157e-07+0j) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.025996177598021218+0j) [Y2 Z3 Z4 Z5 Y6 Z7] +
(-0.005805188989826909+0j) [Y2 Z3 Z4 Z5 Y6 Z8] +
(-0.01756120240964618+0j) [Y2 Z3 Z4 Z5 Y6 Z9] +
(-3.4273231090971276e-07+0j) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12] +
(-7.988770289489799e-07+0j) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12] +
(-0.0008145313270956898+0j) [Y2 Z3 Z4 Z5 Y6 Z10] +
(2.7455184003009737e-06+0j) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13] +
(2.7455184003009737e-06+0j) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13] +
(0.014411099430130893+0j) [Y2 Z3 Z4 Z5 Y6 Z11] +
(0.0006650070219498999+0j) [Y2 Z3 Z4 Z5 Y6 Z12] +
(-0.0034937903598901464+0j) [Y2 Z3 Z4 Z5 Y6 Z13] +
(0.011756013419819272+0j) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9] +
(-0.015225630757226582+0j) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11] +
(3.088250711210686e-06+0j) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(-3.5443954292499532e-06+0j) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12] +
(0.004158797381840046+0j) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13] +
(-0.011756013419819272+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9] +
(0.015225630757226582+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11] +
(-3.088250711210686e-06+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(-3.5443954292499532e-06+0j) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12] +
(-0.004158797381840046+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13] +
(0.014603704729162127+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(-2.8742990714188077e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.014603704729162127+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(-2.8742990714188077e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(-0.2816425776702301+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-1.3002946562939022e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(-1.3002946562939022e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(-0.024282117354693045+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(-0.019538050311314767+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(-0.017091553155898876+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(-0.002446497155415891+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(0.002446497155415891+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(5.775950527268799e-05+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.883676576039133e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(5.146496327547127e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(3.846201671253224e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(-0.03935916802205312+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10] +
(7.979825793387279e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(-0.02475546329289099+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10] +
(5.1055267219684716e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(-0.02143381072160093+0j) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10] +
(5.159350502009226e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(-0.02990378951262485+0j) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10] +
(5.427988656310258e-06+0j) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0016638798784907945+0j) [Y2 Z3 Z4 Y6] +
(-0.018889030304942923+0j) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10] +
(2.9473560116005596e-06+0j) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.003479511890334323+0j) [Y2 Z3 Z5 Y6] +
(-0.028730779551905533+0j) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10] +
(5.935867718178937e-06+0j) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.6021167405610082e-06+0j) [Y2 Y4] +
(0.0004956762314916675+0j) [Y2 Z4 Z5 Y6] +
(-0.03560837898831257+0j) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(7.253273348068956e-06+0j) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.653894222683169+0j) [Z2] +
(1.602116740561008e-06+0j) [Z2 X3 Z4 X5] +
(0.0004956762314916675+0j) [Z2 X3 Z4 Z5 Z6 X7] +
(-0.03560837898831257+0j) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(7.253273348068956e-06+0j) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.602116740561008e-06+0j) [Z2 Y3 Z4 Y5] +
(0.0004956762314916675+0j) [Z2 Y3 Z4 Z5 Z6 Y7] +
(-0.03560837898831257+0j) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(7.253273348068956e-06+0j) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.18189085790751353+0j) [Z2 Z3] +
(-9.509249752218616e-07+0j) [Z2 X4 Z5 X6] +
(-4.7288431471088084e-06+0j) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.024591860883829992+0j) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-9.509249752218616e-07+0j) [Z2 Y4 Z5 Y6] +
(-4.7288431471088084e-06+0j) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.024591860883829992+0j) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.12495807739503213+0j) [Z2 Z4] +
(-1.1708301370624408e-06+0j) [Z2 X5 Z6 X7] +
(-7.089799467399542e-06+0j) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.034903343373661744+0j) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.1708301370624408e-06+0j) [Z2 Y5 Z6 Y7] +
(-7.089799467399542e-06+0j) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.034903343373661744+0j) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.16079764534838564+0j) [Z2 Z5] +
(0.01902042317303997+0j) [Z2 X6 Z7 Z8 Z9 X10] +
(-2.103215604576532e-06+0j) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.01902042317303997+0j) [Z2 Y6 Z7 Z8 Z9 Y10] +
(-2.103215604576532e-06+0j) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.13739104762683224+0j) [Z2 Z6] +
(0.02438908253114947+0j) [Z2 X7 Z8 Z9 Z10 X11] +
(-2.0111220980941287e-06+0j) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.02438908253114947+0j) [Z2 Y7 Z8 Z9 Z10 Y11] +
(-2.0111220980941287e-06+0j) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.16853486561579933+0j) [Z2 Z7] +
(0.15071408121008287+0j) [Z2 Z8] +
(0.18690820476912542+0j) [Z2 Z9] +
(-1.063228342377944e-06+0j) [Z2 X10 Z11 X12] +
(-1.063228342377944e-06+0j) [Z2 Y10 Z11 Y12] +
(0.12799502492468406+0j) [Z2 Z10] +
(1.1094407591282353e-06+0j) [Z2 X11 Z12 X13] +
(1.1094407591282353e-06+0j) [Z2 Y11 Z12 Y13] +
(0.15337968243314154+0j) [Z2 Z11] +
(0.14011289865354815+0j) [Z2 Z12] +
(0.1556901067175246+0j) [Z2 Z13] +
(0.005143391768825118+0j) [X3 X4 Y5 Y6] +
(0.009841749246962612+0j) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10] +
(-2.988511706578377e-06+0j) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.4548424492434675e-06+0j) [X3 X4 X6 X7] +
(-1.5224930676749059e-06+0j) [X3 X4 X6 Z7 Z8 Z9 Z10 X11] +
(-0.008541996625454854+0j) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.151346311214646e-06+0j) [X3 X4 Y7 Z8 Z9 Y10] +
(-0.019257505095251613+0j) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895372954861e-07+0j) [X3 X4 X8 X9] +
(-4.6430510684338575e-06+0j) [X3 X4 X10 X11] +
(-0.008764827575688795+0j) [X3 X4 X10 Z11 Z12 X13] +
(-0.019028242443847317+0j) [X3 X4 Y11 Y12] +
(5.275883122165944e-06+0j) [X3 X4 X12 X13] +
(-0.005143391768825118+0j) [X3 Y4 Y5 X6] +
(-0.009841749246962612+0j) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10] +
(2.988511706578377e-06+0j) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.4548424492434675e-06+0j) [X3 Y4 Y6 X7] +
(-1.5224930676749059e-06+0j) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11] +
(-0.008541996625454854+0j) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.151346311214646e-06+0j) [X3 Y4 Y7 Z8 Z9 X10] +
(0.019257505095251613+0j) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12] +
(-7.956895372954861e-07+0j) [X3 Y4 Y8 X9] +
(-4.6430510684338575e-06+0j) [X3 Y4 Y10 X11] +
(-0.008764827575688795+0j) [X3 Y4 Y10 Z11 Z12 X13] +
(0.019028242443847317+0j) [X3 Y4 Y11 X12] +
(5.275883122165944e-06+0j) [X3 Y4 Y12 X13] +
(-3.8870516735104466e-06+0j) [X3 Z4 X5] +
(3.2118420191711192e-06+0j) [X3 Z4 X5 X6 Z7 Z8 Z9 X10] +
(0.019299560579363797+0j) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.2118420191711192e-06+0j) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10] +
(0.019299560579363797+0j) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.1954890101854388e-06+0j) [X3 Z4 X5 Z6] +
(1.6893489514962132e-06+0j) [X3 Z4 X5 X7 Z8 Z9 Z10 X11] +
(0.010757563953908946+0j) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.537178096713223e-08+0j) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11] +
(4.2055484112186146e-05+0j) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.5935343905802883e-07+0j) [X3 Z4 X5 Z7] +
(-5.770052995914854e-07+0j) [X3 Z4 X5 Z8] +
(2.186842377040007e-07+0j) [X3 Z4 X5 Z9] +
(0.02435307767806899+0j) [X3 Z4 X5 X10 Z11 X12] +
(0.02435307767806899+0j) [X3 Z4 X5 Y10 Z11 Y12] +
(-7.801707500512476e-06+0j) [X3 Z4 X5 Z10] +
(0.015588250102380198+0j) [X3 Z4 X5 X11 Z12 X13] +
(0.005324835234221674+0j) [X3 Z4 X5 Y11 Z12 Y13] +
(-3.158656432078617e-06+0j) [X3 Z4 X5 Z11] +
(8.814937306748825e-06+0j) [X3 Z4 X5 Z12] +
(3.539054184582881e-06+0j) [X3 Z4 X5 Z13] +
(1.6288532435397403e-06+0j) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11] +
(0.010715508469796756+0j) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.010263414868158524+0j) [X3 Z4 Y5 Y11 Z12 X13] +
(0.008469978791023921+0j) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10] +
(-2.6863815430103157e-07+0j) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.011756013419819272+0j) [X3 Z4 Z5 X6 X8 X9] +
(0.015225630757226582+0j) [X3 Z4 Z5 X6 X10 X11] +
(-3.5443954292499532e-06+0j) [X3 Z4 Z5 X6 X10 Z11 Z12 X13] +
(-3.088250711210686e-06+0j) [X3 Z4 Z5 X6 Y11 Y12] +
(-0.004158797381840046+0j) [X3 Z4 Z5 X6 X12 X13] +
(-0.008469978791023921+0j) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10] +
(2.6863815430103157e-07+0j) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.011756013419819272+0j) [X3 Z4 Z5 Y6 Y8 X9] +
(0.015225630757226582+0j) [X3 Z4 Z5 Y6 Y10 X11] +
(-3.5443954292499532e-06+0j) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13] +
(3.088250711210686e-06+0j) [X3 Z4 Z5 Y6 Y11 X12] +
(-0.004158797381840046+0j) [X3 Z4 Z5 Y6 Y12 X13] +
(-0.12133276911042369+0j) [X3 Z4 Z5 Z6 X7] +
(-0.01756120240964618+0j) [X3 Z4 Z5 Z6 X7 Z8] +
(-0.005805188989826909+0j) [X3 Z4 Z5 Z6 X7 Z9] +
(2.7455184003009737e-06+0j) [X3 Z4 Z5 Z6 X7 X10 Z11 X12] +
(2.7455184003009737e-06+0j) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12] +
(0.014411099430130893+0j) [X3 Z4 Z5 Z6 X7 Z10] +
(-7.988770289489799e-07+0j) [X3 Z4 Z5 Z6 X7 X11 Z12 X13] +
(-3.4273231090971276e-07+0j) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13] +
(-0.0008145313270956898+0j) [X3 Z4 Z5 Z6 X7 Z11] +
(-0.0034937903598901464+0j) [X3 Z4 Z5 Z6 X7 Z12] +
(0.0006650070219498999+0j) [X3 Z4 Z5 Z6 X7 Z13] +
(-4.56144718039267e-07+0j) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13] +
(-0.014603704729162127+0j) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10] +
(2.8742990714188077e-06+0j) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(0.014603704729162127+0j) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10] +
(-2.8742990714188077e-06+0j) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(1.3002946562939022e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12] +
(0.002446497155415891+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-1.3002946562939022e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12] +
(0.002446497155415891+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(-0.2816425776702302+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.017091553155898876+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(-0.019538050311314767+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(5.775950527268797e-05+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(2.883676576039133e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(3.846201671253224e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(-0.024282117354693045+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11] +
(5.146496327547127e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(-0.02475546329289099+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11] +
(5.1055267219684716e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(-0.03935916802205312+0j) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11] +
(7.979825793387279e-06+0j) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(-0.02990378951262485+0j) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11] +
(5.427988656310258e-06+0j) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.02599617759802122+0j) [X3 Z4 Z5 X7] +
(-0.02143381072160093+0j) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11] +
(5.159350502009226e-06+0j) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.003479511890334323+0j) [X3 Z4 Z6 X7] +
(-0.028730779551905533+0j) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11] +
(5.935867718178937e-06+0j) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.764994117020342e-07+0j) [X3 X5] +
(0.0016638798784907945+0j) [X3 Z5 Z6 X7] +
(-0.018889030304942923+0j) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.9473560116005596e-06+0j) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.005143391768825118+0j) [Y3 X4 X5 Y6] +
(-0.009841749246962612+0j) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10] +
(2.988511706578377e-06+0j) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.4548424492434675e-06+0j) [Y3 X4 X6 Y7] +
(-1.5224930676749059e-06+0j) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11] +
(-0.008541996625454854+0j) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.151346311214646e-06+0j) [Y3 X4 X7 Z8 Z9 Y10] +
(0.019257505095251613+0j) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895372954861e-07+0j) [Y3 X4 X8 Y9] +
(-4.6430510684338575e-06+0j) [Y3 X4 X10 Y11] +
(-0.008764827575688795+0j) [Y3 X4 X10 Z11 Z12 Y13] +
(0.019028242443847317+0j) [Y3 X4 X11 Y12] +
(5.275883122165944e-06+0j) [Y3 X4 X12 Y13] +
(0.005143391768825118+0j) [Y3 Y4 X5 X6] +
(0.009841749246962612+0j) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10] +
(-2.988511706578377e-06+0j) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.4548424492434675e-06+0j) [Y3 Y4 Y6 Y7] +
(-1.5224930676749059e-06+0j) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11] +
(-0.008541996625454854+0j) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.151346311214646e-06+0j) [Y3 Y4 X7 Z8 Z9 X10] +
(-0.019257505095251613+0j) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12] +
(-7.956895372954861e-07+0j) [Y3 Y4 Y8 Y9] +
(-4.6430510684338575e-06+0j) [Y3 Y4 Y10 Y11] +
(-0.008764827575688795+0j) [Y3 Y4 Y10 Z11 Z12 Y13] +
(-0.019028242443847317+0j) [Y3 Y4 X11 X12] +
(5.275883122165944e-06+0j) [Y3 Y4 Y12 Y13] +
(1.6288532435397403e-06+0j) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11] +
(0.010715508469796756+0j) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.010263414868158524+0j) [Y3 Z4 X5 X11 Z12 Y13] +
(-3.8870516735104466e-06+0j) [Y3 Z4 Y5] +
(3.2118420191711192e-06+0j) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10] +
(0.019299560579363797+0j) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.2118420191711192e-06+0j) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10] +
(0.019299560579363797+0j) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.1954890101854388e-06+0j) [Y3 Z4 Y5 Z6] +
(4.537178096713223e-08+0j) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11] +
(4.2055484112186146e-05+0j) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.6893489514962132e-06+0j) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11] +
(0.010757563953908946+0j) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.5935343905802883e-07+0j) [Y3 Z4 Y5 Z7] +
(-5.770052995914854e-07+0j) [Y3 Z4 Y5 Z8] +
(2.186842377040007e-07+0j) [Y3 Z4 Y5 Z9] +
(0.02435307767806899+0j) [Y3 Z4 Y5 X10 Z11 X12] +
(0.02435307767806899+0j) [Y3 Z4 Y5 Y10 Z11 Y12] +
(-7.801707500512476e-06+0j) [Y3 Z4 Y5 Z10] +
(0.005324835234221674+0j) [Y3 Z4 Y5 X11 Z12 X13] +
(0.015588250102380198+0j) [Y3 Z4 Y5 Y11 Z12 Y13] +
(-3.158656432078617e-06+0j) [Y3 Z4 Y5 Z11] +
(8.814937306748825e-06+0j) [Y3 Z4 Y5 Z12] +
(3.539054184582881e-06+0j) [Y3 Z4 Y5 Z13] +
(-0.008469978791023921+0j) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10] +
(2.6863815430103157e-07+0j) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.011756013419819272+0j) [Y3 Z4 Z5 X6 X8 Y9] +
(0.015225630757226582+0j) [Y3 Z4 Z5 X6 X10 Y11] +
(-3.5443954292499532e-06+0j) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13] +
(3.088250711210686e-06+0j) [Y3 Z4 Z5 X6 X11 Y12] +
(-0.004158797381840046+0j) [Y3 Z4 Z5 X6 X12 Y13] +
(0.008469978791023921+0j) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10] +
(-2.6863815430103157e-07+0j) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-0.011756013419819272+0j) [Y3 Z4 Z5 Y6 Y8 Y9] +
(0.015225630757226582+0j) [Y3 Z4 Z5 Y6 Y10 Y11] +
(-3.5443954292499532e-06+0j) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13] +
(-3.088250711210686e-06+0j) [Y3 Z4 Z5 Y6 X11 X12] +
(-0.004158797381840046+0j) [Y3 Z4 Z5 Y6 Y12 Y13] +
(-4.56144718039267e-07+0j) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13] +
(-0.12133276911042369+0j) [Y3 Z4 Z5 Z6 Y7] +
(-0.01756120240964618+0j) [Y3 Z4 Z5 Z6 Y7 Z8] +
(-0.005805188989826909+0j) [Y3 Z4 Z5 Z6 Y7 Z9] +
(2.7455184003009737e-06+0j) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12] +
(2.7455184003009737e-06+0j) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12] +
(0.014411099430130893+0j) [Y3 Z4 Z5 Z6 Y7 Z10] +
(-3.4273231090971276e-07+0j) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13] +
(-7.988770289489799e-07+0j) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13] +
(-0.0008145313270956898+0j) [Y3 Z4 Z5 Z6 Y7 Z11] +
(-0.0034937903598901464+0j) [Y3 Z4 Z5 Z6 Y7 Z12] +
(0.0006650070219498999+0j) [Y3 Z4 Z5 Z6 Y7 Z13] +
(0.014603704729162127+0j) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10] +
(-2.8742990714188077e-06+0j) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-0.014603704729162127+0j) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10] +
(2.8742990714188077e-06+0j) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-1.3002946562939022e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12] +
(0.002446497155415891+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(1.3002946562939022e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12] +
(0.002446497155415891+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(-0.2816425776702302+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.017091553155898876+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(-0.019538050311314767+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(5.775950527268797e-05+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.883676576039133e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(3.846201671253224e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(-0.024282117354693045+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11] +
(5.146496327547127e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(-0.02475546329289099+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11] +
(5.1055267219684716e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(-0.03935916802205312+0j) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11] +
(7.979825793387279e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(-0.02990378951262485+0j) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11] +
(5.427988656310258e-06+0j) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.02599617759802122+0j) [Y3 Z4 Z5 Y7] +
(-0.02143381072160093+0j) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11] +
(5.159350502009226e-06+0j) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.003479511890334323+0j) [Y3 Z4 Z6 Y7] +
(-0.028730779551905533+0j) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11] +
(5.935867718178937e-06+0j) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.764994117020342e-07+0j) [Y3 Y5] +
(0.0016638798784907945+0j) [Y3 Z5 Z6 Y7] +
(-0.018889030304942923+0j) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.9473560116005596e-06+0j) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.6538942226831692+0j) [Z3] +
(-1.1708301370624408e-06+0j) [Z3 X4 Z5 X6] +
(-7.089799467399542e-06+0j) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.034903343373661744+0j) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.1708301370624408e-06+0j) [Z3 Y4 Z5 Y6] +
(-7.089799467399542e-06+0j) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.034903343373661744+0j) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.16079764534838564+0j) [Z3 Z4] +
(-9.509249752218616e-07+0j) [Z3 X5 Z6 X7] +
(-4.7288431471088084e-06+0j) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.024591860883829992+0j) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-9.509249752218616e-07+0j) [Z3 Y5 Z6 Y7] +
(-4.7288431471088084e-06+0j) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.024591860883829992+0j) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.12495807739503213+0j) [Z3 Z5] +
(0.02438908253114947+0j) [Z3 X6 Z7 Z8 Z9 X10] +
(-2.0111220980941287e-06+0j) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.02438908253114947+0j) [Z3 Y6 Z7 Z8 Z9 Y10] +
(-2.0111220980941287e-06+0j) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.16853486561579933+0j) [Z3 Z6] +
(0.01902042317303997+0j) [Z3 X7 Z8 Z9 Z10 X11] +
(-2.103215604576532e-06+0j) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.01902042317303997+0j) [Z3 Y7 Z8 Z9 Z10 Y11] +
(-2.103215604576532e-06+0j) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.13739104762683224+0j) [Z3 Z7] +
(0.18690820476912542+0j) [Z3 Z8] +
(0.15071408121008287+0j) [Z3 Z9] +
(1.1094407591282353e-06+0j) [Z3 X10 Z11 X12] +
(1.1094407591282353e-06+0j) [Z3 Y10 Z11 Y12] +
(0.15337968243314154+0j) [Z3 Z10] +
(-1.063228342377944e-06+0j) [Z3 X11 Z12 X13] +
(-1.063228342377944e-06+0j) [Z3 Y11 Z12 Y13] +
(0.12799502492468406+0j) [Z3 Z11] +
(0.1556901067175246+0j) [Z3 Z12] +
(0.14011289865354815+0j) [Z3 Z13] +
(-0.011982389010247958+0j) [X4 X5 Y6 Y7] +
(-0.007306759928832963+0j) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11] +
(-2.8882935959112267e-07+0j) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.007306759928832963+0j) [X4 X5 X7 Z8 Z9 X10] +
(-2.8882935959112267e-07+0j) [X4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-0.007156934919856944+0j) [X4 X5 Y8 Y9] +
(-0.0176800679524815+0j) [X4 X5 Y10 Y11] +
(-3.694513294293786e-06+0j) [X4 X5 Y10 Z11 Z12 Y13] +
(-3.6945132942937863e-06+0j) [X4 X5 X11 X12] +
(-0.0383146702948039+0j) [X4 X5 Y12 Y13] +
(0.011982389010247958+0j) [X4 Y5 Y6 X7] +
(0.007306759928832963+0j) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11] +
(2.8882935959112267e-07+0j) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.007306759928832963+0j) [X4 Y5 Y7 Z8 Z9 X10] +
(-2.8882935959112267e-07+0j) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(0.007156934919856944+0j) [X4 Y5 Y8 X9] +
(0.0176800679524815+0j) [X4 Y5 Y10 X11] +
(3.694513294293786e-06+0j) [X4 Y5 Y10 Z11 Z12 X13] +
(-3.6945132942937863e-06+0j) [X4 Y5 Y11 X12] +
(0.0383146702948039+0j) [X4 Y5 Y12 X13] +
(-1.2260484989373826e-05+0j) [X4 Z5 X6] +
(-1.2283337826529685e-06+0j) [X4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(0.00024636437569580476+0j) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2283337826529685e-06+0j) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(0.00024636437569580476+0j) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.8540608580424864e-06+0j) [X4 Z5 X6 Z7] +
(-1.398044908217392e-06+0j) [X4 Z5 X6 Z8] +
(-1.881850183335261e-06+0j) [X4 Z5 X6 Z9] +
(0.007960880725921564+0j) [X4 Z5 X6 X10 Z11 X12] +
(-0.0009298507967730385+0j) [X4 Z5 X6 Y10 Z11 Y12] +
(-1.692397828546632e-06+0j) [X4 Z5 X6 Z10] +
(-0.012215040997613932+0j) [X4 Z5 X6 X11 Z12 X13] +
(-0.012215040997613932+0j) [X4 Z5 X6 Y11 Z12 Y13] +
(4.281913884933537e-06+0j) [X4 Z5 X6 Z11] +
(-4.588855155699114e-06+0j) [X4 Z5 X6 Z13] +
(0.008890731522694602+0j) [X4 Z5 Y6 Y10 Z11 X12] +
(-4.838052751178695e-07+0j) [X4 Z5 Z6 X7 Y8 Y9] +
(5.974311713480168e-06+0j) [X4 Z5 Z6 X7 Y10 Y11] +
(0.011285190200840893+0j) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(0.02017592172353549+0j) [X4 Z5 Z6 X7 X11 X12] +
(-4.556569218056487e-06+0j) [X4 Z5 Z6 X7 Y12 Y13] +
(4.838052751178695e-07+0j) [X4 Z5 Z6 Y7 Y8 X9] +
(-5.974311713480168e-06+0j) [X4 Z5 Z6 Y7 Y10 X11] +
(-0.011285190200840893+0j) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(0.02017592172353549+0j) [X4 Z5 Z6 Y7 Y11 X12] +
(4.556569218056487e-06+0j) [X4 Z5 Z6 Y7 Y12 X13] +
(1.3304731886665585e-06+0j) [X4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(0.005923798336561347+0j) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(1.3304731886665585e-06+0j) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(0.005923798336561347+0j) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(-6.631277928307006e-05+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.016024603689179517+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(-0.016024603689179517+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(3.3343312895647544e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(-4.734622038622324e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(-9.806102775212192e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(-5.071480736589868e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(5.071480736589868e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-0.36937089366156145+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.023145130929529065+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-0.009612634606847349+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(-0.025637238296026866+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(-8.774817864461432e-06+0j) [X4 Z5 Z6 Z7 Z8 X10] +
(-0.047642612176383124+0j) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(-7.444344675794872e-06+0j) [X4 Z5 Z6 Z7 Z9 X10] +
(-0.04171881383982178+0j) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(-6.290028432972391e-06+0j) [X4 Z5 Z6 Z8 Z9 X10] +
(-0.03956441632289339+0j) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(-7.518362215625361e-06+0j) [X4 Z5 Z7 Z8 Z9 X10] +
(-0.039318051947197584+0j) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(-5.929765816771467e-07+0j) [X4 X6] +
(-4.253224225666579e-06+0j) [X4 Z6 Z7 Z8 Z9 X10] +
(-0.022528440196012994+0j) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.011982389010247958+0j) [Y4 X5 X6 Y7] +
(0.007306759928832963+0j) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11] +
(2.8882935959112267e-07+0j) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.007306759928832963+0j) [Y4 X5 X7 Z8 Z9 Y10] +
(-2.8882935959112267e-07+0j) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(0.007156934919856944+0j) [Y4 X5 X8 Y9] +
(0.0176800679524815+0j) [Y4 X5 X10 Y11] +
(3.694513294293786e-06+0j) [Y4 X5 X10 Z11 Z12 Y13] +
(-3.6945132942937863e-06+0j) [Y4 X5 X11 Y12] +
(0.0383146702948039+0j) [Y4 X5 X12 Y13] +
(-0.011982389010247958+0j) [Y4 Y5 X6 X7] +
(-0.007306759928832963+0j) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11] +
(-2.8882935959112267e-07+0j) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.007306759928832963+0j) [Y4 Y5 Y7 Z8 Z9 Y10] +
(-2.8882935959112267e-07+0j) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.007156934919856944+0j) [Y4 Y5 X8 X9] +
(-0.0176800679524815+0j) [Y4 Y5 X10 X11] +
(-3.694513294293786e-06+0j) [Y4 Y5 X10 Z11 Z12 X13] +
(-3.6945132942937863e-06+0j) [Y4 Y5 Y11 Y12] +
(-0.0383146702948039+0j) [Y4 Y5 X12 X13] +
(0.008890731522694602+0j) [Y4 Z5 X6 X10 Z11 Y12] +
(-1.2260484989373826e-05+0j) [Y4 Z5 Y6] +
(-1.2283337826529685e-06+0j) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(0.00024636437569580476+0j) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2283337826529685e-06+0j) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(0.00024636437569580476+0j) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.8540608580424864e-06+0j) [Y4 Z5 Y6 Z7] +
(-1.398044908217392e-06+0j) [Y4 Z5 Y6 Z8] +
(-1.881850183335261e-06+0j) [Y4 Z5 Y6 Z9] +
(-0.0009298507967730385+0j) [Y4 Z5 Y6 X10 Z11 X12] +
(0.007960880725921564+0j) [Y4 Z5 Y6 Y10 Z11 Y12] +
(-1.692397828546632e-06+0j) [Y4 Z5 Y6 Z10] +
(-0.012215040997613932+0j) [Y4 Z5 Y6 X11 Z12 X13] +
(-0.012215040997613932+0j) [Y4 Z5 Y6 Y11 Z12 Y13] +
(4.281913884933537e-06+0j) [Y4 Z5 Y6 Z11] +
(-4.588855155699114e-06+0j) [Y4 Z5 Y6 Z13] +
(4.838052751178695e-07+0j) [Y4 Z5 Z6 X7 X8 Y9] +
(-5.974311713480168e-06+0j) [Y4 Z5 Z6 X7 X10 Y11] +
(-0.011285190200840893+0j) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(0.02017592172353549+0j) [Y4 Z5 Z6 X7 X11 Y12] +
(4.556569218056487e-06+0j) [Y4 Z5 Z6 X7 X12 Y13] +
(-4.838052751178695e-07+0j) [Y4 Z5 Z6 Y7 X8 X9] +
(5.974311713480168e-06+0j) [Y4 Z5 Z6 Y7 X10 X11] +
(0.011285190200840893+0j) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(0.02017592172353549+0j) [Y4 Z5 Z6 Y7 Y11 Y12] +
(-4.556569218056487e-06+0j) [Y4 Z5 Z6 Y7 X12 X13] +
(1.3304731886665585e-06+0j) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(0.005923798336561347+0j) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(1.3304731886665585e-06+0j) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(0.005923798336561347+0j) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(-6.631277928307006e-05+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.016024603689179517+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(-0.016024603689179517+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(3.3343312895647544e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(-4.734622038622324e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(-9.806102775212192e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(5.071480736589868e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-5.071480736589868e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-0.36937089366156145+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.023145130929529065+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-0.009612634606847349+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(-0.025637238296026866+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(-8.774817864461432e-06+0j) [Y4 Z5 Z6 Z7 Z8 Y10] +
(-0.047642612176383124+0j) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(-7.444344675794872e-06+0j) [Y4 Z5 Z6 Z7 Z9 Y10] +
(-0.04171881383982178+0j) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(-6.290028432972391e-06+0j) [Y4 Z5 Z6 Z8 Z9 Y10] +
(-0.03956441632289339+0j) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(-7.518362215625361e-06+0j) [Y4 Z5 Z7 Z8 Z9 Y10] +
(-0.039318051947197584+0j) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(-5.929765816771467e-07+0j) [Y4 Y6] +
(-4.253224225666579e-06+0j) [Y4 Z6 Z7 Z8 Z9 Y10] +
(-0.022528440196012994+0j) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.2034402289145636+0j) [Z4] +
(-5.929765816771467e-07+0j) [Z4 X5 Z6 X7] +
(-4.253224225666579e-06+0j) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.022528440196012994+0j) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-5.929765816771467e-07+0j) [Z4 Y5 Z6 Y7] +
(-4.253224225666579e-06+0j) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.022528440196012994+0j) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.15755314797985673+0j) [Z4 Z5] +
(0.01826683486937555+0j) [Z4 X6 Z7 Z8 Z9 X10] +
(-1.6541174768949816e-06+0j) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.01826683486937555+0j) [Z4 Y6 Z7 Z8 Z9 Y10] +
(-1.6541174768949816e-06+0j) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.13701191674040758+0j) [Z4 Z6] +
(0.010960074940542588+0j) [Z4 X7 Z8 Z9 Z10 X11] +
(-1.9429468364861043e-06+0j) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.010960074940542588+0j) [Z4 Y7 Z8 Z9 Z10 Y11] +
(-1.9429468364861043e-06+0j) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.14899430575065553+0j) [Z4 Z7] +
(0.14960702684445298+0j) [Z4 Z8] +
(0.15676396176430993+0j) [Z4 Z9] +
(1.8782101247701075e-06+0j) [Z4 X10 Z11 X12] +
(1.8782101247701075e-06+0j) [Z4 Y10 Z11 Y12] +
(0.12489990917237606+0j) [Z4 Z10] +
(-1.816303169523678e-06+0j) [Z4 X11 Z12 X13] +
(-1.816303169523678e-06+0j) [Z4 Y11 Z12 Y13] +
(0.14257997712485757+0j) [Z4 Z11] +
(0.11383573679388664+0j) [Z4 Z12] +
(0.15215040708869054+0j) [Z4 Z13] +
(1.2283337826529685e-06+0j) [X5 X6 Y7 Z8 Z9 Y10] +
(-0.0002463643756958048+0j) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-4.838052751178695e-07+0j) [X5 X6 X8 X9] +
(5.974311713480168e-06+0j) [X5 X6 X10 X11] +
(0.02017592172353549+0j) [X5 X6 X10 Z11 Z12 X13] +
(0.011285190200840893+0j) [X5 X6 Y11 Y12] +
(-4.5565692180564866e-06+0j) [X5 X6 X12 X13] +
(-1.2283337826529685e-06+0j) [X5 Y6 Y7 Z8 Z9 X10] +
(0.0002463643756958048+0j) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-4.838052751178695e-07+0j) [X5 Y6 Y8 X9] +
(5.974311713480168e-06+0j) [X5 Y6 Y10 X11] +
(0.02017592172353549+0j) [X5 Y6 Y10 Z11 Z12 X13] +
(-0.011285190200840893+0j) [X5 Y6 Y11 X12] +
(-4.5565692180564866e-06+0j) [X5 Y6 Y12 X13] +
(-1.2260484989373826e-05+0j) [X5 Z6 X7] +
(-1.881850183335261e-06+0j) [X5 Z6 X7 Z8] +
(-1.398044908217392e-06+0j) [X5 Z6 X7 Z9] +
(-0.012215040997613932+0j) [X5 Z6 X7 X10 Z11 X12] +
(-0.012215040997613932+0j) [X5 Z6 X7 Y10 Z11 Y12] +
(4.281913884933537e-06+0j) [X5 Z6 X7 Z10] +
(0.007960880725921564+0j) [X5 Z6 X7 X11 Z12 X13] +
(-0.0009298507967730385+0j) [X5 Z6 X7 Y11 Z12 Y13] +
(-1.692397828546632e-06+0j) [X5 Z6 X7 Z11] +
(-4.588855155699114e-06+0j) [X5 Z6 X7 Z12] +
(0.008890731522694602+0j) [X5 Z6 Y7 Y11 Z12 X13] +
(-1.3304731886665585e-06+0j) [X5 Z6 Z7 X8 Y9 Y10] +
(-0.005923798336561347+0j) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(1.3304731886665585e-06+0j) [X5 Z6 Z7 Y8 Y9 X10] +
(0.005923798336561347+0j) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(0.016024603689179517+0j) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12] +
(-5.071480736589869e-06+0j) [X5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-0.016024603689179517+0j) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12] +
(-5.071480736589869e-06+0j) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(-6.631277928307006e-05+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-9.806102775212192e-06+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(-4.734622038622324e-06+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(-0.36937089366156123+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.023145130929529065+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(-0.025637238296026866+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(3.334331289564755e-06+0j) [X5 Z6 Z7 Z8 Z9 X11] +
(-0.009612634606847349+0j) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(-7.444344675794872e-06+0j) [X5 Z6 Z7 Z8 Z10 X11] +
(-0.04171881383982178+0j) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(-8.774817864461432e-06+0j) [X5 Z6 Z7 Z9 Z10 X11] +
(-0.047642612176383124+0j) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(-7.518362215625361e-06+0j) [X5 Z6 Z8 Z9 Z10 X11] +
(-0.039318051947197584+0j) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.8540608580424864e-06+0j) [X5 X7] +
(-6.290028432972391e-06+0j) [X5 Z7 Z8 Z9 Z10 X11] +
(-0.03956441632289339+0j) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2283337826529685e-06+0j) [Y5 X6 X7 Z8 Z9 Y10] +
(0.0002463643756958048+0j) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-4.838052751178695e-07+0j) [Y5 X6 X8 Y9] +
(5.974311713480168e-06+0j) [Y5 X6 X10 Y11] +
(0.02017592172353549+0j) [Y5 X6 X10 Z11 Z12 Y13] +
(-0.011285190200840893+0j) [Y5 X6 X11 Y12] +
(-4.5565692180564866e-06+0j) [Y5 X6 X12 Y13] +
(1.2283337826529685e-06+0j) [Y5 Y6 X7 Z8 Z9 X10] +
(-0.0002463643756958048+0j) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-4.838052751178695e-07+0j) [Y5 Y6 Y8 Y9] +
(5.974311713480168e-06+0j) [Y5 Y6 Y10 Y11] +
(0.02017592172353549+0j) [Y5 Y6 Y10 Z11 Z12 Y13] +
(0.011285190200840893+0j) [Y5 Y6 X11 X12] +
(-4.5565692180564866e-06+0j) [Y5 Y6 Y12 Y13] +
(0.008890731522694602+0j) [Y5 Z6 X7 X11 Z12 Y13] +
(-1.2260484989373826e-05+0j) [Y5 Z6 Y7] +
(-1.881850183335261e-06+0j) [Y5 Z6 Y7 Z8] +
(-1.398044908217392e-06+0j) [Y5 Z6 Y7 Z9] +
(-0.012215040997613932+0j) [Y5 Z6 Y7 X10 Z11 X12] +
(-0.012215040997613932+0j) [Y5 Z6 Y7 Y10 Z11 Y12] +
(4.281913884933537e-06+0j) [Y5 Z6 Y7 Z10] +
(-0.0009298507967730385+0j) [Y5 Z6 Y7 X11 Z12 X13] +
(0.007960880725921564+0j) [Y5 Z6 Y7 Y11 Z12 Y13] +
(-1.692397828546632e-06+0j) [Y5 Z6 Y7 Z11] +
(-4.588855155699114e-06+0j) [Y5 Z6 Y7 Z12] +
(1.3304731886665585e-06+0j) [Y5 Z6 Z7 X8 X9 Y10] +
(0.005923798336561347+0j) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-1.3304731886665585e-06+0j) [Y5 Z6 Z7 Y8 X9 X10] +
(-0.005923798336561347+0j) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-0.016024603689179517+0j) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12] +
(-5.071480736589869e-06+0j) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(0.016024603689179517+0j) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12] +
(-5.071480736589869e-06+0j) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(-6.631277928307006e-05+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-9.806102775212192e-06+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(-4.734622038622324e-06+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(-0.36937089366156123+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.023145130929529065+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(-0.025637238296026866+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(3.334331289564755e-06+0j) [Y5 Z6 Z7 Z8 Z9 Y11] +
(-0.009612634606847349+0j) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(-7.444344675794872e-06+0j) [Y5 Z6 Z7 Z8 Z10 Y11] +
(-0.04171881383982178+0j) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(-8.774817864461432e-06+0j) [Y5 Z6 Z7 Z9 Z10 Y11] +
(-0.047642612176383124+0j) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(-7.518362215625361e-06+0j) [Y5 Z6 Z8 Z9 Z10 Y11] +
(-0.039318051947197584+0j) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.8540608580424864e-06+0j) [Y5 Y7] +
(-6.290028432972391e-06+0j) [Y5 Z7 Z8 Z9 Z10 Y11] +
(-0.03956441632289339+0j) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.2034402289145634+0j) [Z5] +
(0.010960074940542588+0j) [Z5 X6 Z7 Z8 Z9 X10] +
(-1.9429468364861043e-06+0j) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.010960074940542588+0j) [Z5 Y6 Z7 Z8 Z9 Y10] +
(-1.9429468364861043e-06+0j) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.14899430575065553+0j) [Z5 Z6] +
(0.01826683486937555+0j) [Z5 X7 Z8 Z9 Z10 X11] +
(-1.6541174768949816e-06+0j) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.01826683486937555+0j) [Z5 Y7 Z8 Z9 Z10 Y11] +
(-1.6541174768949816e-06+0j) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.13701191674040758+0j) [Z5 Z7] +
(0.15676396176430993+0j) [Z5 Z8] +
(0.14960702684445298+0j) [Z5 Z9] +
(-1.816303169523678e-06+0j) [Z5 X10 Z11 X12] +
(-1.816303169523678e-06+0j) [Z5 Y10 Z11 Y12] +
(0.14257997712485757+0j) [Z5 Z10] +
(1.8782101247701075e-06+0j) [Z5 X11 Z12 X13] +
(1.8782101247701075e-06+0j) [Z5 Y11 Z12 Y13] +
(0.12489990917237606+0j) [Z5 Z11] +
(0.15215040708869054+0j) [Z5 Z12] +
(0.11383573679388664+0j) [Z5 Z13] +
(-0.01387338174842612+0j) [X6 X7 Y8 Y9] +
(-0.017825140995786446+0j) [X6 X7 Y10 Y11] +
(-1.0358477603561531e-06+0j) [X6 X7 Y10 Z11 Z12 Y13] +
(-1.0358477603561531e-06+0j) [X6 X7 X11 X12] +
(-0.017366118994651413+0j) [X6 X7 Y12 Y13] +
(0.01387338174842612+0j) [X6 Y7 Y8 X9] +
(0.017825140995786446+0j) [X6 Y7 Y10 X11] +
(1.0358477603561531e-06+0j) [X6 Y7 Y10 Z11 Z12 X13] +
(-1.0358477603561531e-06+0j) [X6 Y7 Y11 X12] +
(0.017366118994651413+0j) [X6 Y7 Y12 X13] +
(0.00029219862611107544+0j) [X6 Z7 X8 X9 Z10 X11] +
(-3.3281393510549505e-07+0j) [X6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.00029219862611107544+0j) [X6 Z7 X8 Y9 Z10 Y11] +
(-3.3281393510549505e-07+0j) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(0.22848106564918844+0j) [X6 Z7 Z8 Z9 X10] +
(3.3131455003067762e-06+0j) [X6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(3.3131455003067762e-06+0j) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(0.011307274008848225+0j) [X6 Z7 Z8 Z9 X10 Z11] +
(0.025104957138844555+0j) [X6 Z7 Z8 Z9 X10 Z12] +
(0.010540425907671548+0j) [X6 Z7 Z8 Z9 X10 Z13] +
(-0.014564531231173006+0j) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(0.014564531231173006+0j) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-2.595086006755689e-05+0j) [X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.1839325596024114e-06+0j) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-6.524373848607915e-06+0j) [X6 Z7 Z8 Z9 Z10 X12] +
(-3.211228348301138e-06+0j) [X6 Z7 Z8 Z9 Z11 X12] +
(0.02981242451734576+0j) [X6 Z7 Z8 X10] +
(-3.27748319526306e-06+0j) [X6 Z7 Z8 Z10 Z11 X12] +
(0.030104623143456834+0j) [X6 Z7 Z9 X10] +
(-3.6102971303685546e-06+0j) [X6 Z7 Z9 Z10 Z11 X12] +
(0.03078750538914394+0j) [X6 Z8 Z9 X10] +
(-3.769659451782224e-06+0j) [X6 Z8 Z9 Z10 Z11 X12] +
(0.01387338174842612+0j) [Y6 X7 X8 Y9] +
(0.017825140995786446+0j) [Y6 X7 X10 Y11] +
(1.0358477603561531e-06+0j) [Y6 X7 X10 Z11 Z12 Y13] +
(-1.0358477603561531e-06+0j) [Y6 X7 X11 Y12] +
(0.017366118994651413+0j) [Y6 X7 X12 Y13] +
(-0.01387338174842612+0j) [Y6 Y7 X8 X9] +
(-0.017825140995786446+0j) [Y6 Y7 X10 X11] +
(-1.0358477603561531e-06+0j) [Y6 Y7 X10 Z11 Z12 X13] +
(-1.0358477603561531e-06+0j) [Y6 Y7 Y11 Y12] +
(-0.017366118994651413+0j) [Y6 Y7 X12 X13] +
(0.00029219862611107544+0j) [Y6 Z7 Y8 X9 Z10 X11] +
(-3.3281393510549505e-07+0j) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.00029219862611107544+0j) [Y6 Z7 Y8 Y9 Z10 Y11] +
(-3.3281393510549505e-07+0j) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(0.22848106564918844+0j) [Y6 Z7 Z8 Z9 Y10] +
(3.3131455003067762e-06+0j) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(3.3131455003067762e-06+0j) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(0.011307274008848225+0j) [Y6 Z7 Z8 Z9 Y10 Z11] +
(0.025104957138844555+0j) [Y6 Z7 Z8 Z9 Y10 Z12] +
(0.010540425907671548+0j) [Y6 Z7 Z8 Z9 Y10 Z13] +
(0.014564531231173006+0j) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-0.014564531231173006+0j) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-2.595086006755689e-05+0j) [Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.1839325596024114e-06+0j) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-6.524373848607915e-06+0j) [Y6 Z7 Z8 Z9 Z10 Y12] +
(-3.211228348301138e-06+0j) [Y6 Z7 Z8 Z9 Z11 Y12] +
(0.02981242451734576+0j) [Y6 Z7 Z8 Y10] +
(-3.27748319526306e-06+0j) [Y6 Z7 Z8 Z10 Z11 Y12] +
(0.030104623143456834+0j) [Y6 Z7 Z9 Y10] +
(-3.6102971303685546e-06+0j) [Y6 Z7 Z9 Z10 Z11 Y12] +
(0.03078750538914394+0j) [Y6 Z8 Z9 Y10] +
(-3.769659451782224e-06+0j) [Y6 Z8 Z9 Z10 Z11 Y12] +
(1.309686298861542+0j) [Z6] +
(0.03078750538914394+0j) [Z6 X7 Z8 Z9 Z10 X11] +
(-3.769659451782224e-06+0j) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.03078750538914394+0j) [Z6 Y7 Z8 Z9 Z10 Y11] +
(-3.769659451782224e-06+0j) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.19392534613270213+0j) [Z6 Z7] +
(0.1675665326546127+0j) [Z6 Z8] +
(0.1814399144030388+0j) [Z6 Z9] +
(-1.855120121504858e-06+0j) [Z6 X10 Z11 X12] +
(-1.855120121504858e-06+0j) [Z6 Y10 Z11 Y12] +
(0.11952438964682678+0j) [Z6 Z10] +
(-2.8909678818610113e-06+0j) [Z6 X11 Z12 X13] +
(-2.8909678818610113e-06+0j) [Z6 Y11 Z12 Y13] +
(0.13734953064261324+0j) [Z6 Z11] +
(0.13401715261963712+0j) [Z6 Z12] +
(0.1513832716142885+0j) [Z6 Z13] +
(-0.00029219862611107544+0j) [X7 X8 Y9 Y10] +
(3.3281393510549505e-07+0j) [X7 X8 Y9 Z10 Z11 Y12] +
(0.00029219862611107544+0j) [X7 Y8 Y9 X10] +
(-3.3281393510549505e-07+0j) [X7 Y8 Y9 Z10 Z11 X12] +
(-3.3131455003067767e-06+0j) [X7 Z8 Z9 X10 Y11 Y12] +
(-0.014564531231173006+0j) [X7 Z8 Z9 X10 X12 X13] +
(3.3131455003067767e-06+0j) [X7 Z8 Z9 Y10 Y11 X12] +
(-0.014564531231173006+0j) [X7 Z8 Z9 Y10 Y12 X13] +
(0.22848106564918844+0j) [X7 Z8 Z9 Z10 X11] +
(0.010540425907671548+0j) [X7 Z8 Z9 Z10 X11 Z12] +
(0.025104957138844555+0j) [X7 Z8 Z9 Z10 X11 Z13] +
(-2.5950860067556905e-05+0j) [X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.1839325596024114e-06+0j) [X7 Z8 Z9 Z10 Z11 X13] +
(-3.211228348301138e-06+0j) [X7 Z8 Z9 Z10 Z12 X13] +
(0.011307274008848227+0j) [X7 Z8 Z9 X11] +
(-6.524373848607915e-06+0j) [X7 Z8 Z9 Z11 Z12 X13] +
(0.030104623143456834+0j) [X7 Z8 Z10 X11] +
(-3.6102971303685546e-06+0j) [X7 Z8 Z10 Z11 Z12 X13] +
(0.02981242451734576+0j) [X7 Z9 Z10 X11] +
(-3.27748319526306e-06+0j) [X7 Z9 Z10 Z11 Z12 X13] +
(0.00029219862611107544+0j) [Y7 X8 X9 Y10] +
(-3.3281393510549505e-07+0j) [Y7 X8 X9 Z10 Z11 Y12] +
(-0.00029219862611107544+0j) [Y7 Y8 X9 X10] +
(3.3281393510549505e-07+0j) [Y7 Y8 X9 Z10 Z11 X12] +
(3.3131455003067767e-06+0j) [Y7 Z8 Z9 X10 X11 Y12] +
(-0.014564531231173006+0j) [Y7 Z8 Z9 X10 X12 Y13] +
(-3.3131455003067767e-06+0j) [Y7 Z8 Z9 Y10 X11 X12] +
(-0.014564531231173006+0j) [Y7 Z8 Z9 Y10 Y12 Y13] +
(0.22848106564918844+0j) [Y7 Z8 Z9 Z10 Y11] +
(0.010540425907671548+0j) [Y7 Z8 Z9 Z10 Y11 Z12] +
(0.025104957138844555+0j) [Y7 Z8 Z9 Z10 Y11 Z13] +
(-2.5950860067556905e-05+0j) [Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(4.1839325596024114e-06+0j) [Y7 Z8 Z9 Z10 Z11 Y13] +
(-3.211228348301138e-06+0j) [Y7 Z8 Z9 Z10 Z12 Y13] +
(0.011307274008848227+0j) [Y7 Z8 Z9 Y11] +
(-6.524373848607915e-06+0j) [Y7 Z8 Z9 Z11 Z12 Y13] +
(0.030104623143456834+0j) [Y7 Z8 Z10 Y11] +
(-3.6102971303685546e-06+0j) [Y7 Z8 Z10 Z11 Z12 Y13] +
(0.02981242451734576+0j) [Y7 Z9 Z10 Y11] +
(-3.27748319526306e-06+0j) [Y7 Z9 Z10 Z11 Z12 Y13] +
(1.3096862988615419+0j) [Z7] +
(0.1814399144030388+0j) [Z7 Z8] +
(0.1675665326546127+0j) [Z7 Z9] +
(-2.8909678818610113e-06+0j) [Z7 X10 Z11 X12] +
(-2.8909678818610113e-06+0j) [Z7 Y10 Z11 Y12] +
(0.13734953064261324+0j) [Z7 Z10] +
(-1.855120121504858e-06+0j) [Z7 X11 Z12 X13] +
(-1.855120121504858e-06+0j) [Z7 Y11 Z12 Y13] +
(0.11952438964682678+0j) [Z7 Z11] +
(0.1513832716142885+0j) [Z7 Z12] +
(0.13401715261963712+0j) [Z7 Z13] +
(-0.009560705729135964+0j) [X8 X9 Y10 Y11] +
(6.62861420181944e-07+0j) [X8 X9 Y10 Z11 Z12 Y13] +
(6.62861420181944e-07+0j) [X8 X9 X11 X12] +
(-0.006087822480561861+0j) [X8 X9 Y12 Y13] +
(0.009560705729135964+0j) [X8 Y9 Y10 X11] +
(-6.62861420181944e-07+0j) [X8 Y9 Y10 Z11 Z12 X13] +
(6.62861420181944e-07+0j) [X8 Y9 Y11 X12] +
(0.006087822480561861+0j) [X8 Y9 Y12 X13] +
(0.009560705729135964+0j) [Y8 X9 X10 Y11] +
(-6.62861420181944e-07+0j) [Y8 X9 X10 Z11 Z12 Y13] +
(6.62861420181944e-07+0j) [Y8 X9 X11 Y12] +
(0.006087822480561861+0j) [Y8 X9 X12 Y13] +
(-0.009560705729135964+0j) [Y8 Y9 X10 X11] +
(6.62861420181944e-07+0j) [Y8 Y9 X10 Z11 Z12 X13] +
(6.62861420181944e-07+0j) [Y8 Y9 Y11 Y12] +
(-0.006087822480561861+0j) [Y8 Y9 X12 X13] +
(1.369352563471817+0j) [Z8] +
(0.2200397733437609+0j) [Z8 Z9] +
(-1.597317197818437e-06+0j) [Z8 X10 Z11 X12] +
(-1.597317197818437e-06+0j) [Z8 Y10 Z11 Y12] +
(0.13766872645852576+0j) [Z8 Z10] +
(-9.344557776364932e-07+0j) [Z8 X11 Z12 X13] +
(-9.344557776364932e-07+0j) [Z8 Y11 Z12 Y13] +
(0.14722943218766174+0j) [Z8 Z11] +
(0.14973486803496933+0j) [Z8 Z12] +
(0.1558226905155312+0j) [Z8 Z13] +
(1.3693525634718176+0j) [Z9] +
(-9.344557776364932e-07+0j) [Z9 X10 Z11 X12] +
(-9.344557776364932e-07+0j) [Z9 Y10 Z11 Y12] +
(0.14722943218766174+0j) [Z9 Z10] +
(-1.597317197818437e-06+0j) [Z9 X11 Z12 X13] +
(-1.597317197818437e-06+0j) [Z9 Y11 Z12 Y13] +
(0.13766872645852576+0j) [Z9 Z11] +
(0.1558226905155312+0j) [Z9 Z12] +
(0.14973486803496933+0j) [Z9 Z13] +
(-0.028685183716105903+0j) [X10 X11 Y12 Y13] +
(0.028685183716105903+0j) [X10 Y11 Y12 X13] +
(-1.0722312157797347e-05+0j) [X10 Z11 X12] +
(7.954413176383307e-06+0j) [X10 Z11 X12 Z13] +
(-8.194261372294933e-06+0j) [X10 X12] +
(0.028685183716105903+0j) [Y10 X11 X12 Y13] +
(-0.028685183716105903+0j) [Y10 Y11 X12 X13] +
(-1.0722312157797347e-05+0j) [Y10 Z11 Y12] +
(7.954413176383307e-06+0j) [Y10 Z11 Y12 Z13] +
(-8.194261372294933e-06+0j) [Y10 Y12] +
(0.7829661725950194+0j) [Z10] +
(-8.194261372294933e-06+0j) [Z10 X11 Z12 X13] +
(-8.194261372294933e-06+0j) [Z10 Y11 Z12 Y13] +
(0.14926355147388895+0j) [Z10 Z11] +
(0.1127038692033222+0j) [Z10 Z12] +
(0.1413890529194281+0j) [Z10 Z13] +
(-1.0722312157797347e-05+0j) [X11 Z12 X13] +
(7.954413176383307e-06+0j) [X11 X13] +
(-1.0722312157797347e-05+0j) [Y11 Z12 Y13] +
(7.954413176383307e-06+0j) [Y11 Y13] +
(0.7829661725950194+0j) [Z11] +
(0.1413890529194281+0j) [Z11 Z12] +
(0.1127038692033222+0j) [Z11 Z13] +
(0.8084581961720483+0j) [Z12] +
(0.1543574865722364+0j) [Z12 Z13] +
(0.8084581961720484+0j) [Z13]
Number of qubits: 14
Qubit Hamiltonian
  (-46.46390678868898) [I0]
+ (0.78296617259502) [Z10]
+ (0.7829661725950201) [Z11]
+ (0.8084581961720478) [Z12]
+ (0.808458196172048) [Z13]
+ (1.2034402289145625) [Z4]
+ (1.2034402289145625) [Z5]
+ (1.3096862988615423) [Z6]
+ (1.3096862988615423) [Z7]
+ (1.3693525634718184) [Z8]
+ (1.3693525634718184) [Z9]
+ (1.653894222683171) [Z2]
+ (1.6538942226831714) [Z3]
+ (12.412630742111773) [Z0]
+ (12.412630742111773) [Z1]
+ (-8.194261372459324e-06) [Y10 Y12]
+ (-8.194261372459324e-06) [X10 X12]
+ (-1.8540608581531163e-06) [Y5 Y7]
+ (-1.8540608581531163e-06) [X5 X7]
+ (-7.764994119009936e-07) [Y3 Y5]
+ (-7.764994119009936e-07) [X3 X5]
+ (-5.929765816853344e-07) [Y4 Y6]
+ (-5.929765816853344e-07) [X4 X6]
+ (1.6021167407632491e-06) [Y2 Y4]
+ (1.6021167407632491e-06) [X2 X4]
+ (7.954413176564841e-06) [Y11 Y13]
+ (7.954413176564841e-06) [X11 X13]
+ (0.0032769719312316444) [Y1 Y3]
+ (0.0032769719312316444) [X1 X3]
+ (0.10433064780651399) [Y0 Y2]
+ (0.10433064780651399) [X0 X2]
+ (0.11270386920332218) [Z10 Z12]
+ (0.11270386920332218) [Z11 Z13]
+ (0.11383573679388657) [Z4 Z12]
+ (0.11383573679388657) [Z5 Z13]
+ (0.1195243896468267) [Z6 Z10]
+ (0.1195243896468267) [Z7 Z11]
+ (0.12489990917237605) [Z4 Z10]
+ (0.12489990917237605) [Z5 Z11]
+ (0.12495807739503208) [Z2 Z4]
+ (0.12495807739503208) [Z3 Z5]
+ (0.12799502492468412) [Z2 Z10]
+ (0.12799502492468412) [Z3 Z11]
+ (0.13401715261963693) [Z6 Z12]
+ (0.13401715261963693) [Z7 Z13]
+ (0.13701191674040736) [Z4 Z6]
+ (0.13701191674040736) [Z5 Z7]
+ (0.13734953064261318) [Z6 Z11]
+ (0.13734953064261318) [Z7 Z10]
+ (0.1373910476268321) [Z2 Z6]
+ (0.1373910476268321) [Z3 Z7]
+ (0.1376687264585258) [Z8 Z10]
+ (0.1376687264585258) [Z9 Z11]
+ (0.14011289865354806) [Z2 Z12]
+ (0.14011289865354806) [Z3 Z13]
+ (0.1413890529194281) [Z10 Z13]
+ (0.1413890529194281) [Z11 Z12]
+ (0.14257997712485757) [Z4 Z11]
+ (0.14257997712485757) [Z5 Z10]
+ (0.1472294321876617) [Z8 Z11]
+ (0.1472294321876617) [Z9 Z10]
+ (0.1489943057506553) [Z4 Z7]
+ (0.1489943057506553) [Z5 Z6]
+ (0.14926355147388903) [Z10 Z11]
+ (0.1496070268444529) [Z4 Z8]
+ (0.1496070268444529) [Z5 Z9]
+ (0.14973486803496922) [Z8 Z12]
+ (0.14973486803496922) [Z9 Z13]
+ (0.15071408121008278) [Z2 Z8]
+ (0.15071408121008278) [Z3 Z9]
+ (0.15138327161428833) [Z6 Z13]
+ (0.15138327161428833) [Z7 Z12]
+ (0.15215040708869043) [Z4 Z13]
+ (0.15215040708869043) [Z5 Z12]
+ (0.15337968243314148) [Z2 Z11]
+ (0.15337968243314148) [Z3 Z10]
+ (0.1543574865722363) [Z12 Z13]
+ (0.1556901067175245) [Z2 Z13]
+ (0.1556901067175245) [Z3 Z12]
+ (0.15582269051553108) [Z8 Z13]
+ (0.15582269051553108) [Z9 Z12]
+ (0.15676396176430984) [Z4 Z9]
+ (0.15676396176430984) [Z5 Z8]
+ (0.1575531479798566) [Z4 Z5]
+ (0.16079764534838553) [Z2 Z5]
+ (0.16079764534838553) [Z3 Z4]
+ (0.1675665326546125) [Z6 Z8]
+ (0.1675665326546125) [Z7 Z9]
+ (0.16853486561579917) [Z2 Z7]
+ (0.16853486561579917) [Z3 Z6]
+ (0.18143991440303858) [Z6 Z9]
+ (0.18143991440303858) [Z7 Z8]
+ (0.1818908579075135) [Z2 Z3]
+ (0.1869082047691254) [Z2 Z9]
+ (0.1869082047691254) [Z3 Z8]
+ (0.1929972393536426) [Z0 Z10]
+ (0.1929972393536426) [Z1 Z11]
+ (0.19392534613270168) [Z6 Z7]
+ (0.19661770890342145) [Z0 Z4]
+ (0.19661770890342145) [Z1 Z5]
+ (0.19936354537360823) [Z0 Z5]
+ (0.19936354537360823) [Z1 Z4]
+ (0.20072866460441785) [Z0 Z11]
+ (0.20072866460441785) [Z1 Z10]
+ (0.21102659849791527) [Z0 Z12]
+ (0.21102659849791527) [Z1 Z13]
+ (0.21631037498631822) [Z0 Z13]
+ (0.21631037498631822) [Z1 Z12]
+ (0.2200397733437608) [Z8 Z9]
+ (0.23671080783830423) [Z0 Z2]
+ (0.23671080783830423) [Z1 Z3]
+ (0.24164663936017192) [Z0 Z6]
+ (0.24164663936017192) [Z1 Z7]
+ (0.2485348337131425) [Z0 Z7]
+ (0.2485348337131425) [Z1 Z6]
+ (0.2512944567459169) [Z0 Z3]
+ (0.2512944567459169) [Z1 Z2]
+ (0.2723251830660569) [Z0 Z8]
+ (0.2723251830660569) [Z1 Z9]
+ (0.2788345442672341) [Z0 Z9]
+ (0.2788345442672341) [Z1 Z8]
+ (1.1861763734860509) [Z0 Z1]
+ (-1.2260484990066865e-05) [Y4 Z5 Y6]
+ (-1.2260484990066865e-05) [X4 Z5 X6]
+ (-1.226048499006686e-05) [Y5 Z6 Y7]
+ (-1.226048499006686e-05) [X5 Z6 X7]
+ (-1.0722312157569262e-05) [Y10 Z11 Y12]
+ (-1.0722312157569262e-05) [X10 Z11 X12]
+ (-1.0722312157569258e-05) [Y11 Z12 Y13]
+ (-1.0722312157569258e-05) [X11 Z12 X13]
+ (-3.887051674330942e-06) [Y3 Z4 Y5]
+ (-3.887051674330942e-06) [X3 Z4 X5]
+ (-3.88705167433094e-06) [Y2 Z3 Y4]
+ (-3.88705167433094e-06) [X2 Z3 X4]
+ (0.1250703257977185) [Y1 Z2 Y3]
+ (0.1250703257977185) [X1 Z2 X3]
+ (0.12507032579771857) [Y0 Z1 Y2]
+ (0.12507032579771857) [X0 Z1 X2]
+ (-0.03831467029480388) [Y4 Y5 X12 X13]
+ (-0.03831467029480388) [X4 X5 Y12 Y13]
+ (-0.03619412355904259) [Y2 Y3 X8 X9]
+ (-0.03619412355904259) [X2 X3 Y8 Y9]
+ (-0.03583956795335344) [Y2 Y3 X4 X5]
+ (-0.03583956795335344) [X2 X3 Y4 Y5]
+ (-0.031143817988967086) [Y2 Y3 X6 X7]
+ (-0.031143817988967086) [X2 X3 Y6 Y7]
+ (-0.028685183716105903) [Y10 Y11 X12 X13]
+ (-0.028685183716105903) [X10 X11 Y12 Y13]
+ (-0.025996177598021163) [Y3 Z4 Z5 Y7]
+ (-0.025996177598021163) [X3 Z4 Z5 X7]
+ (-0.025384657508457392) [Y2 Y3 X10 X11]
+ (-0.025384657508457392) [X2 X3 Y10 Y11]
+ (-0.01902824244384726) [Y3 Y4 X11 X12]
+ (-0.01902824244384726) [X3 X4 Y11 Y12]
+ (-0.017825140995786498) [Y6 Y7 X10 X11]
+ (-0.017825140995786498) [X6 X7 Y10 Y11]
+ (-0.017680067952481532) [Y4 Y5 X10 X11]
+ (-0.017680067952481532) [X4 X5 Y10 Y11]
+ (-0.017366118994651427) [Y6 Y7 X12 X13]
+ (-0.017366118994651427) [X6 X7 Y12 Y13]
+ (-0.015577208063976456) [Y2 Y3 X12 X13]
+ (-0.015577208063976456) [X2 X3 Y12 Y13]
+ (-0.014583648907612629) [Y0 Y1 X2 X3]
+ (-0.014583648907612629) [X0 X1 Y2 Y3]
+ (-0.0138733817484261) [Y6 Y7 X8 X9]
+ (-0.0138733817484261) [X6 X7 Y8 Y9]
+ (-0.011982389010247962) [Y4 Y5 X6 X7]
+ (-0.011982389010247962) [X4 X5 Y6 Y7]
+ (-0.011285190200840935) [Y5 X6 X11 Y12]
+ (-0.011285190200840935) [X5 Y6 Y11 X12]
+ (-0.009560705729135928) [Y8 Y9 X10 X11]
+ (-0.009560705729135928) [X8 X9 Y10 Y11]
+ (-0.008125251921381018) [Y1 X2 X8 Y9]
+ (-0.008125251921381018) [Y1 Y2 Y8 Y9]
+ (-0.008125251921381018) [X1 X2 X8 X9]
+ (-0.008125251921381018) [X1 Y2 Y8 X9]
+ (-0.00773142525077527) [Y0 Y1 X10 X11]
+ (-0.00773142525077527) [X0 X1 Y10 Y11]
+ (-0.0071569349198569365) [Y4 Y5 X8 X9]
+ (-0.0071569349198569365) [X4 X5 Y8 Y9]
+ (-0.00688819435297057) [Y0 Y1 X6 X7]
+ (-0.00688819435297057) [X0 X1 Y6 Y7]
+ (-0.006509361201177232) [Y0 Y1 X8 X9]
+ (-0.006509361201177232) [X0 X1 Y8 Y9]
+ (-0.00608782248056186) [Y8 Y9 X12 X13]
+ (-0.00608782248056186) [X8 X9 Y12 Y13]
+ (-0.00528377648840296) [Y0 Y1 X12 X13]
+ (-0.00528377648840296) [X0 X1 Y12 Y13]
+ (-0.005143391768825129) [Y3 X4 X5 Y6]
+ (-0.005143391768825129) [X3 Y4 Y5 X6]
+ (-0.004684903388155184) [Y1 X2 X6 Y7]
+ (-0.004684903388155184) [Y1 Y2 Y6 Y7]
+ (-0.004684903388155184) [X1 X2 X6 X7]
+ (-0.004684903388155184) [X1 Y2 Y6 X7]
+ (-0.004575007626639205) [Y1 X2 X12 Y13]
+ (-0.004575007626639205) [Y1 Y2 Y12 Y13]
+ (-0.004575007626639205) [X1 X2 X12 X13]
+ (-0.004575007626639205) [X1 Y2 Y12 X13]
+ (-0.004424855449441844) [Y1 X2 X4 Y5]
+ (-0.004424855449441844) [Y1 Y2 Y4 Y5]
+ (-0.004424855449441844) [X1 X2 X4 X5]
+ (-0.004424855449441844) [X1 Y2 Y4 X5]
+ (-0.0034795118903343434) [Y2 Z3 Z5 Y6]
+ (-0.0034795118903343434) [X2 Z3 Z5 X6]
+ (-0.0034795118903343434) [Y3 Z4 Z6 Y7]
+ (-0.0034795118903343434) [X3 Z4 Z6 X7]
+ (-0.0027458364701868046) [Y0 Y1 X4 X5]
+ (-0.0027458364701868046) [X0 X1 Y4 Y5]
+ (-0.0017992194936630335) [Y1 X2 X10 Y11]
+ (-0.0017992194936630335) [Y1 Y2 Y10 Y11]
+ (-0.0017992194936630335) [X1 X2 X10 X11]
+ (-0.0017992194936630335) [X1 Y2 Y10 X11]
+ (-0.0002921986261110528) [Y7 Y8 X9 X10]
+ (-0.0002921986261110528) [X7 X8 Y9 Y10]
+ (-8.194261372459324e-06) [Z10 Y11 Z12 Y13]
+ (-8.194261372459324e-06) [Z10 X11 Z12 X13]
+ (-7.801707500712484e-06) [Y2 Z3 Y4 Z11]
+ (-7.801707500712484e-06) [X2 Z3 X4 Z11]
+ (-7.801707500712484e-06) [Y3 Z4 Y5 Z10]
+ (-7.801707500712484e-06) [X3 Z4 X5 Z10]
+ (-4.643051068581625e-06) [Y3 X4 X10 Y11]
+ (-4.643051068581625e-06) [Y3 Y4 Y10 Y11]
+ (-4.643051068581625e-06) [X3 X4 X10 X11]
+ (-4.643051068581625e-06) [X3 Y4 Y10 X11]
+ (-4.588855155870813e-06) [Y4 Z5 Y6 Z13]
+ (-4.588855155870813e-06) [X4 Z5 X6 Z13]
+ (-4.588855155870813e-06) [Y5 Z6 Y7 Z12]
+ (-4.588855155870813e-06) [X5 Z6 X7 Z12]
+ (-4.556569218277352e-06) [Y5 X6 X12 Y13]
+ (-4.556569218277352e-06) [Y5 Y6 Y12 Y13]
+ (-4.556569218277352e-06) [X5 X6 X12 X13]
+ (-4.556569218277352e-06) [X5 Y6 Y12 X13]
+ (-3.6945132944376637e-06) [Y4 X5 X11 Y12]
+ (-3.6945132944376637e-06) [Y4 Y5 Y11 Y12]
+ (-3.6945132944376637e-06) [X4 X5 X11 X12]
+ (-3.6945132944376637e-06) [X4 Y5 Y11 X12]
+ (-3.344081556838618e-06) [Z0 Y5 Z6 Y7]
+ (-3.344081556838618e-06) [Z0 X5 Z6 X7]
+ (-3.344081556838618e-06) [Z1 Y4 Z5 Y6]
+ (-3.344081556838618e-06) [Z1 X4 Z5 X6]
+ (-3.158656432130858e-06) [Y2 Z3 Y4 Z10]
+ (-3.158656432130858e-06) [X2 Z3 X4 Z10]
+ (-3.158656432130858e-06) [Y3 Z4 Y5 Z11]
+ (-3.158656432130858e-06) [X3 Z4 X5 Z11]
+ (-3.0993492439244976e-06) [Z0 Y4 Z5 Y6]
+ (-3.0993492439244976e-06) [Z0 X4 Z5 X6]
+ (-3.0993492439244976e-06) [Z1 Y5 Z6 Y7]
+ (-3.0993492439244976e-06) [Z1 X5 Z6 X7]
+ (-2.89096788182139e-06) [Z6 Y11 Z12 Y13]
+ (-2.89096788182139e-06) [Z6 X11 Z12 X13]
+ (-2.89096788182139e-06) [Z7 Y10 Z11 Y12]
+ (-2.89096788182139e-06) [Z7 X10 Z11 X12]
+ (-2.1776646050845806e-06) [Z0 Y10 Z11 Y12]
+ (-2.1776646050845806e-06) [Z0 X10 Z11 X12]
+ (-2.1776646050845806e-06) [Z1 Y11 Z12 Y13]
+ (-2.1776646050845806e-06) [Z1 X11 Z12 X13]
+ (-1.8818501834029492e-06) [Y4 Z5 Y6 Z9]
+ (-1.8818501834029492e-06) [X4 Z5 X6 Z9]
+ (-1.8818501834029492e-06) [Y5 Z6 Y7 Z8]
+ (-1.8818501834029492e-06) [X5 Z6 X7 Z8]
+ (-1.855120121532225e-06) [Z6 Y10 Z11 Y12]
+ (-1.855120121532225e-06) [Z6 X10 Z11 X12]
+ (-1.855120121532225e-06) [Z7 Y11 Z12 Y13]
+ (-1.855120121532225e-06) [Z7 X11 Z12 X13]
+ (-1.8540608581531165e-06) [Y4 Z5 Y6 Z7]
+ (-1.8540608581531165e-06) [X4 Z5 X6 Z7]
+ (-1.816303169672168e-06) [Z4 Y11 Z12 Y13]
+ (-1.816303169672168e-06) [Z4 X11 Z12 X13]
+ (-1.816303169672168e-06) [Z5 Y10 Z11 Y12]
+ (-1.816303169672168e-06) [Z5 X10 Z11 X12]
+ (-1.6923978286140566e-06) [Y4 Z5 Y6 Z10]
+ (-1.6923978286140566e-06) [X4 Z5 X6 Z10]
+ (-1.6923978286140566e-06) [Y5 Z6 Y7 Z11]
+ (-1.6923978286140566e-06) [X5 Z6 X7 Z11]
+ (-1.6148794139433054e-06) [Z0 Y11 Z12 Y13]
+ (-1.6148794139433054e-06) [Z0 X11 Z12 X13]
+ (-1.6148794139433054e-06) [Z1 Y10 Z11 Y12]
+ (-1.6148794139433054e-06) [Z1 X10 Z11 X12]
+ (-1.5973171978168815e-06) [Z8 Y10 Z11 Y12]
+ (-1.5973171978168815e-06) [Z8 X10 Z11 X12]
+ (-1.5973171978168815e-06) [Z9 Y11 Z12 Y13]
+ (-1.5973171978168815e-06) [Z9 X11 Z12 X13]
+ (-1.4548424492544014e-06) [Y3 X4 X6 Y7]
+ (-1.4548424492544014e-06) [Y3 Y4 Y6 Y7]
+ (-1.4548424492544014e-06) [X3 X4 X6 X7]
+ (-1.4548424492544014e-06) [X3 Y4 Y6 X7]
+ (-1.398044908257213e-06) [Y4 Z5 Y6 Z8]
+ (-1.398044908257213e-06) [X4 Z5 X6 Z8]
+ (-1.398044908257213e-06) [Y5 Z6 Y7 Z9]
+ (-1.398044908257213e-06) [X5 Z6 X7 Z9]
+ (-1.1954890101429948e-06) [Y2 Z3 Y4 Z7]
+ (-1.1954890101429948e-06) [X2 Z3 X4 Z7]
+ (-1.1954890101429948e-06) [Y3 Z4 Y5 Z6]
+ (-1.1954890101429948e-06) [X3 Z4 X5 Z6]
+ (-1.1908508085566147e-06) [Z0 Y3 Z4 Y5]
+ (-1.1908508085566147e-06) [Z0 X3 Z4 X5]
+ (-1.1908508085566147e-06) [Z1 Y2 Z3 Y4]
+ (-1.1908508085566147e-06) [Z1 X2 Z3 X4]
+ (-1.1708301371069631e-06) [Z2 Y5 Z6 Y7]
+ (-1.1708301371069631e-06) [Z2 X5 Z6 X7]
+ (-1.1708301371069631e-06) [Z3 Y4 Z5 Y6]
+ (-1.1708301371069631e-06) [Z3 X4 Z5 X6]
+ (-1.0632283423275522e-06) [Z2 Y10 Z11 Y12]
+ (-1.0632283423275522e-06) [Z2 X10 Z11 X12]
+ (-1.0632283423275522e-06) [Z3 Y11 Z12 Y13]
+ (-1.0632283423275522e-06) [Z3 X11 Z12 X13]
+ (-1.0358477602891653e-06) [Y6 X7 X11 Y12]
+ (-1.0358477602891653e-06) [Y6 Y7 Y11 Y12]
+ (-1.0358477602891653e-06) [X6 X7 X11 X12]
+ (-1.0358477602891653e-06) [X6 Y7 Y11 X12]
+ (-9.509249752862082e-07) [Z2 Y4 Z5 Y6]
+ (-9.509249752862082e-07) [Z2 X4 Z5 X6]
+ (-9.509249752862082e-07) [Z3 Y5 Z6 Y7]
+ (-9.509249752862082e-07) [Z3 X5 Z6 X7]
+ (-9.34455777636315e-07) [Z8 Y11 Z12 Y13]
+ (-9.34455777636315e-07) [Z8 X11 Z12 X13]
+ (-9.34455777636315e-07) [Z9 Y10 Z11 Y12]
+ (-9.34455777636315e-07) [Z9 X10 Z11 X12]
+ (-8.337746755933945e-07) [Z0 Y2 Z3 Y4]
+ (-8.337746755933945e-07) [Z0 X2 Z3 X4]
+ (-8.337746755933945e-07) [Z1 Y3 Z4 Y5]
+ (-8.337746755933945e-07) [Z1 X3 Z4 X5]
+ (-7.956895373611408e-07) [Y3 X4 X8 Y9]
+ (-7.956895373611408e-07) [Y3 Y4 Y8 Y9]
+ (-7.956895373611408e-07) [X3 X4 X8 X9]
+ (-7.956895373611408e-07) [X3 Y4 Y8 X9]
+ (-7.764994119009936e-07) [Y2 Z3 Y4 Z5]
+ (-7.764994119009936e-07) [X2 Z3 X4 Z5]
+ (-5.929765816853344e-07) [Z4 Y5 Z6 Y7]
+ (-5.929765816853344e-07) [Z4 X5 Z6 X7]
+ (-5.770052996030136e-07) [Y2 Z3 Y4 Z9]
+ (-5.770052996030136e-07) [X2 Z3 X4 Z9]
+ (-5.770052996030136e-07) [Y3 Z4 Y5 Z8]
+ (-5.770052996030136e-07) [X3 Z4 X5 Z8]
+ (-5.471647744796989e-07) [Y1 Y2 X11 X12]
+ (-5.471647744796989e-07) [X1 X2 Y11 Y12]
+ (-4.838052751457363e-07) [Y5 X6 X8 Y9]
+ (-4.838052751457363e-07) [Y5 Y6 Y8 Y9]
+ (-4.838052751457363e-07) [X5 X6 X8 X9]
+ (-4.838052751457363e-07) [X5 Y6 Y8 X9]
+ (-3.570761329632201e-07) [Y0 X1 X3 Y4]
+ (-3.570761329632201e-07) [Y0 Y1 Y3 Y4]
+ (-3.570761329632201e-07) [X0 X1 X3 X4]
+ (-3.570761329632201e-07) [X0 Y1 Y3 X4]
+ (-2.4473231291412025e-07) [Y0 X1 X5 Y6]
+ (-2.4473231291412025e-07) [Y0 Y1 Y5 Y6]
+ (-2.4473231291412025e-07) [X0 X1 X5 X6]
+ (-2.4473231291412025e-07) [X0 Y1 Y5 X6]
+ (-2.1990516182075485e-07) [Y2 X3 X5 Y6]
+ (-2.1990516182075485e-07) [Y2 Y3 Y5 Y6]
+ (-2.1990516182075485e-07) [X2 X3 X5 X6]
+ (-2.1990516182075485e-07) [X2 Y3 Y5 X6]
+ (-1.9332412773244167e-07) [Y1 X2 X3 Y4]
+ (-1.9332412773244167e-07) [X1 Y2 Y3 X4]
+ (-1.2919694864073618e-07) [Y1 Z2 Z3 Y5]
+ (-1.2919694864073618e-07) [X1 Z2 Z3 X5]
+ (1.7379332625993532e-07) [Y0 Z1 Z3 Y4]
+ (1.7379332625993532e-07) [X0 Z1 Z3 X4]
+ (1.7379332625993532e-07) [Y1 Z2 Z4 Y5]
+ (1.7379332625993532e-07) [X1 Z2 Z4 X5]
+ (1.9332412773244167e-07) [Y1 Y2 X3 X4]
+ (1.9332412773244167e-07) [X1 X2 Y3 Y4]
+ (2.1868423775812706e-07) [Y2 Z3 Y4 Z8]
+ (2.1868423775812706e-07) [X2 Z3 X4 Z8]
+ (2.1868423775812706e-07) [Y3 Z4 Y5 Z9]
+ (2.1868423775812706e-07) [X3 Z4 X5 Z9]
+ (2.5935343911140673e-07) [Y2 Z3 Y4 Z6]
+ (2.5935343911140673e-07) [X2 Z3 X4 Z6]
+ (2.5935343911140673e-07) [Y3 Z4 Y5 Z7]
+ (2.5935343911140673e-07) [X3 Z4 X5 Z7]
+ (3.606071868346126e-07) [Y0 Z1 Z2 Y4]
+ (3.606071868346126e-07) [X0 Z1 Z2 X4]
+ (3.606071868346126e-07) [Y1 Z3 Z4 Y5]
+ (3.606071868346126e-07) [X1 Z3 Z4 X5]
+ (5.471647744796989e-07) [Y1 X2 X11 Y12]
+ (5.471647744796989e-07) [X1 Y2 Y11 X12]
+ (5.627851911412754e-07) [Y0 X1 X11 Y12]
+ (5.627851911412754e-07) [Y0 Y1 Y11 Y12]
+ (5.627851911412754e-07) [X0 X1 X11 X12]
+ (5.627851911412754e-07) [X0 Y1 Y11 X12]
+ (6.628614201805665e-07) [Y8 X9 X11 Y12]
+ (6.628614201805665e-07) [Y8 Y9 Y11 Y12]
+ (6.628614201805665e-07) [X8 X9 X11 X12]
+ (6.628614201805665e-07) [X8 Y9 Y11 X12]
+ (1.1094407592705214e-06) [Z2 Y11 Z12 Y13]
+ (1.1094407592705214e-06) [Z2 X11 Z12 X13]
+ (1.1094407592705214e-06) [Z3 Y10 Z11 Y12]
+ (1.1094407592705214e-06) [Z3 X10 Z11 X12]
+ (1.6021167407632491e-06) [Z2 Y3 Z4 Y5]
+ (1.6021167407632491e-06) [Z2 X3 Z4 X5]
+ (1.8782101247654954e-06) [Z4 Y10 Z11 Y12]
+ (1.8782101247654954e-06) [Z4 X10 Z11 X12]
+ (1.8782101247654954e-06) [Z5 Y11 Z12 Y13]
+ (1.8782101247654954e-06) [Z5 X11 Z12 X13]
+ (2.1726691015980735e-06) [Y2 X3 X11 Y12]
+ (2.1726691015980735e-06) [Y2 Y3 Y11 Y12]
+ (2.1726691015980735e-06) [X2 X3 X11 X12]
+ (2.1726691015980735e-06) [X2 Y3 Y11 X12]
+ (3.117447946524577e-06) [Y0 Z2 Z3 Y4]
+ (3.117447946524577e-06) [X0 Z2 Z3 X4]
+ (3.5390541846243886e-06) [Y2 Z3 Y4 Z12]
+ (3.5390541846243886e-06) [X2 Z3 X4 Z12]
+ (3.5390541846243886e-06) [Y3 Z4 Y5 Z13]
+ (3.5390541846243886e-06) [X3 Z4 X5 Z13]
+ (4.2819138850029594e-06) [Y4 Z5 Y6 Z11]
+ (4.2819138850029594e-06) [X4 Z5 X6 Z11]
+ (4.2819138850029594e-06) [Y5 Z6 Y7 Z10]
+ (4.2819138850029594e-06) [X5 Z6 X7 Z10]
+ (5.275883122294926e-06) [Y3 X4 X12 Y13]
+ (5.275883122294926e-06) [Y3 Y4 Y12 Y13]
+ (5.275883122294926e-06) [X3 X4 X12 X13]
+ (5.275883122294926e-06) [X3 Y4 Y12 X13]
+ (5.974311713617015e-06) [Y5 X6 X10 Y11]
+ (5.974311713617015e-06) [Y5 Y6 Y10 Y11]
+ (5.974311713617015e-06) [X5 X6 X10 X11]
+ (5.974311713617015e-06) [X5 Y6 Y10 X11]
+ (7.954413176564841e-06) [Y10 Z11 Y12 Z13]
+ (7.954413176564841e-06) [X10 Z11 X12 Z13]
+ (8.814937306919314e-06) [Y2 Z3 Y4 Z13]
+ (8.814937306919314e-06) [X2 Z3 X4 Z13]
+ (8.814937306919314e-06) [Y3 Z4 Y5 Z12]
+ (8.814937306919314e-06) [X3 Z4 X5 Z12]
+ (0.0002921986261110528) [Y7 X8 X9 Y10]
+ (0.0002921986261110528) [X7 Y8 Y9 X10]
+ (0.0004956762314916403) [Y2 Z4 Z5 Y6]
+ (0.0004956762314916403) [X2 Z4 Z5 X6]
+ (0.0011059037691896793) [Y0 Z1 Y2 Z5]
+ (0.0011059037691896793) [X0 Z1 X2 Z5]
+ (0.0011059037691896793) [Y1 Z2 Y3 Z4]
+ (0.0011059037691896793) [X1 Z2 X3 Z4]
+ (0.0016638798784907847) [Y2 Z3 Z4 Y6]
+ (0.0016638798784907847) [X2 Z3 Z4 X6]
+ (0.0016638798784907847) [Y3 Z5 Z6 Y7]
+ (0.0016638798784907847) [X3 Z5 Z6 X7]
+ (0.0017560707018412342) [Y0 Z1 Y2 Z11]
+ (0.0017560707018412342) [X0 Z1 X2 Z11]
+ (0.0017560707018412342) [Y1 Z2 Y3 Z10]
+ (0.0017560707018412342) [X1 Z2 X3 Z10]
+ (0.0023262306231580715) [Y0 Z1 Y2 Z13]
+ (0.0023262306231580715) [X0 Z1 X2 Z13]
+ (0.0023262306231580715) [Y1 Z2 Y3 Z12]
+ (0.0023262306231580715) [X1 Z2 X3 Z12]
+ (0.0027458364701868046) [Y0 X1 X4 Y5]
+ (0.0027458364701868046) [X0 Y1 Y4 X5]
+ (0.0029297686747510464) [Y0 Z1 Y2 Z9]
+ (0.0029297686747510464) [X0 Z1 X2 Z9]
+ (0.0029297686747510464) [Y1 Z2 Y3 Z8]
+ (0.0029297686747510464) [X1 Z2 X3 Z8]
+ (0.003276971931231644) [Y0 Z1 Y2 Z3]
+ (0.003276971931231644) [X0 Z1 X2 Z3]
+ (0.003347617530666174) [Y0 Z1 Y2 Z7]
+ (0.003347617530666174) [X0 Z1 X2 Z7]
+ (0.003347617530666174) [Y1 Z2 Y3 Z6]
+ (0.003347617530666174) [X1 Z2 X3 Z6]
+ (0.0035552901955042673) [Y0 Z1 Y2 Z10]
+ (0.0035552901955042673) [X0 Z1 X2 Z10]
+ (0.0035552901955042673) [Y1 Z2 Y3 Z11]
+ (0.0035552901955042673) [X1 Z2 X3 Z11]
+ (0.005143391768825129) [Y3 Y4 X5 X6]
+ (0.005143391768825129) [X3 X4 Y5 Y6]
+ (0.00528377648840296) [Y0 X1 X12 Y13]
+ (0.00528377648840296) [X0 Y1 Y12 X13]
+ (0.005530759218631524) [Y0 Z1 Y2 Z4]
+ (0.005530759218631524) [X0 Z1 X2 Z4]
+ (0.005530759218631524) [Y1 Z2 Y3 Z5]
+ (0.005530759218631524) [X1 Z2 X3 Z5]
+ (0.00608782248056186) [Y8 X9 X12 Y13]
+ (0.00608782248056186) [X8 Y9 Y12 X13]
+ (0.006509361201177232) [Y0 X1 X8 Y9]
+ (0.006509361201177232) [X0 Y1 Y8 X9]
+ (0.00688819435297057) [Y0 X1 X6 Y7]
+ (0.00688819435297057) [X0 Y1 Y6 X7]
+ (0.006901238249797275) [Y0 Z1 Y2 Z12]
+ (0.006901238249797275) [X0 Z1 X2 Z12]
+ (0.006901238249797275) [Y1 Z2 Y3 Z13]
+ (0.006901238249797275) [X1 Z2 X3 Z13]
+ (0.0071569349198569365) [Y4 X5 X8 Y9]
+ (0.0071569349198569365) [X4 Y5 Y8 X9]
+ (0.00773142525077527) [Y0 X1 X10 Y11]
+ (0.00773142525077527) [X0 Y1 Y10 X11]
+ (0.008032520918821359) [Y0 Z1 Y2 Z6]
+ (0.008032520918821359) [X0 Z1 X2 Z6]
+ (0.008032520918821359) [Y1 Z2 Y3 Z7]
+ (0.008032520918821359) [X1 Z2 X3 Z7]
+ (0.009560705729135928) [Y8 X9 X10 Y11]
+ (0.009560705729135928) [X8 Y9 Y10 X11]
+ (0.011055020596132066) [Y0 Z1 Y2 Z8]
+ (0.011055020596132066) [X0 Z1 X2 Z8]
+ (0.011055020596132066) [Y1 Z2 Y3 Z9]
+ (0.011055020596132066) [X1 Z2 X3 Z9]
+ (0.011285190200840935) [Y5 Y6 X11 X12]
+ (0.011285190200840935) [X5 X6 Y11 Y12]
+ (0.011307274008848161) [Y7 Z8 Z9 Y11]
+ (0.011307274008848161) [X7 Z8 Z9 X11]
+ (0.011982389010247962) [Y4 X5 X6 Y7]
+ (0.011982389010247962) [X4 Y5 Y6 X7]
+ (0.0138733817484261) [Y6 X7 X8 Y9]
+ (0.0138733817484261) [X6 Y7 Y8 X9]
+ (0.014583648907612629) [Y0 X1 X2 Y3]
+ (0.014583648907612629) [X0 Y1 Y2 X3]
+ (0.015577208063976456) [Y2 X3 X12 Y13]
+ (0.015577208063976456) [X2 Y3 Y12 X13]
+ (0.017366118994651427) [Y6 X7 X12 Y13]
+ (0.017366118994651427) [X6 Y7 Y12 X13]
+ (0.017680067952481532) [Y4 X5 X10 Y11]
+ (0.017680067952481532) [X4 Y5 Y10 X11]
+ (0.017825140995786498) [Y6 X7 X10 Y11]
+ (0.017825140995786498) [X6 Y7 Y10 X11]
+ (0.01902824244384726) [Y3 X4 X11 Y12]
+ (0.01902824244384726) [X3 Y4 Y11 X12]
+ (0.025384657508457392) [Y2 X3 X10 Y11]
+ (0.025384657508457392) [X2 Y3 Y10 X11]
+ (0.028685183716105903) [Y10 X11 X12 Y13]
+ (0.028685183716105903) [X10 Y11 Y12 X13]
+ (0.02981242451734576) [Y6 Z7 Z8 Y10]
+ (0.02981242451734576) [X6 Z7 Z8 X10]
+ (0.02981242451734576) [Y7 Z9 Z10 Y11]
+ (0.02981242451734576) [X7 Z9 Z10 X11]
+ (0.03010462314345681) [Y6 Z7 Z9 Y10]
+ (0.03010462314345681) [X6 Z7 Z9 X10]
+ (0.03010462314345681) [Y7 Z8 Z10 Y11]
+ (0.03010462314345681) [X7 Z8 Z10 X11]
+ (0.030787505389143915) [Y6 Z8 Z9 Y10]
+ (0.030787505389143915) [X6 Z8 Z9 X10]
+ (0.031143817988967086) [Y2 X3 X6 Y7]
+ (0.031143817988967086) [X2 Y3 Y6 X7]
+ (0.03583956795335344) [Y2 X3 X4 Y5]
+ (0.03583956795335344) [X2 Y3 Y4 X5]
+ (0.03619412355904259) [Y2 X3 X8 Y9]
+ (0.03619412355904259) [X2 Y3 Y8 X9]
+ (0.03831467029480388) [Y4 X5 X12 Y13]
+ (0.03831467029480388) [X4 Y5 Y12 X13]
+ (0.10433064780651399) [Z0 Y1 Z2 Y3]
+ (0.10433064780651399) [Z0 X1 Z2 X3]
+ (-0.12133276911042355) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042355) [X2 Z3 Z4 Z5 X6]
+ (-0.12133276911042355) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133276911042355) [X3 Z4 Z5 Z6 X7]
+ (3.202076880874746e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.202076880874746e-06) [X1 Z2 Z3 Z4 X5]
+ (3.202076880874747e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.202076880874747e-06) [X0 Z1 Z2 Z3 X4]
+ (0.22848106564918844) [Y7 Z8 Z9 Z10 Y11]
+ (0.22848106564918844) [X7 Z8 Z9 Z10 X11]
+ (0.2284810656491885) [Y6 Z7 Z8 Z9 Y10]
+ (0.2284810656491885) [X6 Z7 Z8 Z9 X10]
+ (-0.03276765782329054) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.03276765782329054) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.03276765782329054) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.03276765782329054) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.02711503684527318) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.02711503684527318) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.02711503684527318) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.02711503684527318) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.025996177598021163) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.025996177598021163) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.01756120240964617) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.01756120240964617) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.01756120240964617) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.01756120240964617) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.01456453123117301) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.01456453123117301) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.01456453123117301) [X7 Z8 Z9 X10 X12 X13]
+ (-0.01456453123117301) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.012215040997613965) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.012215040997613965) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.012215040997613965) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.012215040997613965) [X4 Z5 X6 X11 Z12 X13]
+ (-0.012215040997613965) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.012215040997613965) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.012215040997613965) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.012215040997613965) [X5 Z6 X7 X10 Z11 X12]
+ (-0.01175601341981926) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.01175601341981926) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.01175601341981926) [X3 Z4 Z5 X6 X8 X9]
+ (-0.01175601341981926) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.008764827575688756) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.008764827575688756) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.008764827575688756) [X2 Z3 Z4 X5 X11 X12]
+ (-0.008764827575688756) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.008764827575688756) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.008764827575688756) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.008764827575688756) [X3 X4 X10 Z11 Z12 X13]
+ (-0.008764827575688756) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.008125251921381018) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125251921381018) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.007306759928832983) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306759928832983) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306759928832983) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306759928832983) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.0058051889898269055) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.0058051889898269055) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.0058051889898269055) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.0058051889898269055) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.005652620978017357) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.005652620978017357) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.005652620978017357) [X0 X1 X3 Z4 Z5 X6]
+ (-0.005652620978017357) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.005143391768825129) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.005143391768825129) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.005143391768825129) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.005143391768825129) [X2 Z3 X4 X5 Z6 X7]
+ (-0.004684903388155183) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.004684903388155183) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668620318776299) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668620318776299) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.004575007626639205) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.004575007626639205) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424855449441844) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424855449441844) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.004158797381840061) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.004158797381840061) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.004158797381840061) [X3 Z4 Z5 X6 X12 X13]
+ (-0.004158797381840061) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.003493790359890158) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.003493790359890158) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.003493790359890158) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.003493790359890158) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.002779026799025526) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.002779026799025526) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.002293956611352463) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.002293956611352463) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.0017992194936630333) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.0017992194936630333) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278753941369547) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.0017278753941369547) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298507967730302) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298507967730302) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298507967730302) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298507967730302) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533856254125488) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533856254125488) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008145313270956759) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008145313270956759) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008145313270956759) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008145313270956759) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.73503688059087e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.73503688059087e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.73503688059087e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.73503688059087e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.774817864758996e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.774817864758996e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.774817864758996e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.774817864758996e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.518362215867742e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.518362215867742e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.518362215867742e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.518362215867742e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.444344676031047e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.444344676031047e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.444344676031047e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.444344676031047e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.524373848736428e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.524373848736428e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.524373848736428e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.524373848736428e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.290028433226103e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.290028433226103e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.290028433226103e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.290028433226103e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.974311713617017e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.974311713617017e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.275883122294927e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.275883122294927e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.643051068581625e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.643051068581625e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.556569218277352e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.556569218277352e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.253224225742977e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.253224225742977e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.7696594519579588e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.7696594519579588e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.6945132944376633e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.6945132944376633e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.610297130527386e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.610297130527386e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.610297130527386e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.610297130527386e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.313145500325613e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.313145500325613e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.2774831954438026e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.2774831954438026e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.2774831954438026e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.2774831954438026e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.211228348410816e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.211228348410816e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.211228348410816e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.211228348410816e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.151346311251981e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.151346311251981e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.088250711320387e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.088250711320387e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.1726691015980735e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.1726691015980735e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.4548424492544014e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.4548424492544014e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.3304731887279498e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.3304731887279498e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2283337826416384e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2283337826416384e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.0358477602891653e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.0358477602891653e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956895373611408e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956895373611408e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733197742310912e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733197742310912e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733197742310912e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733197742310912e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.628614201805665e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.628614201805665e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.556281914865094e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.556281914865094e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.556281914865094e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.556281914865094e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.418291574746583e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.418291574746583e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.418291574746583e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.418291574746583e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.927453082734183e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927453082734183e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927453082734183e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927453082734183e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627851911412754e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627851911412754e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660624644981e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660624644981e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660624644981e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660624644981e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660624644981e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660624644981e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660624644981e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660624644981e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.838052751457363e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.838052751457363e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.570761329632201e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.570761329632201e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.328139350835836e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.328139350835836e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.086826565364295e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.086826565364295e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.086826565364295e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.086826565364295e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.447323129141203e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.447323129141203e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.3713289482326798e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.3713289482326798e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.3713289482326798e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.3713289482326798e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.1990516182075482e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.1990516182075482e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.9332412773244167e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.9332412773244167e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.9332412773244167e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.9332412773244167e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.8394209157286567e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.8394209157286567e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.8394209157286567e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.8394209157286567e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.551053917812643e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.551053917812643e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.551053917812643e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.551053917812643e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.380778148242016e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.380778148242016e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.380778148242016e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.380778148242016e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.380778148242016e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.380778148242016e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.380778148242016e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.380778148242016e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.3807781482420159e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.3807781482420159e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.3807781482420159e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.3807781482420159e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.2919694864073618e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.2919694864073618e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.1076325600848534e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.1076325600848534e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.1076325600848534e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.1076325600848534e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.1076325600848534e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.1076325600848534e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.1076325600848534e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.1076325600848534e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.057446595767273e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057446595767273e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057446595767273e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057446595767273e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649310135383392e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649310135383392e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649310135383392e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649310135383392e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.839420915728657e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.839420915728657e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.839420915728657e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.839420915728657e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.1990516182075482e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.1990516182075482e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.447323129141203e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.447323129141203e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.2362599617710186e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.2362599617710186e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.2362599617710186e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.2362599617710186e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.328139350835836e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.328139350835836e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.570761329632201e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.570761329632201e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.838052751457363e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.838052751457363e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.627851911412754e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627851911412754e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.628614201805665e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.628614201805665e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956895373611408e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956895373611408e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.306536652189971e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.306536652189971e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.306536652189971e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.306536652189971e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.0358477602891653e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.0358477602891653e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2283337826416384e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2283337826416384e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.2393363217554266e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393363217554266e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393363217554266e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393363217554266e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.3304731887279498e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.3304731887279498e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.4548424492544014e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.4548424492544014e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.1726691015980735e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.1726691015980735e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.088250711320387e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.088250711320387e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.117447946524577e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.117447946524577e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.151346311251981e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.151346311251981e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.313145500325613e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.313145500325613e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.3343312895670995e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.3343312895670995e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.6945132944376633e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.6945132944376633e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.183932559617733e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.183932559617733e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.556569218277352e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.556569218277352e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.643051068581625e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.643051068581625e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.275883122294927e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.275883122294927e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.974311713617017e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.974311713617017e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.0002921986261110528) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.0002921986261110528) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.0002921986261110528) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.0002921986261110528) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004956762314916404) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004956762314916404) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650070219499025) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650070219499025) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650070219499025) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650070219499025) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533856254125488) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533856254125488) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.0016095313817213828) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.0016095313817213828) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.0016095313817213828) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.0016095313817213828) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0016676041811440703) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.0016676041811440703) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.0016676041811440703) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.0016676041811440703) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.0017278753941369547) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.0017278753941369547) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.0017992194936630333) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.0017992194936630333) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.002293956611352463) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.002293956611352463) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.0024629170071339317) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.0024629170071339317) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.0024629170071339317) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.0024629170071339317) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.003961560792496533) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.003961560792496533) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.003961560792496533) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.003961560792496533) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.004424855449441844) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424855449441844) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004575007626639205) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.004575007626639205) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668620318776299) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668620318776299) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.004684903388155183) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.004684903388155183) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005324835234221687) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.005324835234221687) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.005324835234221687) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.005324835234221687) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.0053686593581095425) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.0053686593581095425) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.0053686593581095425) [X2 X3 X7 Z8 Z9 X10]
+ (0.0053686593581095425) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.007960880725921576) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.007960880725921576) [X4 Z5 X6 X10 Z11 X12]
+ (0.007960880725921576) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.007960880725921576) [X5 Z6 X7 X11 Z12 X13]
+ (0.008125251921381018) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125251921381018) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.008890731522694607) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.008890731522694607) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.008890731522694607) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.008890731522694607) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.010263414868158502) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.010263414868158502) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.010263414868158502) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.010263414868158502) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.01054042590767151) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.01054042590767151) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.01054042590767151) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.01054042590767151) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010960074940542547) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010960074940542547) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010960074940542547) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010960074940542547) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307274008848161) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307274008848161) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.014411099430130882) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411099430130882) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411099430130882) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411099430130882) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.015225630757226558) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.015225630757226558) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.015225630757226558) [X3 Z4 Z5 X6 X10 X11]
+ (0.015225630757226558) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.015588250102380189) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.015588250102380189) [X2 Z3 X4 X10 Z11 X12]
+ (0.015588250102380189) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.015588250102380189) [X3 Z4 X5 X11 Z12 X13]
+ (0.018266834869375533) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.018266834869375533) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.018266834869375533) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.018266834869375533) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.01902042317303992) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.01902042317303992) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.01902042317303992) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.01902042317303992) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.02017592172353554) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.02017592172353554) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.02017592172353554) [X4 Z5 Z6 X7 X11 X12]
+ (0.02017592172353554) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.02017592172353554) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.02017592172353554) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.02017592172353554) [X5 X6 X10 Z11 Z12 X13]
+ (0.02017592172353554) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.024353077678068942) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.024353077678068942) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.024353077678068942) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.024353077678068942) [X2 Z3 X4 X11 Z12 X13]
+ (0.024353077678068942) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.024353077678068942) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.024353077678068942) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.024353077678068942) [X3 Z4 X5 X10 Z11 X12]
+ (0.024389082531149464) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.024389082531149464) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.024389082531149464) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.024389082531149464) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.02510495713884452) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.02510495713884452) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.02510495713884452) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.02510495713884452) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.030787505389143915) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.030787505389143915) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.045879470781298184) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.045879470781298184) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.05600733087780767) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.05600733087780767) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.05600733087780767) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.05600733087780767) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.056084681246613574) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.056084681246613574) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.056084681246613574) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.056084681246613574) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.631277928530539e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631277928530539e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.631277928530538e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631277928530538e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.5950860069221325e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.5950860069221325e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.5950860069221314e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.5950860069221314e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.042743277013783575) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.042743277013783575) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (0.042743277013783575) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.042743277013783575) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.04764261217638309) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.04764261217638309) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.04764261217638309) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.04764261217638309) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.04171881383982175) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.04171881383982175) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.04171881383982175) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.04171881383982175) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.03956441632289329) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.03956441632289329) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.03956441632289329) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03956441632289329) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.039359168022053) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.039359168022053) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.039359168022053) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.039359168022053) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.03931805194719752) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03931805194719752) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.03931805194719752) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03931805194719752) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.035608378988312483) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.035608378988312483) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.029903789512624762) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.029903789512624762) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.029903789512624762) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.029903789512624762) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.028730779551905464) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.028730779551905464) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.028730779551905464) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.028730779551905464) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.025637238296026855) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.025637238296026855) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.025637238296026855) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.025637238296026855) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.024755463292890918) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.024755463292890918) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.024755463292890918) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.024755463292890918) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.024282117354692937) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.024282117354692937) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.02314513092952908) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.02314513092952908) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.022528440196012973) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.022528440196012973) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.021433810721600777) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.021433810721600777) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.021433810721600777) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.021433810721600777) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251592) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.019257505095251592) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.01902824244384726) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.01902824244384726) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.01888903030494283) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.01888903030494283) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.01888903030494283) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.01888903030494283) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.01602460368917955) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.01602460368917955) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.015225630757226558) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225630757226558) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.014603704729162083) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.014603704729162083) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.01456453123117301) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.01456453123117301) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.011756013419819262) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.011756013419819262) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.011285190200840935) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.011285190200840935) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.009841749246962636) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009841749246962636) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.009612634606847309) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612634606847309) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612634606847309) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612634606847309) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.008469978791023984) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.008469978791023984) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.007306759928832983) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.007306759928832983) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005923798336561342) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.005923798336561342) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.005652620978017357) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.005652620978017357) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.0053686593581095425) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0053686593581095425) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.004158797381840061) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.004158797381840061) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0033566705638328896) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.0033566705638328896) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.0033566705638328896) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.0033566705638328896) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.0032675138544235524) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.0032675138544235524) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.0032675138544235524) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.0032675138544235524) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.002779026799025526) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.002779026799025526) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.0026860409778066033) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.0026860409778066033) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.0026860409778066033) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.0026860409778066033) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.002293956611352463) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.002293956611352463) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.002293956611352463) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.002293956611352463) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.0009581655836696478) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.0009581655836696478) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0009581655836696478) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.0009581655836696478) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.0009581655836696478) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.0009581655836696478) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0009581655836696478) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.0009581655836696478) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.0002463643756957708) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0002463643756957708) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303550946) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.00013840177303550946) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.00013840177303550946) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.00013840177303550946) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.73503688059087e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.73503688059087e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.610358530606267e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.610358530606267e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.610358530606267e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.610358530606267e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.531680879560452e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.531680879560452e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.531680879560452e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.531680879560452e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.806102775486927e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.806102775486927e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.806102775486927e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.806102775486927e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.089799467683125e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.089799467683125e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.089799467683125e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.089799467683125e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.6522096693513245e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.6522096693513245e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.6522096693513245e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.6522096693513245e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851833801408e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.481851833801408e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851833801408e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.481851833801408e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.0714807367133e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.0714807367133e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.0714807367133e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.0714807367133e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.734622038773627e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.734622038773627e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.734622038773627e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.734622038773627e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.7288431473347214e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.7288431473347214e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.7288431473347214e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.7288431473347214e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.253224225742977e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.253224225742977e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.7696594519579588e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.7696594519579588e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.544395429371421e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.544395429371421e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.544395429371421e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.544395429371421e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.544395429371421e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.544395429371421e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.544395429371421e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.544395429371421e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.3609563203484036e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563203484036e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563203484036e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.3609563203484036e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.1032156046257645e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.1032156046257645e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.1032156046257645e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.1032156046257645e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220980961645e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.0111220980961645e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220980961645e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.0111220980961645e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9429468366406696e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.9429468366406696e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9429468366406696e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9429468366406696e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.654117477024867e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.654117477024867e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.654117477024867e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.654117477024867e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.5224930676780357e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224930676780357e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224930676780357e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224930676780357e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224930676780357e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930676780357e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930676780357e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224930676780357e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.2283337826416384e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337826416384e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.2283337826416384e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337826416384e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.988770289646428e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988770289646428e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988770289646428e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988770289646428e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.86776510458151e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.86776510458151e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.86776510458151e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.86776510458151e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.189990975359632e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.189990975359632e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246207250776e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246207250776e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.471647744796989e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.471647744796989e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.5614471805103365e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.5614471805103365e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.5614471805103365e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.5614471805103365e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.52338967773446e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.52338967773446e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.427323109136089e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.427323109136089e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.427323109136089e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.427323109136089e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.328139350835836e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350835836e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.328139350835836e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350835836e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.086826565364295e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.086826565364295e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.888293596158028e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.888293596158028e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.888293596158028e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.888293596158028e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.3713289482326798e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.3713289482326798e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.8394209157286573e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.8394209157286573e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.057446595767273e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057446595767273e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.537178096576125e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.537178096576125e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.537178096576125e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.537178096576125e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.057446595767273e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057446595767273e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.20935065295997e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.20935065295997e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.20935065295997e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.20935065295997e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783554991558e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783554991558e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783554991558e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783554991558e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.8394209157286573e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.8394209157286573e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.3713289482326798e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.3713289482326798e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.086826565364295e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.086826565364295e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.52338967773446e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.52338967773446e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.471647744796989e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.471647744796989e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246207250776e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246207250776e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189990975359632e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.189990975359632e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.3304731887279498e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.3304731887279498e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.3304731887279498e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.3304731887279498e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288532435739454e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288532435739454e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288532435739454e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288532435739454e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.6893489515289834e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.6893489515289834e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.6893489515289834e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.6893489515289834e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.745518400406778e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.745518400406778e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.745518400406778e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.745518400406778e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.745518400406778e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.745518400406778e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.745518400406778e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.745518400406778e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.21184201920702e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.21184201920702e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.21184201920702e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.21184201920702e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.21184201920702e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.21184201920702e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.21184201920702e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.21184201920702e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.313145500325613e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.313145500325613e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.313145500325613e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.313145500325613e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.3343312895671e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.3343312895671e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.1839325596177334e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.1839325596177334e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.73503688059087e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.73503688059087e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.0002463643756957708) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.0002463643756957708) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00044585351288407935) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.00044585351288407935) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.00044585351288407935) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.00044585351288407935) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940221543005382) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005382) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005382) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005382) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005382) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005382) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940221543005382) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005382) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533856254125488) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533856254125488) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533856254125488) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533856254125488) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0010435246534907685) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435246534907685) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435246534907685) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435246534907685) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.0012803060973496825) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.0012803060973496825) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.0012803060973496825) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.0012803060973496825) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.0013038004788126934) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.0013038004788126934) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.0013038004788126934) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.0013038004788126934) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.002261966062482341) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.002261966062482341) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.002261966062482341) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.002261966062482341) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.002261966062482341) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.002261966062482341) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.002261966062482341) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.002261966062482341) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.003989841456619295) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.003989841456619295) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.003989841456619295) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.003989841456619295) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.004158797381840061) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.004158797381840061) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.0043110385079143205) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.0043110385079143205) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.0043110385079143205) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.0043110385079143205) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.004636976661182572) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.004636976661182572) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.004636976661182572) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.004636976661182572) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.005114473831660378) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.005114473831660378) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.005114473831660378) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.005114473831660378) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.005114473831660378) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660378) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660378) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005114473831660378) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005241535382803865) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.005241535382803865) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.005241535382803865) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.005241535382803865) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.005262642473076836) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262642473076836) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262642473076836) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262642473076836) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.0053686593581095425) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.0053686593581095425) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005379937155839376) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.005379937155839376) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.005379937155839376) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.005379937155839376) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.005652620978017357) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.005652620978017357) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.005708495985960916) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708495985960916) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708495985960916) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708495985960916) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.005923798336561342) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.005923798336561342) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.007306759928832983) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.007306759928832983) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469978791023984) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.008469978791023984) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.009841749246962636) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.009841749246962636) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.011285190200840935) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.011285190200840935) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.011756013419819262) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.011756013419819262) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.01456453123117301) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.01456453123117301) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603704729162083) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.014603704729162083) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.015225630757226558) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225630757226558) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.01602460368917955) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.01602460368917955) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.01902824244384726) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.01902824244384726) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.019257505095251592) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.019257505095251592) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.045879470781298184) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.045879470781298184) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.3693708936615615) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.3693708936615615) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.3693708936615615) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.3693708936615615) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.28164257767022804) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.28164257767022804) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.28164257767022793) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.28164257767022793) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0906514420703648) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0906514420703648) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0906514420703648) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0906514420703648) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863627) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684737589863627) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863627) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684737589863627) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635021950635001) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635021950635001) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635021950635001) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635021950635001) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0675238509921402) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0675238509921402) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0675238509921402) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0675238509921402) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.035608378988312483) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.035608378988312483) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.034903343373661744) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.034903343373661744) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.034903343373661744) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.034903343373661744) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02459186088382995) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.02459186088382995) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02459186088382995) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.02459186088382995) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.024282117354692937) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.024282117354692937) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.02314513092952908) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.02314513092952908) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.02252844019601297) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.02252844019601297) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019538050311314673) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.019538050311314673) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.019538050311314673) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.019538050311314673) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.017091553155898807) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.017091553155898807) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.017091553155898807) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.017091553155898807) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.016024603689179545) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.016024603689179545) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.016024603689179545) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.016024603689179545) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.010311482489831799) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831799) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831799) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311482489831799) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.009841749246962636) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962636) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.009841749246962636) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962636) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209818) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209818) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209818) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008826368514209818) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008541996625454839) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454839) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454839) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454839) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454839) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454839) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454839) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008541996625454839) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008469978791023982) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023982) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.008469978791023982) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023982) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.004668620318776299) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668620318776299) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.0038764708993369356) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.0038764708993369356) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.0038040661717285364) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0038040661717285364) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0038040661717285364) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0038040661717285364) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.00348415730021788) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.00348415730021788) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0033566705638328896) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.0033566705638328896) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.0032675138544235524) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.0032675138544235524) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.002141361223101596) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.002141361223101596) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.0017278753941369547) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.0017278753941369547) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.0016407548553124033) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0016407548553124033) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214169104) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0014528843214169104) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214169104) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0014528843214169104) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870896771024411) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007870896771024411) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192743499487624) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192743499487624) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.00019400857029755602) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00019400857029755602) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303550946) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.00013840177303550946) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.141625221156146e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.141625221156146e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.141625221156146e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.141625221156146e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.0714807367133e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.0714807367133e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.151346311251981e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.151346311251981e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.088250711320387e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.088250711320387e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.9885117066417424e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.9885117066417424e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8742990714520724e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.8742990714520724e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.3609563203484036e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.3609563203484036e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.3002946562874027e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3002946562874027e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.146837650764137e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.146837650764137e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.146837650764137e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.146837650764137e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.352332103008548e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.352332103008548e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.352332103008548e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.352332103008548e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091637199088482e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637199088482e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637199088482e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637199088482e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637199088482e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637199088482e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637199088482e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091637199088482e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.074305985982489e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.074305985982489e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.074305985982489e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.074305985982489e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128986287347e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128986287347e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128986287347e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128986287347e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867765104581511e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.867765104581511e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.56069246473543e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.56069246473543e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.56069246473543e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.56069246473543e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.56069246473543e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.56069246473543e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.56069246473543e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.56069246473543e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.997018422319176e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.997018422319176e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.997018422319176e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.997018422319176e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.997018422319176e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.997018422319176e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.997018422319176e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.997018422319176e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.5682475213540215e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.5682475213540215e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.5682475213540215e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.5682475213540215e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393085528865e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393085528865e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393085528865e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393085528865e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393085528865e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393085528865e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.3767393085528865e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393085528865e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.888293596158028e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.888293596158028e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6863815440646186e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.6863815440646186e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035783554991558e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7035783554991558e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.20935065295997e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.20935065295997e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244479875e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773244479875e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244479875e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773244479875e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244479875e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773244479875e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773244479875e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773244479875e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253789913317e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253789913317e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9742253789913317e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9742253789913317e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0474716555656467e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.0474716555656467e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.0474716555656467e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.0474716555656467e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.20935065295997e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.20935065295997e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0717282182388812e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0717282182388812e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0717282182388812e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0717282182388812e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.2004287494038445e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.2004287494038445e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.2004287494038445e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.2004287494038445e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.7035783554991558e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7035783554991558e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.3120943051550644e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.3120943051550644e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.3120943051550644e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.3120943051550644e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.6863815440646186e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815440646186e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.888293596158028e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.888293596158028e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.0922506162004024e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506162004024e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.0922506162004024e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506162004024e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.0922506162004024e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506162004024e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.0922506162004024e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506162004024e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.444597854457631e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.444597854457631e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.444597854457631e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.444597854457631e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.6849150953806694e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.6849150953806694e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.6849150953806694e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.6849150953806694e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246974425489658e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246974425489658e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246974425489658e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246974425489658e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246974425489658e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246974425489658e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246974425489658e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246974425489658e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.867765104581511e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.867765104581511e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3002946562874027e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3002946562874027e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.3609563203484036e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.3609563203484036e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.8742990714520724e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.8742990714520724e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.8836765761025975e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.8836765761025975e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.947356011678292e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.947356011678292e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.947356011678292e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.947356011678292e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117066417424e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9885117066417424e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.088250711320387e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.088250711320387e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.151346311251981e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.151346311251981e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.846201671381407e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846201671381407e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846201671381407e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846201671381407e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.0714807367133e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.0714807367133e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.105526722169295e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.105526722169295e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.105526722169295e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.105526722169295e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.1464963276688084e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.1464963276688084e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.1464963276688084e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.1464963276688084e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159350502155866e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159350502155866e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159350502155866e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159350502155866e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427988656562328e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427988656562328e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427988656562328e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427988656562328e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935867718320035e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935867718320035e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935867718320035e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935867718320035e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348363203e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.253273348363203e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979825793621367e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979825793621367e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979825793621367e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979825793621367e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.205548411219959e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.205548411219959e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.205548411219959e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.205548411219959e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00013840177303550946) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.00013840177303550946) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.00018787053389549279) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.00018787053389549279) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.00018787053389549279) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00018787053389549279) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00019400857029755602) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00019400857029755602) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0002463643756957708) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0002463643756957708) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0002463643756957708) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0002463643756957708) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192743499487624) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192743499487624) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156734248908796) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156734248908796) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156734248908796) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156734248908796) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024411) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007870896771024411) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.0015324835230730392) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.0015324835230730392) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.0015324835230730392) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.0015324835230730392) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.0016407548553124033) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0016407548553124033) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0017278753941369547) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.0017278753941369547) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.0024464971554158657) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.0024464971554158657) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.0024464971554158657) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.0024464971554158657) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.0032675138544235524) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.0032675138544235524) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.0033566705638328896) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.0033566705638328896) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.00348415730021788) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.00348415730021788) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0038764708993369356) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.0038764708993369356) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.004668620318776299) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668620318776299) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278109) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.004767272188278109) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.004767272188278109) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278109) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.005286546538226871) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.005286546538226871) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.005286546538226871) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.005286546538226871) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.005408954422409975) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.005408954422409975) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.005408954422409975) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.005408954422409975) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.005923798336561342) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561342) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.005923798336561342) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561342) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.010715508469796756) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010715508469796756) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010715508469796756) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010715508469796756) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.010757563953908953) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010757563953908953) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010757563953908953) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010757563953908953) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.014603704729162083) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.014603704729162083) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.014603704729162083) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.014603704729162083) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.019299560579363793) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363793) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363793) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363793) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363793) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363793) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.019299560579363793) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363793) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.058591988733861775) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.058591988733861775) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.775950527370345e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.775950527370345e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.775950527370348e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.775950527370348e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.07165035181002488) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07165035181002488) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0716503518100249) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0716503518100249) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251592) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019257505095251592) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010311482489831799) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311482489831799) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008826368514209818) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209818) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.007597464029770596) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.007597464029770596) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.007597464029770596) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.007597464029770596) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311867) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311867) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311867) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311867) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311867) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311867) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311867) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311867) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676609) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005348051582676609) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005348051582676609) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676609) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0038040661717285364) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0038040661717285364) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984166168121925) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.002984166168121925) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.002984166168121925) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.002984166168121925) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.0024464971554158657) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.0024464971554158657) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.002249412447093988) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.002249412447093988) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.002249412447093988) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002249412447093988) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.002141361223101596) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.002141361223101596) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.001863894282458729) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.001863894282458729) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.001863894282458729) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.001863894282458729) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.001863894282458729) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.001863894282458729) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.001863894282458729) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.001863894282458729) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0016407548553124033) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124033) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0016407548553124033) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124033) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0012223378081538349) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538349) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538349) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538349) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538349) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538349) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538349) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0012223378081538349) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0010283292378562789) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0010283292378562789) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0010283292378562789) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0010283292378562789) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1463061453081156e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1463061453081156e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.8742990714520724e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990714520724e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.8742990714520724e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990714520724e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3002946562874027e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3002946562874027e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3002946562874027e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3002946562874027e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.0444941297924185e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.0444941297924185e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.0444941297924185e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.0444941297924185e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.956079229950934e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.956079229950934e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.956079229950934e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.956079229950934e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105515036985899e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105515036985899e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105515036985899e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105515036985899e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661347213111502e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661347213111502e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661347213111502e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661347213111502e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341413842183e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341413842183e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189990975359632e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.189990975359632e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876621658207047e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876621658207047e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876621658207047e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876621658207047e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246207250776e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246207250776e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.52338967773446e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.52338967773446e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0767325318824343e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.0767325318824343e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0767325318824343e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.0767325318824343e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.0134714589391405e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0134714589391405e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.904599884082002e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.904599884082002e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.904599884082002e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.904599884082002e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.6667317548202985e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.6667317548202985e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.6667317548202985e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6667317548202985e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.850564192965037e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.850564192965037e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309320937807e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6569309320937807e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309320937807e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6569309320937807e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.850564192965037e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.850564192965037e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815440646186e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815440646186e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.6863815440646186e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815440646186e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714589391405e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.0134714589391405e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.52338967773446e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.52338967773446e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.6704023910329207e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.6704023910329207e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.6704023910329207e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.6704023910329207e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246207250776e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246207250776e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189990975359632e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.189990975359632e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341413842183e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341413842183e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949476487765427e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949476487765427e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.7924939577261044e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939577261044e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939577261044e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.7924939577261044e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.8836765761025975e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.8836765761025975e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.9885117066417424e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117066417424e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117066417424e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117066417424e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348363203e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253273348363203e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4017109735690996e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4017109735690996e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4017109735690996e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4017109735690996e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.58096036934171e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.58096036934171e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.58096036934171e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.58096036934171e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192743499487624) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487624) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192743499487624) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487624) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.0007870896771024411) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024411) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024411) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024411) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0011726348316441863) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0011726348316441863) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0011726348316441863) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0011726348316441863) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012366478019245348) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.0012366478019245348) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.0012366478019245348) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.0012366478019245348) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.002200964069500466) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002200964069500466) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002200964069500466) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002200964069500466) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980214) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980214) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0023949726397980214) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980214) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0023949726397980214) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0023949726397980214) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0023949726397980214) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0023949726397980214) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0024464971554158657) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.0024464971554158657) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.0038040661717285364) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0038040661717285364) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0038764708993369356) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.0038764708993369356) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.0038764708993369356) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.0038764708993369356) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.00422081397004646) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.00422081397004646) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.00422081397004646) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.00422081397004646) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.008826368514209818) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.008826368514209818) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.010311482489831799) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311482489831799) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257505095251592) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019257505095251592) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.05859198873386178) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.05859198873386178) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.398700901322939e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.398700901322939e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.398700901322939e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.398700901322939e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00348415730021788) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00348415730021788) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984166168121925) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.002984166168121925) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.00019400857029755602) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00019400857029755602) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453081156e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453081156e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7924939577261044e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7924939577261044e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540341413842183e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413842183e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540341413842183e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413842183e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.850564192965037e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.850564192965037e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.850564192965037e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.850564192965037e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714589391405e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0134714589391405e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714589391405e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0134714589391405e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949476487765427e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949476487765427e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.7924939577261044e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7924939577261044e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029755602) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029755602) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002984166168121925) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.002984166168121925) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.00348415730021788) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00348415730021788) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
List of core orbitals: [0, 1, 2]
List of active orbitals: [3, 4, 5, 6]
Number of qubits: 8
Number of qubits required to perform quantum simulations: 8
Hamiltonian of the water molecule
  (-73.1387323135253) [I0]
+ (-0.18066792656583341) [Z6]
+ (-0.18066792656583336) [Z7]
+ (-0.15961432501809897) [Z4]
+ (-0.15961432501809897) [Z5]
+ (0.17419956155055727) [Z3]
+ (0.17419956155055732) [Z2]
+ (0.227572690054535) [Z0]
+ (0.22757269005453507) [Z1]
+ (-8.194261372505992e-06) [Y4 Y6]
+ (-8.194261372505992e-06) [X4 X6]
+ (7.954413176424007e-06) [Y5 Y7]
+ (7.954413176424007e-06) [X5 X7]
+ (0.11270386920332191) [Z4 Z6]
+ (0.11270386920332191) [Z5 Z7]
+ (0.1195243896468266) [Z0 Z4]
+ (0.1195243896468266) [Z1 Z5]
+ (0.13401715261963704) [Z0 Z6]
+ (0.13401715261963704) [Z1 Z7]
+ (0.13734953064261302) [Z0 Z5]
+ (0.13734953064261302) [Z1 Z4]
+ (0.13766872645852551) [Z2 Z4]
+ (0.13766872645852551) [Z3 Z5]
+ (0.14138905291942788) [Z4 Z7]
+ (0.14138905291942788) [Z5 Z6]
+ (0.14722943218766144) [Z2 Z5]
+ (0.14722943218766144) [Z3 Z4]
+ (0.14926355147388862) [Z4 Z5]
+ (0.14973486803496916) [Z2 Z6]
+ (0.14973486803496916) [Z3 Z7]
+ (0.15138327161428836) [Z0 Z7]
+ (0.15138327161428836) [Z1 Z6]
+ (0.15435748657223616) [Z6 Z7]
+ (0.155822690515531) [Z2 Z7]
+ (0.155822690515531) [Z3 Z6]
+ (0.16756653265461277) [Z0 Z2]
+ (0.16756653265461277) [Z1 Z3]
+ (0.1814399144030389) [Z0 Z3]
+ (0.1814399144030389) [Z1 Z2]
+ (0.19392534613270224) [Z0 Z1]
+ (0.2200397733437609) [Z2 Z3]
+ (-7.037887510264145e-06) [Y5 Z6 Y7]
+ (-7.037887510264145e-06) [X5 Z6 X7]
+ (-7.037887510264144e-06) [Y4 Z5 Y6]
+ (-7.037887510264144e-06) [X4 Z5 X6]
+ (-0.02868518371610598) [Y4 Y5 X6 X7]
+ (-0.02868518371610598) [X4 X5 Y6 Y7]
+ (-0.0178251409957864) [Y0 Y1 X4 X5]
+ (-0.0178251409957864) [X0 X1 Y4 Y5]
+ (-0.017366118994651333) [Y0 Y1 X6 X7]
+ (-0.017366118994651333) [X0 X1 Y6 Y7]
+ (-0.013873381748426105) [Y0 Y1 X2 X3]
+ (-0.013873381748426105) [X0 X1 Y2 Y3]
+ (-0.009560705729135914) [Y2 Y3 X4 X5]
+ (-0.009560705729135914) [X2 X3 Y4 Y5]
+ (-0.006087822480561853) [Y2 Y3 X6 X7]
+ (-0.006087822480561853) [X2 X3 Y6 Y7]
+ (-0.0002921986261110571) [Y1 Y2 X3 X4]
+ (-0.0002921986261110571) [X1 X2 Y3 Y4]
+ (-8.194261372505992e-06) [Z4 Y5 Z6 Y7]
+ (-8.194261372505992e-06) [Z4 X5 Z6 X7]
+ (-2.8909678816948976e-06) [Z0 Y5 Z6 Y7]
+ (-2.8909678816948976e-06) [Z0 X5 Z6 X7]
+ (-2.8909678816948976e-06) [Z1 Y4 Z5 Y6]
+ (-2.8909678816948976e-06) [Z1 X4 Z5 X6]
+ (-1.8551201216849994e-06) [Z0 Y4 Z5 Y6]
+ (-1.8551201216849994e-06) [Z0 X4 Z5 X6]
+ (-1.8551201216849994e-06) [Z1 Y5 Z6 Y7]
+ (-1.8551201216849994e-06) [Z1 X5 Z6 X7]
+ (-1.5973171979144995e-06) [Z2 Y4 Z5 Y6]
+ (-1.5973171979144995e-06) [Z2 X4 Z5 X6]
+ (-1.5973171979144995e-06) [Z3 Y5 Z6 Y7]
+ (-1.5973171979144995e-06) [Z3 X5 Z6 X7]
+ (-1.035847760009898e-06) [Y0 X1 X5 Y6]
+ (-1.035847760009898e-06) [Y0 Y1 Y5 Y6]
+ (-1.035847760009898e-06) [X0 X1 X5 X6]
+ (-1.035847760009898e-06) [X0 Y1 Y5 X6]
+ (-9.34455777741141e-07) [Z2 Y5 Z6 Y7]
+ (-9.34455777741141e-07) [Z2 X5 Z6 X7]
+ (-9.34455777741141e-07) [Z3 Y4 Z5 Y6]
+ (-9.34455777741141e-07) [Z3 X4 Z5 X6]
+ (6.628614201733583e-07) [Y2 X3 X5 Y6]
+ (6.628614201733583e-07) [Y2 Y3 Y5 Y6]
+ (6.628614201733583e-07) [X2 X3 X5 X6]
+ (6.628614201733583e-07) [X2 Y3 Y5 X6]
+ (7.954413176424007e-06) [Y4 Z5 Y6 Z7]
+ (7.954413176424007e-06) [X4 Z5 X6 Z7]
+ (0.0002921986261110571) [Y1 X2 X3 Y4]
+ (0.0002921986261110571) [X1 Y2 Y3 X4]
+ (0.006087822480561853) [Y2 X3 X6 Y7]
+ (0.006087822480561853) [X2 Y3 Y6 X7]
+ (0.009560705729135914) [Y2 X3 X4 Y5]
+ (0.009560705729135914) [X2 Y3 Y4 X5]
+ (0.011307274008848088) [Y1 Z2 Z3 Y5]
+ (0.011307274008848088) [X1 Z2 Z3 X5]
+ (0.013873381748426105) [Y0 X1 X2 Y3]
+ (0.013873381748426105) [X0 Y1 Y2 X3]
+ (0.017366118994651333) [Y0 X1 X6 Y7]
+ (0.017366118994651333) [X0 Y1 Y6 X7]
+ (0.0178251409957864) [Y0 X1 X4 Y5]
+ (0.0178251409957864) [X0 Y1 Y4 X5]
+ (0.02868518371610598) [Y4 X5 X6 Y7]
+ (0.02868518371610598) [X4 Y5 Y6 X7]
+ (0.02981242451734572) [Y0 Z1 Z2 Y4]
+ (0.02981242451734572) [X0 Z1 Z2 X4]
+ (0.02981242451734572) [Y1 Z3 Z4 Y5]
+ (0.02981242451734572) [X1 Z3 Z4 X5]
+ (0.030104623143456775) [Y0 Z1 Z3 Y4]
+ (0.030104623143456775) [X0 Z1 Z3 X4]
+ (0.030104623143456775) [Y1 Z2 Z4 Y5]
+ (0.030104623143456775) [X1 Z2 Z4 X5]
+ (0.03078750538914391) [Y0 Z2 Z3 Y4]
+ (0.03078750538914391) [X0 Z2 Z3 X4]
+ (0.04375263801065921) [Y1 Z2 Z3 Z4 Y5]
+ (0.04375263801065921) [X1 Z2 Z3 Z4 X5]
+ (0.04375263801065923) [Y0 Z1 Z2 Z3 Y4]
+ (0.04375263801065923) [X0 Z1 Z2 Z3 X4]
+ (-0.014564531231172982) [Y1 Z2 Z3 X4 X6 Y7]
+ (-0.014564531231172982) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-0.014564531231172982) [X1 Z2 Z3 X4 X6 X7]
+ (-0.014564531231172982) [X1 Z2 Z3 Y4 Y6 X7]
+ (-6.524373848781104e-06) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (-6.524373848781104e-06) [X0 Z1 Z2 Z3 Z4 X6]
+ (-6.524373848781104e-06) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (-6.524373848781104e-06) [X1 Z2 Z3 Z5 Z6 X7]
+ (-3.7696594522754826e-06) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (-3.7696594522754826e-06) [X0 Z2 Z3 Z4 Z5 X6]
+ (-3.610297130866349e-06) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (-3.610297130866349e-06) [X0 Z1 Z3 Z4 Z5 X6]
+ (-3.610297130866349e-06) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (-3.610297130866349e-06) [X1 Z2 Z4 Z5 Z6 X7]
+ (-3.31314550014608e-06) [Y1 Z2 Z3 Y4 X5 X6]
+ (-3.31314550014608e-06) [X1 Z2 Z3 X4 Y5 Y6]
+ (-3.2774831958363642e-06) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (-3.2774831958363642e-06) [X0 Z1 Z2 Z4 Z5 X6]
+ (-3.2774831958363642e-06) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (-3.2774831958363642e-06) [X1 Z3 Z4 Z5 Z6 X7]
+ (-3.2112283486350238e-06) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (-3.2112283486350238e-06) [X0 Z1 Z2 Z3 Z5 X6]
+ (-3.2112283486350238e-06) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (-3.2112283486350238e-06) [X1 Z2 Z3 Z4 Z6 X7]
+ (-1.035847760009898e-06) [Y0 Y1 X4 Z5 Z6 X7]
+ (-1.035847760009898e-06) [X0 X1 Y4 Z5 Z6 Y7]
+ (-6.628614201733583e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (-6.628614201733583e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (-3.32813935029985e-07) [Y1 X2 X3 Z4 Z5 Y6]
+ (-3.32813935029985e-07) [X1 Y2 Y3 Z4 Z5 X6]
+ (3.32813935029985e-07) [Y1 Y2 X3 Z4 Z5 X6]
+ (3.32813935029985e-07) [X1 X2 Y3 Z4 Z5 Y6]
+ (6.628614201733583e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (6.628614201733583e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (1.035847760009898e-06) [Y0 X1 X4 Z5 Z6 Y7]
+ (1.035847760009898e-06) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.31314550014608e-06) [Y1 Z2 Z3 X4 X5 Y6]
+ (3.31314550014608e-06) [X1 Z2 Z3 Y4 Y5 X6]
+ (4.1839325594023135e-06) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (4.1839325594023135e-06) [X1 Z2 Z3 Z4 Z5 X7]
+ (0.00029219862611105707) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (0.00029219862611105707) [Y0 Z1 Y2 X3 Z4 X5]
+ (0.00029219862611105707) [X0 Z1 X2 Y3 Z4 Y5]
+ (0.00029219862611105707) [X0 Z1 X2 X3 Z4 X5]
+ (0.010540425907671468) [Y0 Z1 Z2 Z3 Y4 Z7]
+ (0.010540425907671468) [X0 Z1 Z2 Z3 X4 Z7]
+ (0.010540425907671468) [Y1 Z2 Z3 Z4 Y5 Z6]
+ (0.010540425907671468) [X1 Z2 Z3 Z4 X5 Z6]
+ (0.011307274008848088) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (0.011307274008848088) [X0 Z1 Z2 Z3 X4 Z5]
+ (0.02510495713884445) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (0.02510495713884445) [X0 Z1 Z2 Z3 X4 Z6]
+ (0.02510495713884445) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (0.02510495713884445) [X1 Z2 Z3 Z4 X5 Z7]
+ (0.03078750538914391) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (0.03078750538914391) [Z0 X1 Z2 Z3 Z4 X5]
+ (-5.105396550333697e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-5.105396550333697e-06) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-5.105396550333694e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (-5.105396550333694e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (-0.014564531231172982) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-0.014564531231172982) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-3.7696594522754826e-06) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-3.7696594522754826e-06) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-3.32813935029985e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-3.32813935029985e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-3.32813935029985e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-3.32813935029985e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (3.31314550014608e-06) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (3.31314550014608e-06) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (3.31314550014608e-06) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (3.31314550014608e-06) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (4.183932559402313e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (4.183932559402313e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
 </code>
 </pre>
 </details>

---

## 22. tutorial_gbs.html <a name="demo21"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_gbs.html):

```
0.03473293649420271
0.0059573991653360855
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_gbs.html):

```
0.0347329364942027
0.005957399165336084
```

---

