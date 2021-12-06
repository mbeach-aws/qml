Last update: 2021-12-06  03:50:59 (All times shown in Eastern time)
# List of differences in demonstration outputs

# Table of contents

1. [tutorial_jax_transformations.html](#demo0)
2. [tutorial_gbs.html](#demo1)
3. [tutorial_rosalin.html](#demo2)
4. [tutorial_quantum_natural_gradient.html](#demo3)
5. [tutorial_qubit_rotation.html](#demo4)
6. [tutorial_multiclass_classification.html](#demo5)
7. [tutorial_vqe_parallel.html](#demo6)
8. [tutorial_vqt.html](#demo7)
9. [tutorial_qaoa_intro.html](#demo8)
10. [tutorial_measurement_optimize.html](#demo9)
11. [tutorial_adaptive_circuits.html](#demo10)
12. [tutorial_general_parshift.html](#demo11)
13. [tutorial_classical_shadows.html](#demo12)
14. [tutorial_data_reuploading_classifier.html](#demo13)
15. [tutorial_expressivity_fourier_series.html](#demo14)
16. [tutorial_quantum_transfer_learning.html](#demo15)
17. [tutorial_kernel_based_training.html](#demo16)
18. [tutorial_quantum_metrology.html](#demo17)
19. [tutorial_QGAN.html](#demo18)
20. [tutorial_quantum_analytic_descent.html](#demo19)
21. [tutorial_quantum_chemistry.html](#demo20)
22. [tutorial_barren_plateaus.html](#demo21)
23. [tutorial_falqon.html](#demo22)
24. [tutorial_qgrnn.html](#demo23)
25. [tutorial_error_mitigation.html](#demo24)
26. [tutorial_ensemble_multi_qpu.html](#demo25)
27. [tutorial_quanvolution.html](#demo26)
28. [tutorial_qnn_module_tf.html](#demo27)
29. [tutorial_backprop.html](#demo28)
30. [tutorial_adjoint_diff.html](#demo29)
31. [tutorial_doubly_stochastic.html](#demo30)
32. [tutorial_unitary_designs.html](#demo31)


Number of demos different/all demos: 32/57

## 1. tutorial_jax_transformations.html <a name="demo0"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_jax_transformations.html):

```
No jit time: 0.0071 seconds
First run time: 0.0541 seconds
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_jax_transformations.html):

```
No jit time: 0.0125 seconds
First run time: 0.0912 seconds
```

---

## 2. tutorial_gbs.html <a name="demo1"></a>

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

## 3. tutorial_rosalin.html <a name="demo2"></a>

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

## 4. tutorial_quantum_natural_gradient.html <a name="demo3"></a>

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

## 5. tutorial_qubit_rotation.html <a name="demo4"></a>

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

## 6. tutorial_multiclass_classification.html <a name="demo5"></a>

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

## 7. tutorial_vqe_parallel.html <a name="demo6"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_vqe_parallel.html):

```
Speed up: 2.81
Evaluation time: 266.88 s
Evaluation time: 95.06 s
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_vqe_parallel.html):

```
Speed up: 2.93
Evaluation time: 338.15 s
Evaluation time: 115.29 s
```

---

## 8. tutorial_vqt.html <a name="demo7"></a>

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

## 9. tutorial_qaoa_intro.html <a name="demo8"></a>

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
Cost function value: -0.09248671036774045
   (-46.46390678868896) [I0]
+ (0.7829661725950203) [Z10]
+ (0.7829661725950203) [Z11]
+ (0.8084581961720477) [Z13]
+ (0.8084581961720478) [Z12]
+ (1.2034402289145651) [Z4]
+ (1.2034402289145654) [Z5]
+ (1.3096862988615443) [Z7]
+ (1.3096862988615445) [Z6]
+ (1.3693525634718209) [Z8]
+ (1.369352563471821) [Z9]
+ (1.6538942226831734) [Z3]
+ (1.6538942226831737) [Z2]
+ (12.41263074211177) [Z0]
+ (12.41263074211177) [Z1]
+ (-8.194261372485587e-06) [Y10 Y12]
+ (-8.194261372485587e-06) [X10 X12]
+ (-1.8540608578151607e-06) [Y5 Y7]
+ (-1.8540608578151607e-06) [X5 X7]
+ (-7.764994119518015e-07) [Y3 Y5]
+ (-7.764994119518015e-07) [X3 X5]
+ (-5.92976581460238e-07) [Y4 Y6]
+ (-5.92976581460238e-07) [X4 X6]
+ (1.6021167405448002e-06) [Y2 Y4]
+ (1.6021167405448002e-06) [X2 X4]
+ (7.954413176386776e-06) [Y11 Y13]
+ (7.954413176386776e-06) [X11 X13]
+ (0.0032769719312316483) [Y1 Y3]
+ (0.0032769719312316483) [X1 X3]
+ (0.10433064780651374) [Y0 Y2]
+ (0.10433064780651374) [X0 X2]
+ (0.11270386920332187) [Z10 Z12]
+ (0.11270386920332187) [Z11 Z13]
+ (0.11383573679388628) [Z4 Z12]
+ (0.11383573679388628) [Z5 Z13]
+ (0.11952438964682657) [Z6 Z10]
+ (0.11952438964682657) [Z7 Z11]
+ (0.1248999091723758) [Z4 Z10]
+ (0.1248999091723758) [Z5 Z11]
+ (0.12495807739503201) [Z2 Z4]
+ (0.12495807739503201) [Z3 Z5]
+ (0.12799502492468384) [Z2 Z10]
+ (0.12799502492468384) [Z3 Z11]
+ (0.13401715261963673) [Z6 Z12]
+ (0.13401715261963673) [Z7 Z13]
+ (0.13701191674040736) [Z4 Z6]
+ (0.13701191674040736) [Z5 Z7]
+ (0.13734953064261313) [Z6 Z11]
+ (0.13734953064261313) [Z7 Z10]
+ (0.13739104762683202) [Z2 Z6]
+ (0.13739104762683202) [Z3 Z7]
+ (0.13766872645852565) [Z8 Z10]
+ (0.13766872645852565) [Z9 Z11]
+ (0.1401128986535477) [Z2 Z12]
+ (0.1401128986535477) [Z3 Z13]
+ (0.14138905291942772) [Z10 Z13]
+ (0.14138905291942772) [Z11 Z12]
+ (0.1425799771248573) [Z4 Z11]
+ (0.1425799771248573) [Z5 Z10]
+ (0.14722943218766157) [Z8 Z11]
+ (0.14722943218766157) [Z9 Z10]
+ (0.1489943057506553) [Z4 Z7]
+ (0.1489943057506553) [Z5 Z6]
+ (0.14926355147388873) [Z10 Z11]
+ (0.14960702684445287) [Z4 Z8]
+ (0.14960702684445287) [Z5 Z9]
+ (0.149734868034969) [Z8 Z12]
+ (0.149734868034969) [Z9 Z13]
+ (0.15071408121008273) [Z2 Z8]
+ (0.15071408121008273) [Z3 Z9]
+ (0.1513832716142881) [Z6 Z13]
+ (0.1513832716142881) [Z7 Z12]
+ (0.15215040708869007) [Z4 Z13]
+ (0.15215040708869007) [Z5 Z12]
+ (0.15337968243314115) [Z2 Z11]
+ (0.15337968243314115) [Z3 Z10]
+ (0.15435748657223577) [Z12 Z13]
+ (0.15569010671752415) [Z2 Z13]
+ (0.15569010671752415) [Z3 Z12]
+ (0.15582269051553083) [Z8 Z13]
+ (0.15582269051553083) [Z9 Z12]
+ (0.1567639617643098) [Z4 Z9]
+ (0.1567639617643098) [Z5 Z8]
+ (0.1575531479798564) [Z4 Z5]
+ (0.16079764534838537) [Z2 Z5]
+ (0.16079764534838537) [Z3 Z4]
+ (0.16756653265461255) [Z6 Z8]
+ (0.16756653265461255) [Z7 Z9]
+ (0.16853486561579917) [Z2 Z7]
+ (0.16853486561579917) [Z3 Z6]
+ (0.1814399144030386) [Z6 Z9]
+ (0.1814399144030386) [Z7 Z8]
+ (0.18189085790751328) [Z2 Z3]
+ (0.1869082047691254) [Z2 Z9]
+ (0.1869082047691254) [Z3 Z8]
+ (0.19299723935364196) [Z0 Z10]
+ (0.19299723935364196) [Z1 Z11]
+ (0.1939253461327016) [Z6 Z7]
+ (0.19661770890342112) [Z0 Z4]
+ (0.19661770890342112) [Z1 Z5]
+ (0.19936354537360793) [Z0 Z5]
+ (0.19936354537360793) [Z1 Z4]
+ (0.20072866460441724) [Z0 Z11]
+ (0.20072866460441724) [Z1 Z10]
+ (0.21102659849791452) [Z0 Z12]
+ (0.21102659849791452) [Z1 Z13]
+ (0.21631037498631747) [Z0 Z13]
+ (0.21631037498631747) [Z1 Z12]
+ (0.23671080783830387) [Z0 Z2]
+ (0.23671080783830387) [Z1 Z3]
+ (0.24164663936017153) [Z0 Z6]
+ (0.24164663936017153) [Z1 Z7]
+ (0.24853483371314206) [Z0 Z7]
+ (0.24853483371314206) [Z1 Z6]
+ (0.25129445674591644) [Z0 Z3]
+ (0.25129445674591644) [Z1 Z2]
+ (0.2723251830660565) [Z0 Z8]
+ (0.2723251830660565) [Z1 Z9]
+ (0.27883454426723375) [Z0 Z9]
+ (0.27883454426723375) [Z1 Z8]
+ (1.1861763734860475) [Z0 Z1]
+ (-1.2260484988222452e-05) [Y5 Z6 Y7]
+ (-1.2260484988222452e-05) [X5 Z6 X7]
+ (-1.2260484988222449e-05) [Y4 Z5 Y6]
+ (-1.2260484988222449e-05) [X4 Z5 X6]
+ (-1.0722312158171624e-05) [Y11 Z12 Y13]
+ (-1.0722312158171624e-05) [X11 Z12 X13]
+ (-1.0722312158171623e-05) [Y10 Z11 Y12]
+ (-1.0722312158171623e-05) [X10 Z11 X12]
+ (-3.887051672369519e-06) [Y2 Z3 Y4]
+ (-3.887051672369519e-06) [X2 Z3 X4]
+ (-3.887051672369517e-06) [Y3 Z4 Y5]
+ (-3.887051672369517e-06) [X3 Z4 X5]
+ (0.12507032579771774) [Y1 Z2 Y3]
+ (0.12507032579771774) [X1 Z2 X3]
+ (0.12507032579771782) [Y0 Z1 Y2]
+ (0.12507032579771782) [X0 Z1 X2]
+ (-0.038314670294803795) [Y4 Y5 X12 X13]
+ (-0.038314670294803795) [X4 X5 Y12 Y13]
+ (-0.03619412355904264) [Y2 Y3 X8 X9]
+ (-0.03619412355904264) [X2 X3 Y8 Y9]
+ (-0.035839567953353364) [Y2 Y3 X4 X5]
+ (-0.035839567953353364) [X2 X3 Y4 Y5]
+ (-0.031143817988967155) [Y2 Y3 X6 X7]
+ (-0.031143817988967155) [X2 X3 Y6 Y7]
+ (-0.028685183716105844) [Y10 Y11 X12 X13]
+ (-0.028685183716105844) [X10 X11 Y12 Y13]
+ (-0.025996177598021128) [Y3 Z4 Z5 Y7]
+ (-0.025996177598021128) [X3 Z4 Z5 X7]
+ (-0.025384657508457302) [Y2 Y3 X10 X11]
+ (-0.025384657508457302) [X2 X3 Y10 Y11]
+ (-0.019028242443847175) [Y3 Y4 X11 X12]
+ (-0.019028242443847175) [X3 X4 Y11 Y12]
+ (-0.017825140995786536) [Y6 Y7 X10 X11]
+ (-0.017825140995786536) [X6 X7 Y10 Y11]
+ (-0.017680067952481473) [Y4 Y5 X10 X11]
+ (-0.017680067952481473) [X4 X5 Y10 Y11]
+ (-0.01736611899465137) [Y6 Y7 X12 X13]
+ (-0.01736611899465137) [X6 X7 Y12 Y13]
+ (-0.015577208063976418) [Y2 Y3 X12 X13]
+ (-0.015577208063976418) [X2 X3 Y12 Y13]
+ (-0.014583648907612613) [Y0 Y1 X2 X3]
+ (-0.014583648907612613) [X0 X1 Y2 Y3]
+ (-0.013873381748426063) [Y6 Y7 X8 X9]
+ (-0.013873381748426063) [X6 X7 Y8 Y9]
+ (-0.011982389010247962) [Y4 Y5 X6 X7]
+ (-0.011982389010247962) [X4 X5 Y6 Y7]
+ (-0.011285190200840919) [Y5 X6 X11 Y12]
+ (-0.011285190200840919) [X5 Y6 Y11 X12]
+ (-0.009560705729135921) [Y8 Y9 X10 X11]
+ (-0.009560705729135921) [X8 X9 Y10 Y11]
+ (-0.008125251921381015) [Y1 X2 X8 Y9]
+ (-0.008125251921381015) [Y1 Y2 Y8 Y9]
+ (-0.008125251921381015) [X1 X2 X8 X9]
+ (-0.008125251921381015) [X1 Y2 Y8 X9]
+ (-0.007731425250775249) [Y0 Y1 X10 X11]
+ (-0.007731425250775249) [X0 X1 Y10 Y11]
+ (-0.006888194352970534) [Y0 Y1 X6 X7]
+ (-0.006888194352970534) [X0 X1 Y6 Y7]
+ (-0.006509361201177221) [Y0 Y1 X8 X9]
+ (-0.006509361201177221) [X0 X1 Y8 Y9]
+ (-0.006087822480561846) [Y8 Y9 X12 X13]
+ (-0.006087822480561846) [X8 X9 Y12 Y13]
+ (-0.005283776488402935) [Y0 Y1 X12 X13]
+ (-0.005283776488402935) [X0 X1 Y12 Y13]
+ (-0.005143391768825131) [Y3 X4 X5 Y6]
+ (-0.005143391768825131) [X3 Y4 Y5 X6]
+ (-0.0046849033881552005) [Y1 X2 X6 Y7]
+ (-0.0046849033881552005) [Y1 Y2 Y6 Y7]
+ (-0.0046849033881552005) [X1 X2 X6 X7]
+ (-0.0046849033881552005) [X1 Y2 Y6 X7]
+ (-0.004575007626639192) [Y1 X2 X12 Y13]
+ (-0.004575007626639192) [Y1 Y2 Y12 Y13]
+ (-0.004575007626639192) [X1 X2 X12 X13]
+ (-0.004575007626639192) [X1 Y2 Y12 X13]
+ (-0.004424855449441842) [Y1 X2 X4 Y5]
+ (-0.004424855449441842) [Y1 Y2 Y4 Y5]
+ (-0.004424855449441842) [X1 X2 X4 X5]
+ (-0.004424855449441842) [X1 Y2 Y4 X5]
+ (-0.0034795118903343837) [Y2 Z3 Z5 Y6]
+ (-0.0034795118903343837) [X2 Z3 Z5 X6]
+ (-0.0034795118903343837) [Y3 Z4 Z6 Y7]
+ (-0.0034795118903343837) [X3 Z4 Z6 X7]
+ (-0.002745836470186804) [Y0 Y1 X4 X5]
+ (-0.002745836470186804) [X0 X1 Y4 Y5]
+ (-0.0017992194936630218) [Y1 X2 X10 Y11]
+ (-0.0017992194936630218) [Y1 Y2 Y10 Y11]
+ (-0.0017992194936630218) [X1 X2 X10 X11]
+ (-0.0017992194936630218) [X1 Y2 Y10 X11]
+ (-0.000292198626111024) [Y7 Y8 X9 X10]
+ (-0.000292198626111024) [X7 X8 Y9 Y10]
+ (-8.194261372485587e-06) [Z10 Y11 Z12 Y13]
+ (-8.194261372485587e-06) [Z10 X11 Z12 X13]
+ (-7.801707500830777e-06) [Y2 Z3 Y4 Z11]
+ (-7.801707500830777e-06) [X2 Z3 X4 Z11]
+ (-7.801707500830777e-06) [Y3 Z4 Y5 Z10]
+ (-7.801707500830777e-06) [X3 Z4 X5 Z10]
+ (-4.643051068683205e-06) [Y3 X4 X10 Y11]
+ (-4.643051068683205e-06) [Y3 Y4 Y10 Y11]
+ (-4.643051068683205e-06) [X3 X4 X10 X11]
+ (-4.643051068683205e-06) [X3 Y4 Y10 X11]
+ (-4.588855155752813e-06) [Y4 Z5 Y6 Z13]
+ (-4.588855155752813e-06) [X4 Z5 X6 Z13]
+ (-4.588855155752813e-06) [Y5 Z6 Y7 Z12]
+ (-4.588855155752813e-06) [X5 Z6 X7 Z12]
+ (-4.556569218370057e-06) [Y5 X6 X12 Y13]
+ (-4.556569218370057e-06) [Y5 Y6 Y12 Y13]
+ (-4.556569218370057e-06) [X5 X6 X12 X13]
+ (-4.556569218370057e-06) [X5 Y6 Y12 X13]
+ (-3.694513294757279e-06) [Y4 X5 X11 Y12]
+ (-3.694513294757279e-06) [Y4 Y5 Y11 Y12]
+ (-3.694513294757279e-06) [X4 X5 X11 X12]
+ (-3.694513294757279e-06) [X4 Y5 Y11 X12]
+ (-3.3440815562007276e-06) [Z0 Y5 Z6 Y7]
+ (-3.3440815562007276e-06) [Z0 X5 Z6 X7]
+ (-3.3440815562007276e-06) [Z1 Y4 Z5 Y6]
+ (-3.3440815562007276e-06) [Z1 X4 Z5 X6]
+ (-3.158656432147571e-06) [Y2 Z3 Y4 Z10]
+ (-3.158656432147571e-06) [X2 Z3 X4 Z10]
+ (-3.158656432147571e-06) [Y3 Z4 Y5 Z11]
+ (-3.158656432147571e-06) [X3 Z4 X5 Z11]
+ (-3.099349243347665e-06) [Z0 Y4 Z5 Y6]
+ (-3.099349243347665e-06) [Z0 X4 Z5 X6]
+ (-3.099349243347665e-06) [Z1 Y5 Z6 Y7]
+ (-3.099349243347665e-06) [Z1 X5 Z6 X7]
+ (-2.890967881696525e-06) [Z6 Y11 Z12 Y13]
+ (-2.890967881696525e-06) [Z6 X11 Z12 X13]
+ (-2.890967881696525e-06) [Z7 Y10 Z11 Y12]
+ (-2.890967881696525e-06) [Z7 X10 Z11 X12]
+ (-2.1776646052315735e-06) [Z0 Y10 Z11 Y12]
+ (-2.1776646052315735e-06) [Z0 X10 Z11 X12]
+ (-2.1776646052315735e-06) [Z1 Y11 Z12 Y13]
+ (-2.1776646052315735e-06) [Z1 X11 Z12 X13]
+ (-1.881850183055012e-06) [Y4 Z5 Y6 Z9]
+ (-1.881850183055012e-06) [X4 Z5 X6 Z9]
+ (-1.881850183055012e-06) [Y5 Z6 Y7 Z8]
+ (-1.881850183055012e-06) [X5 Z6 X7 Z8]
+ (-1.8551201217069513e-06) [Z6 Y10 Z11 Y12]
+ (-1.8551201217069513e-06) [Z6 X10 Z11 X12]
+ (-1.8551201217069513e-06) [Z7 Y11 Z12 Y13]
+ (-1.8551201217069513e-06) [Z7 X11 Z12 X13]
+ (-1.8540608578151607e-06) [Y4 Z5 Y6 Z7]
+ (-1.8540608578151607e-06) [X4 Z5 X6 Z7]
+ (-1.8163031700001848e-06) [Z4 Y11 Z12 Y13]
+ (-1.8163031700001848e-06) [Z4 X11 Z12 X13]
+ (-1.8163031700001848e-06) [Z5 Y10 Z11 Y12]
+ (-1.8163031700001848e-06) [Z5 X10 Z11 X12]
+ (-1.6923978286291378e-06) [Y4 Z5 Y6 Z10]
+ (-1.6923978286291378e-06) [X4 Z5 X6 Z10]
+ (-1.6923978286291378e-06) [Y5 Z6 Y7 Z11]
+ (-1.6923978286291378e-06) [X5 Z6 X7 Z11]
+ (-1.6148794140877315e-06) [Z0 Y11 Z12 Y13]
+ (-1.6148794140877315e-06) [Z0 X11 Z12 X13]
+ (-1.6148794140877315e-06) [Z1 Y10 Z11 Y12]
+ (-1.6148794140877315e-06) [Z1 X10 Z11 X12]
+ (-1.5973171979345975e-06) [Z8 Y10 Z11 Y12]
+ (-1.5973171979345975e-06) [Z8 X10 Z11 X12]
+ (-1.5973171979345975e-06) [Z9 Y11 Z12 Y13]
+ (-1.5973171979345975e-06) [Z9 X11 Z12 X13]
+ (-1.4548424488772719e-06) [Y3 X4 X6 Y7]
+ (-1.4548424488772719e-06) [Y3 Y4 Y6 Y7]
+ (-1.4548424488772719e-06) [X3 X4 X6 X7]
+ (-1.4548424488772719e-06) [X3 Y4 Y6 X7]
+ (-1.3980449080111195e-06) [Y4 Z5 Y6 Z8]
+ (-1.3980449080111195e-06) [X4 Z5 X6 Z8]
+ (-1.3980449080111195e-06) [Y5 Z6 Y7 Z9]
+ (-1.3980449080111195e-06) [X5 Z6 X7 Z9]
+ (-1.1954890097877183e-06) [Y2 Z3 Y4 Z7]
+ (-1.1954890097877183e-06) [X2 Z3 X4 Z7]
+ (-1.1954890097877183e-06) [Y3 Z4 Y5 Z6]
+ (-1.1954890097877183e-06) [X3 Z4 X5 Z6]
+ (-1.1908508081602683e-06) [Z0 Y3 Z4 Y5]
+ (-1.1908508081602683e-06) [Z0 X3 Z4 X5]
+ (-1.1908508081602683e-06) [Z1 Y2 Z3 Y4]
+ (-1.1908508081602683e-06) [Z1 X2 Z3 X4]
+ (-1.1708301369564683e-06) [Z2 Y5 Z6 Y7]
+ (-1.1708301369564683e-06) [Z2 X5 Z6 X7]
+ (-1.1708301369564683e-06) [Z3 Y4 Z5 Y6]
+ (-1.1708301369564683e-06) [Z3 X4 Z5 X6]
+ (-1.063228342516987e-06) [Z2 Y10 Z11 Y12]
+ (-1.063228342516987e-06) [Z2 X10 Z11 X12]
+ (-1.063228342516987e-06) [Z3 Y11 Z12 Y13]
+ (-1.063228342516987e-06) [Z3 X11 Z12 X13]
+ (-1.0358477599895736e-06) [Y6 X7 X11 Y12]
+ (-1.0358477599895736e-06) [Y6 Y7 Y11 Y12]
+ (-1.0358477599895736e-06) [X6 X7 X11 X12]
+ (-1.0358477599895736e-06) [X6 Y7 Y11 X12]
+ (-9.509249750808989e-07) [Z2 Y4 Z5 Y6]
+ (-9.509249750808989e-07) [Z2 X4 Z5 X6]
+ (-9.509249750808989e-07) [Z3 Y5 Z6 Y7]
+ (-9.509249750808989e-07) [Z3 X5 Z6 X7]
+ (-9.344557777585756e-07) [Z8 Y11 Z12 Y13]
+ (-9.344557777585756e-07) [Z8 X11 Z12 X13]
+ (-9.344557777585756e-07) [Z9 Y10 Z11 Y12]
+ (-9.344557777585756e-07) [Z9 X10 Z11 X12]
+ (-8.337746752968682e-07) [Z0 Y2 Z3 Y4]
+ (-8.337746752968682e-07) [Z0 X2 Z3 X4]
+ (-8.337746752968682e-07) [Z1 Y3 Z4 Y5]
+ (-8.337746752968682e-07) [Z1 X3 Z4 X5]
+ (-7.956895371629258e-07) [Y3 X4 X8 Y9]
+ (-7.956895371629258e-07) [Y3 Y4 Y8 Y9]
+ (-7.956895371629258e-07) [X3 X4 X8 X9]
+ (-7.956895371629258e-07) [X3 Y4 Y8 X9]
+ (-7.764994119518016e-07) [Y2 Z3 Y4 Z5]
+ (-7.764994119518016e-07) [X2 Z3 X4 Z5]
+ (-5.92976581460238e-07) [Z4 Y5 Z6 Y7]
+ (-5.92976581460238e-07) [Z4 X5 Z6 X7]
+ (-5.770052994187334e-07) [Y2 Z3 Y4 Z9]
+ (-5.770052994187334e-07) [X2 Z3 X4 Z9]
+ (-5.770052994187334e-07) [Y3 Z4 Y5 Z8]
+ (-5.770052994187334e-07) [X3 Z4 X5 Z8]
+ (-5.471647744770911e-07) [Y1 Y2 X11 X12]
+ (-5.471647744770911e-07) [X1 X2 Y11 Y12]
+ (-4.838052750438925e-07) [Y5 X6 X8 Y9]
+ (-4.838052750438925e-07) [Y5 Y6 Y8 Y9]
+ (-4.838052750438925e-07) [X5 X6 X8 X9]
+ (-4.838052750438925e-07) [X5 Y6 Y8 X9]
+ (-3.5707613286340023e-07) [Y0 X1 X3 Y4]
+ (-3.5707613286340023e-07) [Y0 Y1 Y3 Y4]
+ (-3.5707613286340023e-07) [X0 X1 X3 X4]
+ (-3.5707613286340023e-07) [X0 Y1 Y3 X4]
+ (-2.447323128530622e-07) [Y0 X1 X5 Y6]
+ (-2.447323128530622e-07) [Y0 Y1 Y5 Y6]
+ (-2.447323128530622e-07) [X0 X1 X5 X6]
+ (-2.447323128530622e-07) [X0 Y1 Y5 X6]
+ (-2.1990516187556934e-07) [Y2 X3 X5 Y6]
+ (-2.1990516187556934e-07) [Y2 Y3 Y5 Y6]
+ (-2.1990516187556934e-07) [X2 X3 X5 X6]
+ (-2.1990516187556934e-07) [X2 Y3 Y5 X6]
+ (-1.933241276878263e-07) [Y1 X2 X3 Y4]
+ (-1.933241276878263e-07) [X1 Y2 Y3 X4]
+ (-1.2919694862814895e-07) [Y1 Z2 Z3 Y5]
+ (-1.2919694862814895e-07) [X1 Z2 Z3 X5]
+ (1.7379332621532424e-07) [Y0 Z1 Z3 Y4]
+ (1.7379332621532424e-07) [X0 Z1 Z3 X4]
+ (1.7379332621532424e-07) [Y1 Z2 Z4 Y5]
+ (1.7379332621532424e-07) [X1 Z2 Z4 X5]
+ (1.933241276878263e-07) [Y1 Y2 X3 X4]
+ (1.933241276878263e-07) [X1 X2 Y3 Y4]
+ (2.1868423774419244e-07) [Y2 Z3 Y4 Z8]
+ (2.1868423774419244e-07) [X2 Z3 X4 Z8]
+ (2.1868423774419244e-07) [Y3 Z4 Y5 Z9]
+ (2.1868423774419244e-07) [X3 Z4 X5 Z9]
+ (2.593534390895537e-07) [Y2 Z3 Y4 Z6]
+ (2.593534390895537e-07) [X2 Z3 X4 Z6]
+ (2.593534390895537e-07) [Y3 Z4 Y5 Z7]
+ (2.593534390895537e-07) [X3 Z4 X5 Z7]
+ (3.606071867453839e-07) [Y0 Z1 Z2 Y4]
+ (3.606071867453839e-07) [X0 Z1 Z2 X4]
+ (3.606071867453839e-07) [Y1 Z3 Z4 Y5]
+ (3.606071867453839e-07) [X1 Z3 Z4 X5]
+ (5.471647744770911e-07) [Y1 X2 X11 Y12]
+ (5.471647744770911e-07) [X1 Y2 Y11 X12]
+ (5.627851911438421e-07) [Y0 X1 X11 Y12]
+ (5.627851911438421e-07) [Y0 Y1 Y11 Y12]
+ (5.627851911438421e-07) [X0 X1 X11 X12]
+ (5.627851911438421e-07) [X0 Y1 Y11 X12]
+ (6.628614201760218e-07) [Y8 X9 X11 Y12]
+ (6.628614201760218e-07) [Y8 Y9 Y11 Y12]
+ (6.628614201760218e-07) [X8 X9 X11 X12]
+ (6.628614201760218e-07) [X8 Y9 Y11 X12]
+ (1.1094407589891013e-06) [Z2 Y11 Z12 Y13]
+ (1.1094407589891013e-06) [Z2 X11 Z12 X13]
+ (1.1094407589891013e-06) [Z3 Y10 Z11 Y12]
+ (1.1094407589891013e-06) [Z3 X10 Z11 X12]
+ (1.6021167405448002e-06) [Z2 Y3 Z4 Y5]
+ (1.6021167405448002e-06) [Z2 X3 Z4 X5]
+ (1.878210124757094e-06) [Z4 Y10 Z11 Y12]
+ (1.878210124757094e-06) [Z4 X10 Z11 X12]
+ (1.878210124757094e-06) [Z5 Y11 Z12 Y13]
+ (1.878210124757094e-06) [Z5 X11 Z12 X13]
+ (2.1726691015060882e-06) [Y2 X3 X11 Y12]
+ (2.1726691015060882e-06) [Y2 Y3 Y11 Y12]
+ (2.1726691015060882e-06) [X2 X3 X11 X12]
+ (2.1726691015060882e-06) [X2 Y3 Y11 X12]
+ (3.1174479456611886e-06) [Y0 Z2 Z3 Y4]
+ (3.1174479456611886e-06) [X0 Z2 Z3 X4]
+ (3.5390541845867367e-06) [Y2 Z3 Y4 Z12]
+ (3.5390541845867367e-06) [X2 Z3 X4 Z12]
+ (3.5390541845867367e-06) [Y3 Z4 Y5 Z13]
+ (3.5390541845867367e-06) [X3 Z4 X5 Z13]
+ (4.281913884942222e-06) [Y4 Z5 Y6 Z11]
+ (4.281913884942222e-06) [X4 Z5 X6 Z11]
+ (4.281913884942222e-06) [Y5 Z6 Y7 Z10]
+ (4.281913884942222e-06) [X5 Z6 X7 Z10]
+ (5.275883122252184e-06) [Y3 X4 X12 Y13]
+ (5.275883122252184e-06) [Y3 Y4 Y12 Y13]
+ (5.275883122252184e-06) [X3 X4 X12 X13]
+ (5.275883122252184e-06) [X3 Y4 Y12 X13]
+ (5.974311713571361e-06) [Y5 X6 X10 Y11]
+ (5.974311713571361e-06) [Y5 Y6 Y10 Y11]
+ (5.974311713571361e-06) [X5 X6 X10 X11]
+ (5.974311713571361e-06) [X5 Y6 Y10 X11]
+ (7.954413176386776e-06) [Y10 Z11 Y12 Z13]
+ (7.954413176386776e-06) [X10 Z11 X12 Z13]
+ (8.81493730683892e-06) [Y2 Z3 Y4 Z13]
+ (8.81493730683892e-06) [X2 Z3 X4 Z13]
+ (8.81493730683892e-06) [Y3 Z4 Y5 Z12]
+ (8.81493730683892e-06) [X3 Z4 X5 Z12]
+ (0.000292198626111024) [Y7 X8 X9 Y10]
+ (0.000292198626111024) [X7 Y8 Y9 X10]
+ (0.000495676231491526) [Y2 Z4 Z5 Y6]
+ (0.000495676231491526) [X2 Z4 Z5 X6]
+ (0.001105903769189679) [Y0 Z1 Y2 Z5]
+ (0.001105903769189679) [X0 Z1 X2 Z5]
+ (0.001105903769189679) [Y1 Z2 Y3 Z4]
+ (0.001105903769189679) [X1 Z2 X3 Z4]
+ (0.0016638798784907474) [Y2 Z3 Z4 Y6]
+ (0.0016638798784907474) [X2 Z3 Z4 X6]
+ (0.0016638798784907474) [Y3 Z5 Z6 Y7]
+ (0.0016638798784907474) [X3 Z5 Z6 X7]
+ (0.0017560707018412314) [Y0 Z1 Y2 Z11]
+ (0.0017560707018412314) [X0 Z1 X2 Z11]
+ (0.0017560707018412314) [Y1 Z2 Y3 Z10]
+ (0.0017560707018412314) [X1 Z2 X3 Z10]
+ (0.0023262306231580637) [Y0 Z1 Y2 Z13]
+ (0.0023262306231580637) [X0 Z1 X2 Z13]
+ (0.0023262306231580637) [Y1 Z2 Y3 Z12]
+ (0.0023262306231580637) [X1 Z2 X3 Z12]
+ (0.002745836470186804) [Y0 X1 X4 Y5]
+ (0.002745836470186804) [X0 Y1 Y4 X5]
+ (0.002929768674751039) [Y0 Z1 Y2 Z9]
+ (0.002929768674751039) [X0 Z1 X2 Z9]
+ (0.002929768674751039) [Y1 Z2 Y3 Z8]
+ (0.002929768674751039) [X1 Z2 X3 Z8]
+ (0.003276971931231648) [Y0 Z1 Y2 Z3]
+ (0.003276971931231648) [X0 Z1 X2 Z3]
+ (0.0033476175306661575) [Y0 Z1 Y2 Z7]
+ (0.0033476175306661575) [X0 Z1 X2 Z7]
+ (0.0033476175306661575) [Y1 Z2 Y3 Z6]
+ (0.0033476175306661575) [X1 Z2 X3 Z6]
+ (0.003555290195504253) [Y0 Z1 Y2 Z10]
+ (0.003555290195504253) [X0 Z1 X2 Z10]
+ (0.003555290195504253) [Y1 Z2 Y3 Z11]
+ (0.003555290195504253) [X1 Z2 X3 Z11]
+ (0.005143391768825131) [Y3 Y4 X5 X6]
+ (0.005143391768825131) [X3 X4 Y5 Y6]
+ (0.005283776488402935) [Y0 X1 X12 Y13]
+ (0.005283776488402935) [X0 Y1 Y12 X13]
+ (0.005530759218631522) [Y0 Z1 Y2 Z4]
+ (0.005530759218631522) [X0 Z1 X2 Z4]
+ (0.005530759218631522) [Y1 Z2 Y3 Z5]
+ (0.005530759218631522) [X1 Z2 X3 Z5]
+ (0.006087822480561846) [Y8 X9 X12 Y13]
+ (0.006087822480561846) [X8 Y9 Y12 X13]
+ (0.006509361201177221) [Y0 X1 X8 Y9]
+ (0.006509361201177221) [X0 Y1 Y8 X9]
+ (0.006888194352970534) [Y0 X1 X6 Y7]
+ (0.006888194352970534) [X0 Y1 Y6 X7]
+ (0.006901238249797256) [Y0 Z1 Y2 Z12]
+ (0.006901238249797256) [X0 Z1 X2 Z12]
+ (0.006901238249797256) [Y1 Z2 Y3 Z13]
+ (0.006901238249797256) [X1 Z2 X3 Z13]
+ (0.007731425250775249) [Y0 X1 X10 Y11]
+ (0.007731425250775249) [X0 Y1 Y10 X11]
+ (0.008032520918821357) [Y0 Z1 Y2 Z6]
+ (0.008032520918821357) [X0 Z1 X2 Z6]
+ (0.008032520918821357) [Y1 Z2 Y3 Z7]
+ (0.008032520918821357) [X1 Z2 X3 Z7]
+ (0.009560705729135921) [Y8 X9 X10 Y11]
+ (0.009560705729135921) [X8 Y9 Y10 X11]
+ (0.011055020596132054) [Y0 Z1 Y2 Z8]
+ (0.011055020596132054) [X0 Z1 X2 Z8]
+ (0.011055020596132054) [Y1 Z2 Y3 Z9]
+ (0.011055020596132054) [X1 Z2 X3 Z9]
+ (0.011285190200840919) [Y5 Y6 X11 X12]
+ (0.011285190200840919) [X5 X6 Y11 Y12]
+ (0.01130727400884822) [Y7 Z8 Z9 Y11]
+ (0.01130727400884822) [X7 Z8 Z9 X11]
+ (0.011982389010247962) [Y4 X5 X6 Y7]
+ (0.011982389010247962) [X4 Y5 Y6 X7]
+ (0.013873381748426063) [Y6 X7 X8 Y9]
+ (0.013873381748426063) [X6 Y7 Y8 X9]
+ (0.014583648907612613) [Y0 X1 X2 Y3]
+ (0.014583648907612613) [X0 Y1 Y2 X3]
+ (0.015577208063976418) [Y2 X3 X12 Y13]
+ (0.015577208063976418) [X2 Y3 Y12 X13]
+ (0.01736611899465137) [Y6 X7 X12 Y13]
+ (0.01736611899465137) [X6 Y7 Y12 X13]
+ (0.017680067952481473) [Y4 X5 X10 Y11]
+ (0.017680067952481473) [X4 Y5 Y10 X11]
+ (0.017825140995786536) [Y6 X7 X10 Y11]
+ (0.017825140995786536) [X6 Y7 Y10 X11]
+ (0.019028242443847175) [Y3 X4 X11 Y12]
+ (0.019028242443847175) [X3 Y4 Y11 X12]
+ (0.025384657508457302) [Y2 X3 X10 Y11]
+ (0.025384657508457302) [X2 Y3 Y10 X11]
+ (0.028685183716105844) [Y10 X11 X12 Y13]
+ (0.028685183716105844) [X10 Y11 Y12 X13]
+ (0.02981242451734588) [Y6 Z7 Z8 Y10]
+ (0.02981242451734588) [X6 Z7 Z8 X10]
+ (0.02981242451734588) [Y7 Z9 Z10 Y11]
+ (0.02981242451734588) [X7 Z9 Z10 X11]
+ (0.030104623143456903) [Y6 Z7 Z9 Y10]
+ (0.030104623143456903) [X6 Z7 Z9 X10]
+ (0.030104623143456903) [Y7 Z8 Z10 Y11]
+ (0.030104623143456903) [X7 Z8 Z10 X11]
+ (0.030787505389143967) [Y6 Z8 Z9 Y10]
+ (0.030787505389143967) [X6 Z8 Z9 X10]
+ (0.031143817988967155) [Y2 X3 X6 Y7]
+ (0.031143817988967155) [X2 Y3 Y6 X7]
+ (0.035839567953353364) [Y2 X3 X4 Y5]
+ (0.035839567953353364) [X2 Y3 Y4 X5]
+ (0.03619412355904264) [Y2 X3 X8 Y9]
+ (0.03619412355904264) [X2 Y3 Y8 X9]
+ (0.038314670294803795) [Y4 X5 X12 Y13]
+ (0.038314670294803795) [X4 Y5 Y12 X13]
+ (0.10433064780651374) [Z0 Y1 Z2 Y3]
+ (0.10433064780651374) [Z0 X1 Z2 X3]
+ (-0.121332769110423) [Y3 Z4 Z5 Z6 Y7]
+ (-0.121332769110423) [X3 Z4 Z5 Z6 X7]
+ (-0.12133276911042296) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042296) [X2 Z3 Z4 Z5 X6]
+ (3.202076879242851e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.202076879242851e-06) [X0 Z1 Z2 Z3 X4]
+ (3.2020768792428514e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.2020768792428514e-06) [X1 Z2 Z3 Z4 X5]
+ (0.22848106564918902) [Y6 Z7 Z8 Z9 Y10]
+ (0.22848106564918902) [X6 Z7 Z8 Z9 X10]
+ (0.22848106564918905) [Y7 Z8 Z9 Z10 Y11]
+ (0.22848106564918905) [X7 Z8 Z9 Z10 X11]
+ (-0.032767657823290476) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.032767657823290476) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.032767657823290476) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.032767657823290476) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273156) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.027115036845273156) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273156) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.027115036845273156) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.025996177598021128) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.025996177598021128) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.01756120240964618) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.01756120240964618) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.01756120240964618) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.01756120240964618) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.014564531231172986) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564531231172986) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564531231172986) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564531231172986) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.01221504099761397) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.01221504099761397) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.01221504099761397) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.01221504099761397) [X4 Z5 X6 X11 Z12 X13]
+ (-0.01221504099761397) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.01221504099761397) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.01221504099761397) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.01221504099761397) [X5 Z6 X7 X10 Z11 X12]
+ (-0.011756013419819225) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.011756013419819225) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.011756013419819225) [X3 Z4 Z5 X6 X8 X9]
+ (-0.011756013419819225) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.008764827575688708) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.008764827575688708) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.008764827575688708) [X2 Z3 Z4 X5 X11 X12]
+ (-0.008764827575688708) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.008764827575688708) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.008764827575688708) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.008764827575688708) [X3 X4 X10 Z11 Z12 X13]
+ (-0.008764827575688708) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.008125251921381015) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125251921381015) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.007306759928832967) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306759928832967) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306759928832967) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306759928832967) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.0058051889898269576) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.0058051889898269576) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.0058051889898269576) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.0058051889898269576) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.00565262097801732) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.00565262097801732) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.00565262097801732) [X0 X1 X3 Z4 Z5 X6]
+ (-0.00565262097801732) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.005143391768825132) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.005143391768825132) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.005143391768825132) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.005143391768825132) [X2 Z3 X4 X5 Z6 X7]
+ (-0.0046849033881552005) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.0046849033881552005) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668620318776283) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668620318776283) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.004575007626639192) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.004575007626639192) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424855449441842) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424855449441842) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.00415879738184005) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.00415879738184005) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.00415879738184005) [X3 Z4 Z5 X6 X12 X13]
+ (-0.00415879738184005) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.0034937903598901495) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.0034937903598901495) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.0034937903598901495) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.0034937903598901495) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.0027790267990255315) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.0027790267990255315) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.0022939566113524467) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.0022939566113524467) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.0017992194936630218) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.0017992194936630218) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278753941369596) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.0017278753941369596) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298507967730515) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298507967730515) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298507967730515) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298507967730515) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533856254125415) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533856254125415) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.000814531327095692) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.000814531327095692) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.000814531327095692) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.000814531327095692) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.735036880588771e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735036880588771e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735036880588771e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735036880588771e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.774817864934127e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.774817864934127e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.774817864934127e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.774817864934127e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.51836221594971e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.51836221594971e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.51836221594971e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.51836221594971e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.444344676250107e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.444344676250107e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.444344676250107e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.444344676250107e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.524373848760506e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.524373848760506e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.524373848760506e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.524373848760506e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.2900284336064e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.2900284336064e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.2900284336064e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.2900284336064e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.97431171357136e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.97431171357136e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.275883122252185e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.275883122252185e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.643051068683206e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.643051068683206e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.556569218370057e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.556569218370057e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.253224225622314e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.253224225622314e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.7696594522891148e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.7696594522891148e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.694513294757278e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.694513294757278e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.6102971308882076e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.6102971308882076e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.6102971308882076e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.6102971308882076e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.3131455001208568e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.3131455001208568e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.277483195855993e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.277483195855993e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.277483195855993e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.277483195855993e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.2112283486396486e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.2112283486396486e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.2112283486396486e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.2112283486396486e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.151346311259942e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.151346311259942e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.088250711480361e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.088250711480361e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.1726691015060882e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.1726691015060882e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.454842448877272e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.454842448877272e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.3304731886840206e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.3304731886840206e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2283337823433098e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2283337823433098e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.0358477599895738e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.0358477599895738e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956895371629258e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956895371629258e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733197743006314e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733197743006314e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733197743006314e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733197743006314e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.628614201760219e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.628614201760219e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.556281914752206e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.556281914752206e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.556281914752206e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.556281914752206e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.418291574800411e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.418291574800411e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.418291574800411e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.418291574800411e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.927453083363105e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927453083363105e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927453083363105e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927453083363105e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627851911438421e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627851911438421e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660624837242e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660624837242e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660624837242e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660624837242e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660624837242e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660624837242e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660624837242e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660624837242e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.838052750438925e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.838052750438925e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.5707613286340023e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.5707613286340023e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.3281393503221495e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.3281393503221495e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.0868265649124696e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.0868265649124696e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.0868265649124696e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.0868265649124696e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.447323128530622e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.447323128530622e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.3713289476497472e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.3713289476497472e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.3713289476497472e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.3713289476497472e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.1990516187556934e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.1990516187556934e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.933241276878263e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.933241276878263e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.933241276878263e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.933241276878263e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.8394209152435983e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.8394209152435983e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.8394209152435983e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.8394209152435983e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.5510539175179712e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.5510539175179712e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.5510539175179712e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.5510539175179712e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.3807781478948845e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.3807781478948845e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.3807781478948845e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.3807781478948845e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.3807781478948845e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.3807781478948845e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.3807781478948845e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.3807781478948845e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.3807781478948845e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.3807781478948845e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.3807781478948845e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.3807781478948845e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.2919694862814893e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.2919694862814893e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.1076325599722316e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.1076325599722316e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.1076325599722316e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.1076325599722316e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.1076325599722316e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.1076325599722316e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.1076325599722316e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.1076325599722316e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.05744659643209e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.05744659643209e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.05744659643209e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.05744659643209e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649310131993454e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649310131993454e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649310131993454e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649310131993454e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.8394209152435983e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.8394209152435983e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.8394209152435983e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.8394209152435983e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.1990516187556934e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.1990516187556934e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.447323128530622e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.447323128530622e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.2362599608490926e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.2362599608490926e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.2362599608490926e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.2362599608490926e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.3281393503221495e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.3281393503221495e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.5707613286340023e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.5707613286340023e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.838052750438925e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.838052750438925e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.627851911438421e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627851911438421e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.628614201760219e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.628614201760219e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956895371629258e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956895371629258e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.306536651947176e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.306536651947176e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.306536651947176e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.306536651947176e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.0358477599895738e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.0358477599895738e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2283337823433098e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2283337823433098e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.2393363216859646e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393363216859646e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393363216859646e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393363216859646e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.3304731886840206e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.3304731886840206e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.454842448877272e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.454842448877272e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.1726691015060882e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.1726691015060882e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.088250711480361e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.088250711480361e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.1174479456611886e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.1174479456611886e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.151346311259942e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.151346311259942e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.3131455001208568e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.3131455001208568e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.3343312894905984e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.3343312894905984e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.694513294757278e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.694513294757278e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.183932559389879e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.183932559389879e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.556569218370057e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.556569218370057e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.643051068683206e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.643051068683206e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.275883122252185e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.275883122252185e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.97431171357136e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.97431171357136e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.000292198626111024) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.000292198626111024) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.000292198626111024) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.000292198626111024) [X6 Z7 X8 X9 Z10 X11]
+ (0.000495676231491526) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.000495676231491526) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650070219499004) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650070219499004) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650070219499004) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650070219499004) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533856254125415) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533856254125415) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.0016095313817213667) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.0016095313817213667) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.0016095313817213667) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.0016095313817213667) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0016676041811440434) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.0016676041811440434) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.0016676041811440434) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.0016676041811440434) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.0017278753941369596) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.0017278753941369596) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.0017992194936630218) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.0017992194936630218) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.0022939566113524467) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.0022939566113524467) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.0024629170071339083) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.0024629170071339083) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.0024629170071339083) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.0024629170071339083) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.0039615607924964906) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.0039615607924964906) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.0039615607924964906) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.0039615607924964906) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.004424855449441842) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424855449441842) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004575007626639192) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.004575007626639192) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668620318776283) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668620318776283) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.0046849033881552005) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.0046849033881552005) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005324835234221677) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.005324835234221677) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.005324835234221677) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.005324835234221677) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.00536865935810959) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.00536865935810959) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.00536865935810959) [X2 X3 X7 Z8 Z9 X10]
+ (0.00536865935810959) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.00796088072592156) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.00796088072592156) [X4 Z5 X6 X10 Z11 X12]
+ (0.00796088072592156) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.00796088072592156) [X5 Z6 X7 X11 Z12 X13]
+ (0.008125251921381015) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125251921381015) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.008890731522694612) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.008890731522694612) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.008890731522694612) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.008890731522694612) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.010263414868158469) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.010263414868158469) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.010263414868158469) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.010263414868158469) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.010540425907671559) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.010540425907671559) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.010540425907671559) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.010540425907671559) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010960074940542646) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010960074940542646) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010960074940542646) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010960074940542646) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307274008848222) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307274008848222) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.014411099430130855) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411099430130855) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411099430130855) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411099430130855) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.015225630757226548) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.015225630757226548) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.015225630757226548) [X3 Z4 Z5 X6 X10 X11]
+ (0.015225630757226548) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.015588250102380144) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.015588250102380144) [X2 Z3 X4 X10 Z11 X12]
+ (0.015588250102380144) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.015588250102380144) [X3 Z4 X5 X11 Z12 X13]
+ (0.018266834869375612) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.018266834869375612) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.018266834869375612) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.018266834869375612) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.019020423173040018) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.019020423173040018) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.019020423173040018) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.019020423173040018) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.020175921723535533) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.020175921723535533) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.020175921723535533) [X4 Z5 Z6 X7 X11 X12]
+ (0.020175921723535533) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.020175921723535533) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.020175921723535533) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.020175921723535533) [X5 X6 X10 Z11 Z12 X13]
+ (0.020175921723535533) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.024353077678068852) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.024353077678068852) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.024353077678068852) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.024353077678068852) [X2 Z3 X4 X11 Z12 X13]
+ (0.024353077678068852) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.024353077678068852) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.024353077678068852) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.024353077678068852) [X3 Z4 X5 X10 Z11 X12]
+ (0.024389082531149606) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.024389082531149606) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.024389082531149606) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.024389082531149606) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.025104957138844544) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.025104957138844544) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.025104957138844544) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.025104957138844544) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.030787505389143967) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.030787505389143967) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.045879470781297796) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.045879470781297796) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.05600733087780778) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.05600733087780778) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.05600733087780778) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.05600733087780778) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661366) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.05608468124661366) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661366) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.05608468124661366) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.631277928746282e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631277928746282e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-6.63127792874628e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.63127792874628e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.5950860072116487e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.5950860072116487e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.595086007211648e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.595086007211648e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.042743277013782235) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.042743277013782235) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (0.042743277013782235) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.042743277013782235) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.047642612176383034) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.047642612176383034) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.047642612176383034) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.047642612176383034) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.041718813839821706) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.041718813839821706) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.041718813839821706) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.041718813839821706) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.03956441632289321) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.03956441632289321) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.03956441632289321) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03956441632289321) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.039359168022052964) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.039359168022052964) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.039359168022052964) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.039359168022052964) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.039318051947197466) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.039318051947197466) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.039318051947197466) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.039318051947197466) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03560837898831244) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.03560837898831244) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.029903789512624762) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.029903789512624762) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.029903789512624762) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.029903789512624762) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.028730779551905443) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.028730779551905443) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.028730779551905443) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.028730779551905443) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.025637238296026786) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.025637238296026786) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.025637238296026786) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.025637238296026786) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.024755463292890887) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.024755463292890887) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.024755463292890887) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.024755463292890887) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.024282117354692948) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.024282117354692948) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.023145130929528947) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.023145130929528947) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.02252844019601298) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.02252844019601298) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.021433810721600804) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.021433810721600804) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.021433810721600804) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.021433810721600804) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.01925750509525157) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.01925750509525157) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019028242443847175) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.019028242443847175) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.01888903030494287) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.01888903030494287) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.01888903030494287) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.01888903030494287) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.01602460368917951) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.01602460368917951) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.015225630757226548) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225630757226548) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.014603704729162078) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.014603704729162078) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564531231172986) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564531231172986) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.011756013419819222) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.011756013419819222) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.011285190200840919) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.011285190200840919) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.009841749246962572) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009841749246962572) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.009612634606847276) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612634606847276) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612634606847276) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612634606847276) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.00846997879102396) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.00846997879102396) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.007306759928832965) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.007306759928832965) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005923798336561336) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.005923798336561336) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.00565262097801732) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.00565262097801732) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.00536865935810959) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.00536865935810959) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.00415879738184005) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.00415879738184005) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.003356670563832872) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.003356670563832872) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.003356670563832872) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.003356670563832872) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.0032675138544235307) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.0032675138544235307) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.0032675138544235307) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.0032675138544235307) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.0027790267990255315) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.0027790267990255315) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.0026860409778066076) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.0026860409778066076) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.0026860409778066076) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.0026860409778066076) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.0022939566113524467) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524467) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.0022939566113524467) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524467) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.000958165583669648) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.000958165583669648) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.000958165583669648) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.000958165583669648) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.000958165583669648) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.000958165583669648) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.000958165583669648) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.000958165583669648) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.0002463643756957401) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0002463643756957401) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0001384017730355096) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.0001384017730355096) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.0001384017730355096) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.0001384017730355096) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.735036880588772e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735036880588772e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.610358530638892e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.610358530638892e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.610358530638892e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.610358530638892e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.53168087959637e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.53168087959637e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.53168087959637e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.53168087959637e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.80610277545787e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.80610277545787e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.80610277545787e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.80610277545787e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.089799467767865e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.089799467767865e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.089799467767865e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.089799467767865e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.652209670110596e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.652209670110596e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.652209670110596e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.652209670110596e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851834589264e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.481851834589264e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851834589264e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.481851834589264e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.0714807365072805e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.0714807365072805e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.0714807365072805e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.0714807365072805e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.734622038950589e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.734622038950589e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.734622038950589e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.734622038950589e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.728843147386222e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.728843147386222e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.728843147386222e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.728843147386222e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.253224225622314e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.253224225622314e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.7696594522891148e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.7696594522891148e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.5443954293996286e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.5443954293996286e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.5443954293996286e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.5443954293996286e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.5443954293996286e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.5443954293996286e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.5443954293996286e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.5443954293996286e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.360956320381643e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.360956320381643e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.360956320381643e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.360956320381643e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.1032156048966584e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.1032156048966584e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.1032156048966584e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.1032156048966584e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220985332145e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.0111220985332145e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220985332145e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.0111220985332145e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9429468367932876e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.9429468367932876e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9429468367932876e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9429468367932876e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.654117477382965e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.654117477382965e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.654117477382965e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.654117477382965e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.5224930676695425e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224930676695425e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224930676695425e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224930676695425e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224930676695425e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930676695425e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930676695425e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224930676695425e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.2283337823433098e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337823433098e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.2283337823433098e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337823433098e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.988770287680579e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988770287680579e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988770287680579e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988770287680579e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.867765104252238e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104252238e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104252238e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.867765104252238e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.189990975667829e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.189990975667829e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246207116615e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246207116615e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.471647744770911e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.471647744770911e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.5614471791926815e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.5614471791926815e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.5614471791926815e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.5614471791926815e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.5233896783569917e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.5233896783569917e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.427323108487896e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.427323108487896e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.427323108487896e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.427323108487896e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.3281393503221495e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.3281393503221495e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.3281393503221495e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.3281393503221495e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.08682656491247e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.08682656491247e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.888293594103228e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.888293594103228e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.888293594103228e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.888293594103228e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.3713289476497472e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.3713289476497472e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.839420915243598e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.839420915243598e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.057446596432091e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057446596432091e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.5371780957876354e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.5371780957876354e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.5371780957876354e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.5371780957876354e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.057446596432091e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057446596432091e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.209350636344388e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350636344388e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350636344388e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.209350636344388e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783552133232e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783552133232e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783552133232e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783552133232e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.839420915243598e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.839420915243598e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.3713289476497472e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.3713289476497472e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.08682656491247e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.08682656491247e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.5233896783569917e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.5233896783569917e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.471647744770911e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.471647744770911e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246207116615e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246207116615e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189990975667829e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.189990975667829e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.3304731886840206e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.3304731886840206e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.3304731886840206e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.3304731886840206e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288532435903998e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288532435903998e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288532435903998e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288532435903998e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.6893489515327425e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.6893489515327425e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.6893489515327425e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.6893489515327425e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.745518400631571e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.745518400631571e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.745518400631571e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.745518400631571e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.745518400631571e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.745518400631571e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.745518400631571e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.745518400631571e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.211842019202284e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.211842019202284e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.211842019202284e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.211842019202284e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.211842019202284e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.211842019202284e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.211842019202284e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.211842019202284e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.3131455001208568e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.3131455001208568e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.3131455001208568e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.3131455001208568e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.3343312894905984e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.3343312894905984e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.183932559389879e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.183932559389879e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.735036880588772e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735036880588772e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.0002463643756957401) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.0002463643756957401) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00044585351288409323) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.00044585351288409323) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.00044585351288409323) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.00044585351288409323) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940221543005467) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005467) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005467) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005467) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005467) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005467) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940221543005467) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005467) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533856254125415) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533856254125415) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533856254125415) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533856254125415) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0010435246534907486) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435246534907486) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435246534907486) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435246534907486) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.001280306097349658) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.001280306097349658) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.001280306097349658) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.001280306097349658) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.0013038004788126889) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.0013038004788126889) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.0013038004788126889) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.0013038004788126889) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.0022619660624823364) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.0022619660624823364) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.0022619660624823364) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.0022619660624823364) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.0022619660624823364) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.0022619660624823364) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.0022619660624823364) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.0022619660624823364) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.003989841456619296) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.003989841456619296) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.003989841456619296) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.003989841456619296) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.00415879738184005) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.00415879738184005) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.004311038507914279) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.004311038507914279) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.004311038507914279) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.004311038507914279) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.004636976661182531) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.004636976661182531) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.004636976661182531) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.004636976661182531) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.005114473831660377) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.005114473831660377) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.005114473831660377) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.005114473831660377) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.005114473831660377) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660377) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660377) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005114473831660377) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005241535382803839) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.005241535382803839) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.005241535382803839) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.005241535382803839) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.00536865935810959) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.00536865935810959) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.00537993715583935) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.00537993715583935) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.00537993715583935) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.00537993715583935) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.00565262097801732) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.00565262097801732) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.005708495985960923) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708495985960923) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708495985960923) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708495985960923) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.005923798336561336) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.005923798336561336) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.007306759928832965) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.007306759928832965) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00846997879102396) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.00846997879102396) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.009841749246962572) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.009841749246962572) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.011285190200840919) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.011285190200840919) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.011756013419819222) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.011756013419819222) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.014564531231172986) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564531231172986) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603704729162078) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.014603704729162078) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.015225630757226548) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225630757226548) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.01602460368917951) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.01602460368917951) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.019028242443847175) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.019028242443847175) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.01925750509525157) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.01925750509525157) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.045879470781297796) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.045879470781297796) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.36937089366156073) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.36937089366156073) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.36937089366156073) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.36937089366156073) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.28164257767022816) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.28164257767022816) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.281642577670228) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.281642577670228) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.09065144207036457) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.09065144207036457) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.09065144207036457) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.09065144207036457) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863605) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684737589863605) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863605) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684737589863605) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635021950634979) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635021950634979) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635021950634979) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635021950634979) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099213999) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.06752385099213999) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099213999) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.06752385099213999) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03560837898831244) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.03560837898831244) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.034903343373661785) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.034903343373661785) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.034903343373661785) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.034903343373661785) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883829957) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.024591860883829957) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883829957) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.024591860883829957) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.024282117354692948) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.024282117354692948) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.023145130929528947) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.023145130929528947) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.02252844019601298) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.02252844019601298) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019538050311314628) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.019538050311314628) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.019538050311314628) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.019538050311314628) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.017091553155898786) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.017091553155898786) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.017091553155898786) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.017091553155898786) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.01602460368917951) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.01602460368917951) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.01602460368917951) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.01602460368917951) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.010311482489831826) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831826) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831826) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311482489831826) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.00984174924696257) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.00984174924696257) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.00984174924696257) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.00984174924696257) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0088263685142098) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0088263685142098) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0088263685142098) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0088263685142098) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008541996625454802) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454802) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454802) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454802) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454802) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454802) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454802) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008541996625454802) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00846997879102396) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.00846997879102396) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.00846997879102396) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.00846997879102396) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.004668620318776283) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668620318776283) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.0038764708993369286) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.0038764708993369286) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.003804066171728528) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728528) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728528) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003804066171728528) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178756) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178756) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0033566705638328723) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.0033566705638328723) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.0032675138544235307) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.0032675138544235307) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0021413612231015646) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.0021413612231015646) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.0017278753941369596) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.0017278753941369596) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.0016407548553124026) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0016407548553124026) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.001452884321416867) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.001452884321416867) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.001452884321416867) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.001452884321416867) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870896771024409) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007870896771024409) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.000519274349948766) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.000519274349948766) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.00019400857029756573) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00019400857029756573) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0001384017730355096) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.0001384017730355096) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.141625221153509e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.141625221153509e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.141625221153509e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.141625221153509e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.0714807365072805e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.0714807365072805e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.151346311259942e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.151346311259942e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.088250711480361e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.088250711480361e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.9885117061192624e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.9885117061192624e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.874299071437731e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.874299071437731e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.360956320381643e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.360956320381643e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.300294656273165e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.300294656273165e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.146837650816845e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.146837650816845e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.146837650816845e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.146837650816845e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.352332103658246e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.352332103658246e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.352332103658246e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.352332103658246e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091637199635416e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637199635416e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637199635416e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637199635416e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637199635416e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637199635416e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637199635416e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091637199635416e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.074305986399847e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.074305986399847e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.074305986399847e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.074305986399847e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128986890025e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128986890025e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128986890025e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128986890025e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.86776510425224e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.86776510425224e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560692465349859e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692465349859e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692465349859e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692465349859e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692465349859e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692465349859e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692465349859e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560692465349859e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.997018422356406e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.997018422356406e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.997018422356406e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.997018422356406e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.997018422356406e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.997018422356406e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.997018422356406e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.997018422356406e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.568247521278426e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.568247521278426e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.568247521278426e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.568247521278426e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393085330326e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393085330326e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393085330326e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393085330326e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393085330326e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393085330326e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.3767393085330326e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393085330326e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.8882935941032275e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.8882935941032275e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.686381547764047e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.686381547764047e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035783552133232e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7035783552133232e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.209350636344386e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.209350636344386e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773245202018e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773245202018e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773245202018e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773245202018e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773245202018e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773245202018e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773245202018e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773245202018e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253798833156e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253798833156e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9742253798833156e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9742253798833156e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.04747165569875e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.04747165569875e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.04747165569875e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.04747165569875e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.209350636344386e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.209350636344386e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0717282186146623e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0717282186146623e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0717282186146623e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0717282186146623e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.2004287493950709e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.2004287493950709e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.2004287493950709e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.2004287493950709e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.7035783552133232e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7035783552133232e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.3120943053155623e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.3120943053155623e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.3120943053155623e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.3120943053155623e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.686381547764047e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.686381547764047e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.8882935941032275e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.8882935941032275e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.0922506161634585e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506161634585e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.0922506161634585e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506161634585e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.0922506161634585e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506161634585e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.0922506161634585e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506161634585e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.4445978539187716e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.4445978539187716e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.4445978539187716e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.4445978539187716e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.684915095208954e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.684915095208954e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.684915095208954e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.684915095208954e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246974425731278e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246974425731278e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246974425731278e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246974425731278e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246974425731278e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246974425731278e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246974425731278e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246974425731278e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.86776510425224e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.86776510425224e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.300294656273165e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.300294656273165e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.360956320381643e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.360956320381643e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.874299071437731e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.874299071437731e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.883676576063036e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.883676576063036e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.947356011854092e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.947356011854092e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.947356011854092e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.947356011854092e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117061192624e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9885117061192624e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.088250711480361e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.088250711480361e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.151346311259942e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.151346311259942e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.846201671324577e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846201671324577e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846201671324577e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846201671324577e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.0714807365072805e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.0714807365072805e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.1055267220314374e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.1055267220314374e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.1055267220314374e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.1055267220314374e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146496327597741e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146496327597741e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146496327597741e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146496327597741e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.15935050186543e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.15935050186543e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.15935050186543e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.15935050186543e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427988656641834e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427988656641834e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427988656641834e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427988656641834e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935867717973354e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935867717973354e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935867717973354e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935867717973354e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348152003e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.253273348152003e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979825793469168e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979825793469168e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979825793469168e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979825793469168e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.2055484112157523e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.2055484112157523e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.2055484112157523e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.2055484112157523e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0001384017730355096) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.0001384017730355096) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.00018787053389553567) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.00018787053389553567) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.00018787053389553567) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00018787053389553567) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00019400857029756573) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00019400857029756573) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0002463643756957401) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0002463643756957401) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0002463643756957401) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0002463643756957401) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.000519274349948766) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.000519274349948766) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.000715673424890906) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.000715673424890906) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.000715673424890906) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.000715673424890906) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024409) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007870896771024409) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.0015324835230730587) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.0015324835230730587) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.0015324835230730587) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.0015324835230730587) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.0016407548553124026) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0016407548553124026) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0017278753941369596) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.0017278753941369596) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.0024464971554158405) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.0024464971554158405) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.0024464971554158405) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.0024464971554158405) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.0032675138544235307) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.0032675138544235307) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.0033566705638328723) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.0033566705638328723) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.0034841573002178756) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0034841573002178756) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0038764708993369286) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.0038764708993369286) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.004668620318776283) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668620318776283) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.00476727218827811) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.00476727218827811) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.00476727218827811) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.00476727218827811) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.005286546538226875) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.005286546538226875) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.005286546538226875) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.005286546538226875) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.005408954422409987) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.005408954422409987) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.005408954422409987) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.005408954422409987) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.005923798336561336) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561336) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.005923798336561336) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561336) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.010715508469796768) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010715508469796768) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010715508469796768) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010715508469796768) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.010757563953908924) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010757563953908924) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010757563953908924) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010757563953908924) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.014603704729162078) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.014603704729162078) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.014603704729162078) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.014603704729162078) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.019299560579363724) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363724) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363724) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363724) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363724) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363724) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.019299560579363724) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363724) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.058591988733861664) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.058591988733861664) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.775950527296244e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.775950527296244e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.775950527296247e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.775950527296247e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.07165035181002552) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07165035181002552) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0716503518100256) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0716503518100256) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.01925750509525157) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.01925750509525157) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010311482489831826) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311482489831826) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008826368514209799) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209799) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0075974640297705774) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0075974640297705774) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0075974640297705774) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0075974640297705774) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311853) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311853) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311853) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311853) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311853) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311853) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311853) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311853) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676601) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005348051582676601) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005348051582676601) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676601) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.003804066171728528) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003804066171728528) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219113) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.0029841661681219113) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.0029841661681219113) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.0029841661681219113) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.0024464971554158405) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.0024464971554158405) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.0022494124470939765) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0022494124470939765) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022494124470939765) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0022494124470939765) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0021413612231015646) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.0021413612231015646) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.0018638942824587253) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587253) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587253) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587253) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587253) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587253) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0018638942824587253) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587253) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0016407548553124026) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124026) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0016407548553124026) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124026) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0012223378081538253) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538253) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538253) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538253) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538253) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538253) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538253) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0012223378081538253) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0010283292378562596) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0010283292378562596) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0010283292378562596) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0010283292378562596) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1463061453223644e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1463061453223644e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.874299071437731e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.874299071437731e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.874299071437731e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.874299071437731e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.300294656273165e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.300294656273165e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.300294656273165e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.300294656273165e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.0444941298258399e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.0444941298258399e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.0444941298258399e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.0444941298258399e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.956079230209606e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.956079230209606e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.956079230209606e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.956079230209606e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.10551503749072e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.10551503749072e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.10551503749072e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.10551503749072e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661347213333026e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661347213333026e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661347213333026e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661347213333026e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341413853733e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341413853733e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189990975667829e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.189990975667829e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876621658486701e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876621658486701e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876621658486701e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876621658486701e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246207116615e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246207116615e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.5233896783569917e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.5233896783569917e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0767325318853127e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.0767325318853127e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0767325318853127e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.0767325318853127e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.013471459014464e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.013471459014464e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.904599884404668e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.904599884404668e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.904599884404668e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.904599884404668e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.6667317545443066e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.6667317545443066e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.6667317545443066e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6667317545443066e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.850564192718885e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.850564192718885e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309313884595e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6569309313884595e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309313884595e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6569309313884595e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.850564192718885e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.850564192718885e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.686381547764047e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.686381547764047e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.686381547764047e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.686381547764047e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471459014464e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.013471459014464e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.5233896783569917e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.5233896783569917e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.6704023904029235e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.6704023904029235e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.6704023904029235e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.6704023904029235e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246207116615e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246207116615e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189990975667829e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.189990975667829e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341413853733e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341413853733e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949476487466726e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949476487466726e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.7924939577237062e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939577237062e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939577237062e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.7924939577237062e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.883676576063036e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.883676576063036e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.988511706119263e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.988511706119263e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.988511706119263e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.988511706119263e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348152003e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253273348152003e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4017109735392597e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4017109735392597e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4017109735392597e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4017109735392597e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603693116307e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.5809603693116307e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603693116307e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.5809603693116307e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.000519274349948766) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.000519274349948766) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.000519274349948766) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.000519274349948766) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.0007870896771024409) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024409) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024409) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024409) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0011726348316441848) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0011726348316441848) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0011726348316441848) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0011726348316441848) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012366478019245524) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.0012366478019245524) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.0012366478019245524) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.0012366478019245524) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.002200964069500444) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002200964069500444) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002200964069500444) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002200964069500444) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980097) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980097) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0023949726397980097) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980097) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0023949726397980097) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0023949726397980097) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0023949726397980097) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0023949726397980097) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0024464971554158405) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.0024464971554158405) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.003804066171728528) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.003804066171728528) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0038764708993369286) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.0038764708993369286) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.0038764708993369286) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.0038764708993369286) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.004220813970046463) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.004220813970046463) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.004220813970046463) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.004220813970046463) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.008826368514209799) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.008826368514209799) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.010311482489831826) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311482489831826) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01925750509525157) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01925750509525157) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.058591988733861664) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.058591988733861664) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3987009014835785e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.3987009014835785e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.3987009014835785e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3987009014835785e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178756) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178756) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219113) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.0029841661681219113) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.00019400857029756573) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00019400857029756573) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453223644e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453223644e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7924939577237062e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7924939577237062e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540341413853733e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413853733e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540341413853733e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413853733e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.850564192718885e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.850564192718885e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.850564192718885e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.850564192718885e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471459014464e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471459014464e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471459014464e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471459014464e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949476487466726e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949476487466726e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.7924939577237062e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7924939577237062e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756573) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756573) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0029841661681219113) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.0029841661681219113) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.0034841573002178756) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0034841573002178756) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
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
   (-46.46390678868899) [I0]
+ (0.7829661725950189) [Z10]
+ (0.782966172595019) [Z11]
+ (0.8084581961720478) [Z12]
+ (0.8084581961720481) [Z13]
+ (1.203440228914562) [Z4]
+ (1.2034402289145623) [Z5]
+ (1.3096862988615434) [Z6]
+ (1.3096862988615439) [Z7]
+ (1.3693525634718169) [Z8]
+ (1.3693525634718173) [Z9]
+ (1.6538942226831692) [Z3]
+ (1.6538942226831694) [Z2]
+ (12.412630742111773) [Z0]
+ (12.412630742111773) [Z1]
+ (-8.194261372184309e-06) [Y10 Y12]
+ (-8.194261372184309e-06) [X10 X12]
+ (-1.8540608579270555e-06) [Y5 Y7]
+ (-1.8540608579270555e-06) [X5 X7]
+ (-7.764994118628142e-07) [Y3 Y5]
+ (-7.764994118628142e-07) [X3 X5]
+ (-5.929765815862276e-07) [Y4 Y6]
+ (-5.929765815862276e-07) [X4 X6]
+ (1.6021167405555709e-06) [Y2 Y4]
+ (1.6021167405555709e-06) [X2 X4]
+ (7.95441317617081e-06) [Y11 Y13]
+ (7.95441317617081e-06) [X11 X13]
+ (0.003276971931231598) [Y1 Y3]
+ (0.003276971931231598) [X1 X3]
+ (0.104330647806514) [Y0 Y2]
+ (0.104330647806514) [X0 X2]
+ (0.11270386920332218) [Z10 Z12]
+ (0.11270386920332218) [Z11 Z13]
+ (0.11383573679388659) [Z4 Z12]
+ (0.11383573679388659) [Z5 Z13]
+ (0.11952438964682689) [Z6 Z10]
+ (0.11952438964682689) [Z7 Z11]
+ (0.12489990917237606) [Z4 Z10]
+ (0.12489990917237606) [Z5 Z11]
+ (0.12495807739503208) [Z2 Z4]
+ (0.12495807739503208) [Z3 Z5]
+ (0.1279950249246841) [Z2 Z10]
+ (0.1279950249246841) [Z3 Z11]
+ (0.13401715261963706) [Z6 Z12]
+ (0.13401715261963706) [Z7 Z13]
+ (0.13701191674040755) [Z4 Z6]
+ (0.13701191674040755) [Z5 Z7]
+ (0.13734953064261324) [Z6 Z11]
+ (0.13734953064261324) [Z7 Z10]
+ (0.13739104762683219) [Z2 Z6]
+ (0.13739104762683219) [Z3 Z7]
+ (0.1376687264585258) [Z8 Z10]
+ (0.1376687264585258) [Z9 Z11]
+ (0.140112898653548) [Z2 Z12]
+ (0.140112898653548) [Z3 Z13]
+ (0.1413890529194281) [Z10 Z13]
+ (0.1413890529194281) [Z11 Z12]
+ (0.14257997712485762) [Z4 Z11]
+ (0.14257997712485762) [Z5 Z10]
+ (0.14722943218766177) [Z8 Z11]
+ (0.14722943218766177) [Z9 Z10]
+ (0.14899430575065545) [Z4 Z7]
+ (0.14899430575065545) [Z5 Z6]
+ (0.14926355147388923) [Z10 Z11]
+ (0.14960702684445298) [Z4 Z8]
+ (0.14960702684445298) [Z5 Z9]
+ (0.14973486803496927) [Z8 Z12]
+ (0.14973486803496927) [Z9 Z13]
+ (0.15071408121008278) [Z2 Z8]
+ (0.15071408121008278) [Z3 Z9]
+ (0.15138327161428847) [Z6 Z13]
+ (0.15138327161428847) [Z7 Z12]
+ (0.15215040708869043) [Z4 Z13]
+ (0.15215040708869043) [Z5 Z12]
+ (0.15337968243314157) [Z2 Z11]
+ (0.15337968243314157) [Z3 Z10]
+ (0.15435748657223625) [Z12 Z13]
+ (0.15569010671752445) [Z2 Z13]
+ (0.15569010671752445) [Z3 Z12]
+ (0.1558226905155311) [Z8 Z13]
+ (0.1558226905155311) [Z9 Z12]
+ (0.1567639617643099) [Z4 Z9]
+ (0.1567639617643099) [Z5 Z8]
+ (0.15755314797985664) [Z4 Z5]
+ (0.16079764534838556) [Z2 Z5]
+ (0.16079764534838556) [Z3 Z4]
+ (0.16756653265461277) [Z6 Z8]
+ (0.16756653265461277) [Z7 Z9]
+ (0.16853486561579922) [Z2 Z7]
+ (0.16853486561579922) [Z3 Z6]
+ (0.18143991440303894) [Z6 Z9]
+ (0.18143991440303894) [Z7 Z8]
+ (0.18189085790751341) [Z2 Z3]
+ (0.18690820476912529) [Z2 Z9]
+ (0.18690820476912529) [Z3 Z8]
+ (0.19299723935364257) [Z0 Z10]
+ (0.19299723935364257) [Z1 Z11]
+ (0.19392534613270243) [Z6 Z7]
+ (0.19661770890342153) [Z0 Z4]
+ (0.19661770890342153) [Z1 Z5]
+ (0.19936354537360834) [Z0 Z5]
+ (0.19936354537360834) [Z1 Z4]
+ (0.2007286646044179) [Z0 Z11]
+ (0.2007286646044179) [Z1 Z10]
+ (0.21102659849791527) [Z0 Z12]
+ (0.21102659849791527) [Z1 Z13]
+ (0.21631037498631822) [Z0 Z13]
+ (0.21631037498631822) [Z1 Z12]
+ (0.23671080783830403) [Z0 Z2]
+ (0.23671080783830403) [Z1 Z3]
+ (0.24164663936017244) [Z0 Z6]
+ (0.24164663936017244) [Z1 Z7]
+ (0.24853483371314306) [Z0 Z7]
+ (0.24853483371314306) [Z1 Z6]
+ (0.25129445674591666) [Z0 Z3]
+ (0.25129445674591666) [Z1 Z2]
+ (0.272325183066057) [Z0 Z8]
+ (0.272325183066057) [Z1 Z9]
+ (0.2788345442672343) [Z0 Z9]
+ (0.2788345442672343) [Z1 Z8]
+ (1.1861763734860518) [Z0 Z1]
+ (-1.2260484988824799e-05) [Y4 Z5 Y6]
+ (-1.2260484988824799e-05) [X4 Z5 X6]
+ (-1.2260484988824799e-05) [Y5 Z6 Y7]
+ (-1.2260484988824799e-05) [X5 Z6 X7]
+ (-1.072231215769171e-05) [Y10 Z11 Y12]
+ (-1.072231215769171e-05) [X10 Z11 X12]
+ (-1.072231215769171e-05) [Y11 Z12 Y13]
+ (-1.072231215769171e-05) [X11 Z12 X13]
+ (-3.8870516733764536e-06) [Y2 Z3 Y4]
+ (-3.8870516733764536e-06) [X2 Z3 X4]
+ (-3.887051673376453e-06) [Y3 Z4 Y5]
+ (-3.887051673376453e-06) [X3 Z4 X5]
+ (0.12507032579771973) [Y0 Z1 Y2]
+ (0.12507032579771973) [X0 Z1 X2]
+ (0.12507032579771976) [Y1 Z2 Y3]
+ (0.12507032579771976) [X1 Z2 X3]
+ (-0.03831467029480385) [Y4 Y5 X12 X13]
+ (-0.03831467029480385) [X4 X5 Y12 Y13]
+ (-0.03619412355904251) [Y2 Y3 X8 X9]
+ (-0.03619412355904251) [X2 X3 Y8 Y9]
+ (-0.035839567953353496) [Y2 Y3 X4 X5]
+ (-0.035839567953353496) [X2 X3 Y4 Y5]
+ (-0.031143817988967034) [Y2 Y3 X6 X7]
+ (-0.031143817988967034) [X2 X3 Y6 Y7]
+ (-0.028685183716105907) [Y10 Y11 X12 X13]
+ (-0.028685183716105907) [X10 X11 Y12 Y13]
+ (-0.025996177598021322) [Y3 Z4 Z5 Y7]
+ (-0.025996177598021322) [X3 Z4 Z5 X7]
+ (-0.025384657508457482) [Y2 Y3 X10 X11]
+ (-0.025384657508457482) [X2 X3 Y10 Y11]
+ (-0.019028242443847383) [Y3 Y4 X11 X12]
+ (-0.019028242443847383) [X3 X4 Y11 Y12]
+ (-0.017825140995786335) [Y6 Y7 X10 X11]
+ (-0.017825140995786335) [X6 X7 Y10 Y11]
+ (-0.017680067952481546) [Y4 Y5 X10 X11]
+ (-0.017680067952481546) [X4 X5 Y10 Y11]
+ (-0.01736611899465138) [Y6 Y7 X12 X13]
+ (-0.01736611899465138) [X6 X7 Y12 Y13]
+ (-0.015577208063976448) [Y2 Y3 X12 X13]
+ (-0.015577208063976448) [X2 X3 Y12 Y13]
+ (-0.014583648907612642) [Y0 Y1 X2 X3]
+ (-0.014583648907612642) [X0 X1 Y2 Y3]
+ (-0.01387338174842616) [Y6 Y7 X8 X9]
+ (-0.01387338174842616) [X6 X7 Y8 Y9]
+ (-0.011982389010247911) [Y4 Y5 X6 X7]
+ (-0.011982389010247911) [X4 X5 Y6 Y7]
+ (-0.011285190200840853) [Y5 X6 X11 Y12]
+ (-0.011285190200840853) [X5 Y6 Y11 X12]
+ (-0.009560705729135976) [Y8 Y9 X10 X11]
+ (-0.009560705729135976) [X8 X9 Y10 Y11]
+ (-0.008125251921381029) [Y1 X2 X8 Y9]
+ (-0.008125251921381029) [Y1 Y2 Y8 Y9]
+ (-0.008125251921381029) [X1 X2 X8 X9]
+ (-0.008125251921381029) [X1 Y2 Y8 X9]
+ (-0.007731425250775325) [Y0 Y1 X10 X11]
+ (-0.007731425250775325) [X0 X1 Y10 Y11]
+ (-0.006888194352970604) [Y0 Y1 X6 X7]
+ (-0.006888194352970604) [X0 X1 Y6 Y7]
+ (-0.006509361201177247) [Y0 Y1 X8 X9]
+ (-0.006509361201177247) [X0 X1 Y8 Y9]
+ (-0.00608782248056186) [Y8 Y9 X12 X13]
+ (-0.00608782248056186) [X8 X9 Y12 Y13]
+ (-0.005283776488402967) [Y0 Y1 X12 X13]
+ (-0.005283776488402967) [X0 X1 Y12 Y13]
+ (-0.005143391768825066) [Y3 X4 X5 Y6]
+ (-0.005143391768825066) [X3 Y4 Y5 X6]
+ (-0.004684903388155201) [Y1 X2 X6 Y7]
+ (-0.004684903388155201) [Y1 Y2 Y6 Y7]
+ (-0.004684903388155201) [X1 X2 X6 X7]
+ (-0.004684903388155201) [X1 Y2 Y6 X7]
+ (-0.004575007626639198) [Y1 X2 X12 Y13]
+ (-0.004575007626639198) [Y1 Y2 Y12 Y13]
+ (-0.004575007626639198) [X1 X2 X12 X13]
+ (-0.004575007626639198) [X1 Y2 Y12 X13]
+ (-0.004424855449441856) [Y1 X2 X4 Y5]
+ (-0.004424855449441856) [Y1 Y2 Y4 Y5]
+ (-0.004424855449441856) [X1 X2 X4 X5]
+ (-0.004424855449441856) [X1 Y2 Y4 X5]
+ (-0.0034795118903342918) [Y2 Z3 Z5 Y6]
+ (-0.0034795118903342918) [X2 Z3 Z5 X6]
+ (-0.0034795118903342918) [Y3 Z4 Z6 Y7]
+ (-0.0034795118903342918) [X3 Z4 Z6 X7]
+ (-0.002745836470186812) [Y0 Y1 X4 X5]
+ (-0.002745836470186812) [X0 X1 Y4 Y5]
+ (-0.001799219493663006) [Y1 X2 X10 Y11]
+ (-0.001799219493663006) [Y1 Y2 Y10 Y11]
+ (-0.001799219493663006) [X1 X2 X10 X11]
+ (-0.001799219493663006) [X1 Y2 Y10 X11]
+ (-0.0002921986261111114) [Y7 Y8 X9 X10]
+ (-0.0002921986261111114) [X7 X8 Y9 Y10]
+ (-8.194261372184309e-06) [Z10 Y11 Z12 Y13]
+ (-8.194261372184309e-06) [Z10 X11 Z12 X13]
+ (-7.801707500478764e-06) [Y2 Z3 Y4 Z11]
+ (-7.801707500478764e-06) [X2 Z3 X4 Z11]
+ (-7.801707500478764e-06) [Y3 Z4 Y5 Z10]
+ (-7.801707500478764e-06) [X3 Z4 X5 Z10]
+ (-4.6430510684535036e-06) [Y3 X4 X10 Y11]
+ (-4.6430510684535036e-06) [Y3 Y4 Y10 Y11]
+ (-4.6430510684535036e-06) [X3 X4 X10 X11]
+ (-4.6430510684535036e-06) [X3 Y4 Y10 X11]
+ (-4.588855155624549e-06) [Y4 Z5 Y6 Z13]
+ (-4.588855155624549e-06) [X4 Z5 X6 Z13]
+ (-4.588855155624549e-06) [Y5 Z6 Y7 Z12]
+ (-4.588855155624549e-06) [X5 Z6 X7 Z12]
+ (-4.55656921808569e-06) [Y5 X6 X12 Y13]
+ (-4.55656921808569e-06) [Y5 Y6 Y12 Y13]
+ (-4.55656921808569e-06) [X5 X6 X12 X13]
+ (-4.55656921808569e-06) [X5 Y6 Y12 X13]
+ (-3.694513294453754e-06) [Y4 X5 X11 Y12]
+ (-3.694513294453754e-06) [Y4 Y5 Y11 Y12]
+ (-3.694513294453754e-06) [X4 X5 X11 X12]
+ (-3.694513294453754e-06) [X4 Y5 Y11 X12]
+ (-3.3440815564253743e-06) [Z0 Y5 Z6 Y7]
+ (-3.3440815564253743e-06) [Z0 X5 Z6 X7]
+ (-3.3440815564253743e-06) [Z1 Y4 Z5 Y6]
+ (-3.3440815564253743e-06) [Z1 X4 Z5 X6]
+ (-3.1586564320252597e-06) [Y2 Z3 Y4 Z10]
+ (-3.1586564320252597e-06) [X2 Z3 X4 Z10]
+ (-3.1586564320252597e-06) [Y3 Z4 Y5 Z11]
+ (-3.1586564320252597e-06) [X3 Z4 X5 Z11]
+ (-3.099349243548008e-06) [Z0 Y4 Z5 Y6]
+ (-3.099349243548008e-06) [Z0 X4 Z5 X6]
+ (-3.099349243548008e-06) [Z1 Y5 Z6 Y7]
+ (-3.099349243548008e-06) [Z1 X5 Z6 X7]
+ (-2.8909678816414972e-06) [Z6 Y11 Z12 Y13]
+ (-2.8909678816414972e-06) [Z6 X11 Z12 X13]
+ (-2.8909678816414972e-06) [Z7 Y10 Z11 Y12]
+ (-2.8909678816414972e-06) [Z7 X10 Z11 X12]
+ (-2.1776646049354236e-06) [Z0 Y10 Z11 Y12]
+ (-2.1776646049354236e-06) [Z0 X10 Z11 X12]
+ (-2.1776646049354236e-06) [Z1 Y11 Z12 Y13]
+ (-2.1776646049354236e-06) [Z1 X11 Z12 X13]
+ (-1.8818501831839495e-06) [Y4 Z5 Y6 Z9]
+ (-1.8818501831839495e-06) [X4 Z5 X6 Z9]
+ (-1.8818501831839495e-06) [Y5 Z6 Y7 Z8]
+ (-1.8818501831839495e-06) [X5 Z6 X7 Z8]
+ (-1.8551201214963247e-06) [Z6 Y10 Z11 Y12]
+ (-1.8551201214963247e-06) [Z6 X10 Z11 X12]
+ (-1.8551201214963247e-06) [Z7 Y11 Z12 Y13]
+ (-1.8551201214963247e-06) [Z7 X11 Z12 X13]
+ (-1.8540608579270557e-06) [Y4 Z5 Y6 Z7]
+ (-1.8540608579270557e-06) [X4 Z5 X6 Z7]
+ (-1.8163031697054762e-06) [Z4 Y11 Z12 Y13]
+ (-1.8163031697054762e-06) [Z4 X11 Z12 X13]
+ (-1.8163031697054762e-06) [Z5 Y10 Z11 Y12]
+ (-1.8163031697054762e-06) [Z5 X10 Z11 X12]
+ (-1.692397828572066e-06) [Y4 Z5 Y6 Z10]
+ (-1.692397828572066e-06) [X4 Z5 X6 Z10]
+ (-1.692397828572066e-06) [Y5 Z6 Y7 Z11]
+ (-1.692397828572066e-06) [X5 Z6 X7 Z11]
+ (-1.6148794137840558e-06) [Z0 Y11 Z12 Y13]
+ (-1.6148794137840558e-06) [Z0 X11 Z12 X13]
+ (-1.6148794137840558e-06) [Z1 Y10 Z11 Y12]
+ (-1.6148794137840558e-06) [Z1 X10 Z11 X12]
+ (-1.5973171977592672e-06) [Z8 Y10 Z11 Y12]
+ (-1.5973171977592672e-06) [Z8 X10 Z11 X12]
+ (-1.5973171977592672e-06) [Z9 Y11 Z12 Y13]
+ (-1.5973171977592672e-06) [Z9 X11 Z12 X13]
+ (-1.4548424490259027e-06) [Y3 X4 X6 Y7]
+ (-1.4548424490259027e-06) [Y3 Y4 Y6 Y7]
+ (-1.4548424490259027e-06) [X3 X4 X6 X7]
+ (-1.4548424490259027e-06) [X3 Y4 Y6 X7]
+ (-1.398044908100772e-06) [Y4 Z5 Y6 Z8]
+ (-1.398044908100772e-06) [X4 Z5 X6 Z8]
+ (-1.398044908100772e-06) [Y5 Z6 Y7 Z9]
+ (-1.398044908100772e-06) [X5 Z6 X7 Z9]
+ (-1.1954890099697997e-06) [Y2 Z3 Y4 Z7]
+ (-1.1954890099697997e-06) [X2 Z3 X4 Z7]
+ (-1.1954890099697997e-06) [Y3 Z4 Y5 Z6]
+ (-1.1954890099697997e-06) [X3 Z4 X5 Z6]
+ (-1.1908508084394254e-06) [Z0 Y3 Z4 Y5]
+ (-1.1908508084394254e-06) [Z0 X3 Z4 X5]
+ (-1.1908508084394254e-06) [Z1 Y2 Z3 Y4]
+ (-1.1908508084394254e-06) [Z1 X2 Z3 X4]
+ (-1.1708301370187362e-06) [Z2 Y5 Z6 Y7]
+ (-1.1708301370187362e-06) [Z2 X5 Z6 X7]
+ (-1.1708301370187362e-06) [Z3 Y4 Z5 Y6]
+ (-1.1708301370187362e-06) [Z3 X4 Z5 X6]
+ (-1.0632283423425125e-06) [Z2 Y10 Z11 Y12]
+ (-1.0632283423425125e-06) [Z2 X10 Z11 X12]
+ (-1.0632283423425125e-06) [Z3 Y11 Z12 Y13]
+ (-1.0632283423425125e-06) [Z3 X11 Z12 X13]
+ (-1.0358477601451723e-06) [Y6 X7 X11 Y12]
+ (-1.0358477601451723e-06) [Y6 Y7 Y11 Y12]
+ (-1.0358477601451723e-06) [X6 X7 X11 X12]
+ (-1.0358477601451723e-06) [X6 Y7 Y11 X12]
+ (-9.50924975136642e-07) [Z2 Y4 Z5 Y6]
+ (-9.50924975136642e-07) [Z2 X4 Z5 X6]
+ (-9.50924975136642e-07) [Z3 Y5 Z6 Y7]
+ (-9.50924975136642e-07) [Z3 X5 Z6 X7]
+ (-9.344557775841352e-07) [Z8 Y11 Z12 Y13]
+ (-9.344557775841352e-07) [Z8 X11 Z12 X13]
+ (-9.344557775841352e-07) [Z9 Y10 Z11 Y12]
+ (-9.344557775841352e-07) [Z9 X10 Z11 X12]
+ (-8.337746755247127e-07) [Z0 Y2 Z3 Y4]
+ (-8.337746755247127e-07) [Z0 X2 Z3 X4]
+ (-8.337746755247127e-07) [Z1 Y3 Z4 Y5]
+ (-8.337746755247127e-07) [Z1 X3 Z4 X5]
+ (-7.956895372539645e-07) [Y3 X4 X8 Y9]
+ (-7.956895372539645e-07) [Y3 Y4 Y8 Y9]
+ (-7.956895372539645e-07) [X3 X4 X8 X9]
+ (-7.956895372539645e-07) [X3 Y4 Y8 X9]
+ (-7.764994118628141e-07) [Y2 Z3 Y4 Z5]
+ (-7.764994118628141e-07) [X2 Z3 X4 Z5]
+ (-5.929765815862276e-07) [Z4 Y5 Z6 Y7]
+ (-5.929765815862276e-07) [Z4 X5 Z6 X7]
+ (-5.770052995374555e-07) [Y2 Z3 Y4 Z9]
+ (-5.770052995374555e-07) [X2 Z3 X4 Z9]
+ (-5.770052995374555e-07) [Y3 Z4 Y5 Z8]
+ (-5.770052995374555e-07) [X3 Z4 X5 Z8]
+ (-5.47164774460319e-07) [Y1 Y2 X11 X12]
+ (-5.47164774460319e-07) [X1 X2 Y11 Y12]
+ (-4.838052750831773e-07) [Y5 X6 X8 Y9]
+ (-4.838052750831773e-07) [Y5 Y6 Y8 Y9]
+ (-4.838052750831773e-07) [X5 X6 X8 X9]
+ (-4.838052750831773e-07) [X5 Y6 Y8 X9]
+ (-3.57076132914713e-07) [Y0 X1 X3 Y4]
+ (-3.57076132914713e-07) [Y0 Y1 Y3 Y4]
+ (-3.57076132914713e-07) [X0 X1 X3 X4]
+ (-3.57076132914713e-07) [X0 Y1 Y3 X4]
+ (-2.4473231287736574e-07) [Y0 X1 X5 Y6]
+ (-2.4473231287736574e-07) [Y0 Y1 Y5 Y6]
+ (-2.4473231287736574e-07) [X0 X1 X5 X6]
+ (-2.4473231287736574e-07) [X0 Y1 Y5 X6]
+ (-2.1990516188209422e-07) [Y2 X3 X5 Y6]
+ (-2.1990516188209422e-07) [Y2 Y3 Y5 Y6]
+ (-2.1990516188209422e-07) [X2 X3 X5 X6]
+ (-2.1990516188209422e-07) [X2 Y3 Y5 X6]
+ (-1.9332412770343196e-07) [Y1 X2 X3 Y4]
+ (-1.9332412770343196e-07) [X1 Y2 Y3 X4]
+ (-1.2919694862616044e-07) [Y1 Z2 Z3 Y5]
+ (-1.2919694862616044e-07) [X1 Z2 Z3 X5]
+ (1.7379332623482667e-07) [Y0 Z1 Z3 Y4]
+ (1.7379332623482667e-07) [X0 Z1 Z3 X4]
+ (1.7379332623482667e-07) [Y1 Z2 Z4 Y5]
+ (1.7379332623482667e-07) [X1 Z2 Z4 X5]
+ (1.9332412770343196e-07) [Y1 Y2 X3 X4]
+ (1.9332412770343196e-07) [X1 X2 Y3 Y4]
+ (2.186842377165091e-07) [Y2 Z3 Y4 Z8]
+ (2.186842377165091e-07) [X2 Z3 X4 Z8]
+ (2.186842377165091e-07) [Y3 Z4 Y5 Z9]
+ (2.186842377165091e-07) [X3 Z4 X5 Z9]
+ (2.59353439056103e-07) [Y2 Z3 Y4 Z6]
+ (2.59353439056103e-07) [X2 Z3 X4 Z6]
+ (2.59353439056103e-07) [Y3 Z4 Y5 Z7]
+ (2.59353439056103e-07) [X3 Z4 X5 Z7]
+ (3.6060718678176324e-07) [Y0 Z1 Z2 Y4]
+ (3.6060718678176324e-07) [X0 Z1 Z2 X4]
+ (3.6060718678176324e-07) [Y1 Z3 Z4 Y5]
+ (3.6060718678176324e-07) [X1 Z3 Z4 X5]
+ (5.47164774460319e-07) [Y1 X2 X11 Y12]
+ (5.47164774460319e-07) [X1 Y2 Y11 X12]
+ (5.627851911513675e-07) [Y0 X1 X11 Y12]
+ (5.627851911513675e-07) [Y0 Y1 Y11 Y12]
+ (5.627851911513675e-07) [X0 X1 X11 X12]
+ (5.627851911513675e-07) [X0 Y1 Y11 X12]
+ (6.62861420175132e-07) [Y8 X9 X11 Y12]
+ (6.62861420175132e-07) [Y8 Y9 Y11 Y12]
+ (6.62861420175132e-07) [X8 X9 X11 X12]
+ (6.62861420175132e-07) [X8 Y9 Y11 X12]
+ (1.109440759157188e-06) [Z2 Y11 Z12 Y13]
+ (1.109440759157188e-06) [Z2 X11 Z12 X13]
+ (1.109440759157188e-06) [Z3 Y10 Z11 Y12]
+ (1.109440759157188e-06) [Z3 X10 Z11 X12]
+ (1.6021167405555709e-06) [Z2 Y3 Z4 Y5]
+ (1.6021167405555709e-06) [Z2 X3 Z4 X5]
+ (1.8782101247482778e-06) [Z4 Y10 Z11 Y12]
+ (1.8782101247482778e-06) [Z4 X10 Z11 X12]
+ (1.8782101247482778e-06) [Z5 Y11 Z12 Y13]
+ (1.8782101247482778e-06) [Z5 X11 Z12 X13]
+ (2.1726691014997003e-06) [Y2 X3 X11 Y12]
+ (2.1726691014997003e-06) [Y2 Y3 Y11 Y12]
+ (2.1726691014997003e-06) [X2 X3 X11 X12]
+ (2.1726691014997003e-06) [X2 Y3 Y11 X12]
+ (3.117447946082594e-06) [Y0 Z2 Z3 Y4]
+ (3.117447946082594e-06) [X0 Z2 Z3 X4]
+ (3.5390541844438363e-06) [Y2 Z3 Y4 Z12]
+ (3.5390541844438363e-06) [X2 Z3 X4 Z12]
+ (3.5390541844438363e-06) [Y3 Z4 Y5 Z13]
+ (3.5390541844438363e-06) [X3 Z4 X5 Z13]
+ (4.281913884814063e-06) [Y4 Z5 Y6 Z11]
+ (4.281913884814063e-06) [X4 Z5 X6 Z11]
+ (4.281913884814063e-06) [Y5 Z6 Y7 Z10]
+ (4.281913884814063e-06) [X5 Z6 X7 Z10]
+ (5.27588312211807e-06) [Y3 X4 X12 Y13]
+ (5.27588312211807e-06) [Y3 Y4 Y12 Y13]
+ (5.27588312211807e-06) [X3 X4 X12 X13]
+ (5.27588312211807e-06) [X3 Y4 Y12 X13]
+ (5.9743117133861294e-06) [Y5 X6 X10 Y11]
+ (5.9743117133861294e-06) [Y5 Y6 Y10 Y11]
+ (5.9743117133861294e-06) [X5 X6 X10 X11]
+ (5.9743117133861294e-06) [X5 Y6 Y10 X11]
+ (7.95441317617081e-06) [Y10 Z11 Y12 Z13]
+ (7.95441317617081e-06) [X10 Z11 X12 Z13]
+ (8.814937306561905e-06) [Y2 Z3 Y4 Z13]
+ (8.814937306561905e-06) [X2 Z3 X4 Z13]
+ (8.814937306561905e-06) [Y3 Z4 Y5 Z12]
+ (8.814937306561905e-06) [X3 Z4 X5 Z12]
+ (0.0002921986261111114) [Y7 X8 X9 Y10]
+ (0.0002921986261111114) [X7 Y8 Y9 X10]
+ (0.0004956762314917652) [Y2 Z4 Z5 Y6]
+ (0.0004956762314917652) [X2 Z4 Z5 X6]
+ (0.0011059037691896457) [Y0 Z1 Y2 Z5]
+ (0.0011059037691896457) [X0 Z1 X2 Z5]
+ (0.0011059037691896457) [Y1 Z2 Y3 Z4]
+ (0.0011059037691896457) [X1 Z2 X3 Z4]
+ (0.001663879878490774) [Y2 Z3 Z4 Y6]
+ (0.001663879878490774) [X2 Z3 Z4 X6]
+ (0.001663879878490774) [Y3 Z5 Z6 Y7]
+ (0.001663879878490774) [X3 Z5 Z6 X7]
+ (0.001756070701841196) [Y0 Z1 Y2 Z11]
+ (0.001756070701841196) [X0 Z1 X2 Z11]
+ (0.001756070701841196) [Y1 Z2 Y3 Z10]
+ (0.001756070701841196) [X1 Z2 X3 Z10]
+ (0.0023262306231580376) [Y0 Z1 Y2 Z13]
+ (0.0023262306231580376) [X0 Z1 X2 Z13]
+ (0.0023262306231580376) [Y1 Z2 Y3 Z12]
+ (0.0023262306231580376) [X1 Z2 X3 Z12]
+ (0.002745836470186812) [Y0 X1 X4 Y5]
+ (0.002745836470186812) [X0 Y1 Y4 X5]
+ (0.002929768674751006) [Y0 Z1 Y2 Z9]
+ (0.002929768674751006) [X0 Z1 X2 Z9]
+ (0.002929768674751006) [Y1 Z2 Y3 Z8]
+ (0.002929768674751006) [X1 Z2 X3 Z8]
+ (0.003276971931231598) [Y0 Z1 Y2 Z3]
+ (0.003276971931231598) [X0 Z1 X2 Z3]
+ (0.0033476175306661592) [Y0 Z1 Y2 Z7]
+ (0.0033476175306661592) [X0 Z1 X2 Z7]
+ (0.0033476175306661592) [Y1 Z2 Y3 Z6]
+ (0.0033476175306661592) [X1 Z2 X3 Z6]
+ (0.0035552901955042022) [Y0 Z1 Y2 Z10]
+ (0.0035552901955042022) [X0 Z1 X2 Z10]
+ (0.0035552901955042022) [Y1 Z2 Y3 Z11]
+ (0.0035552901955042022) [X1 Z2 X3 Z11]
+ (0.005143391768825066) [Y3 Y4 X5 X6]
+ (0.005143391768825066) [X3 X4 Y5 Y6]
+ (0.005283776488402967) [Y0 X1 X12 Y13]
+ (0.005283776488402967) [X0 Y1 Y12 X13]
+ (0.005530759218631502) [Y0 Z1 Y2 Z4]
+ (0.005530759218631502) [X0 Z1 X2 Z4]
+ (0.005530759218631502) [Y1 Z2 Y3 Z5]
+ (0.005530759218631502) [X1 Z2 X3 Z5]
+ (0.00608782248056186) [Y8 X9 X12 Y13]
+ (0.00608782248056186) [X8 Y9 Y12 X13]
+ (0.006509361201177247) [Y0 X1 X8 Y9]
+ (0.006509361201177247) [X0 Y1 Y8 X9]
+ (0.006888194352970604) [Y0 X1 X6 Y7]
+ (0.006888194352970604) [X0 Y1 Y6 X7]
+ (0.006901238249797237) [Y0 Z1 Y2 Z12]
+ (0.006901238249797237) [X0 Z1 X2 Z12]
+ (0.006901238249797237) [Y1 Z2 Y3 Z13]
+ (0.006901238249797237) [X1 Z2 X3 Z13]
+ (0.007731425250775325) [Y0 X1 X10 Y11]
+ (0.007731425250775325) [X0 Y1 Y10 X11]
+ (0.00803252091882136) [Y0 Z1 Y2 Z6]
+ (0.00803252091882136) [X0 Z1 X2 Z6]
+ (0.00803252091882136) [Y1 Z2 Y3 Z7]
+ (0.00803252091882136) [X1 Z2 X3 Z7]
+ (0.009560705729135976) [Y8 X9 X10 Y11]
+ (0.009560705729135976) [X8 Y9 Y10 X11]
+ (0.011055020596132035) [Y0 Z1 Y2 Z8]
+ (0.011055020596132035) [X0 Z1 X2 Z8]
+ (0.011055020596132035) [Y1 Z2 Y3 Z9]
+ (0.011055020596132035) [X1 Z2 X3 Z9]
+ (0.011285190200840853) [Y5 Y6 X11 X12]
+ (0.011285190200840853) [X5 X6 Y11 Y12]
+ (0.011307274008848097) [Y7 Z8 Z9 Y11]
+ (0.011307274008848097) [X7 Z8 Z9 X11]
+ (0.011982389010247911) [Y4 X5 X6 Y7]
+ (0.011982389010247911) [X4 Y5 Y6 X7]
+ (0.01387338174842616) [Y6 X7 X8 Y9]
+ (0.01387338174842616) [X6 Y7 Y8 X9]
+ (0.014583648907612642) [Y0 X1 X2 Y3]
+ (0.014583648907612642) [X0 Y1 Y2 X3]
+ (0.015577208063976448) [Y2 X3 X12 Y13]
+ (0.015577208063976448) [X2 Y3 Y12 X13]
+ (0.01736611899465138) [Y6 X7 X12 Y13]
+ (0.01736611899465138) [X6 Y7 Y12 X13]
+ (0.017680067952481546) [Y4 X5 X10 Y11]
+ (0.017680067952481546) [X4 Y5 Y10 X11]
+ (0.017825140995786335) [Y6 X7 X10 Y11]
+ (0.017825140995786335) [X6 Y7 Y10 X11]
+ (0.019028242443847383) [Y3 X4 X11 Y12]
+ (0.019028242443847383) [X3 Y4 Y11 X12]
+ (0.025384657508457482) [Y2 X3 X10 Y11]
+ (0.025384657508457482) [X2 Y3 Y10 X11]
+ (0.028685183716105907) [Y10 X11 X12 Y13]
+ (0.028685183716105907) [X10 Y11 Y12 X13]
+ (0.029812424517345594) [Y6 Z7 Z8 Y10]
+ (0.029812424517345594) [X6 Z7 Z8 X10]
+ (0.029812424517345594) [Y7 Z9 Z10 Y11]
+ (0.029812424517345594) [X7 Z9 Z10 X11]
+ (0.030104623143456702) [Y6 Z7 Z9 Y10]
+ (0.030104623143456702) [X6 Z7 Z9 X10]
+ (0.030104623143456702) [Y7 Z8 Z10 Y11]
+ (0.030104623143456702) [X7 Z8 Z10 X11]
+ (0.030787505389143842) [Y6 Z8 Z9 Y10]
+ (0.030787505389143842) [X6 Z8 Z9 X10]
+ (0.031143817988967034) [Y2 X3 X6 Y7]
+ (0.031143817988967034) [X2 Y3 Y6 X7]
+ (0.035839567953353496) [Y2 X3 X4 Y5]
+ (0.035839567953353496) [X2 Y3 Y4 X5]
+ (0.03619412355904251) [Y2 X3 X8 Y9]
+ (0.03619412355904251) [X2 Y3 Y8 X9]
+ (0.03831467029480385) [Y4 X5 X12 Y13]
+ (0.03831467029480385) [X4 Y5 Y12 X13]
+ (0.104330647806514) [Z0 Y1 Z2 Y3]
+ (0.104330647806514) [Z0 X1 Z2 X3]
+ (-0.12133276911042434) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042434) [X2 Z3 Z4 Z5 X6]
+ (-0.12133276911042433) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133276911042433) [X3 Z4 Z5 Z6 X7]
+ (3.20207688024345e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.20207688024345e-06) [X1 Z2 Z3 Z4 X5]
+ (3.202076880243451e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.202076880243451e-06) [X0 Z1 Z2 Z3 X4]
+ (0.22848106564918663) [Y6 Z7 Z8 Z9 Y10]
+ (0.22848106564918663) [X6 Z7 Z8 Z9 X10]
+ (0.2284810656491867) [Y7 Z8 Z9 Z10 Y11]
+ (0.2284810656491867) [X7 Z8 Z9 Z10 X11]
+ (-0.0327676578232905) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.0327676578232905) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.0327676578232905) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.0327676578232905) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.02711503684527313) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.02711503684527313) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.02711503684527313) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.02711503684527313) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.025996177598021326) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.025996177598021326) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.017561202409646162) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.017561202409646162) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.017561202409646162) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.017561202409646162) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.014564531231172979) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564531231172979) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564531231172979) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564531231172979) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.012215040997613863) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.012215040997613863) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.012215040997613863) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.012215040997613863) [X4 Z5 X6 X11 Z12 X13]
+ (-0.012215040997613863) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.012215040997613863) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.012215040997613863) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.012215040997613863) [X5 Z6 X7 X10 Z11 X12]
+ (-0.011756013419819288) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.011756013419819288) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.011756013419819288) [X3 Z4 Z5 X6 X8 X9]
+ (-0.011756013419819288) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.008764827575688833) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.008764827575688833) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.008764827575688833) [X2 Z3 Z4 X5 X11 X12]
+ (-0.008764827575688833) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.008764827575688833) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.008764827575688833) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.008764827575688833) [X3 X4 X10 Z11 Z12 X13]
+ (-0.008764827575688833) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.008125251921381029) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125251921381029) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.007306759928832947) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306759928832947) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306759928832947) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306759928832947) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005805188989826871) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.005805188989826871) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.005805188989826871) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.005805188989826871) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.0056526209780173664) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.0056526209780173664) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.0056526209780173664) [X0 X1 X3 Z4 Z5 X6]
+ (-0.0056526209780173664) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.005143391768825066) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.005143391768825066) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.005143391768825066) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.005143391768825066) [X2 Z3 X4 X5 Z6 X7]
+ (-0.004684903388155201) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.004684903388155201) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668620318776304) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668620318776304) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.004575007626639198) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.004575007626639198) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424855449441856) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424855449441856) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.004158797381840043) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.004158797381840043) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.004158797381840043) [X3 Z4 Z5 X6 X12 X13]
+ (-0.004158797381840043) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.0034937903598901564) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.0034937903598901564) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.0034937903598901564) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.0034937903598901564) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.00277902679902557) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.00277902679902557) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.002293956611352474) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.002293956611352474) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.001799219493663006) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.001799219493663006) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278753941369692) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.0017278753941369692) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298507967730094) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298507967730094) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298507967730094) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298507967730094) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533856254125581) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533856254125581) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008145313270956707) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008145313270956707) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008145313270956707) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008145313270956707) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.735036880593522e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735036880593522e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735036880593522e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735036880593522e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.77481786457423e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.77481786457423e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.77481786457423e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.77481786457423e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.518362215672723e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.518362215672723e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.518362215672723e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.518362215672723e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.444344675912398e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.444344675912398e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.444344675912398e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.444344675912398e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.524373848531878e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.524373848531878e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.524373848531878e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.524373848531878e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.290028433210406e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.290028433210406e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.290028433210406e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.290028433210406e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.974311713386128e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.974311713386128e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.27588312211807e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.27588312211807e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.643051068453503e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.643051068453503e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.55656921808569e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.55656921808569e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.253224225614046e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.253224225614046e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.7696594519199714e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.7696594519199714e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.694513294453754e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.694513294453754e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.6102971305249524e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.6102971305249524e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.6102971305249524e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.6102971305249524e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.3131455001698733e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.3131455001698733e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.2774831954589352e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.2774831954589352e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.2774831954589352e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.2774831954589352e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.211228348362005e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.211228348362005e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.211228348362005e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.211228348362005e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.1513463111130774e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.1513463111130774e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.0882507112803782e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.0882507112803782e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.1726691014997003e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.1726691014997003e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.4548424490259024e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.4548424490259024e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.3304731886618313e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.3304731886618313e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2283337824623164e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2283337824623164e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.0358477601451723e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.0358477601451723e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956895372539645e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956895372539645e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.73319774229444e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.73319774229444e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.73319774229444e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.73319774229444e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.62861420175132e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.62861420175132e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.556281914474453e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.556281914474453e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.556281914474453e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.556281914474453e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.418291574540821e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.418291574540821e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.418291574540821e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.418291574540821e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.927453082800975e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927453082800975e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927453082800975e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927453082800975e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627851911513675e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627851911513675e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660624706512e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660624706512e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660624706512e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660624706512e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660624706512e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660624706512e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660624706512e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660624706512e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.838052750831773e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.838052750831773e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.57076132914713e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.57076132914713e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.328139350660173e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.328139350660173e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.086826565155536e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.086826565155536e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.086826565155536e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.086826565155536e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.4473231287736574e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.4473231287736574e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.3713289478874337e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.3713289478874337e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.3713289478874337e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.3713289478874337e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.1990516188209422e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.1990516188209422e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.9332412770343196e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.9332412770343196e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.9332412770343196e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.9332412770343196e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.8394209154393264e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.8394209154393264e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.8394209154393264e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.8394209154393264e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.551053917580762e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.551053917580762e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.551053917580762e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.551053917580762e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.380778148042671e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.380778148042671e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.380778148042671e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.380778148042671e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.380778148042671e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.380778148042671e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.380778148042671e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.380778148042671e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.380778148042671e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.380778148042671e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.380778148042671e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.380778148042671e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.2919694862616044e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.2919694862616044e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.1076325598527862e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.1076325598527862e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.1076325598527862e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.1076325598527862e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.1076325598527862e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.1076325598527862e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.1076325598527862e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.1076325598527862e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.057446594934644e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057446594934644e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057446594934644e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057446594934644e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649310134861333e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649310134861333e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649310134861333e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649310134861333e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.8394209154393264e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.8394209154393264e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.8394209154393264e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.8394209154393264e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.1990516188209422e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.1990516188209422e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.4473231287736574e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.4473231287736574e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.2362599613735664e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.2362599613735664e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.2362599613735664e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.2362599613735664e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.328139350660173e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.328139350660173e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.57076132914713e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.57076132914713e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.838052750831773e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.838052750831773e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.627851911513675e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627851911513675e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.62861420175132e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.62861420175132e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956895372539645e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956895372539645e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.306536651882868e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.306536651882868e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.306536651882868e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.306536651882868e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.0358477601451723e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.0358477601451723e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2283337824623164e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2283337824623164e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.2393363217038404e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393363217038404e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393363217038404e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393363217038404e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.3304731886618313e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.3304731886618313e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.4548424490259024e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.4548424490259024e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.1726691014997003e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.1726691014997003e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.0882507112803782e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.0882507112803782e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.117447946082594e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.117447946082594e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.1513463111130774e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.1513463111130774e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.3131455001698733e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.3131455001698733e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.334331289350555e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.334331289350555e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.694513294453754e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.694513294453754e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.183932559373912e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.183932559373912e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.55656921808569e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.55656921808569e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.643051068453503e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.643051068453503e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.27588312211807e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.27588312211807e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.974311713386128e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.974311713386128e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.0002921986261111114) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.0002921986261111114) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.0002921986261111114) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.0002921986261111114) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004956762314917651) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004956762314917651) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650070219498863) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650070219498863) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650070219498863) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650070219498863) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533856254125581) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533856254125581) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.0016095313817213821) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.0016095313817213821) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.0016095313817213821) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.0016095313817213821) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0016676041811440777) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.0016676041811440777) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.0016676041811440777) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.0016676041811440777) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.0017278753941369692) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.0017278753941369692) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.001799219493663006) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.001799219493663006) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.002293956611352474) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.002293956611352474) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.00246291700713394) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.00246291700713394) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.00246291700713394) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.00246291700713394) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.003961560792496552) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.003961560792496552) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.003961560792496552) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.003961560792496552) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.004424855449441856) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424855449441856) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004575007626639198) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.004575007626639198) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668620318776304) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668620318776304) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.004684903388155201) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.004684903388155201) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005324835234221661) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.005324835234221661) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.005324835234221661) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.005324835234221661) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.005368659358109438) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368659358109438) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368659358109438) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368659358109438) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.007960880725921542) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.007960880725921542) [X4 Z5 X6 X10 Z11 X12]
+ (0.007960880725921542) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.007960880725921542) [X5 Z6 X7 X11 Z12 X13]
+ (0.008125251921381029) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125251921381029) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.00889073152269455) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.00889073152269455) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.00889073152269455) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.00889073152269455) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.01026341486815855) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.01026341486815855) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.01026341486815855) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.01026341486815855) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.01054042590767145) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.01054042590767145) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.01054042590767145) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.01054042590767145) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010960074940542477) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010960074940542477) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010960074940542477) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010960074940542477) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307274008848097) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307274008848097) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.014411099430130922) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411099430130922) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411099430130922) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411099430130922) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.015225630757226596) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.015225630757226596) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.015225630757226596) [X3 Z4 Z5 X6 X10 X11]
+ (0.015225630757226596) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.015588250102380212) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.015588250102380212) [X2 Z3 X4 X10 Z11 X12]
+ (0.015588250102380212) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.015588250102380212) [X3 Z4 X5 X11 Z12 X13]
+ (0.018266834869375425) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.018266834869375425) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.018266834869375425) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.018266834869375425) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.01902042317303984) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.01902042317303984) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.01902042317303984) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.01902042317303984) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.020175921723535405) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.020175921723535405) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.020175921723535405) [X4 Z5 Z6 X7 X11 X12]
+ (0.020175921723535405) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.020175921723535405) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.020175921723535405) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.020175921723535405) [X5 X6 X10 Z11 Z12 X13]
+ (0.020175921723535405) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.024353077678069043) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.024353077678069043) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.024353077678069043) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.024353077678069043) [X2 Z3 X4 X11 Z12 X13]
+ (0.024353077678069043) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.024353077678069043) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.024353077678069043) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.024353077678069043) [X3 Z4 X5 X10 Z11 X12]
+ (0.024389082531149284) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.024389082531149284) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.024389082531149284) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.024389082531149284) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.025104957138844426) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.025104957138844426) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.025104957138844426) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.025104957138844426) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.030787505389143845) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.030787505389143845) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.045879470781298295) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.045879470781298295) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.0560073308778074) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.0560073308778074) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.0560073308778074) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.0560073308778074) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661333) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.05608468124661333) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661333) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.05608468124661333) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.631277928389464e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631277928389464e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.631277928389464e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631277928389464e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.5950860069511864e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.5950860069511864e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.595086006951186e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.595086006951186e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.04274327701378314) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.04274327701378314) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (0.04274327701378314) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04274327701378314) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.04764261217638309) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.04764261217638309) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.04764261217638309) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.04764261217638309) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.04171881383982175) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.04171881383982175) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.04171881383982175) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.04171881383982175) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.03956441632289347) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.03956441632289347) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.03956441632289347) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03956441632289347) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.039359168022053186) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.039359168022053186) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.039359168022053186) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.039359168022053186) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.039318051947197584) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.039318051947197584) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.039318051947197584) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.039318051947197584) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.035608378988312574) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.035608378988312574) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.029903789512624887) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.029903789512624887) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.029903789512624887) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.029903789512624887) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.02873077955190554) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.02873077955190554) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.02873077955190554) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.02873077955190554) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0256372382960268) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.0256372382960268) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.0256372382960268) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.0256372382960268) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.024755463292891033) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.024755463292891033) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.024755463292891033) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.024755463292891033) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.024282117354692996) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.024282117354692996) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.02314513092952905) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.02314513092952905) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.022528440196012953) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.022528440196012953) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02143381072160104) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.02143381072160104) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.02143381072160104) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.02143381072160104) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.01925750509525159) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.01925750509525159) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019028242443847383) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.019028242443847383) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.018889030304942902) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.018889030304942902) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.018889030304942902) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.018889030304942902) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.01602460368917955) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.01602460368917955) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.015225630757226596) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225630757226596) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.014603704729162151) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.014603704729162151) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564531231172979) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564531231172979) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.01175601341981929) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.01175601341981929) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.011285190200840853) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.011285190200840853) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.00984174924696264) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.00984174924696264) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.009612634606847251) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612634606847251) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612634606847251) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612634606847251) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.008469978791023847) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.008469978791023847) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.007306759928832947) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.007306759928832947) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005923798336561344) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.005923798336561344) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.0056526209780173664) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.0056526209780173664) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368659358109438) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368659358109438) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.004158797381840043) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.004158797381840043) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0033566705638329083) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.0033566705638329083) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.0033566705638329083) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.0033566705638329083) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.0032675138544235646) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.0032675138544235646) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.0032675138544235646) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.0032675138544235646) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.00277902679902557) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.00277902679902557) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.0026860409778066262) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.0026860409778066262) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.0026860409778066262) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.0026860409778066262) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.002293956611352474) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.002293956611352474) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.002293956611352474) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.002293956611352474) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.0009581655836696564) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.0009581655836696564) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0009581655836696564) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.0009581655836696564) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.0009581655836696564) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.0009581655836696564) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0009581655836696564) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.0009581655836696564) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.00024636437569588223) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00024636437569588223) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0001384017730354989) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.0001384017730354989) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.0001384017730354989) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.0001384017730354989) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.735036880593522e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735036880593522e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.610358530567431e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.610358530567431e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.610358530567431e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.610358530567431e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808795257385e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.5316808795257385e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808795257385e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5316808795257385e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.806102775116861e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.806102775116861e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.806102775116861e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.806102775116861e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.089799467547044e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.089799467547044e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.089799467547044e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.089799467547044e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.652209669382072e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.652209669382072e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.652209669382072e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.652209669382072e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851833831607e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.481851833831607e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851833831607e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.481851833831607e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.071480736350722e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.071480736350722e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.071480736350722e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.071480736350722e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.734622038766141e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.734622038766141e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.734622038766141e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.734622038766141e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.728843147219539e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.728843147219539e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.728843147219539e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.728843147219539e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.253224225614046e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.253224225614046e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.7696594519199714e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.7696594519199714e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.544395429269026e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.544395429269026e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.544395429269026e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.544395429269026e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.544395429269026e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.544395429269026e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.544395429269026e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.544395429269026e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.3609563203275043e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563203275043e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563203275043e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.3609563203275043e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.1032156046602705e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.1032156046602705e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.1032156046602705e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.1032156046602705e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.011122098216548e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.011122098216548e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.011122098216548e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.011122098216548e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9429468366140706e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.9429468366140706e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9429468366140706e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9429468366140706e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174771081917e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.6541174771081917e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174771081917e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.6541174771081917e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.5224930676046794e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224930676046794e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224930676046794e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224930676046794e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224930676046794e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930676046794e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930676046794e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224930676046794e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.2283337824623164e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337824623164e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.2283337824623164e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337824623164e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.98877028850672e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.98877028850672e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.98877028850672e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.98877028850672e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.867765104169252e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104169252e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104169252e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.867765104169252e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.189990975284269e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.189990975284269e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246206996904e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246206996904e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.47164774460319e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.47164774460319e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.561447179886472e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.561447179886472e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.561447179886472e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.561447179886472e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.5233896778198134e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.5233896778198134e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.4273231086202454e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.4273231086202454e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.4273231086202454e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.4273231086202454e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.328139350660173e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350660173e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.328139350660173e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350660173e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.086826565155536e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.086826565155536e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.8882935950587886e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935950587886e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935950587886e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.8882935950587886e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.3713289478874334e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.3713289478874334e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.8394209154393264e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.8394209154393264e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.057446594934646e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057446594934646e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.537178096449279e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.537178096449279e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.537178096449279e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.537178096449279e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.057446594934646e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057446594934646e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.209350644372254e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350644372254e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350644372254e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.209350644372254e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.703578355504652e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.703578355504652e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.703578355504652e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.703578355504652e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.8394209154393264e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.8394209154393264e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.3713289478874334e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.3713289478874334e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.086826565155536e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.086826565155536e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.5233896778198134e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.5233896778198134e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.47164774460319e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.47164774460319e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246206996904e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246206996904e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189990975284269e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.189990975284269e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.3304731886618313e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.3304731886618313e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.3304731886618313e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.3304731886618313e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288532435083973e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288532435083973e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288532435083973e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288532435083973e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.6893489514604618e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.6893489514604618e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.6893489514604618e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.6893489514604618e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.7455184004183543e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.7455184004183543e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.7455184004183543e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.7455184004183543e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.7455184004183543e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.7455184004183543e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.7455184004183543e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.7455184004183543e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.2118420190651406e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420190651406e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.2118420190651406e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420190651406e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.2118420190651406e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420190651406e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.2118420190651406e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420190651406e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.3131455001698733e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.3131455001698733e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.3131455001698733e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.3131455001698733e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.334331289350555e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.334331289350555e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.183932559373912e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.183932559373912e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.735036880593522e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735036880593522e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00024636437569588223) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00024636437569588223) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0004458535128840798) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.0004458535128840798) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.0004458535128840798) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.0004458535128840798) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940221543005277) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005277) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005277) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005277) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005277) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005277) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940221543005277) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005277) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533856254125581) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533856254125581) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533856254125581) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533856254125581) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.001043524653490759) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.001043524653490759) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.001043524653490759) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.001043524653490759) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.001280306097349675) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.001280306097349675) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.001280306097349675) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.001280306097349675) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.0013038004788126997) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.0013038004788126997) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.0013038004788126997) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.0013038004788126997) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.002261966062482356) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.002261966062482356) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.002261966062482356) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.002261966062482356) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.002261966062482356) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.002261966062482356) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.002261966062482356) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.002261966062482356) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.003989841456619325) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.003989841456619325) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.003989841456619325) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.003989841456619325) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.004158797381840043) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.004158797381840043) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.004311038507914324) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.004311038507914324) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.004311038507914324) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.004311038507914324) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.004636976661182584) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.004636976661182584) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.004636976661182584) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.004636976661182584) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.005114473831660384) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.005114473831660384) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.005114473831660384) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.005114473831660384) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.005114473831660384) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660384) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660384) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005114473831660384) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005241535382803887) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.005241535382803887) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.005241535382803887) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.005241535382803887) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.005368659358109438) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368659358109438) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005379937155839385) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.005379937155839385) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.005379937155839385) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.005379937155839385) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.0056526209780173664) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.0056526209780173664) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.005708495985960913) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708495985960913) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708495985960913) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708495985960913) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.005923798336561344) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.005923798336561344) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.007306759928832947) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.007306759928832947) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469978791023847) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.008469978791023847) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.00984174924696264) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.00984174924696264) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.011285190200840853) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.011285190200840853) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.01175601341981929) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.01175601341981929) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.014564531231172979) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564531231172979) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603704729162151) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.014603704729162151) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.015225630757226596) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225630757226596) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.01602460368917955) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.01602460368917955) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.019028242443847383) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.019028242443847383) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.01925750509525159) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.01925750509525159) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.045879470781298295) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.045879470781298295) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.3693708936615623) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.3693708936615623) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.3693708936615623) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.3693708936615623) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.28164257767023076) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.28164257767023076) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.28164257767023065) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.28164257767023065) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0906514420703648) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0906514420703648) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0906514420703648) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0906514420703648) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863623) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684737589863623) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863623) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684737589863623) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635021950635033) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635021950635033) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635021950635033) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635021950635033) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214045) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.06752385099214045) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214045) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.06752385099214045) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.035608378988312574) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.035608378988312574) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.034903343373661605) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.034903343373661605) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.034903343373661605) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.034903343373661605) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883829916) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.024591860883829916) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883829916) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.024591860883829916) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.024282117354692996) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.024282117354692996) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.02314513092952905) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.02314513092952905) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.022528440196012953) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.022528440196012953) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019538050311314763) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.019538050311314763) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.019538050311314763) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.019538050311314763) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.017091553155898855) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.017091553155898855) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.017091553155898855) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.017091553155898855) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.01602460368917955) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.01602460368917955) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.01602460368917955) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.01602460368917955) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.010311482489831686) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831686) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831686) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311482489831686) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.00984174924696264) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.00984174924696264) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.00984174924696264) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.00984174924696264) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209882) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209882) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209882) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008826368514209882) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008541996625454861) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454861) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454861) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454861) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454861) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454861) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454861) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008541996625454861) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008469978791023847) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023847) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.008469978791023847) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023847) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.004668620318776304) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668620318776304) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.0038764708993369638) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.0038764708993369638) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.0038040661717285438) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0038040661717285438) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0038040661717285438) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0038040661717285438) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.00348415730021788) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.00348415730021788) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0033566705638329083) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.0033566705638329083) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.0032675138544235646) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.0032675138544235646) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0021413612231016045) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.0021413612231016045) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.0017278753941369692) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.0017278753941369692) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.0016407548553123998) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0016407548553123998) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.001452884321416911) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.001452884321416911) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.001452884321416911) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.001452884321416911) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870896771024517) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007870896771024517) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192743499487772) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192743499487772) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.00019400857029756177) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00019400857029756177) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0001384017730354989) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.0001384017730354989) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.141625221155926e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.141625221155926e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.141625221155926e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.141625221155926e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.071480736350721e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.071480736350721e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.1513463111130774e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.1513463111130774e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0882507112803782e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.0882507112803782e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.988511706314494e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.988511706314494e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8742990713826555e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.8742990713826555e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.3609563203275043e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.3609563203275043e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.3002946562549126e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3002946562549126e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.146837650736174e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.146837650736174e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.146837650736174e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.146837650736174e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.352332103025088e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.352332103025088e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.352332103025088e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.352332103025088e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091637198928584e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637198928584e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637198928584e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637198928584e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637198928584e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637198928584e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637198928584e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091637198928584e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.074305985834059e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.074305985834059e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.074305985834059e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.074305985834059e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128986252971e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128986252971e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128986252971e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128986252971e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867765104169252e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.867765104169252e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.56069246487915e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.56069246487915e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.56069246487915e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.56069246487915e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.56069246487915e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.56069246487915e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.56069246487915e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.56069246487915e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.997018422054988e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.997018422054988e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.997018422054988e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.997018422054988e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.997018422054988e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.997018422054988e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.997018422054988e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.997018422054988e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.5682475211087695e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.5682475211087695e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.5682475211087695e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.5682475211087695e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393084331595e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393084331595e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393084331595e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393084331595e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393084331595e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393084331595e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.3767393084331595e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393084331595e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.8882935950587886e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.8882935950587886e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6863815453689403e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.6863815453689403e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035783555046517e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7035783555046517e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.209350644372254e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.209350644372254e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773243478647e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773243478647e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773243478647e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773243478647e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773243478647e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773243478647e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773243478647e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773243478647e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253793929193e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253793929193e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9742253793929193e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9742253793929193e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0474716554407001e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.0474716554407001e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.0474716554407001e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.0474716554407001e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.209350644372254e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.209350644372254e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0717282183882848e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0717282183882848e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0717282183882848e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0717282183882848e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.2004287494016025e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.2004287494016025e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.2004287494016025e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.2004287494016025e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.7035783555046517e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7035783555046517e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.3120943052429303e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.3120943052429303e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.3120943052429303e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.3120943052429303e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.6863815453689403e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815453689403e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.8882935950587886e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.8882935950587886e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.09225061608399e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.09225061608399e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.09225061608399e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.09225061608399e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.09225061608399e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.09225061608399e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.09225061608399e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.09225061608399e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.4445978540818207e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.4445978540818207e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.4445978540818207e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.4445978540818207e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.684915095181134e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.684915095181134e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.684915095181134e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.684915095181134e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246974425385189e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246974425385189e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246974425385189e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246974425385189e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246974425385189e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246974425385189e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246974425385189e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246974425385189e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.867765104169252e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.867765104169252e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3002946562549126e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3002946562549126e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.3609563203275043e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.3609563203275043e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.8742990713826555e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.8742990713826555e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.8836765760678606e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.8836765760678606e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.9473560116749312e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9473560116749312e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.9473560116749312e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9473560116749312e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.988511706314494e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.988511706314494e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.0882507112803782e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.0882507112803782e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.1513463111130774e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.1513463111130774e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.846201671239626e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846201671239626e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846201671239626e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846201671239626e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.071480736350721e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.071480736350721e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.105526721960647e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.105526721960647e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.105526721960647e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.105526721960647e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146496327494538e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146496327494538e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146496327494538e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146496327494538e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159350501885149e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159350501885149e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159350501885149e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159350501885149e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427988656422042e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427988656422042e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427988656422042e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427988656422042e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935867717989426e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935867717989426e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935867717989426e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935867717989426e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348019476e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.253273348019476e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979825793343303e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979825793343303e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979825793343303e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979825793343303e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.205548411219395e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.205548411219395e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.205548411219395e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.205548411219395e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0001384017730354989) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.0001384017730354989) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.00018787053389548883) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.00018787053389548883) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.00018787053389548883) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00018787053389548883) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00019400857029756177) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00019400857029756177) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0002463643756958822) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0002463643756958822) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0002463643756958822) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0002463643756958822) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192743499487772) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192743499487772) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156734248908924) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156734248908924) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156734248908924) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156734248908924) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024517) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007870896771024517) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.00153248352307304) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.00153248352307304) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.00153248352307304) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.00153248352307304) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.0016407548553123998) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0016407548553123998) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0017278753941369692) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.0017278753941369692) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.0024464971554159077) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.0024464971554159077) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.0024464971554159077) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.0024464971554159077) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.0032675138544235646) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.0032675138544235646) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.0033566705638329083) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.0033566705638329083) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.00348415730021788) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.00348415730021788) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0038764708993369638) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.0038764708993369638) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.004668620318776304) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668620318776304) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278137) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.004767272188278137) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.004767272188278137) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278137) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.005286546538226914) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.005286546538226914) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.005286546538226914) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.005286546538226914) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.005408954422410003) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.005408954422410003) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.005408954422410003) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.005408954422410003) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.005923798336561344) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561344) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.005923798336561344) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561344) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.010715508469796723) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010715508469796723) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010715508469796723) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010715508469796723) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.010757563953908918) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010757563953908918) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010757563953908918) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010757563953908918) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01460370472916215) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.01460370472916215) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.01460370472916215) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.01460370472916215) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
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
+ (5.7759505272456694e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.7759505272456694e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.7759505272456715e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.7759505272456715e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.07165035181002737) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.07165035181002737) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.07165035181002741) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07165035181002741) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.01925750509525159) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.01925750509525159) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010311482489831686) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311482489831686) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008826368514209884) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209884) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.007597464029770602) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.007597464029770602) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.007597464029770602) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.007597464029770602) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311878) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311878) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311878) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311878) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311878) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311878) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311878) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311878) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676604) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005348051582676604) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005348051582676604) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676604) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0038040661717285433) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0038040661717285433) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219473) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.0029841661681219473) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.0029841661681219473) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.0029841661681219473) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.0024464971554159077) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.0024464971554159077) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.0022494124470939987) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0022494124470939987) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022494124470939987) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0022494124470939987) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0021413612231016045) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.0021413612231016045) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.0018638942824587244) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587244) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587244) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587244) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587244) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587244) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0018638942824587244) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587244) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0016407548553123998) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553123998) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0016407548553123998) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553123998) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0012223378081538468) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538468) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538468) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538468) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538468) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538468) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538468) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0012223378081538468) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0010283292378562854) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0010283292378562854) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0010283292378562854) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0010283292378562854) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1463061453080053e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1463061453080053e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.8742990713826555e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990713826555e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.8742990713826555e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990713826555e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3002946562549126e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3002946562549126e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3002946562549126e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3002946562549126e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.0444941298034481e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.0444941298034481e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.0444941298034481e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.0444941298034481e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.956079229971841e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.956079229971841e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.956079229971841e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.956079229971841e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105515037091189e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105515037091189e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105515037091189e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105515037091189e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661347213070065e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661347213070065e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661347213070065e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661347213070065e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341413697515e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341413697515e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189990975284269e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.189990975284269e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876621658147973e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876621658147973e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876621658147973e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876621658147973e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246206996904e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246206996904e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.5233896778198134e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.5233896778198134e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0767325319320736e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.0767325319320736e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0767325319320736e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.0767325319320736e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.013471458946008e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.013471458946008e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.904599884336966e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.904599884336966e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.904599884336966e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.904599884336966e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.666731754599585e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.666731754599585e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.666731754599585e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.666731754599585e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.850564192880653e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.850564192880653e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309314381306e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6569309314381306e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309314381306e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6569309314381306e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.850564192880653e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.850564192880653e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815453689403e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815453689403e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.6863815453689403e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815453689403e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458946008e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.013471458946008e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.5233896778198134e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.5233896778198134e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.670402390384138e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.670402390384138e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.670402390384138e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.670402390384138e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246206996904e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246206996904e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189990975284269e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.189990975284269e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341413697515e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341413697515e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949476487346439e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949476487346439e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.7924939577000316e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939577000316e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939577000316e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.7924939577000316e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.8836765760678606e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.8836765760678606e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.988511706314494e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.988511706314494e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.988511706314494e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.988511706314494e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348019476e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253273348019476e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.401710973511916e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.401710973511916e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.401710973511916e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.401710973511916e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.580960369281919e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.580960369281919e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.580960369281919e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.580960369281919e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192743499487772) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487772) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192743499487772) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487772) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.000787089677102452) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.000787089677102452) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.000787089677102452) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.000787089677102452) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0011726348316441824) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0011726348316441824) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0011726348316441824) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0011726348316441824) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012366478019245316) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.0012366478019245316) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.0012366478019245316) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.0012366478019245316) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.0022009640695004676) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0022009640695004676) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0022009640695004676) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0022009640695004676) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798029) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798029) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798029) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798029) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798029) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798029) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002394972639798029) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798029) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0024464971554159077) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.0024464971554159077) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.0038040661717285433) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0038040661717285433) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0038764708993369638) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.0038764708993369638) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.0038764708993369638) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.0038764708993369638) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.004220813970046479) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.004220813970046479) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.004220813970046479) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.004220813970046479) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.008826368514209884) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.008826368514209884) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.010311482489831686) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311482489831686) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01925750509525159) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01925750509525159) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.05859198873386222) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.05859198873386222) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3987009016006722e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.3987009016006722e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.3987009016006718e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3987009016006718e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00348415730021788) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00348415730021788) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984166168121947) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.002984166168121947) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.00019400857029756177) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00019400857029756177) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453080053e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453080053e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7924939577000316e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7924939577000316e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540341413697515e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413697515e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540341413697515e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413697515e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.850564192880653e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.850564192880653e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.850564192880653e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.850564192880653e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458946008e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471458946008e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458946008e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471458946008e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.94947648734644e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.94947648734644e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.7924939577000316e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7924939577000316e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756177) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756177) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002984166168121947) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.002984166168121947) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.00348415730021788) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00348415730021788) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
Expectation value of XIZ =  0.07715357869738937
 </code>
 </pre>
 </details>

---

## 11. tutorial_adaptive_circuits.html <a name="demo10"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_adaptive_circuits.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Excitation : [0, 1, 2, 3], Gradient: -0.01278217515766197
Excitation : [0, 1, 2, 5], Gradient: -1.3552527156068685e-20
Excitation : [0, 1, 2, 7], Gradient: -2.0328790734103208e-19
Excitation : [0, 1, 2, 9], Gradient: 0.03426451170167995
Excitation : [0, 1, 3, 4], Gradient: 1.0842021724855054e-19
Excitation : [0, 1, 3, 6], Gradient: -1.2197274440461947e-19
Excitation : [0, 1, 3, 8], Gradient: -0.034264511701679747
Excitation : [0, 1, 4, 5], Gradient: -0.023581529020676416
Excitation : [0, 1, 5, 8], Gradient: 1.4907779871675732e-19
Excitation : [0, 1, 6, 7], Gradient: -0.023581529020676402
Excitation : [0, 1, 7, 8], Gradient: -1.2197274440461872e-19
Excitation : [0, 1, 8, 9], Gradient: -0.1236227348559901
[[0, 1, 2, 3], [0, 1, 2, 9], [0, 1, 3, 8], [0, 1, 4, 5], [0, 1, 6, 7], [0, 1, 8, 9]]
Excitation : [0, 2], Gradient: -0.005062536239335471
Excitation : [0, 4], Gradient: -1.42793439237632e-17
Excitation : [0, 6], Gradient: -1.2314399299283899e-18
Excitation : [0, 8], Gradient: -0.0009448044625779991
Excitation : [1, 3], Gradient: 0.00492661687700407
Excitation : [1, 5], Gradient: 2.5723665298511912e-18
Excitation : [1, 7], Gradient: 1.9507656084335405e-18
Excitation : [1, 9], Gradient: 0.0014535534854046436
[[0, 2], [0, 8], [1, 3], [1, 9]]
n = 0,  E = -7.86266587 H, t = 2.31 s
n = 1,  E = -7.87094621 H, t = 2.88 s
n = 2,  E = -7.87563100 H, t = 2.30 s
n = 3,  E = -7.87829146 H, t = 2.87 s
n = 4,  E = -7.87981705 H, t = 2.30 s
n = 5,  E = -7.88070477 H, t = 2.84 s
n = 6,  E = -7.88123143 H, t = 2.29 s
n = 7,  E = -7.88155161 H, t = 2.83 s
n = 8,  E = -7.88175217 H, t = 2.88 s
n = 9,  E = -7.88188237 H, t = 2.29 s
n = 10,  E = -7.88197041 H, t = 2.88 s
n = 11,  E = -7.88203267 H, t = 2.29 s
n = 12,  E = -7.88207879 H, t = 2.87 s
n = 13,  E = -7.88211452 H, t = 2.29 s
n = 14,  E = -7.88214335 H, t = 2.85 s
n = 15,  E = -7.88216743 H, t = 2.28 s
n = 16,  E = -7.88218814 H, t = 2.81 s
n = 17,  E = -7.88220634 H, t = 2.89 s
n = 18,  E = -7.88222261 H, t = 2.28 s
n = 19,  E = -7.88223734 H, t = 2.88 s
n = 0,  E = -7.86266587 H, t = 0.13 s
n = 1,  E = -7.87094621 H, t = 0.12 s
n = 2,  E = -7.87563100 H, t = 0.13 s
n = 3,  E = -7.87829146 H, t = 0.12 s
n = 4,  E = -7.87981705 H, t = 0.12 s
n = 5,  E = -7.88070477 H, t = 0.13 s
n = 6,  E = -7.88123143 H, t = 0.13 s
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
Excitation : [0, 1, 2, 9], Gradient: 0.034264511701695255
Excitation : [0, 1, 3, 4], Gradient: 0.0
Excitation : [0, 1, 3, 6], Gradient: 0.0
Excitation : [0, 1, 3, 8], Gradient: -0.00856612792542371
Excitation : [0, 1, 4, 5], Gradient: 0.0
Excitation : [0, 1, 5, 8], Gradient: 0.0
Excitation : [0, 1, 6, 7], Gradient: 0.0
Excitation : [0, 1, 7, 8], Gradient: 0.0
Excitation : [0, 1, 8, 9], Gradient: 0.0
[[0, 1, 2, 9], [0, 1, 3, 8]]
Excitation : [0, 2], Gradient: -0.013361843799979095
Excitation : [0, 4], Gradient: 0.0
Excitation : [0, 6], Gradient: 0.0
Excitation : [0, 8], Gradient: 0.008127419311877796
Excitation : [1, 3], Gradient: 9.609881567228473e-06
Excitation : [1, 5], Gradient: -0.004875127086708602
Excitation : [1, 7], Gradient: -0.004875127086708602
Excitation : [1, 9], Gradient: -0.007509748822113286
[[0, 2], [0, 8], [1, 5], [1, 7], [1, 9]]
n = 0,  E = -7.85513767 H, t = 2.55 s
n = 1,  E = -7.85585993 H, t = 2.50 s
n = 2,  E = -7.85642249 H, t = 2.54 s
n = 3,  E = -7.85686535 H, t = 2.49 s
n = 4,  E = -7.85721832 H, t = 2.53 s
n = 5,  E = -7.85750361 H, t = 2.87 s
n = 6,  E = -7.85773773 H, t = 2.22 s
n = 7,  E = -7.85793296 H, t = 2.79 s
n = 8,  E = -7.85809846 H, t = 2.51 s
n = 9,  E = -7.85824102 H, t = 2.22 s
n = 10,  E = -7.85836572 H, t = 2.81 s
n = 11,  E = -7.85847636 H, t = 2.52 s
n = 12,  E = -7.85857579 H, t = 2.22 s
n = 13,  E = -7.85866614 H, t = 2.75 s
n = 14,  E = -7.85874902 H, t = 2.55 s
n = 15,  E = -7.85882566 H, t = 2.19 s
n = 16,  E = -7.85889701 H, t = 2.81 s
n = 17,  E = -7.85896378 H, t = 2.55 s
n = 18,  E = -7.85902654 H, t = 2.23 s
n = 19,  E = -7.85908573 H, t = 2.84 s
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/math/multi_dispatch.py:66: UserWarning: Contains tensors of types {'autograd', 'scipy'}; dispatch will prioritize TensorFlow and PyTorch over autograd. Consider replacing Autograd with vanilla NumPy.
n = 0,  E = -7.86266587 H, t = 0.12 s
n = 1,  E = -7.86373056 H, t = 0.12 s
n = 2,  E = -7.86443636 H, t = 0.12 s
n = 3,  E = -7.86490587 H, t = 0.12 s
n = 4,  E = -7.86521992 H, t = 0.12 s
n = 5,  E = -7.86543166 H, t = 0.12 s
n = 6,  E = -7.86557597 H, t = 0.12 s
n = 7,  E = -7.86567575 H, t = 0.12 s
n = 8,  E = -7.86574604 H, t = 0.12 s
n = 9,  E = -7.86579669 H, t = 0.12 s
n = 10,  E = -7.86583418 H, t = 0.12 s
n = 11,  E = -7.86586277 H, t = 0.12 s
n = 12,  E = -7.86588528 H, t = 0.12 s
n = 13,  E = -7.86590357 H, t = 0.12 s
n = 14,  E = -7.86591886 H, t = 0.12 s
n = 15,  E = -7.86593199 H, t = 0.12 s
n = 16,  E = -7.86594350 H, t = 0.12 s
n = 17,  E = -7.86595377 H, t = 0.12 s
n = 18,  E = -7.86596307 H, t = 0.12 s
 </code>
 </pre>
 </details>

---

## 12. tutorial_general_parshift.html <a name="demo11"></a>

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

## 13. tutorial_classical_shadows.html <a name="demo12"></a>

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

## 14. tutorial_data_reuploading_classifier.html <a name="demo13"></a>

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

## 15. tutorial_expressivity_fourier_series.html <a name="demo14"></a>

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

## 16. tutorial_quantum_transfer_learning.html <a name="demo15"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
  9%|8         | 3.95M/44.7M [00:00<00:01, 41.3MB/s]
 18%|#7        | 7.90M/44.7M [00:00<00:00, 39.5MB/s]
 30%|###       | 13.4M/44.7M [00:00<00:00, 47.4MB/s]
 43%|####2     | 19.0M/44.7M [00:00<00:00, 51.6MB/s]
 55%|#####5    | 24.6M/44.7M [00:00<00:00, 53.5MB/s]
 67%|######7   | 29.9M/44.7M [00:00<00:00, 54.4MB/s]
 82%|########1 | 36.5M/44.7M [00:00<00:00, 59.1MB/s]
 94%|#########4| 42.2M/44.7M [00:00<00:00, 56.4MB/s]
100%|##########| 44.7M/44.7M [00:00<00:00, 54.2MB/s]
Training started:
Phase: train Epoch: 1/3 Iter: 1/62 Batch time: 0.2180
Phase: train Epoch: 1/3 Iter: 2/62 Batch time: 0.1936
Phase: train Epoch: 1/3 Iter: 3/62 Batch time: 0.1970
Phase: train Epoch: 1/3 Iter: 4/62 Batch time: 0.1925
Phase: train Epoch: 1/3 Iter: 5/62 Batch time: 0.1944
Phase: train Epoch: 1/3 Iter: 6/62 Batch time: 0.1875
Phase: train Epoch: 1/3 Iter: 7/62 Batch time: 0.1904
Phase: train Epoch: 1/3 Iter: 8/62 Batch time: 0.1876
Phase: train Epoch: 1/3 Iter: 9/62 Batch time: 0.1888
Phase: train Epoch: 1/3 Iter: 10/62 Batch time: 0.1909
Phase: train Epoch: 1/3 Iter: 11/62 Batch time: 0.1890
Phase: train Epoch: 1/3 Iter: 12/62 Batch time: 0.1854
Phase: train Epoch: 1/3 Iter: 13/62 Batch time: 0.2038
Phase: train Epoch: 1/3 Iter: 14/62 Batch time: 0.1984
Phase: train Epoch: 1/3 Iter: 15/62 Batch time: 0.1900
Phase: train Epoch: 1/3 Iter: 16/62 Batch time: 0.1868
Phase: train Epoch: 1/3 Iter: 17/62 Batch time: 0.2021
Phase: train Epoch: 1/3 Iter: 18/62 Batch time: 0.2204
Phase: train Epoch: 1/3 Iter: 19/62 Batch time: 0.1899
Phase: train Epoch: 1/3 Iter: 20/62 Batch time: 0.1875
Phase: train Epoch: 1/3 Iter: 21/62 Batch time: 0.1889
Phase: train Epoch: 1/3 Iter: 22/62 Batch time: 0.1888
Phase: train Epoch: 1/3 Iter: 23/62 Batch time: 0.1942
Phase: train Epoch: 1/3 Iter: 24/62 Batch time: 0.1919
Phase: train Epoch: 1/3 Iter: 25/62 Batch time: 0.1879
Phase: train Epoch: 1/3 Iter: 26/62 Batch time: 0.1874
Phase: train Epoch: 1/3 Iter: 27/62 Batch time: 0.1944
Phase: train Epoch: 1/3 Iter: 28/62 Batch time: 0.1922
Phase: train Epoch: 1/3 Iter: 29/62 Batch time: 0.1891
Phase: train Epoch: 1/3 Iter: 30/62 Batch time: 0.1879
Phase: train Epoch: 1/3 Iter: 31/62 Batch time: 0.1931
Phase: train Epoch: 1/3 Iter: 32/62 Batch time: 0.1892
Phase: train Epoch: 1/3 Iter: 33/62 Batch time: 0.1886
Phase: train Epoch: 1/3 Iter: 34/62 Batch time: 0.1890
Phase: train Epoch: 1/3 Iter: 35/62 Batch time: 0.1879
Phase: train Epoch: 1/3 Iter: 36/62 Batch time: 0.1964
Phase: train Epoch: 1/3 Iter: 37/62 Batch time: 0.1982
Phase: train Epoch: 1/3 Iter: 38/62 Batch time: 0.1905
Phase: train Epoch: 1/3 Iter: 39/62 Batch time: 0.1877
Phase: train Epoch: 1/3 Iter: 40/62 Batch time: 0.1954
Phase: train Epoch: 1/3 Iter: 41/62 Batch time: 0.1891
Phase: train Epoch: 1/3 Iter: 42/62 Batch time: 0.1850
Phase: train Epoch: 1/3 Iter: 43/62 Batch time: 0.1965
Phase: train Epoch: 1/3 Iter: 44/62 Batch time: 0.1940
Phase: train Epoch: 1/3 Iter: 45/62 Batch time: 0.1926
Phase: train Epoch: 1/3 Iter: 46/62 Batch time: 0.1884
Phase: train Epoch: 1/3 Iter: 47/62 Batch time: 0.1932
Phase: train Epoch: 1/3 Iter: 48/62 Batch time: 0.1909
Phase: train Epoch: 1/3 Iter: 49/62 Batch time: 0.1914
Phase: train Epoch: 1/3 Iter: 50/62 Batch time: 0.1935
Phase: train Epoch: 1/3 Iter: 51/62 Batch time: 0.1903
Phase: train Epoch: 1/3 Iter: 52/62 Batch time: 0.1882
Phase: train Epoch: 1/3 Iter: 53/62 Batch time: 0.1878
Phase: train Epoch: 1/3 Iter: 54/62 Batch time: 0.1875
Phase: train Epoch: 1/3 Iter: 55/62 Batch time: 0.1889
Phase: train Epoch: 1/3 Iter: 56/62 Batch time: 0.1876
Phase: train Epoch: 1/3 Iter: 57/62 Batch time: 0.1911
Phase: train Epoch: 1/3 Iter: 58/62 Batch time: 0.1906
Phase: train Epoch: 1/3 Iter: 59/62 Batch time: 0.1910
Phase: train Epoch: 1/3 Iter: 60/62 Batch time: 0.1893
Phase: train Epoch: 1/3 Iter: 61/62 Batch time: 0.1899
Phase: train Epoch: 1/3 Loss: 0.6993 Acc: 0.5246
Phase: validation Epoch: 1/3 Iter: 1/39 Batch time: 0.1387
Phase: validation Epoch: 1/3 Iter: 2/39 Batch time: 0.1367
Phase: validation Epoch: 1/3 Iter: 3/39 Batch time: 0.1366
Phase: validation Epoch: 1/3 Iter: 4/39 Batch time: 0.1375
Phase: validation Epoch: 1/3 Iter: 5/39 Batch time: 0.1337
Phase: validation Epoch: 1/3 Iter: 6/39 Batch time: 0.1343
Phase: validation Epoch: 1/3 Iter: 7/39 Batch time: 0.1329
Phase: validation Epoch: 1/3 Iter: 8/39 Batch time: 0.1343
Phase: validation Epoch: 1/3 Iter: 9/39 Batch time: 0.1341
Phase: validation Epoch: 1/3 Iter: 10/39 Batch time: 0.1357
Phase: validation Epoch: 1/3 Iter: 11/39 Batch time: 0.1339
Phase: validation Epoch: 1/3 Iter: 12/39 Batch time: 0.1382
Phase: validation Epoch: 1/3 Iter: 13/39 Batch time: 0.1346
Phase: validation Epoch: 1/3 Iter: 14/39 Batch time: 0.1400
Phase: validation Epoch: 1/3 Iter: 15/39 Batch time: 0.1390
Phase: validation Epoch: 1/3 Iter: 16/39 Batch time: 0.1355
Phase: validation Epoch: 1/3 Iter: 17/39 Batch time: 0.1346
Phase: validation Epoch: 1/3 Iter: 18/39 Batch time: 0.1356
Phase: validation Epoch: 1/3 Iter: 19/39 Batch time: 0.1326
Phase: validation Epoch: 1/3 Iter: 20/39 Batch time: 0.1341
Phase: validation Epoch: 1/3 Iter: 21/39 Batch time: 0.1343
Phase: validation Epoch: 1/3 Iter: 22/39 Batch time: 0.1347
Phase: validation Epoch: 1/3 Iter: 23/39 Batch time: 0.1338
Phase: validation Epoch: 1/3 Iter: 24/39 Batch time: 0.1350
Phase: validation Epoch: 1/3 Iter: 25/39 Batch time: 0.1339
Phase: validation Epoch: 1/3 Iter: 26/39 Batch time: 0.1354
Phase: validation Epoch: 1/3 Iter: 27/39 Batch time: 0.1374
Phase: validation Epoch: 1/3 Iter: 28/39 Batch time: 0.1431
Phase: validation Epoch: 1/3 Iter: 29/39 Batch time: 0.1375
Phase: validation Epoch: 1/3 Iter: 30/39 Batch time: 0.1382
Phase: validation Epoch: 1/3 Iter: 31/39 Batch time: 0.1366
Phase: validation Epoch: 1/3 Iter: 32/39 Batch time: 0.1367
Phase: validation Epoch: 1/3 Iter: 33/39 Batch time: 0.2580
Phase: validation Epoch: 1/3 Iter: 34/39 Batch time: 0.1365
Phase: validation Epoch: 1/3 Iter: 35/39 Batch time: 0.1347
Phase: validation Epoch: 1/3 Iter: 36/39 Batch time: 0.1370
Phase: validation Epoch: 1/3 Iter: 37/39 Batch time: 0.1368
Phase: validation Epoch: 1/3 Iter: 38/39 Batch time: 0.1404
Phase: validation Epoch: 1/3 Iter: 39/39 Batch time: 0.0525
Phase: validation   Epoch: 1/3 Loss: 0.6432 Acc: 0.6536
Phase: train Epoch: 2/3 Iter: 1/62 Batch time: 0.1898
Phase: train Epoch: 2/3 Iter: 2/62 Batch time: 0.1915
Phase: train Epoch: 2/3 Iter: 3/62 Batch time: 0.1967
Phase: train Epoch: 2/3 Iter: 4/62 Batch time: 0.1903
Phase: train Epoch: 2/3 Iter: 5/62 Batch time: 0.1890
Phase: train Epoch: 2/3 Iter: 6/62 Batch time: 0.1897
Phase: train Epoch: 2/3 Iter: 7/62 Batch time: 0.1888
Phase: train Epoch: 2/3 Iter: 8/62 Batch time: 0.1942
Phase: train Epoch: 2/3 Iter: 9/62 Batch time: 0.1918
Phase: train Epoch: 2/3 Iter: 10/62 Batch time: 0.1904
Phase: train Epoch: 2/3 Iter: 11/62 Batch time: 0.1868
Phase: train Epoch: 2/3 Iter: 12/62 Batch time: 0.1886
Phase: train Epoch: 2/3 Iter: 13/62 Batch time: 0.1881
Phase: train Epoch: 2/3 Iter: 14/62 Batch time: 0.1888
Phase: train Epoch: 2/3 Iter: 15/62 Batch time: 0.1862
Phase: train Epoch: 2/3 Iter: 16/62 Batch time: 0.1866
Phase: train Epoch: 2/3 Iter: 17/62 Batch time: 0.1926
Phase: train Epoch: 2/3 Iter: 18/62 Batch time: 0.1883
Phase: train Epoch: 2/3 Iter: 19/62 Batch time: 0.1919
Phase: train Epoch: 2/3 Iter: 20/62 Batch time: 0.1882
Phase: train Epoch: 2/3 Iter: 21/62 Batch time: 0.1887
Phase: train Epoch: 2/3 Iter: 22/62 Batch time: 0.1905
Phase: train Epoch: 2/3 Iter: 23/62 Batch time: 0.1890
Phase: train Epoch: 2/3 Iter: 24/62 Batch time: 0.1946
Phase: train Epoch: 2/3 Iter: 25/62 Batch time: 0.1906
Phase: train Epoch: 2/3 Iter: 26/62 Batch time: 0.1951
Phase: train Epoch: 2/3 Iter: 27/62 Batch time: 0.2007
Phase: train Epoch: 2/3 Iter: 28/62 Batch time: 0.1978
Phase: train Epoch: 2/3 Iter: 29/62 Batch time: 0.1921
Phase: train Epoch: 2/3 Iter: 30/62 Batch time: 0.1879
Phase: train Epoch: 2/3 Iter: 31/62 Batch time: 0.1902
Phase: train Epoch: 2/3 Iter: 32/62 Batch time: 0.1860
Phase: train Epoch: 2/3 Iter: 33/62 Batch time: 0.1880
Phase: train Epoch: 2/3 Iter: 34/62 Batch time: 0.1914
Phase: train Epoch: 2/3 Iter: 35/62 Batch time: 0.1872
Phase: train Epoch: 2/3 Iter: 36/62 Batch time: 0.1917
Phase: train Epoch: 2/3 Iter: 37/62 Batch time: 0.1911
Phase: train Epoch: 2/3 Iter: 38/62 Batch time: 0.1911
Phase: train Epoch: 2/3 Iter: 39/62 Batch time: 0.1881
Phase: train Epoch: 2/3 Iter: 40/62 Batch time: 0.1880
Phase: train Epoch: 2/3 Iter: 41/62 Batch time: 0.1896
Phase: train Epoch: 2/3 Iter: 42/62 Batch time: 0.1904
Phase: train Epoch: 2/3 Iter: 43/62 Batch time: 0.1911
Phase: train Epoch: 2/3 Iter: 44/62 Batch time: 0.1881
Phase: train Epoch: 2/3 Iter: 45/62 Batch time: 0.1900
Phase: train Epoch: 2/3 Iter: 46/62 Batch time: 0.1955
Phase: train Epoch: 2/3 Iter: 47/62 Batch time: 0.1939
Phase: train Epoch: 2/3 Iter: 48/62 Batch time: 0.2031
Phase: train Epoch: 2/3 Iter: 49/62 Batch time: 0.1920
Phase: train Epoch: 2/3 Iter: 50/62 Batch time: 0.1919
Phase: train Epoch: 2/3 Iter: 51/62 Batch time: 0.1913
Phase: train Epoch: 2/3 Iter: 52/62 Batch time: 0.1972
Phase: train Epoch: 2/3 Iter: 53/62 Batch time: 0.1911
Phase: train Epoch: 2/3 Iter: 54/62 Batch time: 0.1991
Phase: train Epoch: 2/3 Iter: 55/62 Batch time: 0.1977
Phase: train Epoch: 2/3 Iter: 56/62 Batch time: 0.1918
Phase: train Epoch: 2/3 Iter: 57/62 Batch time: 0.1887
Phase: train Epoch: 2/3 Iter: 58/62 Batch time: 0.1883
Phase: train Epoch: 2/3 Iter: 59/62 Batch time: 0.1882
Phase: train Epoch: 2/3 Iter: 60/62 Batch time: 0.1890
Phase: train Epoch: 2/3 Iter: 61/62 Batch time: 0.1878
Phase: train Epoch: 2/3 Loss: 0.6141 Acc: 0.7049
Phase: validation Epoch: 2/3 Iter: 1/39 Batch time: 0.1364
Phase: validation Epoch: 2/3 Iter: 2/39 Batch time: 0.1325
Phase: validation Epoch: 2/3 Iter: 3/39 Batch time: 0.1424
Phase: validation Epoch: 2/3 Iter: 4/39 Batch time: 0.1342
Phase: validation Epoch: 2/3 Iter: 5/39 Batch time: 0.1342
Phase: validation Epoch: 2/3 Iter: 6/39 Batch time: 0.1367
Phase: validation Epoch: 2/3 Iter: 7/39 Batch time: 0.1345
Phase: validation Epoch: 2/3 Iter: 8/39 Batch time: 0.1430
Phase: validation Epoch: 2/3 Iter: 9/39 Batch time: 0.1354
Phase: validation Epoch: 2/3 Iter: 10/39 Batch time: 0.1336
Phase: validation Epoch: 2/3 Iter: 11/39 Batch time: 0.1318
Phase: validation Epoch: 2/3 Iter: 12/39 Batch time: 0.1347
Phase: validation Epoch: 2/3 Iter: 13/39 Batch time: 0.1320
Phase: validation Epoch: 2/3 Iter: 14/39 Batch time: 0.1343
Phase: validation Epoch: 2/3 Iter: 15/39 Batch time: 0.1335
Phase: validation Epoch: 2/3 Iter: 16/39 Batch time: 0.1334
Phase: validation Epoch: 2/3 Iter: 17/39 Batch time: 0.1339
Phase: validation Epoch: 2/3 Iter: 18/39 Batch time: 0.1341
Phase: validation Epoch: 2/3 Iter: 19/39 Batch time: 0.1513
Phase: validation Epoch: 2/3 Iter: 20/39 Batch time: 0.1337
Phase: validation Epoch: 2/3 Iter: 21/39 Batch time: 0.1330
Phase: validation Epoch: 2/3 Iter: 22/39 Batch time: 0.1373
Phase: validation Epoch: 2/3 Iter: 23/39 Batch time: 0.1416
Phase: validation Epoch: 2/3 Iter: 24/39 Batch time: 0.1379
Phase: validation Epoch: 2/3 Iter: 25/39 Batch time: 0.1333
Phase: validation Epoch: 2/3 Iter: 26/39 Batch time: 0.1348
Phase: validation Epoch: 2/3 Iter: 27/39 Batch time: 0.1340
Phase: validation Epoch: 2/3 Iter: 28/39 Batch time: 0.1338
Phase: validation Epoch: 2/3 Iter: 29/39 Batch time: 0.1331
Phase: validation Epoch: 2/3 Iter: 30/39 Batch time: 0.1340
Phase: validation Epoch: 2/3 Iter: 31/39 Batch time: 0.1324
Phase: validation Epoch: 2/3 Iter: 32/39 Batch time: 0.1321
Phase: validation Epoch: 2/3 Iter: 33/39 Batch time: 0.1337
Phase: validation Epoch: 2/3 Iter: 34/39 Batch time: 0.1340
Phase: validation Epoch: 2/3 Iter: 35/39 Batch time: 0.1326
Phase: validation Epoch: 2/3 Iter: 36/39 Batch time: 0.1335
Phase: validation Epoch: 2/3 Iter: 37/39 Batch time: 0.1324
Phase: validation Epoch: 2/3 Iter: 38/39 Batch time: 0.1326
Phase: validation Epoch: 2/3 Iter: 39/39 Batch time: 0.0481
Phase: validation   Epoch: 2/3 Loss: 0.5392 Acc: 0.8235
Phase: train Epoch: 3/3 Iter: 1/62 Batch time: 0.1921
Phase: train Epoch: 3/3 Iter: 2/62 Batch time: 0.1779
Phase: train Epoch: 3/3 Iter: 3/62 Batch time: 0.1828
Phase: train Epoch: 3/3 Iter: 4/62 Batch time: 0.1801
Phase: train Epoch: 3/3 Iter: 5/62 Batch time: 0.1757
Phase: train Epoch: 3/3 Iter: 6/62 Batch time: 0.1895
Phase: train Epoch: 3/3 Iter: 7/62 Batch time: 0.1882
Phase: train Epoch: 3/3 Iter: 8/62 Batch time: 0.1855
Phase: train Epoch: 3/3 Iter: 9/62 Batch time: 0.1880
Phase: train Epoch: 3/3 Iter: 10/62 Batch time: 0.1898
Phase: train Epoch: 3/3 Iter: 11/62 Batch time: 0.1891
Phase: train Epoch: 3/3 Iter: 12/62 Batch time: 0.1884
Phase: train Epoch: 3/3 Iter: 13/62 Batch time: 0.1907
Phase: train Epoch: 3/3 Iter: 14/62 Batch time: 0.1891
Phase: train Epoch: 3/3 Iter: 15/62 Batch time: 0.1902
Phase: train Epoch: 3/3 Iter: 16/62 Batch time: 0.1876
Phase: train Epoch: 3/3 Iter: 17/62 Batch time: 0.1890
Phase: train Epoch: 3/3 Iter: 18/62 Batch time: 0.1876
Phase: train Epoch: 3/3 Iter: 19/62 Batch time: 0.1881
Phase: train Epoch: 3/3 Iter: 20/62 Batch time: 0.1874
Phase: train Epoch: 3/3 Iter: 21/62 Batch time: 0.1859
Phase: train Epoch: 3/3 Iter: 22/62 Batch time: 0.1869
Phase: train Epoch: 3/3 Iter: 23/62 Batch time: 0.1864
Phase: train Epoch: 3/3 Iter: 24/62 Batch time: 0.1856
Phase: train Epoch: 3/3 Iter: 25/62 Batch time: 0.1885
Phase: train Epoch: 3/3 Iter: 26/62 Batch time: 0.1887
Phase: train Epoch: 3/3 Iter: 27/62 Batch time: 0.1890
Phase: train Epoch: 3/3 Iter: 28/62 Batch time: 0.1869
Phase: train Epoch: 3/3 Iter: 29/62 Batch time: 0.1930
Phase: train Epoch: 3/3 Iter: 30/62 Batch time: 0.1882
Phase: train Epoch: 3/3 Iter: 31/62 Batch time: 0.1880
Phase: train Epoch: 3/3 Iter: 32/62 Batch time: 0.1869
Phase: train Epoch: 3/3 Iter: 33/62 Batch time: 0.1884
Phase: train Epoch: 3/3 Iter: 34/62 Batch time: 0.1885
Phase: train Epoch: 3/3 Iter: 35/62 Batch time: 0.1882
Phase: train Epoch: 3/3 Iter: 36/62 Batch time: 0.1897
Phase: train Epoch: 3/3 Iter: 37/62 Batch time: 0.1880
Phase: train Epoch: 3/3 Iter: 38/62 Batch time: 0.1893
Phase: train Epoch: 3/3 Iter: 39/62 Batch time: 0.1878
Phase: train Epoch: 3/3 Iter: 40/62 Batch time: 0.1877
Phase: train Epoch: 3/3 Iter: 41/62 Batch time: 0.1884
Phase: train Epoch: 3/3 Iter: 42/62 Batch time: 0.1863
Phase: train Epoch: 3/3 Iter: 43/62 Batch time: 0.1869
Phase: train Epoch: 3/3 Iter: 44/62 Batch time: 0.1857
Phase: train Epoch: 3/3 Iter: 45/62 Batch time: 0.1855
Phase: train Epoch: 3/3 Iter: 46/62 Batch time: 0.1895
Phase: train Epoch: 3/3 Iter: 47/62 Batch time: 0.1878
Phase: train Epoch: 3/3 Iter: 48/62 Batch time: 0.1890
Phase: train Epoch: 3/3 Iter: 49/62 Batch time: 0.1886
Phase: train Epoch: 3/3 Iter: 50/62 Batch time: 0.1899
Phase: train Epoch: 3/3 Iter: 51/62 Batch time: 0.1870
Phase: train Epoch: 3/3 Iter: 52/62 Batch time: 0.1883
Phase: train Epoch: 3/3 Iter: 53/62 Batch time: 0.1884
Phase: train Epoch: 3/3 Iter: 54/62 Batch time: 0.1886
Phase: train Epoch: 3/3 Iter: 55/62 Batch time: 0.1871
Phase: train Epoch: 3/3 Iter: 56/62 Batch time: 0.1871
Phase: train Epoch: 3/3 Iter: 57/62 Batch time: 0.1923
Phase: train Epoch: 3/3 Iter: 58/62 Batch time: 0.1889
Phase: train Epoch: 3/3 Iter: 59/62 Batch time: 0.1875
Phase: train Epoch: 3/3 Iter: 60/62 Batch time: 0.1878
Phase: train Epoch: 3/3 Iter: 61/62 Batch time: 0.1882
Phase: train Epoch: 3/3 Loss: 0.5652 Acc: 0.7336
Phase: validation Epoch: 3/3 Iter: 1/39 Batch time: 0.1394
Phase: validation Epoch: 3/3 Iter: 2/39 Batch time: 0.1341
Phase: validation Epoch: 3/3 Iter: 3/39 Batch time: 0.1350
Phase: validation Epoch: 3/3 Iter: 4/39 Batch time: 0.1346
Phase: validation Epoch: 3/3 Iter: 5/39 Batch time: 0.1340
Phase: validation Epoch: 3/3 Iter: 6/39 Batch time: 0.1393
Phase: validation Epoch: 3/3 Iter: 7/39 Batch time: 0.1359
Phase: validation Epoch: 3/3 Iter: 8/39 Batch time: 0.1378
Phase: validation Epoch: 3/3 Iter: 9/39 Batch time: 0.1364
Phase: validation Epoch: 3/3 Iter: 10/39 Batch time: 0.1349
Phase: validation Epoch: 3/3 Iter: 11/39 Batch time: 0.1357
Phase: validation Epoch: 3/3 Iter: 12/39 Batch time: 0.1348
Phase: validation Epoch: 3/3 Iter: 13/39 Batch time: 0.1345
Phase: validation Epoch: 3/3 Iter: 14/39 Batch time: 0.1293
Phase: validation Epoch: 3/3 Iter: 15/39 Batch time: 0.1298
Phase: validation Epoch: 3/3 Iter: 16/39 Batch time: 0.1282
Phase: validation Epoch: 3/3 Iter: 17/39 Batch time: 0.1300
Phase: validation Epoch: 3/3 Iter: 18/39 Batch time: 0.1289
Phase: validation Epoch: 3/3 Iter: 19/39 Batch time: 0.1296
Phase: validation Epoch: 3/3 Iter: 20/39 Batch time: 0.1304
Phase: validation Epoch: 3/3 Iter: 21/39 Batch time: 0.1299
Phase: validation Epoch: 3/3 Iter: 22/39 Batch time: 0.1265
Phase: validation Epoch: 3/3 Iter: 23/39 Batch time: 0.1274
Phase: validation Epoch: 3/3 Iter: 24/39 Batch time: 0.1278
Phase: validation Epoch: 3/3 Iter: 25/39 Batch time: 0.1284
Phase: validation Epoch: 3/3 Iter: 26/39 Batch time: 0.1271
Phase: validation Epoch: 3/3 Iter: 27/39 Batch time: 0.1302
Phase: validation Epoch: 3/3 Iter: 28/39 Batch time: 0.1294
Phase: validation Epoch: 3/3 Iter: 29/39 Batch time: 0.1321
Phase: validation Epoch: 3/3 Iter: 30/39 Batch time: 0.1310
Phase: validation Epoch: 3/3 Iter: 31/39 Batch time: 0.1282
Phase: validation Epoch: 3/3 Iter: 32/39 Batch time: 0.1275
Phase: validation Epoch: 3/3 Iter: 33/39 Batch time: 0.1263
Phase: validation Epoch: 3/3 Iter: 34/39 Batch time: 0.1241
Phase: validation Epoch: 3/3 Iter: 35/39 Batch time: 0.1252
Phase: validation Epoch: 3/3 Iter: 36/39 Batch time: 0.1250
Phase: validation Epoch: 3/3 Iter: 37/39 Batch time: 0.1251
Phase: validation Epoch: 3/3 Iter: 38/39 Batch time: 0.1253
Phase: validation Epoch: 3/3 Iter: 39/39 Batch time: 0.0432
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
 18%|#7        | 7.94M/44.7M [00:00<00:00, 82.9MB/s]
 35%|###5      | 15.9M/44.7M [00:00<00:00, 65.3MB/s]
 50%|####9     | 22.3M/44.7M [00:00<00:00, 64.8MB/s]
 66%|######5   | 29.3M/44.7M [00:00<00:00, 67.8MB/s]
 84%|########4 | 37.7M/44.7M [00:00<00:00, 74.7MB/s]
100%|##########| 44.7M/44.7M [00:00<00:00, 65.3MB/s]
Training started:
Phase: train Epoch: 1/3 Iter: 1/62 Batch time: 0.3963
Phase: train Epoch: 1/3 Iter: 2/62 Batch time: 0.3845
Phase: train Epoch: 1/3 Iter: 3/62 Batch time: 0.3864
Phase: train Epoch: 1/3 Iter: 4/62 Batch time: 0.3811
Phase: train Epoch: 1/3 Iter: 5/62 Batch time: 0.3880
Phase: train Epoch: 1/3 Iter: 6/62 Batch time: 0.3849
Phase: train Epoch: 1/3 Iter: 7/62 Batch time: 0.3795
Phase: train Epoch: 1/3 Iter: 8/62 Batch time: 0.3832
Phase: train Epoch: 1/3 Iter: 9/62 Batch time: 0.3791
Phase: train Epoch: 1/3 Iter: 10/62 Batch time: 0.3800
Phase: train Epoch: 1/3 Iter: 11/62 Batch time: 0.3745
Phase: train Epoch: 1/3 Iter: 12/62 Batch time: 0.3962
Phase: train Epoch: 1/3 Iter: 13/62 Batch time: 0.3739
Phase: train Epoch: 1/3 Iter: 14/62 Batch time: 0.3976
Phase: train Epoch: 1/3 Iter: 15/62 Batch time: 0.3838
Phase: train Epoch: 1/3 Iter: 16/62 Batch time: 0.3759
Phase: train Epoch: 1/3 Iter: 17/62 Batch time: 0.3810
Phase: train Epoch: 1/3 Iter: 18/62 Batch time: 0.3776
Phase: train Epoch: 1/3 Iter: 19/62 Batch time: 0.3795
Phase: train Epoch: 1/3 Iter: 20/62 Batch time: 0.3841
Phase: train Epoch: 1/3 Iter: 21/62 Batch time: 0.3754
Phase: train Epoch: 1/3 Iter: 22/62 Batch time: 0.3905
Phase: train Epoch: 1/3 Iter: 23/62 Batch time: 0.3813
Phase: train Epoch: 1/3 Iter: 24/62 Batch time: 0.3762
Phase: train Epoch: 1/3 Iter: 25/62 Batch time: 0.3772
Phase: train Epoch: 1/3 Iter: 26/62 Batch time: 0.3857
Phase: train Epoch: 1/3 Iter: 27/62 Batch time: 0.3849
Phase: train Epoch: 1/3 Iter: 28/62 Batch time: 0.4035
Phase: train Epoch: 1/3 Iter: 29/62 Batch time: 0.4236
Phase: train Epoch: 1/3 Iter: 30/62 Batch time: 0.3853
Phase: train Epoch: 1/3 Iter: 31/62 Batch time: 0.3825
Phase: train Epoch: 1/3 Iter: 32/62 Batch time: 0.3859
Phase: train Epoch: 1/3 Iter: 33/62 Batch time: 0.3798
Phase: train Epoch: 1/3 Iter: 34/62 Batch time: 0.3868
Phase: train Epoch: 1/3 Iter: 35/62 Batch time: 0.3890
Phase: train Epoch: 1/3 Iter: 36/62 Batch time: 0.3924
Phase: train Epoch: 1/3 Iter: 37/62 Batch time: 0.3987
Phase: train Epoch: 1/3 Iter: 38/62 Batch time: 0.3833
Phase: train Epoch: 1/3 Iter: 39/62 Batch time: 0.3800
Phase: train Epoch: 1/3 Iter: 40/62 Batch time: 0.3749
Phase: train Epoch: 1/3 Iter: 41/62 Batch time: 0.3892
Phase: train Epoch: 1/3 Iter: 42/62 Batch time: 0.3933
Phase: train Epoch: 1/3 Iter: 43/62 Batch time: 0.3770
Phase: train Epoch: 1/3 Iter: 44/62 Batch time: 0.3803
Phase: train Epoch: 1/3 Iter: 45/62 Batch time: 0.3858
Phase: train Epoch: 1/3 Iter: 46/62 Batch time: 0.3990
Phase: train Epoch: 1/3 Iter: 47/62 Batch time: 0.3859
Phase: train Epoch: 1/3 Iter: 48/62 Batch time: 0.3780
Phase: train Epoch: 1/3 Iter: 49/62 Batch time: 0.4060
Phase: train Epoch: 1/3 Iter: 50/62 Batch time: 0.3794
Phase: train Epoch: 1/3 Iter: 51/62 Batch time: 0.3774
Phase: train Epoch: 1/3 Iter: 52/62 Batch time: 0.3842
Phase: train Epoch: 1/3 Iter: 53/62 Batch time: 0.3915
Phase: train Epoch: 1/3 Iter: 54/62 Batch time: 0.3910
Phase: train Epoch: 1/3 Iter: 55/62 Batch time: 0.3880
Phase: train Epoch: 1/3 Iter: 56/62 Batch time: 0.3864
Phase: train Epoch: 1/3 Iter: 57/62 Batch time: 0.4135
Phase: train Epoch: 1/3 Iter: 58/62 Batch time: 0.3924
Phase: train Epoch: 1/3 Iter: 59/62 Batch time: 0.3956
Phase: train Epoch: 1/3 Iter: 60/62 Batch time: 0.3821
Phase: train Epoch: 1/3 Iter: 61/62 Batch time: 0.3851
Phase: train Epoch: 1/3 Loss: 0.6993 Acc: 0.5246
Phase: validation Epoch: 1/3 Iter: 1/39 Batch time: 0.3327
Phase: validation Epoch: 1/3 Iter: 2/39 Batch time: 0.3088
Phase: validation Epoch: 1/3 Iter: 3/39 Batch time: 0.3232
Phase: validation Epoch: 1/3 Iter: 4/39 Batch time: 0.3012
Phase: validation Epoch: 1/3 Iter: 5/39 Batch time: 0.3207
Phase: validation Epoch: 1/3 Iter: 6/39 Batch time: 0.3036
Phase: validation Epoch: 1/3 Iter: 7/39 Batch time: 0.3147
Phase: validation Epoch: 1/3 Iter: 8/39 Batch time: 0.3056
Phase: validation Epoch: 1/3 Iter: 9/39 Batch time: 0.3125
Phase: validation Epoch: 1/3 Iter: 10/39 Batch time: 0.3104
Phase: validation Epoch: 1/3 Iter: 11/39 Batch time: 0.2999
Phase: validation Epoch: 1/3 Iter: 12/39 Batch time: 0.3064
Phase: validation Epoch: 1/3 Iter: 13/39 Batch time: 0.3017
Phase: validation Epoch: 1/3 Iter: 14/39 Batch time: 0.3039
Phase: validation Epoch: 1/3 Iter: 15/39 Batch time: 0.3049
Phase: validation Epoch: 1/3 Iter: 16/39 Batch time: 0.3049
Phase: validation Epoch: 1/3 Iter: 17/39 Batch time: 0.3104
Phase: validation Epoch: 1/3 Iter: 18/39 Batch time: 0.3131
Phase: validation Epoch: 1/3 Iter: 19/39 Batch time: 0.2991
Phase: validation Epoch: 1/3 Iter: 20/39 Batch time: 0.3036
Phase: validation Epoch: 1/3 Iter: 21/39 Batch time: 0.3004
Phase: validation Epoch: 1/3 Iter: 22/39 Batch time: 0.3037
Phase: validation Epoch: 1/3 Iter: 23/39 Batch time: 0.3041
Phase: validation Epoch: 1/3 Iter: 24/39 Batch time: 0.3278
Phase: validation Epoch: 1/3 Iter: 25/39 Batch time: 0.3294
Phase: validation Epoch: 1/3 Iter: 26/39 Batch time: 0.3063
Phase: validation Epoch: 1/3 Iter: 27/39 Batch time: 0.3052
Phase: validation Epoch: 1/3 Iter: 28/39 Batch time: 0.3059
Phase: validation Epoch: 1/3 Iter: 29/39 Batch time: 0.3024
Phase: validation Epoch: 1/3 Iter: 30/39 Batch time: 0.3128
Phase: validation Epoch: 1/3 Iter: 31/39 Batch time: 0.3128
Phase: validation Epoch: 1/3 Iter: 32/39 Batch time: 0.3020
Phase: validation Epoch: 1/3 Iter: 33/39 Batch time: 0.3035
Phase: validation Epoch: 1/3 Iter: 34/39 Batch time: 0.3030
Phase: validation Epoch: 1/3 Iter: 35/39 Batch time: 0.3160
Phase: validation Epoch: 1/3 Iter: 36/39 Batch time: 0.3750
Phase: validation Epoch: 1/3 Iter: 37/39 Batch time: 0.3334
Phase: validation Epoch: 1/3 Iter: 38/39 Batch time: 0.3128
Phase: validation Epoch: 1/3 Iter: 39/39 Batch time: 0.0947
Phase: validation   Epoch: 1/3 Loss: 0.6432 Acc: 0.6536
Phase: train Epoch: 2/3 Iter: 1/62 Batch time: 0.3745
Phase: train Epoch: 2/3 Iter: 2/62 Batch time: 0.3836
Phase: train Epoch: 2/3 Iter: 3/62 Batch time: 0.3798
Phase: train Epoch: 2/3 Iter: 4/62 Batch time: 0.3868
Phase: train Epoch: 2/3 Iter: 5/62 Batch time: 0.3852
Phase: train Epoch: 2/3 Iter: 6/62 Batch time: 0.3853
Phase: train Epoch: 2/3 Iter: 7/62 Batch time: 0.3837
Phase: train Epoch: 2/3 Iter: 8/62 Batch time: 0.4050
Phase: train Epoch: 2/3 Iter: 9/62 Batch time: 0.3868
Phase: train Epoch: 2/3 Iter: 10/62 Batch time: 0.3929
Phase: train Epoch: 2/3 Iter: 11/62 Batch time: 0.3781
Phase: train Epoch: 2/3 Iter: 12/62 Batch time: 0.3856
Phase: train Epoch: 2/3 Iter: 13/62 Batch time: 0.3807
Phase: train Epoch: 2/3 Iter: 14/62 Batch time: 0.3874
Phase: train Epoch: 2/3 Iter: 15/62 Batch time: 0.3911
Phase: train Epoch: 2/3 Iter: 16/62 Batch time: 0.4021
Phase: train Epoch: 2/3 Iter: 17/62 Batch time: 0.3785
Phase: train Epoch: 2/3 Iter: 18/62 Batch time: 0.3972
Phase: train Epoch: 2/3 Iter: 19/62 Batch time: 0.3841
Phase: train Epoch: 2/3 Iter: 20/62 Batch time: 0.3864
Phase: train Epoch: 2/3 Iter: 21/62 Batch time: 0.3917
Phase: train Epoch: 2/3 Iter: 22/62 Batch time: 0.3897
Phase: train Epoch: 2/3 Iter: 23/62 Batch time: 0.3945
Phase: train Epoch: 2/3 Iter: 24/62 Batch time: 0.3840
Phase: train Epoch: 2/3 Iter: 25/62 Batch time: 0.4009
Phase: train Epoch: 2/3 Iter: 26/62 Batch time: 0.3976
Phase: train Epoch: 2/3 Iter: 27/62 Batch time: 0.3743
Phase: train Epoch: 2/3 Iter: 28/62 Batch time: 0.3799
Phase: train Epoch: 2/3 Iter: 29/62 Batch time: 0.3835
Phase: train Epoch: 2/3 Iter: 30/62 Batch time: 0.3983
Phase: train Epoch: 2/3 Iter: 31/62 Batch time: 0.3798
Phase: train Epoch: 2/3 Iter: 32/62 Batch time: 0.3780
Phase: train Epoch: 2/3 Iter: 33/62 Batch time: 0.3796
Phase: train Epoch: 2/3 Iter: 34/62 Batch time: 0.4157
Phase: train Epoch: 2/3 Iter: 35/62 Batch time: 0.3923
Phase: train Epoch: 2/3 Iter: 36/62 Batch time: 0.3958
Phase: train Epoch: 2/3 Iter: 37/62 Batch time: 0.3844
Phase: train Epoch: 2/3 Iter: 38/62 Batch time: 0.4265
Phase: train Epoch: 2/3 Iter: 39/62 Batch time: 0.3948
Phase: train Epoch: 2/3 Iter: 40/62 Batch time: 0.3899
Phase: train Epoch: 2/3 Iter: 41/62 Batch time: 0.3826
Phase: train Epoch: 2/3 Iter: 42/62 Batch time: 0.3969
Phase: train Epoch: 2/3 Iter: 43/62 Batch time: 0.3778
Phase: train Epoch: 2/3 Iter: 44/62 Batch time: 0.3814
Phase: train Epoch: 2/3 Iter: 45/62 Batch time: 0.3828
Phase: train Epoch: 2/3 Iter: 46/62 Batch time: 0.3891
Phase: train Epoch: 2/3 Iter: 47/62 Batch time: 0.3960
Phase: train Epoch: 2/3 Iter: 48/62 Batch time: 0.3869
Phase: train Epoch: 2/3 Iter: 49/62 Batch time: 0.3880
Phase: train Epoch: 2/3 Iter: 50/62 Batch time: 0.3814
Phase: train Epoch: 2/3 Iter: 51/62 Batch time: 0.3799
Phase: train Epoch: 2/3 Iter: 52/62 Batch time: 0.3791
Phase: train Epoch: 2/3 Iter: 53/62 Batch time: 0.3847
Phase: train Epoch: 2/3 Iter: 54/62 Batch time: 0.3886
Phase: train Epoch: 2/3 Iter: 55/62 Batch time: 0.3844
Phase: train Epoch: 2/3 Iter: 56/62 Batch time: 0.3831
Phase: train Epoch: 2/3 Iter: 57/62 Batch time: 0.3824
Phase: train Epoch: 2/3 Iter: 58/62 Batch time: 0.3908
Phase: train Epoch: 2/3 Iter: 59/62 Batch time: 0.4103
Phase: train Epoch: 2/3 Iter: 60/62 Batch time: 0.3824
Phase: train Epoch: 2/3 Iter: 61/62 Batch time: 0.4093
Phase: train Epoch: 2/3 Loss: 0.6141 Acc: 0.7049
Phase: validation Epoch: 2/3 Iter: 1/39 Batch time: 0.3122
Phase: validation Epoch: 2/3 Iter: 2/39 Batch time: 0.3081
Phase: validation Epoch: 2/3 Iter: 3/39 Batch time: 0.3020
Phase: validation Epoch: 2/3 Iter: 4/39 Batch time: 0.3134
Phase: validation Epoch: 2/3 Iter: 5/39 Batch time: 0.3039
Phase: validation Epoch: 2/3 Iter: 6/39 Batch time: 0.3030
Phase: validation Epoch: 2/3 Iter: 7/39 Batch time: 0.3013
Phase: validation Epoch: 2/3 Iter: 8/39 Batch time: 0.2962
Phase: validation Epoch: 2/3 Iter: 9/39 Batch time: 0.3044
Phase: validation Epoch: 2/3 Iter: 10/39 Batch time: 0.3094
Phase: validation Epoch: 2/3 Iter: 11/39 Batch time: 0.3132
Phase: validation Epoch: 2/3 Iter: 12/39 Batch time: 0.3129
Phase: validation Epoch: 2/3 Iter: 13/39 Batch time: 0.3167
Phase: validation Epoch: 2/3 Iter: 14/39 Batch time: 0.3107
Phase: validation Epoch: 2/3 Iter: 15/39 Batch time: 0.3327
Phase: validation Epoch: 2/3 Iter: 16/39 Batch time: 0.3420
Phase: validation Epoch: 2/3 Iter: 17/39 Batch time: 0.3136
Phase: validation Epoch: 2/3 Iter: 18/39 Batch time: 0.3043
Phase: validation Epoch: 2/3 Iter: 19/39 Batch time: 0.3014
Phase: validation Epoch: 2/3 Iter: 20/39 Batch time: 0.3166
Phase: validation Epoch: 2/3 Iter: 21/39 Batch time: 0.3065
Phase: validation Epoch: 2/3 Iter: 22/39 Batch time: 0.3235
Phase: validation Epoch: 2/3 Iter: 23/39 Batch time: 0.3093
Phase: validation Epoch: 2/3 Iter: 24/39 Batch time: 0.3047
Phase: validation Epoch: 2/3 Iter: 25/39 Batch time: 0.3077
Phase: validation Epoch: 2/3 Iter: 26/39 Batch time: 0.3080
Phase: validation Epoch: 2/3 Iter: 27/39 Batch time: 0.3076
Phase: validation Epoch: 2/3 Iter: 28/39 Batch time: 0.3226
Phase: validation Epoch: 2/3 Iter: 29/39 Batch time: 0.3022
Phase: validation Epoch: 2/3 Iter: 30/39 Batch time: 0.3010
Phase: validation Epoch: 2/3 Iter: 31/39 Batch time: 0.3085
Phase: validation Epoch: 2/3 Iter: 32/39 Batch time: 0.3108
Phase: validation Epoch: 2/3 Iter: 33/39 Batch time: 0.3122
Phase: validation Epoch: 2/3 Iter: 34/39 Batch time: 0.3054
Phase: validation Epoch: 2/3 Iter: 35/39 Batch time: 0.3143
Phase: validation Epoch: 2/3 Iter: 36/39 Batch time: 0.3043
Phase: validation Epoch: 2/3 Iter: 37/39 Batch time: 0.3074
Phase: validation Epoch: 2/3 Iter: 38/39 Batch time: 0.3095
Phase: validation Epoch: 2/3 Iter: 39/39 Batch time: 0.0919
Phase: validation   Epoch: 2/3 Loss: 0.5392 Acc: 0.8235
Phase: train Epoch: 3/3 Iter: 1/62 Batch time: 0.3766
Phase: train Epoch: 3/3 Iter: 2/62 Batch time: 0.4079
Phase: train Epoch: 3/3 Iter: 3/62 Batch time: 0.3971
Phase: train Epoch: 3/3 Iter: 4/62 Batch time: 0.3875
Phase: train Epoch: 3/3 Iter: 5/62 Batch time: 0.3856
Phase: train Epoch: 3/3 Iter: 6/62 Batch time: 0.3937
Phase: train Epoch: 3/3 Iter: 7/62 Batch time: 0.4006
Phase: train Epoch: 3/3 Iter: 8/62 Batch time: 0.3881
Phase: train Epoch: 3/3 Iter: 9/62 Batch time: 0.4226
Phase: train Epoch: 3/3 Iter: 10/62 Batch time: 0.3831
Phase: train Epoch: 3/3 Iter: 11/62 Batch time: 0.3805
Phase: train Epoch: 3/3 Iter: 12/62 Batch time: 0.3859
Phase: train Epoch: 3/3 Iter: 13/62 Batch time: 0.3804
Phase: train Epoch: 3/3 Iter: 14/62 Batch time: 0.3847
Phase: train Epoch: 3/3 Iter: 15/62 Batch time: 0.3913
Phase: train Epoch: 3/3 Iter: 16/62 Batch time: 0.3759
Phase: train Epoch: 3/3 Iter: 17/62 Batch time: 0.4023
Phase: train Epoch: 3/3 Iter: 18/62 Batch time: 0.3803
Phase: train Epoch: 3/3 Iter: 19/62 Batch time: 0.3836
Phase: train Epoch: 3/3 Iter: 20/62 Batch time: 0.3827
Phase: train Epoch: 3/3 Iter: 21/62 Batch time: 0.3798
Phase: train Epoch: 3/3 Iter: 22/62 Batch time: 0.4022
Phase: train Epoch: 3/3 Iter: 23/62 Batch time: 0.3870
Phase: train Epoch: 3/3 Iter: 24/62 Batch time: 0.3774
Phase: train Epoch: 3/3 Iter: 25/62 Batch time: 0.3881
Phase: train Epoch: 3/3 Iter: 26/62 Batch time: 0.3906
Phase: train Epoch: 3/3 Iter: 27/62 Batch time: 0.3879
Phase: train Epoch: 3/3 Iter: 28/62 Batch time: 0.3836
Phase: train Epoch: 3/3 Iter: 29/62 Batch time: 0.3828
Phase: train Epoch: 3/3 Iter: 30/62 Batch time: 0.3937
Phase: train Epoch: 3/3 Iter: 31/62 Batch time: 0.3784
Phase: train Epoch: 3/3 Iter: 32/62 Batch time: 0.4138
Phase: train Epoch: 3/3 Iter: 33/62 Batch time: 0.3795
Phase: train Epoch: 3/3 Iter: 34/62 Batch time: 0.3773
Phase: train Epoch: 3/3 Iter: 35/62 Batch time: 0.3833
Phase: train Epoch: 3/3 Iter: 36/62 Batch time: 0.3797
Phase: train Epoch: 3/3 Iter: 37/62 Batch time: 0.4010
Phase: train Epoch: 3/3 Iter: 38/62 Batch time: 0.3875
Phase: train Epoch: 3/3 Iter: 39/62 Batch time: 0.3832
Phase: train Epoch: 3/3 Iter: 40/62 Batch time: 0.3879
Phase: train Epoch: 3/3 Iter: 41/62 Batch time: 0.3698
Phase: train Epoch: 3/3 Iter: 42/62 Batch time: 0.3784
Phase: train Epoch: 3/3 Iter: 43/62 Batch time: 0.4071
Phase: train Epoch: 3/3 Iter: 44/62 Batch time: 0.3816
Phase: train Epoch: 3/3 Iter: 45/62 Batch time: 0.3995
Phase: train Epoch: 3/3 Iter: 46/62 Batch time: 0.4080
Phase: train Epoch: 3/3 Iter: 47/62 Batch time: 0.4056
Phase: train Epoch: 3/3 Iter: 48/62 Batch time: 0.4442
Phase: train Epoch: 3/3 Iter: 49/62 Batch time: 0.3811
Phase: train Epoch: 3/3 Iter: 50/62 Batch time: 0.4054
Phase: train Epoch: 3/3 Iter: 51/62 Batch time: 0.3862
Phase: train Epoch: 3/3 Iter: 52/62 Batch time: 0.4103
Phase: train Epoch: 3/3 Iter: 53/62 Batch time: 0.3755
Phase: train Epoch: 3/3 Iter: 54/62 Batch time: 0.3836
Phase: train Epoch: 3/3 Iter: 55/62 Batch time: 0.3750
Phase: train Epoch: 3/3 Iter: 56/62 Batch time: 0.3903
Phase: train Epoch: 3/3 Iter: 57/62 Batch time: 0.3862
Phase: train Epoch: 3/3 Iter: 58/62 Batch time: 0.4022
Phase: train Epoch: 3/3 Iter: 59/62 Batch time: 0.4161
Phase: train Epoch: 3/3 Iter: 60/62 Batch time: 0.3941
Phase: train Epoch: 3/3 Iter: 61/62 Batch time: 0.4140
Phase: train Epoch: 3/3 Loss: 0.5652 Acc: 0.7336
Phase: validation Epoch: 3/3 Iter: 1/39 Batch time: 0.3313
Phase: validation Epoch: 3/3 Iter: 2/39 Batch time: 0.3061
Phase: validation Epoch: 3/3 Iter: 3/39 Batch time: 0.3367
Phase: validation Epoch: 3/3 Iter: 4/39 Batch time: 0.3082
Phase: validation Epoch: 3/3 Iter: 5/39 Batch time: 0.3133
Phase: validation Epoch: 3/3 Iter: 6/39 Batch time: 0.3206
Phase: validation Epoch: 3/3 Iter: 7/39 Batch time: 0.3144
Phase: validation Epoch: 3/3 Iter: 8/39 Batch time: 0.3160
Phase: validation Epoch: 3/3 Iter: 9/39 Batch time: 0.3270
Phase: validation Epoch: 3/3 Iter: 10/39 Batch time: 0.3237
Phase: validation Epoch: 3/3 Iter: 11/39 Batch time: 0.3240
Phase: validation Epoch: 3/3 Iter: 12/39 Batch time: 0.3250
Phase: validation Epoch: 3/3 Iter: 13/39 Batch time: 0.3170
Phase: validation Epoch: 3/3 Iter: 14/39 Batch time: 0.3123
Phase: validation Epoch: 3/3 Iter: 15/39 Batch time: 0.3205
Phase: validation Epoch: 3/3 Iter: 16/39 Batch time: 0.3475
Phase: validation Epoch: 3/3 Iter: 17/39 Batch time: 0.3345
Phase: validation Epoch: 3/3 Iter: 18/39 Batch time: 0.3178
Phase: validation Epoch: 3/3 Iter: 19/39 Batch time: 0.3576
Phase: validation Epoch: 3/3 Iter: 20/39 Batch time: 0.3263
Phase: validation Epoch: 3/3 Iter: 21/39 Batch time: 0.3200
Phase: validation Epoch: 3/3 Iter: 22/39 Batch time: 0.3208
Phase: validation Epoch: 3/3 Iter: 23/39 Batch time: 0.3115
Phase: validation Epoch: 3/3 Iter: 24/39 Batch time: 0.3202
Phase: validation Epoch: 3/3 Iter: 25/39 Batch time: 0.3368
Phase: validation Epoch: 3/3 Iter: 26/39 Batch time: 0.3180
Phase: validation Epoch: 3/3 Iter: 27/39 Batch time: 0.3086
Phase: validation Epoch: 3/3 Iter: 28/39 Batch time: 0.3284
Phase: validation Epoch: 3/3 Iter: 29/39 Batch time: 0.3134
Phase: validation Epoch: 3/3 Iter: 30/39 Batch time: 0.3103
Phase: validation Epoch: 3/3 Iter: 31/39 Batch time: 0.3313
Phase: validation Epoch: 3/3 Iter: 32/39 Batch time: 0.3119
Phase: validation Epoch: 3/3 Iter: 33/39 Batch time: 0.3260
Phase: validation Epoch: 3/3 Iter: 34/39 Batch time: 0.3324
Phase: validation Epoch: 3/3 Iter: 35/39 Batch time: 0.3126
Phase: validation Epoch: 3/3 Iter: 36/39 Batch time: 0.3290
Phase: validation Epoch: 3/3 Iter: 37/39 Batch time: 0.3150
Phase: validation Epoch: 3/3 Iter: 38/39 Batch time: 0.3104
Phase: validation Epoch: 3/3 Iter: 39/39 Batch time: 0.0866
Phase: validation   Epoch: 3/3 Loss: 0.4484 Acc: 0.8497
Training completed in 1m 57s
Best test loss: 0.4484 | Best test accuracy: 0.8497
 </code>
 </pre>
 </details>

---

## 17. tutorial_kernel_based_training.html <a name="demo16"></a>

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

## 18. tutorial_quantum_metrology.html <a name="demo17"></a>

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

## 19. tutorial_QGAN.html <a name="demo18"></a>

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

## 20. tutorial_quantum_analytic_descent.html <a name="demo19"></a>

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

## 21. tutorial_quantum_chemistry.html <a name="demo20"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quantum_chemistry.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
(-46.463906788688895+0j) [] +
(-3.5707613291130964e-07+0j) [X0 X1 Y2 Z3 Z4 Y5] +
(-0.00565262097801735+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7] +
(-0.008826368514209834+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.7924939576933602e-06+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.5707613291130964e-07+0j) [X0 X1 X3 X4] +
(-0.00565262097801735+0j) [X0 X1 X3 Z4 Z5 X6] +
(-0.008826368514209834+0j) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.7924939576933602e-06+0j) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.0027458364701868146+0j) [X0 X1 Y4 Y5] +
(-2.447323128922324e-07+0j) [X0 X1 Y4 Z5 Z6 Y7] +
(-7.867765104184687e-07+0j) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0038040661717285394+0j) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.447323128922324e-07+0j) [X0 X1 X5 X6] +
(-7.867765104184687e-07+0j) [X0 X1 X5 Z6 Z7 Z8 Z9 X10] +
(-0.0038040661717285394+0j) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.006888194352970554+0j) [X0 X1 Y6 Y7] +
(-7.735036880589873e-05+0j) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11] +
(1.703578355473936e-07+0j) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.735036880589871e-05+0j) [X0 X1 X7 Z8 Z9 X10] +
(1.703578355473936e-07+0j) [X0 X1 X7 Z8 Z9 Z10 Z11 X12] +
(-0.006509361201177227+0j) [X0 X1 Y8 Y9] +
(-0.007731425250775268+0j) [X0 X1 Y10 Y11] +
(5.627851911272413e-07+0j) [X0 X1 Y10 Z11 Z12 Y13] +
(5.627851911272413e-07+0j) [X0 X1 X11 X12] +
(-0.005283776488402949+0j) [X0 X1 Y12 Y13] +
(3.5707613291130964e-07+0j) [X0 Y1 Y2 Z3 Z4 X5] +
(0.00565262097801735+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7] +
(0.008826368514209834+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-1.7924939576933602e-06+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.5707613291130964e-07+0j) [X0 Y1 Y3 X4] +
(-0.00565262097801735+0j) [X0 Y1 Y3 Z4 Z5 X6] +
(-0.008826368514209834+0j) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.7924939576933602e-06+0j) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0027458364701868146+0j) [X0 Y1 Y4 X5] +
(2.447323128922324e-07+0j) [X0 Y1 Y4 Z5 Z6 X7] +
(7.867765104184687e-07+0j) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0038040661717285394+0j) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.447323128922324e-07+0j) [X0 Y1 Y5 X6] +
(-7.867765104184687e-07+0j) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.0038040661717285394+0j) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.006888194352970554+0j) [X0 Y1 Y6 X7] +
(7.735036880589873e-05+0j) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11] +
(-1.703578355473936e-07+0j) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.735036880589871e-05+0j) [X0 Y1 Y7 Z8 Z9 X10] +
(1.703578355473936e-07+0j) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12] +
(0.006509361201177227+0j) [X0 Y1 Y8 X9] +
(0.007731425250775268+0j) [X0 Y1 Y10 X11] +
(-5.627851911272413e-07+0j) [X0 Y1 Y10 Z11 Z12 X13] +
(5.627851911272413e-07+0j) [X0 Y1 Y11 X12] +
(0.005283776488402949+0j) [X0 Y1 Y12 X13] +
(0.1250703257977212+0j) [X0 Z1 X2] +
(-1.9332412771466073e-07+0j) [X0 Z1 X2 X3 Z4 X5] +
(-0.0022939566113524667+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 X7] +
(-0.0016407548553124258+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(3.013471458900149e-07+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.9332412771466073e-07+0j) [X0 Z1 X2 Y3 Z4 Y5] +
(-0.0022939566113524667+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7] +
(-0.0016407548553124258+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(3.013471458900149e-07+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0032769719312317077+0j) [X0 Z1 X2 Z3] +
(-1.5510539177775966e-07+0j) [X0 Z1 X2 X4 Z5 X6] +
(-1.1468376507499873e-06+0j) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.007597464029770618+0j) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.3807781481354972e-07+0j) [X0 Z1 X2 Y4 Z5 Y6] +
(-7.900128986262792e-07+0j) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.005348051582676636+0j) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0055307592186315865+0j) [X0 Z1 X2 Z4] +
(-1.3807781481354972e-07+0j) [X0 Z1 X2 X5 Z6 X7] +
(-3.376739308448536e-07+0j) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0018638942824587533+0j) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.3807781481354972e-07+0j) [X0 Z1 X2 Y5 Z6 Y7] +
(-3.376739308448536e-07+0j) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0018638942824587533+0j) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0011059037691897291+0j) [X0 Z1 X2 Z5] +
(0.005708495985960939+0j) [X0 Z1 X2 X6 Z7 Z8 Z9 X10] +
(-8.352332102925357e-07+0j) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
1.9742253792497535e-08j [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005262642473076857+0j) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10] +
(-8.074305985832789e-07+0j) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.00803252091882142+0j) [X0 Z1 X2 Z6] +
(0.0005940221543005574+0j) [X0 Z1 X2 X7 Z8 Z9 Z10 X11] +
(-8.379773244268824e-08+0j) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005940221543005574+0j) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11] +
(-8.379773244268824e-08+0j) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0033476175306662234+0j) [X0 Z1 X2 Z7] +
(0.011055020596132125+0j) [X0 Z1 X2 Z8] +
(0.0029297686747511067+0j) [X0 Z1 X2 Z9] +
(-6.41829157478176e-07+0j) [X0 Z1 X2 X10 Z11 X12] +
(-6.556281914765591e-07+0j) [X0 Z1 X2 Y10 Z11 Y12] +
(0.003555290195504293+0j) [X0 Z1 X2 Z10] +
(-1.1076325600224075e-07+0j) [X0 Z1 X2 X11 Z12 X13] +
(-1.1076325600224075e-07+0j) [X0 Z1 X2 Y11 Z12 Y13] +
(0.001756070701841274+0j) [X0 Z1 X2 Z11] +
(0.006901238249797319+0j) [X0 Z1 X2 Z12] +
(0.0023262306231581205+0j) [X0 Z1 X2 Z13] +
(-3.5682475212370807e-07+0j) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.047471655553153e-08+0j) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0004458535128840815+0j) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10] +
(-1.9742253792147155e-08+0j) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441857+0j) [X0 Z1 Z2 X3 Y4 Y5] +
(-4.523389677814255e-07+0j) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0034841573002178834+0j) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-8.091637199051336e-07+0j) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10] +
(-0.005733569747311865+0j) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004684903388155195+0j) [X0 Z1 Z2 X3 Y6 Y7] +
(0.0046686203187763+0j) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11] +
(-7.189990975221074e-07+0j) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005114473831660382+0j) [X0 Z1 Z2 X3 X7 Z8 Z9 X10] +
(-7.560692464683303e-07+0j) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381018+0j) [X0 Z1 Z2 X3 Y8 Y9] +
(-0.0017992194936630188+0j) [X0 Z1 Z2 X3 Y10 Y11] +
(-5.471647744738668e-07+0j) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13] +
(-5.287660624763868e-07+0j) [X0 Z1 Z2 X3 X11 X12] +
(-0.004575007626639198+0j) [X0 Z1 Z2 X3 Y12 Y13] +
(0.004424855449441857+0j) [X0 Z1 Z2 Y3 Y4 X5] +
(4.523389677814255e-07+0j) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0034841573002178834+0j) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.091637199051336e-07+0j) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.005733569747311865+0j) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.004684903388155195+0j) [X0 Z1 Z2 Y3 Y6 X7] +
(-0.0046686203187763+0j) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11] +
(7.189990975221074e-07+0j) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005114473831660382+0j) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10] +
(-7.560692464683303e-07+0j) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12] +
(0.008125251921381018+0j) [X0 Z1 Z2 Y3 Y8 X9] +
(0.0017992194936630188+0j) [X0 Z1 Z2 Y3 Y10 X11] +
(5.471647744738668e-07+0j) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13] +
(-5.287660624763868e-07+0j) [X0 Z1 Z2 Y3 Y11 X12] +
(0.004575007626639198+0j) [X0 Z1 Z2 Y3 Y12 X13] +
(3.2020768798887158e-06+0j) [X0 Z1 Z2 Z3 X4] +
(0.000853385625412547+0j) [X0 Z1 Z2 Z3 X4 X5 Z6 X7] +
(0.0007870896771024419+0j) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.000853385625412547+0j) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7] +
(0.0007870896771024419+0j) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.2919694862618413e-07+0j) [X0 Z1 Z2 Z3 X4 Z5] +
(4.444597854217967e-07+0j) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10] +
(0.0011726348316441885+0j) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.6849150951772586e-07+0j) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10] +
(0.00220096406950046+0j) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.8394209155545287e-07+0j) [X0 Z1 Z2 Z3 X4 Z6] +
(4.092250616101356e-07+0j) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11] +
(0.0023949726397980214+0j) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.092250616101356e-07+0j) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11] +
(0.0023949726397980214+0j) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.236259961356317e-07+0j) [X0 Z1 Z2 Z3 X4 Z8] +
(8.649310134011179e-08+0j) [X0 Z1 Z2 Z3 X4 Z9] +
(0.0013038004788126951+0j) [X0 Z1 Z2 Z3 X4 X10 Z11 X12] +
(0.00398984145661931+0j) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12] +
(-6.733197742323143e-07+0j) [X0 Z1 Z2 Z3 X4 Z10] +
(0.0022619660624823438+0j) [X0 Z1 Z2 Z3 X4 X11 Z12 X13] +
(0.0022619660624823438+0j) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13] +
(-5.927453082724306e-07+0j) [X0 Z1 Z2 Z3 X4 Z11] +
(1.2393363217132198e-06+0j) [X0 Z1 Z2 Z3 X4 Z12] +
(9.306536651883631e-07+0j) [X0 Z1 Z2 Z3 X4 Z13] +
(-0.001028329237856272+0j) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.002686040977806615+0j) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12] +
(-1.8394209155545282e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7] +
(-0.00019400857029756068+0j) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0012223378081538325+0j) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-2.3713289479551997e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9] +
(8.057446595988366e-08+0j) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11] +
(0.0017278753941369664+0j) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13] +
(-0.0009581655836696487+0j) [X0 Z1 Z2 Z3 Z4 X5 X11 X12] +
(-3.086826565248567e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13] +
(1.8394209155545282e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7] +
(0.00019400857029756068+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0012223378081538325+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(2.3713289479551997e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9] +
(-8.057446595988366e-08+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11] +
(-0.0017278753941369664+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13] +
(-0.0009581655836696487+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12] +
(3.086826565248567e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13] +
(0.04274327701378333+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6] +
(0.0005192743499487632+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(-1.850564192860225e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005192743499487632+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(-1.850564192860225e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.002779026799025518+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7] +
(0.004636976661182568+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8] +
(0.0012803060973496888+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9] +
(2.312094305145714e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12] +
(1.071728218241235e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12] +
(0.005379937155839381+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10] +
(7.246974425367426e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13] +
(7.246974425367426e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13] +
(0.005241535382803865+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11] +
(0.004311038507914312+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12] +
(0.00104352465349077+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13] +
(1.2004287493900927e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12] +
(-0.0033566705638328792+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9] +
(-0.0001384017730355146+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11] +
(-6.17524620712619e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(-4.997018422187658e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12] +
(-0.003267513854423542+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13] +
(0.0033566705638328792+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9] +
(0.0001384017730355146+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11] +
(6.17524620712619e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(-4.997018422187658e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12] +
(0.003267513854423542+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13] +
(0.0038764708993369295+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(-7.540341413642074e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.0038764708993369295+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(-7.540341413642074e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(0.07165035181002657+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0021413612231015646+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(0.004220813970046476+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(0.0012366478019245584+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(-0.0029841661681219173+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(0.0029841661681219173+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-1.3987009014851748e-05+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(8.949476487207354e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-6.876621658391887e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(-7.661347213203925e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(0.00153248352307307+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10] +
(-2.9045998843794783e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(0.00540895442241+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10] +
(-1.0444941298021551e-06+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(0.004767272188278125+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10] +
(-8.105515037141971e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(0.005286546538226888+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10] +
(-9.956079230002197e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0016095313817213902+0j) [X0 Z1 Z2 Z3 Z4 X6] +
(-7.14162522115256e-05+0j) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10] +
(-2.666731754812456e-07+0j) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0024629170071339373+0j) [X0 Z1 Z2 Z3 Z5 X6] +
(0.000715673424890916+0j) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10] +
(-3.0767325318978334e-07+0j) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.6060718679978207e-07+0j) [X0 Z1 Z2 X4] +
(0.003961560792496535+0j) [X0 Z1 Z2 Z4 Z5 X6] +
(0.00018787053389553542+0j) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.6569309315550103e-07+0j) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.7379332624231772e-07+0j) [X0 Z1 Z3 X4] +
(0.0016676041811440672+0j) [X0 Z1 Z3 Z4 Z5 X6] +
(-0.0014528843214168905+0j) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(4.670402390455159e-07+0j) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.10433064780651422+0j) [X0 X2] +
(3.1174479461046654e-06+0j) [X0 Z2 Z3 X4] +
(0.045879470781298+0j) [X0 Z2 Z3 Z4 Z5 X6] +
(0.058591988733861795+0j) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-1.1463061452964958e-05+0j) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.5707613291130964e-07+0j) [Y0 X1 X2 Z3 Z4 Y5] +
(0.00565262097801735+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7] +
(0.008826368514209834+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.7924939576933602e-06+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.5707613291130964e-07+0j) [Y0 X1 X3 Y4] +
(-0.00565262097801735+0j) [Y0 X1 X3 Z4 Z5 Y6] +
(-0.008826368514209834+0j) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.7924939576933602e-06+0j) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0027458364701868146+0j) [Y0 X1 X4 Y5] +
(2.447323128922324e-07+0j) [Y0 X1 X4 Z5 Z6 Y7] +
(7.867765104184687e-07+0j) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.0038040661717285394+0j) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.447323128922324e-07+0j) [Y0 X1 X5 Y6] +
(-7.867765104184687e-07+0j) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.0038040661717285394+0j) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.006888194352970554+0j) [Y0 X1 X6 Y7] +
(7.735036880589873e-05+0j) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11] +
(-1.703578355473936e-07+0j) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.735036880589871e-05+0j) [Y0 X1 X7 Z8 Z9 Y10] +
(1.703578355473936e-07+0j) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12] +
(0.006509361201177227+0j) [Y0 X1 X8 Y9] +
(0.007731425250775268+0j) [Y0 X1 X10 Y11] +
(-5.627851911272413e-07+0j) [Y0 X1 X10 Z11 Z12 Y13] +
(5.627851911272413e-07+0j) [Y0 X1 X11 Y12] +
(0.005283776488402949+0j) [Y0 X1 X12 Y13] +
(-3.5707613291130964e-07+0j) [Y0 Y1 X2 Z3 Z4 X5] +
(-0.00565262097801735+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7] +
(-0.008826368514209834+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.7924939576933602e-06+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.5707613291130964e-07+0j) [Y0 Y1 Y3 Y4] +
(-0.00565262097801735+0j) [Y0 Y1 Y3 Z4 Z5 Y6] +
(-0.008826368514209834+0j) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.7924939576933602e-06+0j) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.0027458364701868146+0j) [Y0 Y1 X4 X5] +
(-2.447323128922324e-07+0j) [Y0 Y1 X4 Z5 Z6 X7] +
(-7.867765104184687e-07+0j) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0038040661717285394+0j) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.447323128922324e-07+0j) [Y0 Y1 Y5 Y6] +
(-7.867765104184687e-07+0j) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.0038040661717285394+0j) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.006888194352970554+0j) [Y0 Y1 X6 X7] +
(-7.735036880589873e-05+0j) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11] +
(1.703578355473936e-07+0j) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.735036880589871e-05+0j) [Y0 Y1 Y7 Z8 Z9 Y10] +
(1.703578355473936e-07+0j) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.006509361201177227+0j) [Y0 Y1 X8 X9] +
(-0.007731425250775268+0j) [Y0 Y1 X10 X11] +
(5.627851911272413e-07+0j) [Y0 Y1 X10 Z11 Z12 X13] +
(5.627851911272413e-07+0j) [Y0 Y1 Y11 Y12] +
(-0.005283776488402949+0j) [Y0 Y1 X12 X13] +
(-3.5682475212370807e-07+0j) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10] +
(0.0004458535128840815+0j) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10] +
(-1.9742253792147155e-08+0j) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.047471655553153e-08+0j) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.1250703257977212+0j) [Y0 Z1 Y2] +
(-1.9332412771466073e-07+0j) [Y0 Z1 Y2 X3 Z4 X5] +
(-0.0022939566113524667+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7] +
(-0.0016407548553124258+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(3.013471458900149e-07+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.9332412771466073e-07+0j) [Y0 Z1 Y2 Y3 Z4 Y5] +
(-0.0022939566113524667+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7] +
(-0.0016407548553124258+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(3.013471458900149e-07+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0032769719312317077+0j) [Y0 Z1 Y2 Z3] +
(-1.3807781481354972e-07+0j) [Y0 Z1 Y2 X4 Z5 X6] +
(-7.900128986262792e-07+0j) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.005348051582676636+0j) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.5510539177775966e-07+0j) [Y0 Z1 Y2 Y4 Z5 Y6] +
(-1.1468376507499873e-06+0j) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.007597464029770618+0j) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0055307592186315865+0j) [Y0 Z1 Y2 Z4] +
(-1.3807781481354972e-07+0j) [Y0 Z1 Y2 X5 Z6 X7] +
(-3.376739308448536e-07+0j) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0018638942824587533+0j) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.3807781481354972e-07+0j) [Y0 Z1 Y2 Y5 Z6 Y7] +
(-3.376739308448536e-07+0j) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0018638942824587533+0j) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0011059037691897291+0j) [Y0 Z1 Y2 Z5] +
(0.005262642473076857+0j) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10] +
(-8.074305985832789e-07+0j) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.005708495985960939+0j) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10] +
-1.9742253792497535e-08j [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-8.352332102925357e-07+0j) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.00803252091882142+0j) [Y0 Z1 Y2 Z6] +
(0.0005940221543005574+0j) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11] +
(-8.379773244268824e-08+0j) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005940221543005574+0j) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11] +
(-8.379773244268824e-08+0j) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0033476175306662234+0j) [Y0 Z1 Y2 Z7] +
(0.011055020596132125+0j) [Y0 Z1 Y2 Z8] +
(0.0029297686747511067+0j) [Y0 Z1 Y2 Z9] +
(-6.556281914765591e-07+0j) [Y0 Z1 Y2 X10 Z11 X12] +
(-6.41829157478176e-07+0j) [Y0 Z1 Y2 Y10 Z11 Y12] +
(0.003555290195504293+0j) [Y0 Z1 Y2 Z10] +
(-1.1076325600224075e-07+0j) [Y0 Z1 Y2 X11 Z12 X13] +
(-1.1076325600224075e-07+0j) [Y0 Z1 Y2 Y11 Z12 Y13] +
(0.001756070701841274+0j) [Y0 Z1 Y2 Z11] +
(0.006901238249797319+0j) [Y0 Z1 Y2 Z12] +
(0.0023262306231581205+0j) [Y0 Z1 Y2 Z13] +
(0.004424855449441857+0j) [Y0 Z1 Z2 X3 X4 Y5] +
(4.523389677814255e-07+0j) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.0034841573002178834+0j) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-8.091637199051336e-07+0j) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.005733569747311865+0j) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.004684903388155195+0j) [Y0 Z1 Z2 X3 X6 Y7] +
(-0.0046686203187763+0j) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11] +
(7.189990975221074e-07+0j) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005114473831660382+0j) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10] +
(-7.560692464683303e-07+0j) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12] +
(0.008125251921381018+0j) [Y0 Z1 Z2 X3 X8 Y9] +
(0.0017992194936630188+0j) [Y0 Z1 Z2 X3 X10 Y11] +
(5.471647744738668e-07+0j) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13] +
(-5.287660624763868e-07+0j) [Y0 Z1 Z2 X3 X11 Y12] +
(0.004575007626639198+0j) [Y0 Z1 Z2 X3 X12 Y13] +
(-0.004424855449441857+0j) [Y0 Z1 Z2 Y3 X4 X5] +
(-4.523389677814255e-07+0j) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0034841573002178834+0j) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.091637199051336e-07+0j) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.005733569747311865+0j) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004684903388155195+0j) [Y0 Z1 Z2 Y3 X6 X7] +
(0.0046686203187763+0j) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11] +
(-7.189990975221074e-07+0j) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005114473831660382+0j) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10] +
(-7.560692464683303e-07+0j) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381018+0j) [Y0 Z1 Z2 Y3 X8 X9] +
(-0.0017992194936630188+0j) [Y0 Z1 Z2 Y3 X10 X11] +
(-5.471647744738668e-07+0j) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13] +
(-5.287660624763868e-07+0j) [Y0 Z1 Z2 Y3 Y11 Y12] +
(-0.004575007626639198+0j) [Y0 Z1 Z2 Y3 X12 X13] +
(-0.001028329237856272+0j) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.002686040977806615+0j) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12] +
(3.2020768798887158e-06+0j) [Y0 Z1 Z2 Z3 Y4] +
(0.000853385625412547+0j) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7] +
(0.0007870896771024419+0j) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.000853385625412547+0j) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7] +
(0.0007870896771024419+0j) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.2919694862618413e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z5] +
(4.6849150951772586e-07+0j) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10] +
(0.00220096406950046+0j) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.444597854217967e-07+0j) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10] +
(0.0011726348316441885+0j) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.8394209155545287e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z6] +
(4.092250616101356e-07+0j) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11] +
(0.0023949726397980214+0j) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.092250616101356e-07+0j) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11] +
(0.0023949726397980214+0j) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.236259961356317e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z8] +
(8.649310134011179e-08+0j) [Y0 Z1 Z2 Z3 Y4 Z9] +
(0.00398984145661931+0j) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12] +
(0.0013038004788126951+0j) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12] +
(-6.733197742323143e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z10] +
(0.0022619660624823438+0j) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13] +
(0.0022619660624823438+0j) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13] +
(-5.927453082724306e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z11] +
(1.2393363217132198e-06+0j) [Y0 Z1 Z2 Z3 Y4 Z12] +
(9.306536651883631e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z13] +
(1.8394209155545282e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7] +
(0.00019400857029756068+0j) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0012223378081538325+0j) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(2.3713289479551997e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9] +
(-8.057446595988366e-08+0j) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11] +
(-0.0017278753941369664+0j) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13] +
(-0.0009581655836696487+0j) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12] +
(3.086826565248567e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13] +
(-1.8394209155545282e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7] +
(-0.00019400857029756068+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0012223378081538325+0j) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-2.3713289479551997e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9] +
(8.057446595988366e-08+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11] +
(0.0017278753941369664+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13] +
(-0.0009581655836696487+0j) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12] +
(-3.086826565248567e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13] +
(1.2004287493900927e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12] +
(0.04274327701378333+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6] +
(0.0005192743499487632+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(-1.850564192860225e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005192743499487632+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(-1.850564192860225e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.002779026799025518+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7] +
(0.004636976661182568+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8] +
(0.0012803060973496888+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9] +
(1.071728218241235e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12] +
(2.312094305145714e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12] +
(0.005379937155839381+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10] +
(7.246974425367426e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13] +
(7.246974425367426e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13] +
(0.005241535382803865+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11] +
(0.004311038507914312+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12] +
(0.00104352465349077+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13] +
(0.0033566705638328792+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9] +
(0.0001384017730355146+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11] +
(6.17524620712619e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(-4.997018422187658e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12] +
(0.003267513854423542+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13] +
(-0.0033566705638328792+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9] +
(-0.0001384017730355146+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11] +
(-6.17524620712619e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(-4.997018422187658e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12] +
(-0.003267513854423542+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13] +
(0.0038764708993369295+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(-7.540341413642074e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.0038764708993369295+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(-7.540341413642074e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(0.07165035181002657+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0021413612231015646+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(0.004220813970046476+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(0.0012366478019245584+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(0.0029841661681219173+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-0.0029841661681219173+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-1.3987009014851748e-05+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(8.949476487207354e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-6.876621658391887e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(-7.661347213203925e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(0.00153248352307307+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10] +
(-2.9045998843794783e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(0.00540895442241+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10] +
(-1.0444941298021551e-06+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(0.004767272188278125+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10] +
(-8.105515037141971e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(0.005286546538226888+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10] +
(-9.956079230002197e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0016095313817213902+0j) [Y0 Z1 Z2 Z3 Z4 Y6] +
(-7.14162522115256e-05+0j) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10] +
(-2.666731754812456e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0024629170071339373+0j) [Y0 Z1 Z2 Z3 Z5 Y6] +
(0.000715673424890916+0j) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10] +
(-3.0767325318978334e-07+0j) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(3.6060718679978207e-07+0j) [Y0 Z1 Z2 Y4] +
(0.003961560792496535+0j) [Y0 Z1 Z2 Z4 Z5 Y6] +
(0.00018787053389553542+0j) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.6569309315550103e-07+0j) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.7379332624231772e-07+0j) [Y0 Z1 Z3 Y4] +
(0.0016676041811440672+0j) [Y0 Z1 Z3 Z4 Z5 Y6] +
(-0.0014528843214168905+0j) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(4.670402390455159e-07+0j) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.10433064780651422+0j) [Y0 Y2] +
(3.1174479461046654e-06+0j) [Y0 Z2 Z3 Y4] +
(0.045879470781298+0j) [Y0 Z2 Z3 Z4 Z5 Y6] +
(0.058591988733861795+0j) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-1.1463061452964958e-05+0j) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.10433064780651422+0j) [Z0 X1 Z2 X3] +
(3.1174479461046654e-06+0j) [Z0 X1 Z2 Z3 Z4 X5] +
(0.045879470781298+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7] +
(0.058591988733861795+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-1.1463061452964958e-05+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.10433064780651422+0j) [Z0 Y1 Z2 Y3] +
(3.1174479461046654e-06+0j) [Z0 Y1 Z2 Z3 Z4 Y5] +
(0.045879470781298+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7] +
(0.058591988733861795+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.1463061452964958e-05+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.1861763734860493+0j) [Z0 Z1] +
(-8.337746754109954e-07+0j) [Z0 X2 Z3 X4] +
(-0.027115036845273135+0j) [Z0 X2 Z3 Z4 Z5 X6] +
(-0.0675238509921401+0j) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.4017109735203908e-05+0j) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-8.337746754109954e-07+0j) [Z0 Y2 Z3 Y4] +
(-0.027115036845273135+0j) [Z0 Y2 Z3 Z4 Z5 Y6] +
(-0.0675238509921401+0j) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.4017109735203908e-05+0j) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.1908508083223051e-06+0j) [Z0 X3 Z4 X5] +
(-0.03276765782329048+0j) [Z0 X3 Z4 Z5 Z6 X7] +
(-0.07635021950634995+0j) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.5809603692897268e-05+0j) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.1908508083223051e-06+0j) [Z0 Y3 Z4 Y5] +
(-0.03276765782329048+0j) [Z0 Y3 Z4 Z5 Z6 Y7] +
(-0.07635021950634995+0j) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.5809603692897268e-05+0j) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.099349243799313e-06+0j) [Z0 X4 Z5 X6] +
(-1.531680879528886e-05+0j) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-3.099349243799313e-06+0j) [Z0 Y4 Z5 Y6] +
(-1.531680879528886e-05+0j) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(0.19661770890342162+0j) [Z0 Z4] +
(-3.344081556691545e-06+0j) [Z0 X5 Z6 X7] +
(-1.6103585305707328e-05+0j) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-3.344081556691545e-06+0j) [Z0 Y5 Z6 Y7] +
(-1.6103585305707328e-05+0j) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.19936354537360845+0j) [Z0 Z5] +
(0.05608468124661361+0j) [Z0 X6 Z7 Z8 Z9 X10] +
(-6.652209669292463e-06+0j) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.05608468124661361+0j) [Z0 Y6 Z7 Z8 Z9 Y10] +
(-6.652209669292463e-06+0j) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.2416466393601718+0j) [Z0 Z6] +
(0.0560073308778077+0j) [Z0 X7 Z8 Z9 Z10 X11] +
(-6.4818518337450695e-06+0j) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0560073308778077+0j) [Z0 Y7 Z8 Z9 Z10 Y11] +
(-6.4818518337450695e-06+0j) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.24853483371314236+0j) [Z0 Z7] +
(-2.1776646053217677e-06+0j) [Z0 X10 Z11 X12] +
(-2.1776646053217677e-06+0j) [Z0 Y10 Z11 Y12] +
(0.1929972393536423+0j) [Z0 Z10] +
(-1.6148794141945263e-06+0j) [Z0 X11 Z12 X13] +
(-1.6148794141945263e-06+0j) [Z0 Y11 Z12 Y13] +
(0.20072866460441757+0j) [Z0 Z11] +
(0.21102659849791497+0j) [Z0 Z12] +
(0.21631037498631792+0j) [Z0 Z13] +
(1.9332412771466073e-07+0j) [X1 X2 Y3 Y4] +
(0.002293956611352467+0j) [X1 X2 Y3 Z4 Z5 Y6] +
(0.0016407548553124258+0j) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-3.013471458900149e-07+0j) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004424855449441857+0j) [X1 X2 X4 X5] +
(-8.091637199051336e-07+0j) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.005733569747311865+0j) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-4.523389677814255e-07+0j) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.0034841573002178834+0j) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004684903388155195+0j) [X1 X2 X6 X7] +
(0.005114473831660382+0j) [X1 X2 X6 Z7 Z8 Z9 Z10 X11] +
(-7.560692464683303e-07+0j) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0046686203187763+0j) [X1 X2 Y7 Z8 Z9 Y10] +
(-7.189990975221074e-07+0j) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381018+0j) [X1 X2 X8 X9] +
(-0.0017992194936630184+0j) [X1 X2 X10 X11] +
(-5.287660624763868e-07+0j) [X1 X2 X10 Z11 Z12 X13] +
(-5.471647744738668e-07+0j) [X1 X2 Y11 Y12] +
(-0.004575007626639198+0j) [X1 X2 X12 X13] +
(-1.9332412771466073e-07+0j) [X1 Y2 Y3 X4] +
(-0.002293956611352467+0j) [X1 Y2 Y3 Z4 Z5 X6] +
(-0.0016407548553124258+0j) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(3.013471458900149e-07+0j) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441857+0j) [X1 Y2 Y4 X5] +
(-8.091637199051336e-07+0j) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.005733569747311865+0j) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.523389677814255e-07+0j) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10] +
(0.0034841573002178834+0j) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004684903388155195+0j) [X1 Y2 Y6 X7] +
(0.005114473831660382+0j) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11] +
(-7.560692464683303e-07+0j) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0046686203187763+0j) [X1 Y2 Y7 Z8 Z9 X10] +
(7.189990975221074e-07+0j) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381018+0j) [X1 Y2 Y8 X9] +
(-0.0017992194936630184+0j) [X1 Y2 Y10 X11] +
(-5.287660624763868e-07+0j) [X1 Y2 Y10 Z11 Z12 X13] +
(5.471647744738668e-07+0j) [X1 Y2 Y11 X12] +
(-0.004575007626639198+0j) [X1 Y2 Y12 X13] +
(0.12507032579772132+0j) [X1 Z2 X3] +
(-1.3807781481354972e-07+0j) [X1 Z2 X3 X4 Z5 X6] +
(-3.376739308448536e-07+0j) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0018638942824587533+0j) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.3807781481354972e-07+0j) [X1 Z2 X3 Y4 Z5 Y6] +
(-3.376739308448536e-07+0j) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0018638942824587533+0j) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0011059037691897291+0j) [X1 Z2 X3 Z4] +
(-1.5510539177775966e-07+0j) [X1 Z2 X3 X5 Z6 X7] +
(-1.1468376507499873e-06+0j) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.007597464029770618+0j) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.3807781481354972e-07+0j) [X1 Z2 X3 Y5 Z6 Y7] +
(-7.900128986262792e-07+0j) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.005348051582676636+0j) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0055307592186315865+0j) [X1 Z2 X3 Z5] +
(0.0005940221543005574+0j) [X1 Z2 X3 X6 Z7 Z8 Z9 X10] +
(-8.379773244268824e-08+0j) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0005940221543005574+0j) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10] +
(-8.379773244268824e-08+0j) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0033476175306662234+0j) [X1 Z2 X3 Z6] +
(0.005708495985960939+0j) [X1 Z2 X3 X7 Z8 Z9 Z10 X11] +
(-8.352332102925357e-07+0j) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
1.9742253792497535e-08j [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005262642473076857+0j) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11] +
(-8.074305985832789e-07+0j) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.00803252091882142+0j) [X1 Z2 X3 Z7] +
(0.0029297686747511067+0j) [X1 Z2 X3 Z8] +
(0.011055020596132125+0j) [X1 Z2 X3 Z9] +
(-1.1076325600224075e-07+0j) [X1 Z2 X3 X10 Z11 X12] +
(-1.1076325600224075e-07+0j) [X1 Z2 X3 Y10 Z11 Y12] +
(0.001756070701841274+0j) [X1 Z2 X3 Z10] +
(-6.41829157478176e-07+0j) [X1 Z2 X3 X11 Z12 X13] +
(-6.556281914765591e-07+0j) [X1 Z2 X3 Y11 Z12 Y13] +
(0.003555290195504293+0j) [X1 Z2 X3 Z11] +
(0.0023262306231581205+0j) [X1 Z2 X3 Z12] +
(0.006901238249797319+0j) [X1 Z2 X3 Z13] +
(-3.5682475212370807e-07+0j) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.047471655553153e-08+0j) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0004458535128840815+0j) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11] +
(-1.9742253792147155e-08+0j) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0008533856254125472+0j) [X1 Z2 Z3 X4 Y5 Y6] +
(-0.0007870896771024418+0j) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10] +
(-1.8394209155545287e-07+0j) [X1 Z2 Z3 X4 X6 X7] +
(-0.0012223378081538325+0j) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.00019400857029756068+0j) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12] +
(-2.3713289479551997e-07+0j) [X1 Z2 Z3 X4 X8 X9] +
(8.057446595988366e-08+0j) [X1 Z2 Z3 X4 X10 X11] +
(-0.0009581655836696487+0j) [X1 Z2 Z3 X4 X10 Z11 Z12 X13] +
(0.0017278753941369664+0j) [X1 Z2 Z3 X4 Y11 Y12] +
(-3.0868265652485675e-07+0j) [X1 Z2 Z3 X4 X12 X13] +
(0.0008533856254125472+0j) [X1 Z2 Z3 Y4 Y5 X6] +
(0.0007870896771024418+0j) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10] +
(-1.8394209155545287e-07+0j) [X1 Z2 Z3 Y4 Y6 X7] +
(-0.0012223378081538325+0j) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.00019400857029756068+0j) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12] +
(-2.3713289479551997e-07+0j) [X1 Z2 Z3 Y4 Y8 X9] +
(8.057446595988366e-08+0j) [X1 Z2 Z3 Y4 Y10 X11] +
(-0.0009581655836696487+0j) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13] +
(-0.0017278753941369664+0j) [X1 Z2 Z3 Y4 Y11 X12] +
(-3.0868265652485675e-07+0j) [X1 Z2 Z3 Y4 Y12 X13] +
(3.2020768798887158e-06+0j) [X1 Z2 Z3 Z4 X5] +
(4.092250616101356e-07+0j) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10] +
(0.0023949726397980214+0j) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.092250616101356e-07+0j) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10] +
(0.0023949726397980214+0j) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.444597854217967e-07+0j) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11] +
(0.0011726348316441885+0j) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.6849150951772586e-07+0j) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11] +
(0.00220096406950046+0j) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.8394209155545287e-07+0j) [X1 Z2 Z3 Z4 X5 Z7] +
(8.649310134011179e-08+0j) [X1 Z2 Z3 Z4 X5 Z8] +
(3.236259961356317e-07+0j) [X1 Z2 Z3 Z4 X5 Z9] +
(0.0022619660624823438+0j) [X1 Z2 Z3 Z4 X5 X10 Z11 X12] +
(0.0022619660624823438+0j) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12] +
(-5.927453082724306e-07+0j) [X1 Z2 Z3 Z4 X5 Z10] +
(0.0013038004788126951+0j) [X1 Z2 Z3 Z4 X5 X11 Z12 X13] +
(0.00398984145661931+0j) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13] +
(-6.733197742323143e-07+0j) [X1 Z2 Z3 Z4 X5 Z11] +
(9.306536651883631e-07+0j) [X1 Z2 Z3 Z4 X5 Z12] +
(1.2393363217132198e-06+0j) [X1 Z2 Z3 Z4 X5 Z13] +
(-0.001028329237856272+0j) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.002686040977806615+0j) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13] +
(-0.0005192743499487632+0j) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10] +
(1.850564192860225e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.003356670563832879+0j) [X1 Z2 Z3 Z4 Z5 X6 X8 X9] +
(-0.0001384017730355146+0j) [X1 Z2 Z3 Z4 Z5 X6 X10 X11] +
(-4.997018422187658e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13] +
(-6.17524620712619e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12] +
(-0.0032675138544235416+0j) [X1 Z2 Z3 Z4 Z5 X6 X12 X13] +
(0.0005192743499487632+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10] +
(-1.850564192860225e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.003356670563832879+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9] +
(-0.0001384017730355146+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11] +
(-4.997018422187658e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13] +
(6.17524620712619e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12] +
(-0.0032675138544235416+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13] +
(0.04274327701378333+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7] +
(0.0012803060973496888+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8] +
(0.004636976661182568+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9] +
(7.246974425367426e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12] +
(7.246974425367426e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12] +
(0.005241535382803865+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10] +
(2.312094305145714e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13] +
(1.071728218241235e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13] +
(0.005379937155839381+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11] +
(0.00104352465349077+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12] +
(0.004311038507914312+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13] +
(1.2004287493900927e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13] +
(-0.0038764708993369295+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10] +
(7.540341413642074e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(0.0038764708993369295+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10] +
(-7.540341413642074e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(-0.0029841661681219173+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-0.0029841661681219173+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(0.07165035181002663+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0012366478019245584+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(0.004220813970046476+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(-1.3987009014851745e-05+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(8.949476487207354e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(-7.661347213203925e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(-0.0021413612231015646+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11] +
(-6.876621658391887e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(0.00540895442241+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11] +
(-1.0444941298021551e-06+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(0.00153248352307307+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11] +
(-2.9045998843794783e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(0.005286546538226888+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11] +
(-9.956079230002197e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.002779026799025518+0j) [X1 Z2 Z3 Z4 Z5 X7] +
(0.004767272188278125+0j) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11] +
(-8.105515037141971e-07+0j) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0024629170071339373+0j) [X1 Z2 Z3 Z4 Z6 X7] +
(0.000715673424890916+0j) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11] +
(-3.0767325318978334e-07+0j) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2919694862618413e-07+0j) [X1 Z2 Z3 X5] +
(0.0016095313817213902+0j) [X1 Z2 Z3 Z5 Z6 X7] +
(-7.14162522115256e-05+0j) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-2.666731754812456e-07+0j) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.7379332624231772e-07+0j) [X1 Z2 Z4 X5] +
(0.0016676041811440672+0j) [X1 Z2 Z4 Z5 Z6 X7] +
(-0.0014528843214168905+0j) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(4.670402390455159e-07+0j) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0032769719312317077+0j) [X1 X3] +
(3.6060718679978207e-07+0j) [X1 Z3 Z4 X5] +
(0.003961560792496535+0j) [X1 Z3 Z4 Z5 Z6 X7] +
(0.00018787053389553542+0j) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.6569309315550103e-07+0j) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.9332412771466073e-07+0j) [Y1 X2 X3 Y4] +
(-0.002293956611352467+0j) [Y1 X2 X3 Z4 Z5 Y6] +
(-0.0016407548553124258+0j) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(3.013471458900149e-07+0j) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004424855449441857+0j) [Y1 X2 X4 Y5] +
(-8.091637199051336e-07+0j) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.005733569747311865+0j) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(4.523389677814255e-07+0j) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10] +
(0.0034841573002178834+0j) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004684903388155195+0j) [Y1 X2 X6 Y7] +
(0.005114473831660382+0j) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11] +
(-7.560692464683303e-07+0j) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0046686203187763+0j) [Y1 X2 X7 Z8 Z9 Y10] +
(7.189990975221074e-07+0j) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381018+0j) [Y1 X2 X8 Y9] +
(-0.0017992194936630184+0j) [Y1 X2 X10 Y11] +
(-5.287660624763868e-07+0j) [Y1 X2 X10 Z11 Z12 Y13] +
(5.471647744738668e-07+0j) [Y1 X2 X11 Y12] +
(-0.004575007626639198+0j) [Y1 X2 X12 Y13] +
(1.9332412771466073e-07+0j) [Y1 Y2 X3 X4] +
(0.002293956611352467+0j) [Y1 Y2 X3 Z4 Z5 X6] +
(0.0016407548553124258+0j) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-3.013471458900149e-07+0j) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441857+0j) [Y1 Y2 Y4 Y5] +
(-8.091637199051336e-07+0j) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.005733569747311865+0j) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-4.523389677814255e-07+0j) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10] +
(-0.0034841573002178834+0j) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004684903388155195+0j) [Y1 Y2 Y6 Y7] +
(0.005114473831660382+0j) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11] +
(-7.560692464683303e-07+0j) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0046686203187763+0j) [Y1 Y2 X7 Z8 Z9 X10] +
(-7.189990975221074e-07+0j) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381018+0j) [Y1 Y2 Y8 Y9] +
(-0.0017992194936630184+0j) [Y1 Y2 Y10 Y11] +
(-5.287660624763868e-07+0j) [Y1 Y2 Y10 Z11 Z12 Y13] +
(-5.471647744738668e-07+0j) [Y1 Y2 X11 X12] +
(-0.004575007626639198+0j) [Y1 Y2 Y12 Y13] +
(-3.5682475212370807e-07+0j) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.0004458535128840815+0j) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11] +
(-1.9742253792147155e-08+0j) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.047471655553153e-08+0j) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.12507032579772132+0j) [Y1 Z2 Y3] +
(-1.3807781481354972e-07+0j) [Y1 Z2 Y3 X4 Z5 X6] +
(-3.376739308448536e-07+0j) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0018638942824587533+0j) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.3807781481354972e-07+0j) [Y1 Z2 Y3 Y4 Z5 Y6] +
(-3.376739308448536e-07+0j) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0018638942824587533+0j) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0011059037691897291+0j) [Y1 Z2 Y3 Z4] +
(-1.3807781481354972e-07+0j) [Y1 Z2 Y3 X5 Z6 X7] +
(-7.900128986262792e-07+0j) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.005348051582676636+0j) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.5510539177775966e-07+0j) [Y1 Z2 Y3 Y5 Z6 Y7] +
(-1.1468376507499873e-06+0j) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.007597464029770618+0j) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0055307592186315865+0j) [Y1 Z2 Y3 Z5] +
(0.0005940221543005574+0j) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10] +
(-8.379773244268824e-08+0j) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0005940221543005574+0j) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10] +
(-8.379773244268824e-08+0j) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0033476175306662234+0j) [Y1 Z2 Y3 Z6] +
(0.005262642473076857+0j) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11] +
(-8.074305985832789e-07+0j) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005708495985960939+0j) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11] +
-1.9742253792497535e-08j [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.352332102925357e-07+0j) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.00803252091882142+0j) [Y1 Z2 Y3 Z7] +
(0.0029297686747511067+0j) [Y1 Z2 Y3 Z8] +
(0.011055020596132125+0j) [Y1 Z2 Y3 Z9] +
(-1.1076325600224075e-07+0j) [Y1 Z2 Y3 X10 Z11 X12] +
(-1.1076325600224075e-07+0j) [Y1 Z2 Y3 Y10 Z11 Y12] +
(0.001756070701841274+0j) [Y1 Z2 Y3 Z10] +
(-6.556281914765591e-07+0j) [Y1 Z2 Y3 X11 Z12 X13] +
(-6.41829157478176e-07+0j) [Y1 Z2 Y3 Y11 Z12 Y13] +
(0.003555290195504293+0j) [Y1 Z2 Y3 Z11] +
(0.0023262306231581205+0j) [Y1 Z2 Y3 Z12] +
(0.006901238249797319+0j) [Y1 Z2 Y3 Z13] +
(0.0008533856254125472+0j) [Y1 Z2 Z3 X4 X5 Y6] +
(0.0007870896771024418+0j) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10] +
(-1.8394209155545287e-07+0j) [Y1 Z2 Z3 X4 X6 Y7] +
(-0.0012223378081538325+0j) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.00019400857029756068+0j) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12] +
(-2.3713289479551997e-07+0j) [Y1 Z2 Z3 X4 X8 Y9] +
(8.057446595988366e-08+0j) [Y1 Z2 Z3 X4 X10 Y11] +
(-0.0009581655836696487+0j) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13] +
(-0.0017278753941369664+0j) [Y1 Z2 Z3 X4 X11 Y12] +
(-3.0868265652485675e-07+0j) [Y1 Z2 Z3 X4 X12 Y13] +
(-0.0008533856254125472+0j) [Y1 Z2 Z3 Y4 X5 X6] +
(-0.0007870896771024418+0j) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10] +
(-1.8394209155545287e-07+0j) [Y1 Z2 Z3 Y4 Y6 Y7] +
(-0.0012223378081538325+0j) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.00019400857029756068+0j) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12] +
(-2.3713289479551997e-07+0j) [Y1 Z2 Z3 Y4 Y8 Y9] +
(8.057446595988366e-08+0j) [Y1 Z2 Z3 Y4 Y10 Y11] +
(-0.0009581655836696487+0j) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13] +
(0.0017278753941369664+0j) [Y1 Z2 Z3 Y4 X11 X12] +
(-3.0868265652485675e-07+0j) [Y1 Z2 Z3 Y4 Y12 Y13] +
(-0.001028329237856272+0j) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.002686040977806615+0j) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13] +
(3.2020768798887158e-06+0j) [Y1 Z2 Z3 Z4 Y5] +
(4.092250616101356e-07+0j) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10] +
(0.0023949726397980214+0j) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.092250616101356e-07+0j) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10] +
(0.0023949726397980214+0j) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.6849150951772586e-07+0j) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11] +
(0.00220096406950046+0j) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.444597854217967e-07+0j) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11] +
(0.0011726348316441885+0j) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.8394209155545287e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z7] +
(8.649310134011179e-08+0j) [Y1 Z2 Z3 Z4 Y5 Z8] +
(3.236259961356317e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z9] +
(0.0022619660624823438+0j) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12] +
(0.0022619660624823438+0j) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12] +
(-5.927453082724306e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z10] +
(0.00398984145661931+0j) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13] +
(0.0013038004788126951+0j) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13] +
(-6.733197742323143e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z11] +
(9.306536651883631e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z12] +
(1.2393363217132198e-06+0j) [Y1 Z2 Z3 Z4 Y5 Z13] +
(0.0005192743499487632+0j) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10] +
(-1.850564192860225e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.003356670563832879+0j) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9] +
(-0.0001384017730355146+0j) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11] +
(-4.997018422187658e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13] +
(6.17524620712619e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12] +
(-0.0032675138544235416+0j) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13] +
(-0.0005192743499487632+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10] +
(1.850564192860225e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-0.003356670563832879+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9] +
(-0.0001384017730355146+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11] +
(-4.997018422187658e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13] +
(-6.17524620712619e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12] +
(-0.0032675138544235416+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13] +
(1.2004287493900927e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13] +
(0.04274327701378333+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7] +
(0.0012803060973496888+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8] +
(0.004636976661182568+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9] +
(7.246974425367426e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12] +
(7.246974425367426e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12] +
(0.005241535382803865+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10] +
(1.071728218241235e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13] +
(2.312094305145714e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13] +
(0.005379937155839381+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11] +
(0.00104352465349077+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12] +
(0.004311038507914312+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13] +
(0.0038764708993369295+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10] +
(-7.540341413642074e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-0.0038764708993369295+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10] +
(7.540341413642074e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-0.0029841661681219173+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(-0.0029841661681219173+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(0.07165035181002663+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.0012366478019245584+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(0.004220813970046476+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(-1.3987009014851745e-05+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(8.949476487207354e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(-7.661347213203925e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(-0.0021413612231015646+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11] +
(-6.876621658391887e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(0.00540895442241+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11] +
(-1.0444941298021551e-06+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(0.00153248352307307+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11] +
(-2.9045998843794783e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(0.005286546538226888+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11] +
(-9.956079230002197e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.002779026799025518+0j) [Y1 Z2 Z3 Z4 Z5 Y7] +
(0.004767272188278125+0j) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11] +
(-8.105515037141971e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0024629170071339373+0j) [Y1 Z2 Z3 Z4 Z6 Y7] +
(0.000715673424890916+0j) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11] +
(-3.0767325318978334e-07+0j) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.2919694862618413e-07+0j) [Y1 Z2 Z3 Y5] +
(0.0016095313817213902+0j) [Y1 Z2 Z3 Z5 Z6 Y7] +
(-7.14162522115256e-05+0j) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-2.666731754812456e-07+0j) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.7379332624231772e-07+0j) [Y1 Z2 Z4 Y5] +
(0.0016676041811440672+0j) [Y1 Z2 Z4 Z5 Z6 Y7] +
(-0.0014528843214168905+0j) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(4.670402390455159e-07+0j) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0032769719312317077+0j) [Y1 Y3] +
(3.6060718679978207e-07+0j) [Y1 Z3 Z4 Y5] +
(0.003961560792496535+0j) [Y1 Z3 Z4 Z5 Z6 Y7] +
(0.00018787053389553542+0j) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.6569309315550103e-07+0j) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.1908508083223051e-06+0j) [Z1 X2 Z3 X4] +
(-0.03276765782329048+0j) [Z1 X2 Z3 Z4 Z5 X6] +
(-0.07635021950634995+0j) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.5809603692897268e-05+0j) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.1908508083223051e-06+0j) [Z1 Y2 Z3 Y4] +
(-0.03276765782329048+0j) [Z1 Y2 Z3 Z4 Z5 Y6] +
(-0.07635021950634995+0j) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.5809603692897268e-05+0j) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-8.337746754109954e-07+0j) [Z1 X3 Z4 X5] +
(-0.027115036845273135+0j) [Z1 X3 Z4 Z5 Z6 X7] +
(-0.0675238509921401+0j) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.4017109735203908e-05+0j) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.337746754109954e-07+0j) [Z1 Y3 Z4 Y5] +
(-0.027115036845273135+0j) [Z1 Y3 Z4 Z5 Z6 Y7] +
(-0.0675238509921401+0j) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.4017109735203908e-05+0j) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.344081556691545e-06+0j) [Z1 X4 Z5 X6] +
(-1.6103585305707328e-05+0j) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-3.344081556691545e-06+0j) [Z1 Y4 Z5 Y6] +
(-1.6103585305707328e-05+0j) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(0.19936354537360845+0j) [Z1 Z4] +
(-3.099349243799313e-06+0j) [Z1 X5 Z6 X7] +
(-1.531680879528886e-05+0j) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-3.099349243799313e-06+0j) [Z1 Y5 Z6 Y7] +
(-1.531680879528886e-05+0j) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.19661770890342162+0j) [Z1 Z5] +
(0.0560073308778077+0j) [Z1 X6 Z7 Z8 Z9 X10] +
(-6.4818518337450695e-06+0j) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0560073308778077+0j) [Z1 Y6 Z7 Z8 Z9 Y10] +
(-6.4818518337450695e-06+0j) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.24853483371314236+0j) [Z1 Z6] +
(0.05608468124661361+0j) [Z1 X7 Z8 Z9 Z10 X11] +
(-6.652209669292463e-06+0j) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.05608468124661361+0j) [Z1 Y7 Z8 Z9 Z10 Y11] +
(-6.652209669292463e-06+0j) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.2416466393601718+0j) [Z1 Z7] +
(-1.6148794141945263e-06+0j) [Z1 X10 Z11 X12] +
(-1.6148794141945263e-06+0j) [Z1 Y10 Z11 Y12] +
(0.20072866460441757+0j) [Z1 Z10] +
(-2.1776646053217677e-06+0j) [Z1 X11 Z12 X13] +
(-2.1776646053217677e-06+0j) [Z1 Y11 Z12 Y13] +
(0.1929972393536423+0j) [Z1 Z11] +
(0.21631037498631792+0j) [Z1 Z12] +
(0.21102659849791497+0j) [Z1 Z13] +
(-0.03583956795335344+0j) [X2 X3 Y4 Y5] +
(-2.199051618123706e-07+0j) [X2 X3 Y4 Z5 Z6 Y7] +
(-2.3609563202857227e-06+0j) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.010311482489831839+0j) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.199051618123706e-07+0j) [X2 X3 X5 X6] +
(-2.3609563202857223e-06+0j) [X2 X3 X5 Z6 Z7 Z8 Z9 X10] +
(-0.010311482489831839+0j) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.031143817988967117+0j) [X2 X3 Y6 Y7] +
(0.0053686593581095815+0j) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11] +
(9.209350648316568e-08+0j) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0053686593581095815+0j) [X2 X3 X7 Z8 Z9 X10] +
(9.209350648316568e-08+0j) [X2 X3 X7 Z8 Z9 Z10 Z11 X12] +
(-0.03619412355904262+0j) [X2 X3 Y8 Y9] +
(-0.02538465750845734+0j) [X2 X3 Y10 Y11] +
(2.172669101460603e-06+0j) [X2 X3 Y10 Z11 Z12 Y13] +
(2.172669101460603e-06+0j) [X2 X3 X11 X12] +
(-0.01557720806397644+0j) [X2 X3 Y12 Y13] +
(0.03583956795335344+0j) [X2 Y3 Y4 X5] +
(2.199051618123706e-07+0j) [X2 Y3 Y4 Z5 Z6 X7] +
(2.3609563202857227e-06+0j) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.010311482489831839+0j) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.199051618123706e-07+0j) [X2 Y3 Y5 X6] +
(-2.3609563202857223e-06+0j) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.010311482489831839+0j) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.031143817988967117+0j) [X2 Y3 Y6 X7] +
(-0.0053686593581095815+0j) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11] +
(-9.209350648316568e-08+0j) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0053686593581095815+0j) [X2 Y3 Y7 Z8 Z9 X10] +
(9.209350648316568e-08+0j) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12] +
(0.03619412355904262+0j) [X2 Y3 Y8 X9] +
(0.02538465750845734+0j) [X2 Y3 Y10 X11] +
(-2.172669101460603e-06+0j) [X2 Y3 Y10 Z11 Z12 X13] +
(2.172669101460603e-06+0j) [X2 Y3 Y11 X12] +
(0.01557720806397644+0j) [X2 Y3 Y12 X13] +
(-3.887051673092265e-06+0j) [X2 Z3 X4] +
(-0.005143391768825116+0j) [X2 Z3 X4 X5 Z6 X7] +
(-0.0098417492469626+0j) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.9885117063999666e-06+0j) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.005143391768825116+0j) [X2 Z3 X4 Y5 Z6 Y7] +
(-0.0098417492469626+0j) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.9885117063999666e-06+0j) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.764994118591163e-07+0j) [X2 Z3 X4 Z5] +
(1.6893489514970217e-06+0j) [X2 Z3 X4 X6 Z7 Z8 Z9 X10] +
(0.010757563953908953+0j) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.5371780964700184e-08+0j) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10] +
(4.205548411217834e-05+0j) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.5935343911386345e-07+0j) [X2 Z3 X4 Z6] +
(3.211842019175555e-06+0j) [X2 Z3 X4 X7 Z8 Z9 Z10 X11] +
(0.019299560579363776+0j) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.211842019175555e-06+0j) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11] +
(0.019299560579363776+0j) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.1954890100469523e-06+0j) [X2 Z3 X4 Z7] +
(2.1868423776946327e-07+0j) [X2 Z3 X4 Z8] +
(-5.770052994946738e-07+0j) [X2 Z3 X4 Z9] +
(0.015588250102380193+0j) [X2 Z3 X4 X10 Z11 X12] +
(0.005324835234221694+0j) [X2 Z3 X4 Y10 Z11 Y12] +
(-3.158656432038324e-06+0j) [X2 Z3 X4 Z10] +
(0.024353077678068925+0j) [X2 Z3 X4 X11 Z12 X13] +
(0.024353077678068925+0j) [X2 Z3 X4 Y11 Z12 Y13] +
(-7.801707500529877e-06+0j) [X2 Z3 X4 Z11] +
(3.5390541845541437e-06+0j) [X2 Z3 X4 Z12] +
(8.814937306676361e-06+0j) [X2 Z3 X4 Z13] +
(1.628853243541976e-06+0j) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10] +
(0.010715508469796776+0j) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.010263414868158498+0j) [X2 Z3 Y4 Y10 Z11 X12] +
(-1.454842449160816e-06+0j) [X2 Z3 Z4 X5 Y6 Y7] +
(-3.15134631122051e-06+0j) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11] +
(-0.0192575050952516+0j) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.5224930676785333e-06+0j) [X2 Z3 Z4 X5 X7 Z8 Z9 X10] +
(-0.008541996625454826+0j) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-7.95689537264137e-07+0j) [X2 Z3 Z4 X5 Y8 Y9] +
(-4.643051068491552e-06+0j) [X2 Z3 Z4 X5 Y10 Y11] +
(-0.019028242443847227+0j) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13] +
(-0.008764827575688729+0j) [X2 Z3 Z4 X5 X11 X12] +
(5.275883122122217e-06+0j) [X2 Z3 Z4 X5 Y12 Y13] +
(1.454842449160816e-06+0j) [X2 Z3 Z4 Y5 Y6 X7] +
(3.15134631122051e-06+0j) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11] +
(0.0192575050952516+0j) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.5224930676785333e-06+0j) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10] +
(-0.008541996625454826+0j) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(7.95689537264137e-07+0j) [X2 Z3 Z4 Y5 Y8 X9] +
(4.643051068491552e-06+0j) [X2 Z3 Z4 Y5 Y10 X11] +
(0.019028242443847227+0j) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13] +
(-0.008764827575688729+0j) [X2 Z3 Z4 Y5 Y11 X12] +
(-5.275883122122217e-06+0j) [X2 Z3 Z4 Y5 Y12 X13] +
(-0.12133276911042329+0j) [X2 Z3 Z4 Z5 X6] +
(-0.00846997879102398+0j) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(2.686381544586611e-07+0j) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.00846997879102398+0j) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(2.686381544586611e-07+0j) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.025996177598021135+0j) [X2 Z3 Z4 Z5 X6 Z7] +
(-0.005805188989826926+0j) [X2 Z3 Z4 Z5 X6 Z8] +
(-0.017561202409646166+0j) [X2 Z3 Z4 Z5 X6 Z9] +
(-7.988770288949553e-07+0j) [X2 Z3 Z4 Z5 X6 X10 Z11 X12] +
(-3.427323108857073e-07+0j) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12] +
(-0.0008145313270956603+0j) [X2 Z3 Z4 Z5 X6 Z10] +
(2.745518400383146e-06+0j) [X2 Z3 Z4 Z5 X6 X11 Z12 X13] +
(2.745518400383146e-06+0j) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13] +
(0.014411099430130893+0j) [X2 Z3 Z4 Z5 X6 Z11] +
(0.0006650070219499358+0j) [X2 Z3 Z4 Z5 X6 Z12] +
(-0.003493790359890116+0j) [X2 Z3 Z4 Z5 X6 Z13] +
(-4.561447180092482e-07+0j) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12] +
(-0.011756013419819241+0j) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9] +
(0.015225630757226554+0j) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11] +
(-3.088250711268853e-06+0j) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(-3.544395429278102e-06+0j) [X2 Z3 Z4 Z5 Z6 X7 X11 X12] +
(-0.004158797381840051+0j) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13] +
(0.011756013419819241+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9] +
(-0.015225630757226554+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11] +
(3.088250711268853e-06+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(-3.544395429278102e-06+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12] +
(0.004158797381840051+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13] +
(0.014603704729162066+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(-2.8742990713630047e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.014603704729162066+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(-2.8742990713630047e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(-0.2816425776702277+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-1.3002946561942242e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(-1.3002946561942242e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(-0.024282117354692948+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(-0.019538050311314645+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(-0.017091553155898782+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(0.0024464971554158613+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(-0.0024464971554158613+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(5.775950527158138e-05+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(2.8836765760017514e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(5.146496327477979e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(3.846201671283755e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(-0.039359168022052984+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10] +
(7.97982579332848e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(-0.02475546329289091+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10] +
(5.1055267219654764e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(-0.021433810721600808+0j) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10] +
(5.1593505019361425e-06+0j) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(-0.02990378951262479+0j) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10] +
(5.427988656394803e-06+0j) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0016638798784907498+0j) [X2 Z3 Z4 X6] +
(-0.0188890303049429+0j) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10] +
(2.9473560116387083e-06+0j) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.003479511890334366+0j) [X2 Z3 Z5 X6] +
(-0.0287307795519055+0j) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10] +
(5.935867718038675e-06+0j) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.6021167406798884e-06+0j) [X2 X4] +
(0.0004956762314915926+0j) [X2 Z4 Z5 X6] +
(-0.03560837898831252+0j) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(7.2532733480562396e-06+0j) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.03583956795335344+0j) [Y2 X3 X4 Y5] +
(2.199051618123706e-07+0j) [Y2 X3 X4 Z5 Z6 Y7] +
(2.3609563202857227e-06+0j) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.010311482489831839+0j) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.199051618123706e-07+0j) [Y2 X3 X5 Y6] +
(-2.3609563202857223e-06+0j) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.010311482489831839+0j) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.031143817988967117+0j) [Y2 X3 X6 Y7] +
(-0.0053686593581095815+0j) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11] +
(-9.209350648316568e-08+0j) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0053686593581095815+0j) [Y2 X3 X7 Z8 Z9 Y10] +
(9.209350648316568e-08+0j) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12] +
(0.03619412355904262+0j) [Y2 X3 X8 Y9] +
(0.02538465750845734+0j) [Y2 X3 X10 Y11] +
(-2.172669101460603e-06+0j) [Y2 X3 X10 Z11 Z12 Y13] +
(2.172669101460603e-06+0j) [Y2 X3 X11 Y12] +
(0.01557720806397644+0j) [Y2 X3 X12 Y13] +
(-0.03583956795335344+0j) [Y2 Y3 X4 X5] +
(-2.199051618123706e-07+0j) [Y2 Y3 X4 Z5 Z6 X7] +
(-2.3609563202857227e-06+0j) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.010311482489831839+0j) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.199051618123706e-07+0j) [Y2 Y3 Y5 Y6] +
(-2.3609563202857223e-06+0j) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.010311482489831839+0j) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.031143817988967117+0j) [Y2 Y3 X6 X7] +
(0.0053686593581095815+0j) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11] +
(9.209350648316568e-08+0j) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0053686593581095815+0j) [Y2 Y3 Y7 Z8 Z9 Y10] +
(9.209350648316568e-08+0j) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.03619412355904262+0j) [Y2 Y3 X8 X9] +
(-0.02538465750845734+0j) [Y2 Y3 X10 X11] +
(2.172669101460603e-06+0j) [Y2 Y3 X10 Z11 Z12 X13] +
(2.172669101460603e-06+0j) [Y2 Y3 Y11 Y12] +
(-0.01557720806397644+0j) [Y2 Y3 X12 X13] +
(1.628853243541976e-06+0j) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10] +
(0.010715508469796776+0j) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.010263414868158498+0j) [Y2 Z3 X4 X10 Z11 Y12] +
(-3.887051673092265e-06+0j) [Y2 Z3 Y4] +
(-0.005143391768825116+0j) [Y2 Z3 Y4 X5 Z6 X7] +
(-0.0098417492469626+0j) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.9885117063999666e-06+0j) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.005143391768825116+0j) [Y2 Z3 Y4 Y5 Z6 Y7] +
(-0.0098417492469626+0j) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.9885117063999666e-06+0j) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.764994118591163e-07+0j) [Y2 Z3 Y4 Z5] +
(4.5371780964700184e-08+0j) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10] +
(4.205548411217834e-05+0j) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.6893489514970217e-06+0j) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10] +
(0.010757563953908953+0j) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.5935343911386345e-07+0j) [Y2 Z3 Y4 Z6] +
(3.211842019175555e-06+0j) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11] +
(0.019299560579363776+0j) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.211842019175555e-06+0j) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11] +
(0.019299560579363776+0j) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.1954890100469523e-06+0j) [Y2 Z3 Y4 Z7] +
(2.1868423776946327e-07+0j) [Y2 Z3 Y4 Z8] +
(-5.770052994946738e-07+0j) [Y2 Z3 Y4 Z9] +
(0.005324835234221694+0j) [Y2 Z3 Y4 X10 Z11 X12] +
(0.015588250102380193+0j) [Y2 Z3 Y4 Y10 Z11 Y12] +
(-3.158656432038324e-06+0j) [Y2 Z3 Y4 Z10] +
(0.024353077678068925+0j) [Y2 Z3 Y4 X11 Z12 X13] +
(0.024353077678068925+0j) [Y2 Z3 Y4 Y11 Z12 Y13] +
(-7.801707500529877e-06+0j) [Y2 Z3 Y4 Z11] +
(3.5390541845541437e-06+0j) [Y2 Z3 Y4 Z12] +
(8.814937306676361e-06+0j) [Y2 Z3 Y4 Z13] +
(1.454842449160816e-06+0j) [Y2 Z3 Z4 X5 X6 Y7] +
(3.15134631122051e-06+0j) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11] +
(0.0192575050952516+0j) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.5224930676785333e-06+0j) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10] +
(-0.008541996625454826+0j) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(7.95689537264137e-07+0j) [Y2 Z3 Z4 X5 X8 Y9] +
(4.643051068491552e-06+0j) [Y2 Z3 Z4 X5 X10 Y11] +
(0.019028242443847227+0j) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13] +
(-0.008764827575688729+0j) [Y2 Z3 Z4 X5 X11 Y12] +
(-5.275883122122217e-06+0j) [Y2 Z3 Z4 X5 X12 Y13] +
(-1.454842449160816e-06+0j) [Y2 Z3 Z4 Y5 X6 X7] +
(-3.15134631122051e-06+0j) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11] +
(-0.0192575050952516+0j) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.5224930676785333e-06+0j) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10] +
(-0.008541996625454826+0j) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-7.95689537264137e-07+0j) [Y2 Z3 Z4 Y5 X8 X9] +
(-4.643051068491552e-06+0j) [Y2 Z3 Z4 Y5 X10 X11] +
(-0.019028242443847227+0j) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13] +
(-0.008764827575688729+0j) [Y2 Z3 Z4 Y5 Y11 Y12] +
(5.275883122122217e-06+0j) [Y2 Z3 Z4 Y5 X12 X13] +
(-4.561447180092482e-07+0j) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12] +
(-0.12133276911042329+0j) [Y2 Z3 Z4 Z5 Y6] +
(-0.00846997879102398+0j) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(2.686381544586611e-07+0j) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.00846997879102398+0j) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(2.686381544586611e-07+0j) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.025996177598021135+0j) [Y2 Z3 Z4 Z5 Y6 Z7] +
(-0.005805188989826926+0j) [Y2 Z3 Z4 Z5 Y6 Z8] +
(-0.017561202409646166+0j) [Y2 Z3 Z4 Z5 Y6 Z9] +
(-3.427323108857073e-07+0j) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12] +
(-7.988770288949553e-07+0j) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12] +
(-0.0008145313270956603+0j) [Y2 Z3 Z4 Z5 Y6 Z10] +
(2.745518400383146e-06+0j) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13] +
(2.745518400383146e-06+0j) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13] +
(0.014411099430130893+0j) [Y2 Z3 Z4 Z5 Y6 Z11] +
(0.0006650070219499358+0j) [Y2 Z3 Z4 Z5 Y6 Z12] +
(-0.003493790359890116+0j) [Y2 Z3 Z4 Z5 Y6 Z13] +
(0.011756013419819241+0j) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9] +
(-0.015225630757226554+0j) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11] +
(3.088250711268853e-06+0j) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(-3.544395429278102e-06+0j) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12] +
(0.004158797381840051+0j) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13] +
(-0.011756013419819241+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9] +
(0.015225630757226554+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11] +
(-3.088250711268853e-06+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(-3.544395429278102e-06+0j) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12] +
(-0.004158797381840051+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13] +
(0.014603704729162066+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(-2.8742990713630047e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.014603704729162066+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(-2.8742990713630047e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(-0.2816425776702277+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-1.3002946561942242e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(-1.3002946561942242e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(-0.024282117354692948+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(-0.019538050311314645+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(-0.017091553155898782+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(-0.0024464971554158613+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(0.0024464971554158613+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(5.775950527158138e-05+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.8836765760017514e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(5.146496327477979e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(3.846201671283755e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(-0.039359168022052984+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10] +
(7.97982579332848e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(-0.02475546329289091+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10] +
(5.1055267219654764e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(-0.021433810721600808+0j) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10] +
(5.1593505019361425e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(-0.02990378951262479+0j) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10] +
(5.427988656394803e-06+0j) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0016638798784907498+0j) [Y2 Z3 Z4 Y6] +
(-0.0188890303049429+0j) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10] +
(2.9473560116387083e-06+0j) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.003479511890334366+0j) [Y2 Z3 Z5 Y6] +
(-0.0287307795519055+0j) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10] +
(5.935867718038675e-06+0j) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.6021167406798884e-06+0j) [Y2 Y4] +
(0.0004956762314915926+0j) [Y2 Z4 Z5 Y6] +
(-0.03560837898831252+0j) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(7.2532733480562396e-06+0j) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.65389422268317+0j) [Z2] +
(1.6021167406798884e-06+0j) [Z2 X3 Z4 X5] +
(0.0004956762314915926+0j) [Z2 X3 Z4 Z5 Z6 X7] +
(-0.03560837898831252+0j) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(7.2532733480562396e-06+0j) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.6021167406798884e-06+0j) [Z2 Y3 Z4 Y5] +
(0.0004956762314915926+0j) [Z2 Y3 Z4 Z5 Z6 Y7] +
(-0.03560837898831252+0j) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(7.2532733480562396e-06+0j) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.18189085790751341+0j) [Z2 Z3] +
(-9.509249752351701e-07+0j) [Z2 X4 Z5 X6] +
(-4.72884314720647e-06+0j) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.024591860883829992+0j) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-9.509249752351701e-07+0j) [Z2 Y4 Z5 Y6] +
(-4.72884314720647e-06+0j) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.024591860883829992+0j) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.12495807739503222+0j) [Z2 Z4] +
(-1.1708301370475408e-06+0j) [Z2 X5 Z6 X7] +
(-7.0897994674921915e-06+0j) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.03490334337366183+0j) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.1708301370475408e-06+0j) [Z2 Y5 Z6 Y7] +
(-7.0897994674921915e-06+0j) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.03490334337366183+0j) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.16079764534838567+0j) [Z2 Z5] +
(0.019020423173039924+0j) [Z2 X6 Z7 Z8 Z9 X10] +
(-2.103215604640571e-06+0j) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.019020423173039924+0j) [Z2 Y6 Z7 Z8 Z9 Y10] +
(-2.103215604640571e-06+0j) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.13739104762683216+0j) [Z2 Z6] +
(0.024389082531149506+0j) [Z2 X7 Z8 Z9 Z10 X11] +
(-2.0111220981574054e-06+0j) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.024389082531149506+0j) [Z2 Y7 Z8 Z9 Z10 Y11] +
(-2.0111220981574054e-06+0j) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.16853486561579928+0j) [Z2 Z7] +
(0.15071408121008278+0j) [Z2 Z8] +
(0.1869082047691254+0j) [Z2 Z9] +
(-1.0632283424949317e-06+0j) [Z2 X10 Z11 X12] +
(-1.0632283424949317e-06+0j) [Z2 Y10 Z11 Y12] +
(1.109440758965671e-06+0j) [Z2 X11 Z12 X13] +
(1.109440758965671e-06+0j) [Z2 Y11 Z12 Y13] +
(0.14011289865354798+0j) [Z2 Z12] +
(0.15569010671752442+0j) [Z2 Z13] +
(0.005143391768825115+0j) [X3 X4 Y5 Y6] +
(0.009841749246962602+0j) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10] +
(-2.9885117063999666e-06+0j) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.4548424491608157e-06+0j) [X3 X4 X6 X7] +
(-1.5224930676785333e-06+0j) [X3 X4 X6 Z7 Z8 Z9 Z10 X11] +
(-0.008541996625454826+0j) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.15134631122051e-06+0j) [X3 X4 Y7 Z8 Z9 Y10] +
(-0.0192575050952516+0j) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12] +
(-7.95689537264137e-07+0j) [X3 X4 X8 X9] +
(-4.643051068491553e-06+0j) [X3 X4 X10 X11] +
(-0.008764827575688729+0j) [X3 X4 X10 Z11 Z12 X13] +
(-0.019028242443847227+0j) [X3 X4 Y11 Y12] +
(5.275883122122217e-06+0j) [X3 X4 X12 X13] +
(-0.005143391768825115+0j) [X3 Y4 Y5 X6] +
(-0.009841749246962602+0j) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10] +
(2.9885117063999666e-06+0j) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.4548424491608157e-06+0j) [X3 Y4 Y6 X7] +
(-1.5224930676785333e-06+0j) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11] +
(-0.008541996625454826+0j) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.15134631122051e-06+0j) [X3 Y4 Y7 Z8 Z9 X10] +
(0.0192575050952516+0j) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12] +
(-7.95689537264137e-07+0j) [X3 Y4 Y8 X9] +
(-4.643051068491553e-06+0j) [X3 Y4 Y10 X11] +
(-0.008764827575688729+0j) [X3 Y4 Y10 Z11 Z12 X13] +
(0.019028242443847227+0j) [X3 Y4 Y11 X12] +
(5.275883122122217e-06+0j) [X3 Y4 Y12 X13] +
(-3.887051673092265e-06+0j) [X3 Z4 X5] +
(3.211842019175555e-06+0j) [X3 Z4 X5 X6 Z7 Z8 Z9 X10] +
(0.019299560579363776+0j) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.211842019175555e-06+0j) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10] +
(0.019299560579363776+0j) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.1954890100469523e-06+0j) [X3 Z4 X5 Z6] +
(1.6893489514970217e-06+0j) [X3 Z4 X5 X7 Z8 Z9 Z10 X11] +
(0.010757563953908953+0j) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.5371780964700184e-08+0j) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11] +
(4.205548411217834e-05+0j) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.5935343911386345e-07+0j) [X3 Z4 X5 Z7] +
(-5.770052994946738e-07+0j) [X3 Z4 X5 Z8] +
(2.1868423776946327e-07+0j) [X3 Z4 X5 Z9] +
(0.024353077678068925+0j) [X3 Z4 X5 X10 Z11 X12] +
(0.024353077678068925+0j) [X3 Z4 X5 Y10 Z11 Y12] +
(-7.801707500529877e-06+0j) [X3 Z4 X5 Z10] +
(0.015588250102380193+0j) [X3 Z4 X5 X11 Z12 X13] +
(0.005324835234221694+0j) [X3 Z4 X5 Y11 Z12 Y13] +
(-3.158656432038324e-06+0j) [X3 Z4 X5 Z11] +
(8.814937306676361e-06+0j) [X3 Z4 X5 Z12] +
(3.5390541845541437e-06+0j) [X3 Z4 X5 Z13] +
(1.628853243541976e-06+0j) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11] +
(0.010715508469796776+0j) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.010263414868158498+0j) [X3 Z4 Y5 Y11 Z12 X13] +
(0.00846997879102398+0j) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10] +
(-2.686381544586611e-07+0j) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.011756013419819241+0j) [X3 Z4 Z5 X6 X8 X9] +
(0.015225630757226554+0j) [X3 Z4 Z5 X6 X10 X11] +
(-3.544395429278102e-06+0j) [X3 Z4 Z5 X6 X10 Z11 Z12 X13] +
(-3.088250711268853e-06+0j) [X3 Z4 Z5 X6 Y11 Y12] +
(-0.004158797381840051+0j) [X3 Z4 Z5 X6 X12 X13] +
(-0.00846997879102398+0j) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10] +
(2.686381544586611e-07+0j) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.011756013419819241+0j) [X3 Z4 Z5 Y6 Y8 X9] +
(0.015225630757226554+0j) [X3 Z4 Z5 Y6 Y10 X11] +
(-3.544395429278102e-06+0j) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13] +
(3.088250711268853e-06+0j) [X3 Z4 Z5 Y6 Y11 X12] +
(-0.004158797381840051+0j) [X3 Z4 Z5 Y6 Y12 X13] +
(-0.12133276911042327+0j) [X3 Z4 Z5 Z6 X7] +
(-0.017561202409646166+0j) [X3 Z4 Z5 Z6 X7 Z8] +
(-0.005805188989826926+0j) [X3 Z4 Z5 Z6 X7 Z9] +
(2.745518400383146e-06+0j) [X3 Z4 Z5 Z6 X7 X10 Z11 X12] +
(2.745518400383146e-06+0j) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12] +
(0.014411099430130893+0j) [X3 Z4 Z5 Z6 X7 Z10] +
(-7.988770288949553e-07+0j) [X3 Z4 Z5 Z6 X7 X11 Z12 X13] +
(-3.427323108857073e-07+0j) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13] +
(-0.0008145313270956603+0j) [X3 Z4 Z5 Z6 X7 Z11] +
(-0.003493790359890116+0j) [X3 Z4 Z5 Z6 X7 Z12] +
(0.0006650070219499358+0j) [X3 Z4 Z5 Z6 X7 Z13] +
(-4.561447180092482e-07+0j) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13] +
(-0.01460370472916207+0j) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10] +
(2.8742990713630047e-06+0j) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(0.01460370472916207+0j) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10] +
(-2.8742990713630047e-06+0j) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(1.3002946561942242e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12] +
(0.0024464971554158613+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-1.3002946561942242e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12] +
(0.0024464971554158613+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(-0.2816425776702276+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.017091553155898782+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(-0.019538050311314645+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(5.775950527158135e-05+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(2.8836765760017514e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(3.846201671283755e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(-0.024282117354692948+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11] +
(5.146496327477979e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(-0.02475546329289091+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11] +
(5.1055267219654764e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(-0.039359168022052984+0j) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11] +
(7.97982579332848e-06+0j) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(-0.02990378951262479+0j) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11] +
(5.427988656394803e-06+0j) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.025996177598021135+0j) [X3 Z4 Z5 X7] +
(-0.021433810721600808+0j) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11] +
(5.1593505019361425e-06+0j) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.003479511890334366+0j) [X3 Z4 Z6 X7] +
(-0.0287307795519055+0j) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11] +
(5.935867718038675e-06+0j) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.764994118591163e-07+0j) [X3 X5] +
(0.0016638798784907498+0j) [X3 Z5 Z6 X7] +
(-0.0188890303049429+0j) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.9473560116387083e-06+0j) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.005143391768825115+0j) [Y3 X4 X5 Y6] +
(-0.009841749246962602+0j) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10] +
(2.9885117063999666e-06+0j) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.4548424491608157e-06+0j) [Y3 X4 X6 Y7] +
(-1.5224930676785333e-06+0j) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11] +
(-0.008541996625454826+0j) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.15134631122051e-06+0j) [Y3 X4 X7 Z8 Z9 Y10] +
(0.0192575050952516+0j) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12] +
(-7.95689537264137e-07+0j) [Y3 X4 X8 Y9] +
(-4.643051068491553e-06+0j) [Y3 X4 X10 Y11] +
(-0.008764827575688729+0j) [Y3 X4 X10 Z11 Z12 Y13] +
(0.019028242443847227+0j) [Y3 X4 X11 Y12] +
(5.275883122122217e-06+0j) [Y3 X4 X12 Y13] +
(0.005143391768825115+0j) [Y3 Y4 X5 X6] +
(0.009841749246962602+0j) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10] +
(-2.9885117063999666e-06+0j) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.4548424491608157e-06+0j) [Y3 Y4 Y6 Y7] +
(-1.5224930676785333e-06+0j) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11] +
(-0.008541996625454826+0j) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.15134631122051e-06+0j) [Y3 Y4 X7 Z8 Z9 X10] +
(-0.0192575050952516+0j) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12] +
(-7.95689537264137e-07+0j) [Y3 Y4 Y8 Y9] +
(-4.643051068491553e-06+0j) [Y3 Y4 Y10 Y11] +
(-0.008764827575688729+0j) [Y3 Y4 Y10 Z11 Z12 Y13] +
(-0.019028242443847227+0j) [Y3 Y4 X11 X12] +
(5.275883122122217e-06+0j) [Y3 Y4 Y12 Y13] +
(1.628853243541976e-06+0j) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11] +
(0.010715508469796776+0j) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.010263414868158498+0j) [Y3 Z4 X5 X11 Z12 Y13] +
(-3.887051673092265e-06+0j) [Y3 Z4 Y5] +
(3.211842019175555e-06+0j) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10] +
(0.019299560579363776+0j) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.211842019175555e-06+0j) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10] +
(0.019299560579363776+0j) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.1954890100469523e-06+0j) [Y3 Z4 Y5 Z6] +
(4.5371780964700184e-08+0j) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11] +
(4.205548411217834e-05+0j) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.6893489514970217e-06+0j) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11] +
(0.010757563953908953+0j) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.5935343911386345e-07+0j) [Y3 Z4 Y5 Z7] +
(-5.770052994946738e-07+0j) [Y3 Z4 Y5 Z8] +
(2.1868423776946327e-07+0j) [Y3 Z4 Y5 Z9] +
(0.024353077678068925+0j) [Y3 Z4 Y5 X10 Z11 X12] +
(0.024353077678068925+0j) [Y3 Z4 Y5 Y10 Z11 Y12] +
(-7.801707500529877e-06+0j) [Y3 Z4 Y5 Z10] +
(0.005324835234221694+0j) [Y3 Z4 Y5 X11 Z12 X13] +
(0.015588250102380193+0j) [Y3 Z4 Y5 Y11 Z12 Y13] +
(-3.158656432038324e-06+0j) [Y3 Z4 Y5 Z11] +
(8.814937306676361e-06+0j) [Y3 Z4 Y5 Z12] +
(3.5390541845541437e-06+0j) [Y3 Z4 Y5 Z13] +
(-0.00846997879102398+0j) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10] +
(2.686381544586611e-07+0j) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.011756013419819241+0j) [Y3 Z4 Z5 X6 X8 Y9] +
(0.015225630757226554+0j) [Y3 Z4 Z5 X6 X10 Y11] +
(-3.544395429278102e-06+0j) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13] +
(3.088250711268853e-06+0j) [Y3 Z4 Z5 X6 X11 Y12] +
(-0.004158797381840051+0j) [Y3 Z4 Z5 X6 X12 Y13] +
(0.00846997879102398+0j) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10] +
(-2.686381544586611e-07+0j) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-0.011756013419819241+0j) [Y3 Z4 Z5 Y6 Y8 Y9] +
(0.015225630757226554+0j) [Y3 Z4 Z5 Y6 Y10 Y11] +
(-3.544395429278102e-06+0j) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13] +
(-3.088250711268853e-06+0j) [Y3 Z4 Z5 Y6 X11 X12] +
(-0.004158797381840051+0j) [Y3 Z4 Z5 Y6 Y12 Y13] +
(-4.561447180092482e-07+0j) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13] +
(-0.12133276911042327+0j) [Y3 Z4 Z5 Z6 Y7] +
(-0.017561202409646166+0j) [Y3 Z4 Z5 Z6 Y7 Z8] +
(-0.005805188989826926+0j) [Y3 Z4 Z5 Z6 Y7 Z9] +
(2.745518400383146e-06+0j) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12] +
(2.745518400383146e-06+0j) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12] +
(0.014411099430130893+0j) [Y3 Z4 Z5 Z6 Y7 Z10] +
(-3.427323108857073e-07+0j) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13] +
(-7.988770288949553e-07+0j) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13] +
(-0.0008145313270956603+0j) [Y3 Z4 Z5 Z6 Y7 Z11] +
(-0.003493790359890116+0j) [Y3 Z4 Z5 Z6 Y7 Z12] +
(0.0006650070219499358+0j) [Y3 Z4 Z5 Z6 Y7 Z13] +
(0.01460370472916207+0j) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10] +
(-2.8742990713630047e-06+0j) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-0.01460370472916207+0j) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10] +
(2.8742990713630047e-06+0j) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-1.3002946561942242e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12] +
(0.0024464971554158613+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(1.3002946561942242e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12] +
(0.0024464971554158613+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(-0.2816425776702276+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.017091553155898782+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(-0.019538050311314645+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(5.775950527158135e-05+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.8836765760017514e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(3.846201671283755e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(-0.024282117354692948+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11] +
(5.146496327477979e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(-0.02475546329289091+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11] +
(5.1055267219654764e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(-0.039359168022052984+0j) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11] +
(7.97982579332848e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(-0.02990378951262479+0j) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11] +
(5.427988656394803e-06+0j) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.025996177598021135+0j) [Y3 Z4 Z5 Y7] +
(-0.021433810721600808+0j) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11] +
(5.1593505019361425e-06+0j) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.003479511890334366+0j) [Y3 Z4 Z6 Y7] +
(-0.0287307795519055+0j) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11] +
(5.935867718038675e-06+0j) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.764994118591163e-07+0j) [Y3 Y5] +
(0.0016638798784907498+0j) [Y3 Z5 Z6 Y7] +
(-0.0188890303049429+0j) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.9473560116387083e-06+0j) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.6538942226831699+0j) [Z3] +
(-1.1708301370475408e-06+0j) [Z3 X4 Z5 X6] +
(-7.0897994674921915e-06+0j) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.03490334337366183+0j) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.1708301370475408e-06+0j) [Z3 Y4 Z5 Y6] +
(-7.0897994674921915e-06+0j) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.03490334337366183+0j) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.16079764534838567+0j) [Z3 Z4] +
(-9.509249752351701e-07+0j) [Z3 X5 Z6 X7] +
(-4.72884314720647e-06+0j) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.024591860883829992+0j) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-9.509249752351701e-07+0j) [Z3 Y5 Z6 Y7] +
(-4.72884314720647e-06+0j) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.024591860883829992+0j) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.12495807739503222+0j) [Z3 Z5] +
(0.024389082531149506+0j) [Z3 X6 Z7 Z8 Z9 X10] +
(-2.0111220981574054e-06+0j) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.024389082531149506+0j) [Z3 Y6 Z7 Z8 Z9 Y10] +
(-2.0111220981574054e-06+0j) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.16853486561579928+0j) [Z3 Z6] +
(0.019020423173039924+0j) [Z3 X7 Z8 Z9 Z10 X11] +
(-2.103215604640571e-06+0j) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.019020423173039924+0j) [Z3 Y7 Z8 Z9 Z10 Y11] +
(-2.103215604640571e-06+0j) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.13739104762683216+0j) [Z3 Z7] +
(0.1869082047691254+0j) [Z3 Z8] +
(0.15071408121008278+0j) [Z3 Z9] +
(1.109440758965671e-06+0j) [Z3 X10 Z11 X12] +
(1.109440758965671e-06+0j) [Z3 Y10 Z11 Y12] +
(-1.0632283424949317e-06+0j) [Z3 X11 Z12 X13] +
(-1.0632283424949317e-06+0j) [Z3 Y11 Z12 Y13] +
(0.15569010671752442+0j) [Z3 Z12] +
(0.14011289865354798+0j) [Z3 Z13] +
(-0.01198238901024798+0j) [X4 X5 Y6 Y7] +
(-0.007306759928832977+0j) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11] +
(-2.888293596076952e-07+0j) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.007306759928832977+0j) [X4 X5 X7 Z8 Z9 X10] +
(-2.888293596076952e-07+0j) [X4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-0.0071569349198569564+0j) [X4 X5 Y8 Y9] +
(-3.6945132944554667e-06+0j) [X4 X5 Y10 Z11 Z12 Y13] +
(-3.6945132944554667e-06+0j) [X4 X5 X11 X12] +
(-0.038314670294803906+0j) [X4 X5 Y12 Y13] +
(0.01198238901024798+0j) [X4 Y5 Y6 X7] +
(0.007306759928832977+0j) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11] +
(2.888293596076952e-07+0j) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.007306759928832977+0j) [X4 Y5 Y7 Z8 Z9 X10] +
(-2.888293596076952e-07+0j) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(0.0071569349198569564+0j) [X4 Y5 Y8 X9] +
(3.6945132944554667e-06+0j) [X4 Y5 Y10 Z11 Z12 X13] +
(-3.6945132944554667e-06+0j) [X4 Y5 Y11 X12] +
(0.038314670294803906+0j) [X4 Y5 Y12 X13] +
(-1.2260484989589458e-05+0j) [X4 Z5 X6] +
(-1.2283337825748155e-06+0j) [X4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(0.00024636437569575104+0j) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2283337825748155e-06+0j) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(0.00024636437569575104+0j) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.854060858053329e-06+0j) [X4 Z5 X6 Z7] +
(-1.3980449082091647e-06+0j) [X4 Z5 X6 Z8] +
(-1.8818501833174753e-06+0j) [X4 Z5 X6 Z9] +
(0.00796088072592157+0j) [X4 Z5 X6 X10 Z11 X12] +
(-0.0009298507967730387+0j) [X4 Z5 X6 Y10 Z11 Y12] +
(-1.6923978285902755e-06+0j) [X4 Z5 X6 Z10] +
(-0.012215040997613995+0j) [X4 Z5 X6 X11 Z12 X13] +
(-0.012215040997613995+0j) [X4 Z5 X6 Y11 Z12 Y13] +
(4.281913884876456e-06+0j) [X4 Z5 X6 Z11] +
(-4.588855155727868e-06+0j) [X4 Z5 X6 Z13] +
(0.008890731522694609+0j) [X4 Z5 Y6 Y10 Z11 X12] +
(-4.838052751083105e-07+0j) [X4 Z5 Z6 X7 Y8 Y9] +
(5.974311713466731e-06+0j) [X4 Z5 Z6 X7 Y10 Y11] +
(0.011285190200840957+0j) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(0.020175921723535564+0j) [X4 Z5 Z6 X7 X11 X12] +
(-4.556569218146057e-06+0j) [X4 Z5 Z6 X7 Y12 Y13] +
(4.838052751083105e-07+0j) [X4 Z5 Z6 Y7 Y8 X9] +
(-5.974311713466731e-06+0j) [X4 Z5 Z6 Y7 Y10 X11] +
(-0.011285190200840957+0j) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(0.020175921723535564+0j) [X4 Z5 Z6 Y7 Y11 X12] +
(4.556569218146057e-06+0j) [X4 Z5 Z6 Y7 Y12 X13] +
(1.330473188676827e-06+0j) [X4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(0.005923798336561343+0j) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(1.330473188676827e-06+0j) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(0.005923798336561343+0j) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(-6.631277928457139e-05+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.01602460368917958+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(-0.01602460368917958+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(3.3343312894609523e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(-4.734622038677317e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(-9.806102775255643e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(-5.071480736578327e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(5.071480736578327e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-0.3693708936615615+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.02314513092952899+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-0.009612634606847284+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(-0.025637238296026866+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(-8.774817864592504e-06+0j) [X4 Z5 Z6 Z7 Z8 X10] +
(-0.04764261217638309+0j) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(-7.444344675915676e-06+0j) [X4 Z5 Z6 Z7 Z9 X10] +
(-0.04171881383982175+0j) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(-6.290028433146429e-06+0j) [X4 Z5 Z6 Z8 Z9 X10] +
(-0.0395644163228933+0j) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(-7.518362215721244e-06+0j) [X4 Z5 Z7 Z8 Z9 X10] +
(-0.03931805194719755+0j) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(-5.929765816128054e-07+0j) [X4 X6] +
(-4.253224225592533e-06+0j) [X4 Z6 Z7 Z8 Z9 X10] +
(-0.02252844019601306+0j) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.01198238901024798+0j) [Y4 X5 X6 Y7] +
(0.007306759928832977+0j) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11] +
(2.888293596076952e-07+0j) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.007306759928832977+0j) [Y4 X5 X7 Z8 Z9 Y10] +
(-2.888293596076952e-07+0j) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(0.0071569349198569564+0j) [Y4 X5 X8 Y9] +
(3.6945132944554667e-06+0j) [Y4 X5 X10 Z11 Z12 Y13] +
(-3.6945132944554667e-06+0j) [Y4 X5 X11 Y12] +
(0.038314670294803906+0j) [Y4 X5 X12 Y13] +
(-0.01198238901024798+0j) [Y4 Y5 X6 X7] +
(-0.007306759928832977+0j) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11] +
(-2.888293596076952e-07+0j) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.007306759928832977+0j) [Y4 Y5 Y7 Z8 Z9 Y10] +
(-2.888293596076952e-07+0j) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.0071569349198569564+0j) [Y4 Y5 X8 X9] +
(-3.6945132944554667e-06+0j) [Y4 Y5 X10 Z11 Z12 X13] +
(-3.6945132944554667e-06+0j) [Y4 Y5 Y11 Y12] +
(-0.038314670294803906+0j) [Y4 Y5 X12 X13] +
(0.008890731522694609+0j) [Y4 Z5 X6 X10 Z11 Y12] +
(-1.2260484989589458e-05+0j) [Y4 Z5 Y6] +
(-1.2283337825748155e-06+0j) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(0.00024636437569575104+0j) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2283337825748155e-06+0j) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(0.00024636437569575104+0j) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.854060858053329e-06+0j) [Y4 Z5 Y6 Z7] +
(-1.3980449082091647e-06+0j) [Y4 Z5 Y6 Z8] +
(-1.8818501833174753e-06+0j) [Y4 Z5 Y6 Z9] +
(-0.0009298507967730387+0j) [Y4 Z5 Y6 X10 Z11 X12] +
(0.00796088072592157+0j) [Y4 Z5 Y6 Y10 Z11 Y12] +
(-1.6923978285902755e-06+0j) [Y4 Z5 Y6 Z10] +
(-0.012215040997613995+0j) [Y4 Z5 Y6 X11 Z12 X13] +
(-0.012215040997613995+0j) [Y4 Z5 Y6 Y11 Z12 Y13] +
(4.281913884876456e-06+0j) [Y4 Z5 Y6 Z11] +
(-4.588855155727868e-06+0j) [Y4 Z5 Y6 Z13] +
(4.838052751083105e-07+0j) [Y4 Z5 Z6 X7 X8 Y9] +
(-5.974311713466731e-06+0j) [Y4 Z5 Z6 X7 X10 Y11] +
(-0.011285190200840957+0j) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(0.020175921723535564+0j) [Y4 Z5 Z6 X7 X11 Y12] +
(4.556569218146057e-06+0j) [Y4 Z5 Z6 X7 X12 Y13] +
(-4.838052751083105e-07+0j) [Y4 Z5 Z6 Y7 X8 X9] +
(5.974311713466731e-06+0j) [Y4 Z5 Z6 Y7 X10 X11] +
(0.011285190200840957+0j) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(0.020175921723535564+0j) [Y4 Z5 Z6 Y7 Y11 Y12] +
(-4.556569218146057e-06+0j) [Y4 Z5 Z6 Y7 X12 X13] +
(1.330473188676827e-06+0j) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(0.005923798336561343+0j) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(1.330473188676827e-06+0j) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(0.005923798336561343+0j) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(-6.631277928457139e-05+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.01602460368917958+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(-0.01602460368917958+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(3.3343312894609523e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(-4.734622038677317e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(-9.806102775255643e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(5.071480736578327e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-5.071480736578327e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-0.3693708936615615+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.02314513092952899+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-0.009612634606847284+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(-0.025637238296026866+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(-8.774817864592504e-06+0j) [Y4 Z5 Z6 Z7 Z8 Y10] +
(-0.04764261217638309+0j) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(-7.444344675915676e-06+0j) [Y4 Z5 Z6 Z7 Z9 Y10] +
(-0.04171881383982175+0j) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(-6.290028433146429e-06+0j) [Y4 Z5 Z6 Z8 Z9 Y10] +
(-0.0395644163228933+0j) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(-7.518362215721244e-06+0j) [Y4 Z5 Z7 Z8 Z9 Y10] +
(-0.03931805194719755+0j) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(-5.929765816128054e-07+0j) [Y4 Y6] +
(-4.253224225592533e-06+0j) [Y4 Z6 Z7 Z8 Z9 Y10] +
(-0.02252844019601306+0j) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.2034402289145645+0j) [Z4] +
(-5.929765816128054e-07+0j) [Z4 X5 Z6 X7] +
(-4.253224225592533e-06+0j) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.02252844019601306+0j) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-5.929765816128054e-07+0j) [Z4 Y5 Z6 Y7] +
(-4.253224225592533e-06+0j) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.02252844019601306+0j) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.1575531479798569+0j) [Z4 Z5] +
(0.01826683486937558+0j) [Z4 X6 Z7 Z8 Z9 X10] +
(-1.6541174770109495e-06+0j) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.01826683486937558+0j) [Z4 Y6 Z7 Z8 Z9 Y10] +
(-1.6541174770109495e-06+0j) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.1370119167404076+0j) [Z4 Z6] +
(0.010960074940542604+0j) [Z4 X7 Z8 Z9 Z10 X11] +
(-1.9429468366186446e-06+0j) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.010960074940542604+0j) [Z4 Y7 Z8 Z9 Z10 Y11] +
(-1.9429468366186446e-06+0j) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.1489943057506556+0j) [Z4 Z7] +
(0.14960702684445307+0j) [Z4 Z8] +
(0.15676396176431+0j) [Z4 Z9] +
(1.8782101246069198e-06+0j) [Z4 X10 Z11 X12] +
(1.8782101246069198e-06+0j) [Z4 Y10 Z11 Y12] +
(0.12489990917237616+0j) [Z4 Z10] +
(-1.8163031698485477e-06+0j) [Z4 X11 Z12 X13] +
(-1.8163031698485477e-06+0j) [Z4 Y11 Z12 Y13] +
(0.14257997712485768+0j) [Z4 Z11] +
(0.11383573679388667+0j) [Z4 Z12] +
(0.1521504070886906+0j) [Z4 Z13] +
(1.2283337825748155e-06+0j) [X5 X6 Y7 Z8 Z9 Y10] +
(-0.00024636437569575104+0j) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-4.838052751083105e-07+0j) [X5 X6 X8 X9] +
(5.974311713466732e-06+0j) [X5 X6 X10 X11] +
(0.020175921723535564+0j) [X5 X6 X10 Z11 Z12 X13] +
(0.011285190200840957+0j) [X5 X6 Y11 Y12] +
(-4.556569218146056e-06+0j) [X5 X6 X12 X13] +
(-1.2283337825748155e-06+0j) [X5 Y6 Y7 Z8 Z9 X10] +
(0.00024636437569575104+0j) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-4.838052751083105e-07+0j) [X5 Y6 Y8 X9] +
(5.974311713466732e-06+0j) [X5 Y6 Y10 X11] +
(0.020175921723535564+0j) [X5 Y6 Y10 Z11 Z12 X13] +
(-0.011285190200840957+0j) [X5 Y6 Y11 X12] +
(-4.556569218146056e-06+0j) [X5 Y6 Y12 X13] +
(-1.2260484989589461e-05+0j) [X5 Z6 X7] +
(-1.8818501833174753e-06+0j) [X5 Z6 X7 Z8] +
(-1.3980449082091647e-06+0j) [X5 Z6 X7 Z9] +
(-0.012215040997613995+0j) [X5 Z6 X7 X10 Z11 X12] +
(-0.012215040997613995+0j) [X5 Z6 X7 Y10 Z11 Y12] +
(4.281913884876456e-06+0j) [X5 Z6 X7 Z10] +
(0.00796088072592157+0j) [X5 Z6 X7 X11 Z12 X13] +
(-0.0009298507967730387+0j) [X5 Z6 X7 Y11 Z12 Y13] +
(-1.6923978285902755e-06+0j) [X5 Z6 X7 Z11] +
(-4.588855155727868e-06+0j) [X5 Z6 X7 Z12] +
(0.008890731522694609+0j) [X5 Z6 Y7 Y11 Z12 X13] +
(-1.330473188676827e-06+0j) [X5 Z6 Z7 X8 Y9 Y10] +
(-0.005923798336561343+0j) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(1.330473188676827e-06+0j) [X5 Z6 Z7 Y8 Y9 X10] +
(0.005923798336561343+0j) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(0.01602460368917958+0j) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12] +
(-5.071480736578327e-06+0j) [X5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-0.01602460368917958+0j) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12] +
(-5.071480736578327e-06+0j) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(-6.631277928457136e-05+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-9.806102775255643e-06+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(-4.734622038677317e-06+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(-0.3693708936615616+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.023145130929528992+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(-0.025637238296026866+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(3.334331289460952e-06+0j) [X5 Z6 Z7 Z8 Z9 X11] +
(-0.009612634606847284+0j) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(-7.444344675915676e-06+0j) [X5 Z6 Z7 Z8 Z10 X11] +
(-0.04171881383982175+0j) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(-8.774817864592504e-06+0j) [X5 Z6 Z7 Z9 Z10 X11] +
(-0.04764261217638309+0j) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(-7.518362215721244e-06+0j) [X5 Z6 Z8 Z9 Z10 X11] +
(-0.03931805194719755+0j) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.854060858053329e-06+0j) [X5 X7] +
(-6.290028433146429e-06+0j) [X5 Z7 Z8 Z9 Z10 X11] +
(-0.0395644163228933+0j) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2283337825748155e-06+0j) [Y5 X6 X7 Z8 Z9 Y10] +
(0.00024636437569575104+0j) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-4.838052751083105e-07+0j) [Y5 X6 X8 Y9] +
(5.974311713466732e-06+0j) [Y5 X6 X10 Y11] +
(0.020175921723535564+0j) [Y5 X6 X10 Z11 Z12 Y13] +
(-0.011285190200840957+0j) [Y5 X6 X11 Y12] +
(-4.556569218146056e-06+0j) [Y5 X6 X12 Y13] +
(1.2283337825748155e-06+0j) [Y5 Y6 X7 Z8 Z9 X10] +
(-0.00024636437569575104+0j) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-4.838052751083105e-07+0j) [Y5 Y6 Y8 Y9] +
(5.974311713466732e-06+0j) [Y5 Y6 Y10 Y11] +
(0.020175921723535564+0j) [Y5 Y6 Y10 Z11 Z12 Y13] +
(0.011285190200840957+0j) [Y5 Y6 X11 X12] +
(-4.556569218146056e-06+0j) [Y5 Y6 Y12 Y13] +
(0.008890731522694609+0j) [Y5 Z6 X7 X11 Z12 Y13] +
(-1.2260484989589461e-05+0j) [Y5 Z6 Y7] +
(-1.8818501833174753e-06+0j) [Y5 Z6 Y7 Z8] +
(-1.3980449082091647e-06+0j) [Y5 Z6 Y7 Z9] +
(-0.012215040997613995+0j) [Y5 Z6 Y7 X10 Z11 X12] +
(-0.012215040997613995+0j) [Y5 Z6 Y7 Y10 Z11 Y12] +
(4.281913884876456e-06+0j) [Y5 Z6 Y7 Z10] +
(-0.0009298507967730387+0j) [Y5 Z6 Y7 X11 Z12 X13] +
(0.00796088072592157+0j) [Y5 Z6 Y7 Y11 Z12 Y13] +
(-1.6923978285902755e-06+0j) [Y5 Z6 Y7 Z11] +
(-4.588855155727868e-06+0j) [Y5 Z6 Y7 Z12] +
(1.330473188676827e-06+0j) [Y5 Z6 Z7 X8 X9 Y10] +
(0.005923798336561343+0j) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-1.330473188676827e-06+0j) [Y5 Z6 Z7 Y8 X9 X10] +
(-0.005923798336561343+0j) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-0.01602460368917958+0j) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12] +
(-5.071480736578327e-06+0j) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(0.01602460368917958+0j) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12] +
(-5.071480736578327e-06+0j) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(-6.631277928457136e-05+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-9.806102775255643e-06+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(-4.734622038677317e-06+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(-0.3693708936615616+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.023145130929528992+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(-0.025637238296026866+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(3.334331289460952e-06+0j) [Y5 Z6 Z7 Z8 Z9 Y11] +
(-0.009612634606847284+0j) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(-7.444344675915676e-06+0j) [Y5 Z6 Z7 Z8 Z10 Y11] +
(-0.04171881383982175+0j) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(-8.774817864592504e-06+0j) [Y5 Z6 Z7 Z9 Z10 Y11] +
(-0.04764261217638309+0j) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(-7.518362215721244e-06+0j) [Y5 Z6 Z8 Z9 Z10 Y11] +
(-0.03931805194719755+0j) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.854060858053329e-06+0j) [Y5 Y7] +
(-6.290028433146429e-06+0j) [Y5 Z7 Z8 Z9 Z10 Y11] +
(-0.0395644163228933+0j) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.203440228914565+0j) [Z5] +
(0.010960074940542604+0j) [Z5 X6 Z7 Z8 Z9 X10] +
(-1.9429468366186446e-06+0j) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.010960074940542604+0j) [Z5 Y6 Z7 Z8 Z9 Y10] +
(-1.9429468366186446e-06+0j) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.1489943057506556+0j) [Z5 Z6] +
(0.01826683486937558+0j) [Z5 X7 Z8 Z9 Z10 X11] +
(-1.6541174770109495e-06+0j) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.01826683486937558+0j) [Z5 Y7 Z8 Z9 Z10 Y11] +
(-1.6541174770109495e-06+0j) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.1370119167404076+0j) [Z5 Z7] +
(0.15676396176431+0j) [Z5 Z8] +
(0.14960702684445307+0j) [Z5 Z9] +
(-1.8163031698485477e-06+0j) [Z5 X10 Z11 X12] +
(-1.8163031698485477e-06+0j) [Z5 Y10 Z11 Y12] +
(0.14257997712485768+0j) [Z5 Z10] +
(1.8782101246069198e-06+0j) [Z5 X11 Z12 X13] +
(1.8782101246069198e-06+0j) [Z5 Y11 Z12 Y13] +
(0.12489990917237616+0j) [Z5 Z11] +
(0.1521504070886906+0j) [Z5 Z12] +
(0.11383573679388667+0j) [Z5 Z13] +
(-0.013873381748426077+0j) [X6 X7 Y8 Y9] +
(-0.01782514099578653+0j) [X6 X7 Y10 Y11] +
(-1.0358477602385953e-06+0j) [X6 X7 Y10 Z11 Z12 Y13] +
(-1.0358477602385953e-06+0j) [X6 X7 X11 X12] +
(-0.01736611899465142+0j) [X6 X7 Y12 Y13] +
(0.013873381748426077+0j) [X6 Y7 Y8 X9] +
(0.01782514099578653+0j) [X6 Y7 Y10 X11] +
(1.0358477602385953e-06+0j) [X6 Y7 Y10 Z11 Z12 X13] +
(-1.0358477602385953e-06+0j) [X6 Y7 Y11 X12] +
(0.01736611899465142+0j) [X6 Y7 Y12 X13] +
(0.0002921986261110318+0j) [X6 Z7 X8 X9 Z10 X11] +
(-3.32813935066297e-07+0j) [X6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.0002921986261110318+0j) [X6 Z7 X8 Y9 Z10 Y11] +
(-3.32813935066297e-07+0j) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(0.22848106564918869+0j) [X6 Z7 Z8 Z9 X10] +
(3.3131455001735515e-06+0j) [X6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(3.3131455001735515e-06+0j) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(0.011307274008848118+0j) [X6 Z7 Z8 Z9 X10 Z11] +
(0.025104957138844523+0j) [X6 Z7 Z8 Z9 X10 Z12] +
(0.010540425907671475+0j) [X6 Z7 Z8 Z9 X10 Z13] +
(-0.014564531231173048+0j) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(0.014564531231173048+0j) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-2.595086006874674e-05+0j) [X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.183932559490294e-06+0j) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-6.524373848544512e-06+0j) [X6 Z7 Z8 Z9 Z10 X12] +
(-3.2112283483709604e-06+0j) [X6 Z7 Z8 Z9 Z11 X12] +
(0.029812424517345785+0j) [X6 Z7 Z8 X10] +
(-3.277483195419628e-06+0j) [X6 Z7 Z8 Z10 Z11 X12] +
(0.030104623143456816+0j) [X6 Z7 Z9 X10] +
(-3.6102971304859245e-06+0j) [X6 Z7 Z9 Z10 Z11 X12] +
(0.030787505389143915+0j) [X6 Z8 Z9 X10] +
(-3.769659451912474e-06+0j) [X6 Z8 Z9 Z10 Z11 X12] +
(0.013873381748426077+0j) [Y6 X7 X8 Y9] +
(0.01782514099578653+0j) [Y6 X7 X10 Y11] +
(1.0358477602385953e-06+0j) [Y6 X7 X10 Z11 Z12 Y13] +
(-1.0358477602385953e-06+0j) [Y6 X7 X11 Y12] +
(0.01736611899465142+0j) [Y6 X7 X12 Y13] +
(-0.013873381748426077+0j) [Y6 Y7 X8 X9] +
(-0.01782514099578653+0j) [Y6 Y7 X10 X11] +
(-1.0358477602385953e-06+0j) [Y6 Y7 X10 Z11 Z12 X13] +
(-1.0358477602385953e-06+0j) [Y6 Y7 Y11 Y12] +
(-0.01736611899465142+0j) [Y6 Y7 X12 X13] +
(0.0002921986261110318+0j) [Y6 Z7 Y8 X9 Z10 X11] +
(-3.32813935066297e-07+0j) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.0002921986261110318+0j) [Y6 Z7 Y8 Y9 Z10 Y11] +
(-3.32813935066297e-07+0j) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(0.22848106564918869+0j) [Y6 Z7 Z8 Z9 Y10] +
(3.3131455001735515e-06+0j) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(3.3131455001735515e-06+0j) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(0.011307274008848118+0j) [Y6 Z7 Z8 Z9 Y10 Z11] +
(0.025104957138844523+0j) [Y6 Z7 Z8 Z9 Y10 Z12] +
(0.010540425907671475+0j) [Y6 Z7 Z8 Z9 Y10 Z13] +
(0.014564531231173048+0j) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-0.014564531231173048+0j) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-2.595086006874674e-05+0j) [Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.183932559490294e-06+0j) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-6.524373848544512e-06+0j) [Y6 Z7 Z8 Z9 Z10 Y12] +
(-3.2112283483709604e-06+0j) [Y6 Z7 Z8 Z9 Z11 Y12] +
(0.029812424517345785+0j) [Y6 Z7 Z8 Y10] +
(-3.277483195419628e-06+0j) [Y6 Z7 Z8 Z10 Z11 Y12] +
(0.030104623143456816+0j) [Y6 Z7 Z9 Y10] +
(-3.6102971304859245e-06+0j) [Y6 Z7 Z9 Z10 Z11 Y12] +
(0.030787505389143915+0j) [Y6 Z8 Z9 Y10] +
(-3.769659451912474e-06+0j) [Y6 Z8 Z9 Z10 Z11 Y12] +
(1.30968629886154+0j) [Z6] +
(0.030787505389143915+0j) [Z6 X7 Z8 Z9 Z10 X11] +
(-3.769659451912474e-06+0j) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.030787505389143915+0j) [Z6 Y7 Z8 Z9 Z10 Y11] +
(-3.769659451912474e-06+0j) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.19392534613270182+0j) [Z6 Z7] +
(0.16756653265461255+0j) [Z6 Z8] +
(0.18143991440303864+0j) [Z6 Z9] +
(-1.8551201216753566e-06+0j) [Z6 X10 Z11 X12] +
(-1.8551201216753566e-06+0j) [Z6 Y10 Z11 Y12] +
(0.11952438964682674+0j) [Z6 Z10] +
(-2.890967881913952e-06+0j) [Z6 X11 Z12 X13] +
(-2.890967881913952e-06+0j) [Z6 Y11 Z12 Y13] +
(0.1373495306426133+0j) [Z6 Z11] +
(0.13401715261963695+0j) [Z6 Z12] +
(0.15138327161428838+0j) [Z6 Z13] +
(-0.0002921986261110318+0j) [X7 X8 Y9 Y10] +
(3.32813935066297e-07+0j) [X7 X8 Y9 Z10 Z11 Y12] +
(0.0002921986261110318+0j) [X7 Y8 Y9 X10] +
(-3.32813935066297e-07+0j) [X7 Y8 Y9 Z10 Z11 X12] +
(-3.313145500173552e-06+0j) [X7 Z8 Z9 X10 Y11 Y12] +
(-0.014564531231173048+0j) [X7 Z8 Z9 X10 X12 X13] +
(3.313145500173552e-06+0j) [X7 Z8 Z9 Y10 Y11 X12] +
(-0.014564531231173048+0j) [X7 Z8 Z9 Y10 Y12 X13] +
(0.22848106564918877+0j) [X7 Z8 Z9 Z10 X11] +
(0.010540425907671475+0j) [X7 Z8 Z9 Z10 X11 Z12] +
(0.025104957138844523+0j) [X7 Z8 Z9 Z10 X11 Z13] +
(-2.5950860068746752e-05+0j) [X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.183932559490295e-06+0j) [X7 Z8 Z9 Z10 Z11 X13] +
(-3.2112283483709604e-06+0j) [X7 Z8 Z9 Z10 Z12 X13] +
(0.011307274008848116+0j) [X7 Z8 Z9 X11] +
(-6.524373848544512e-06+0j) [X7 Z8 Z9 Z11 Z12 X13] +
(0.030104623143456816+0j) [X7 Z8 Z10 X11] +
(-3.6102971304859245e-06+0j) [X7 Z8 Z10 Z11 Z12 X13] +
(0.029812424517345785+0j) [X7 Z9 Z10 X11] +
(-3.277483195419628e-06+0j) [X7 Z9 Z10 Z11 Z12 X13] +
(0.0002921986261110318+0j) [Y7 X8 X9 Y10] +
(-3.32813935066297e-07+0j) [Y7 X8 X9 Z10 Z11 Y12] +
(-0.0002921986261110318+0j) [Y7 Y8 X9 X10] +
(3.32813935066297e-07+0j) [Y7 Y8 X9 Z10 Z11 X12] +
(3.313145500173552e-06+0j) [Y7 Z8 Z9 X10 X11 Y12] +
(-0.014564531231173048+0j) [Y7 Z8 Z9 X10 X12 Y13] +
(-3.313145500173552e-06+0j) [Y7 Z8 Z9 Y10 X11 X12] +
(-0.014564531231173048+0j) [Y7 Z8 Z9 Y10 Y12 Y13] +
(0.22848106564918877+0j) [Y7 Z8 Z9 Z10 Y11] +
(0.010540425907671475+0j) [Y7 Z8 Z9 Z10 Y11 Z12] +
(0.025104957138844523+0j) [Y7 Z8 Z9 Z10 Y11 Z13] +
(-2.5950860068746752e-05+0j) [Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(4.183932559490295e-06+0j) [Y7 Z8 Z9 Z10 Z11 Y13] +
(-3.2112283483709604e-06+0j) [Y7 Z8 Z9 Z10 Z12 Y13] +
(0.011307274008848116+0j) [Y7 Z8 Z9 Y11] +
(-6.524373848544512e-06+0j) [Y7 Z8 Z9 Z11 Z12 Y13] +
(0.030104623143456816+0j) [Y7 Z8 Z10 Y11] +
(-3.6102971304859245e-06+0j) [Y7 Z8 Z10 Z11 Z12 Y13] +
(0.029812424517345785+0j) [Y7 Z9 Z10 Y11] +
(-3.277483195419628e-06+0j) [Y7 Z9 Z10 Z11 Z12 Y13] +
(1.3096862988615396+0j) [Z7] +
(0.18143991440303864+0j) [Z7 Z8] +
(0.16756653265461255+0j) [Z7 Z9] +
(-2.890967881913952e-06+0j) [Z7 X10 Z11 X12] +
(-2.890967881913952e-06+0j) [Z7 Y10 Z11 Y12] +
(0.1373495306426133+0j) [Z7 Z10] +
(-1.8551201216753566e-06+0j) [Z7 X11 Z12 X13] +
(-1.8551201216753566e-06+0j) [Z7 Y11 Z12 Y13] +
(0.11952438964682674+0j) [Z7 Z11] +
(0.15138327161428838+0j) [Z7 Z12] +
(0.13401715261963695+0j) [Z7 Z13] +
(-0.009560705729135912+0j) [X8 X9 Y10 Y11] +
(6.628614201452515e-07+0j) [X8 X9 Y10 Z11 Z12 Y13] +
(6.628614201452514e-07+0j) [X8 X9 X11 X12] +
(-0.006087822480561848+0j) [X8 X9 Y12 Y13] +
(0.009560705729135912+0j) [X8 Y9 Y10 X11] +
(-6.628614201452515e-07+0j) [X8 Y9 Y10 Z11 Z12 X13] +
(6.628614201452514e-07+0j) [X8 Y9 Y11 X12] +
(0.006087822480561848+0j) [X8 Y9 Y12 X13] +
(0.009560705729135912+0j) [Y8 X9 X10 Y11] +
(-6.628614201452515e-07+0j) [Y8 X9 X10 Z11 Z12 Y13] +
(6.628614201452514e-07+0j) [Y8 X9 X11 Y12] +
(0.006087822480561848+0j) [Y8 X9 X12 Y13] +
(-0.009560705729135912+0j) [Y8 Y9 X10 X11] +
(6.628614201452515e-07+0j) [Y8 Y9 X10 Z11 Z12 X13] +
(6.628614201452514e-07+0j) [Y8 Y9 Y11 Y12] +
(-0.006087822480561848+0j) [Y8 Y9 X12 X13] +
(1.3693525634718193+0j) [Z8] +
(0.2200397733437608+0j) [Z8 Z9] +
(-1.5973171979678916e-06+0j) [Z8 X10 Z11 X12] +
(-1.5973171979678916e-06+0j) [Z8 Y10 Z11 Y12] +
(0.1376687264585258+0j) [Z8 Z10] +
(-9.344557778226398e-07+0j) [Z8 X11 Z12 X13] +
(-9.344557778226398e-07+0j) [Z8 Y11 Z12 Y13] +
(0.1472294321876617+0j) [Z8 Z11] +
(0.14973486803496922+0j) [Z8 Z12] +
(0.15582269051553105+0j) [Z8 Z13] +
(-9.344557778226398e-07+0j) [Z9 X10 Z11 X12] +
(-9.344557778226398e-07+0j) [Z9 Y10 Z11 Y12] +
(0.1472294321876617+0j) [Z9 Z10] +
(-1.5973171979678916e-06+0j) [Z9 X11 Z12 X13] +
(-1.5973171979678916e-06+0j) [Z9 Y11 Z12 Y13] +
(0.1376687264585258+0j) [Z9 Z11] +
(0.15582269051553105+0j) [Z9 Z12] +
(0.14973486803496922+0j) [Z9 Z13] +
(-0.028685183716105962+0j) [X10 X11 Y12 Y13] +
(0.028685183716105962+0j) [X10 Y11 Y12 X13] +
(-1.0722312158715286e-05+0j) [X10 Z11 X12] +
(7.954413176223767e-06+0j) [X10 Z11 X12 Z13] +
(-8.19426137235684e-06+0j) [X10 X12] +
(0.028685183716105962+0j) [Y10 X11 X12 Y13] +
(-0.028685183716105962+0j) [Y10 Y11 X12 X13] +
(-1.0722312158715286e-05+0j) [Y10 Z11 Y12] +
(7.954413176223767e-06+0j) [Y10 Z11 Y12 Z13] +
(-8.19426137235684e-06+0j) [Y10 Y12] +
(0.7829661725950193+0j) [Z10] +
(-8.19426137235684e-06+0j) [Z10 X11 Z12 X13] +
(-8.19426137235684e-06+0j) [Z10 Y11 Z12 Y13] +
(0.1492635514738891+0j) [Z10 Z11] +
(0.1127038692033222+0j) [Z10 Z12] +
(0.14138905291942816+0j) [Z10 Z13] +
(-1.0722312158715282e-05+0j) [X11 Z12 X13] +
(7.954413176223767e-06+0j) [X11 X13] +
(-1.0722312158715282e-05+0j) [Y11 Z12 Y13] +
(7.954413176223767e-06+0j) [Y11 Y13] +
(0.7829661725950192+0j) [Z11] +
(0.14138905291942816+0j) [Z11 Z12] +
(0.1127038692033222+0j) [Z11 Z13] +
(0.8084581961720478+0j) [Z12] +
(0.1543574865722363+0j) [Z12 Z13] +
(0.808458196172048+0j) [Z13]
  (-46.463906788688945) [I0]
+ (0.7829661725950198) [Z10]
+ (0.7829661725950199) [Z11]
+ (0.8084581961720492) [Z12]
+ (0.8084581961720493) [Z13]
+ (1.203440228914564) [Z4]
+ (1.203440228914564) [Z5]
+ (1.3096862988615452) [Z6]
+ (1.3096862988615454) [Z7]
+ (1.3693525634718184) [Z8]
+ (1.3693525634718189) [Z9]
+ (1.6538942226831712) [Z2]
+ (1.6538942226831714) [Z3]
+ (12.412630742111759) [Z0]
+ (12.412630742111759) [Z1]
+ (-8.194261372076604e-06) [Y10 Y12]
+ (-8.194261372076604e-06) [X10 X12]
+ (-1.854060858037578e-06) [Y5 Y7]
+ (-1.854060858037578e-06) [X5 X7]
+ (-7.764994119392702e-07) [Y3 Y5]
+ (-7.764994119392702e-07) [X3 X5]
+ (-5.929765816219532e-07) [Y4 Y6]
+ (-5.929765816219532e-07) [X4 X6]
+ (1.602116740616277e-06) [Y2 Y4]
+ (1.602116740616277e-06) [X2 X4]
+ (7.954413175916487e-06) [Y11 Y13]
+ (7.954413175916487e-06) [X11 X13]
+ (0.0032769719312315208) [Y1 Y3]
+ (0.0032769719312315208) [X1 X3]
+ (0.10433064780651358) [Y0 Y2]
+ (0.10433064780651358) [X0 X2]
+ (0.11270386920332227) [Z10 Z12]
+ (0.11270386920332227) [Z11 Z13]
+ (0.11383573679388652) [Z4 Z12]
+ (0.11383573679388652) [Z5 Z13]
+ (0.11952438964682686) [Z6 Z10]
+ (0.11952438964682686) [Z7 Z11]
+ (0.12489990917237599) [Z4 Z10]
+ (0.12489990917237599) [Z5 Z11]
+ (0.12495807739503192) [Z2 Z4]
+ (0.12495807739503192) [Z3 Z5]
+ (0.12799502492468404) [Z2 Z10]
+ (0.12799502492468404) [Z3 Z11]
+ (0.1340171526196371) [Z6 Z12]
+ (0.1340171526196371) [Z7 Z13]
+ (0.13701191674040744) [Z4 Z6]
+ (0.13701191674040744) [Z5 Z7]
+ (0.1373495306426132) [Z6 Z11]
+ (0.1373495306426132) [Z7 Z10]
+ (0.13739104762683207) [Z2 Z6]
+ (0.13739104762683207) [Z3 Z7]
+ (0.13766872645852588) [Z8 Z10]
+ (0.13766872645852588) [Z9 Z11]
+ (0.140112898653548) [Z2 Z12]
+ (0.140112898653548) [Z3 Z13]
+ (0.1413890529194282) [Z10 Z13]
+ (0.1413890529194282) [Z11 Z12]
+ (0.14257997712485754) [Z4 Z11]
+ (0.14257997712485754) [Z5 Z10]
+ (0.14722943218766182) [Z8 Z11]
+ (0.14722943218766182) [Z9 Z10]
+ (0.14899430575065534) [Z4 Z7]
+ (0.14899430575065534) [Z5 Z6]
+ (0.14926355147388917) [Z10 Z11]
+ (0.14960702684445287) [Z4 Z8]
+ (0.14960702684445287) [Z5 Z9]
+ (0.14973486803496933) [Z8 Z12]
+ (0.14973486803496933) [Z9 Z13]
+ (0.15071408121008273) [Z2 Z8]
+ (0.15071408121008273) [Z3 Z9]
+ (0.1513832716142885) [Z6 Z13]
+ (0.1513832716142885) [Z7 Z12]
+ (0.15215040708869038) [Z4 Z13]
+ (0.15215040708869038) [Z5 Z12]
+ (0.15337968243314154) [Z2 Z11]
+ (0.15337968243314154) [Z3 Z10]
+ (0.1543574865722364) [Z12 Z13]
+ (0.15569010671752448) [Z2 Z13]
+ (0.15569010671752448) [Z3 Z12]
+ (0.15582269051553121) [Z8 Z13]
+ (0.15582269051553121) [Z9 Z12]
+ (0.1567639617643098) [Z4 Z9]
+ (0.1567639617643098) [Z5 Z8]
+ (0.15755314797985648) [Z4 Z5]
+ (0.1607976453483854) [Z2 Z5]
+ (0.1607976453483854) [Z3 Z4]
+ (0.16756653265461274) [Z6 Z8]
+ (0.16756653265461274) [Z7 Z9]
+ (0.1685348656157991) [Z2 Z7]
+ (0.1685348656157991) [Z3 Z6]
+ (0.18143991440303886) [Z6 Z9]
+ (0.18143991440303886) [Z7 Z8]
+ (0.18189085790751328) [Z2 Z3]
+ (0.18690820476912526) [Z2 Z9]
+ (0.18690820476912526) [Z3 Z8]
+ (0.19299723935364244) [Z0 Z10]
+ (0.19299723935364244) [Z1 Z11]
+ (0.1939253461327022) [Z6 Z7]
+ (0.19661770890342117) [Z0 Z4]
+ (0.19661770890342117) [Z1 Z5]
+ (0.19936354537360795) [Z0 Z5]
+ (0.19936354537360795) [Z1 Z4]
+ (0.20072866460441777) [Z0 Z11]
+ (0.20072866460441777) [Z1 Z10]
+ (0.21102659849791516) [Z0 Z12]
+ (0.21102659849791516) [Z1 Z13]
+ (0.2163103749863181) [Z0 Z13]
+ (0.2163103749863181) [Z1 Z12]
+ (0.23671080783830373) [Z0 Z2]
+ (0.23671080783830373) [Z1 Z3]
+ (0.24164663936017203) [Z0 Z6]
+ (0.24164663936017203) [Z1 Z7]
+ (0.2485348337131426) [Z0 Z7]
+ (0.2485348337131426) [Z1 Z6]
+ (0.25129445674591633) [Z0 Z3]
+ (0.25129445674591633) [Z1 Z2]
+ (0.27232518306605663) [Z0 Z8]
+ (0.27232518306605663) [Z1 Z9]
+ (0.27883454426723386) [Z0 Z9]
+ (0.27883454426723386) [Z1 Z8]
+ (1.1861763734860484) [Z0 Z1]
+ (-1.2260484989339883e-05) [Y4 Z5 Y6]
+ (-1.2260484989339883e-05) [X4 Z5 X6]
+ (-1.2260484989339878e-05) [Y5 Z6 Y7]
+ (-1.2260484989339878e-05) [X5 Z6 X7]
+ (-1.0722312157960718e-05) [Y10 Z11 Y12]
+ (-1.0722312157960718e-05) [X10 Z11 X12]
+ (-1.0722312157960711e-05) [Y11 Z12 Y13]
+ (-1.0722312157960711e-05) [X11 Z12 X13]
+ (-3.8870516742878195e-06) [Y2 Z3 Y4]
+ (-3.8870516742878195e-06) [X2 Z3 X4]
+ (-3.887051674287818e-06) [Y3 Z4 Y5]
+ (-3.887051674287818e-06) [X3 Z4 X5]
+ (0.1250703257977179) [Y0 Z1 Y2]
+ (0.1250703257977179) [X0 Z1 X2]
+ (0.12507032579771793) [Y1 Z2 Y3]
+ (0.12507032579771793) [X1 Z2 X3]
+ (-0.03831467029480387) [Y4 Y5 X12 X13]
+ (-0.03831467029480387) [X4 X5 Y12 Y13]
+ (-0.03619412355904253) [Y2 Y3 X8 X9]
+ (-0.03619412355904253) [X2 X3 Y8 Y9]
+ (-0.03583956795335346) [Y2 Y3 X4 X5]
+ (-0.03583956795335346) [X2 X3 Y4 Y5]
+ (-0.031143817988967037) [Y2 Y3 X6 X7]
+ (-0.031143817988967037) [X2 X3 Y6 Y7]
+ (-0.02868518371610592) [Y10 Y11 X12 X13]
+ (-0.02868518371610592) [X10 X11 Y12 Y13]
+ (-0.02599617759802131) [Y3 Z4 Z5 Y7]
+ (-0.02599617759802131) [X3 Z4 Z5 X7]
+ (-0.025384657508457482) [Y2 Y3 X10 X11]
+ (-0.025384657508457482) [X2 X3 Y10 Y11]
+ (-0.019028242443847362) [Y3 Y4 X11 X12]
+ (-0.019028242443847362) [X3 X4 Y11 Y12]
+ (-0.017825140995786342) [Y6 Y7 X10 X11]
+ (-0.017825140995786342) [X6 X7 Y10 Y11]
+ (-0.01768006795248153) [Y4 Y5 X10 X11]
+ (-0.01768006795248153) [X4 X5 Y10 Y11]
+ (-0.017366118994651375) [Y6 Y7 X12 X13]
+ (-0.017366118994651375) [X6 X7 Y12 Y13]
+ (-0.014583648907612587) [Y0 Y1 X2 X3]
+ (-0.014583648907612587) [X0 X1 Y2 Y3]
+ (-0.013873381748426145) [Y6 Y7 X8 X9]
+ (-0.013873381748426145) [X6 X7 Y8 Y9]
+ (-0.011982389010247911) [Y4 Y5 X6 X7]
+ (-0.011982389010247911) [X4 X5 Y6 Y7]
+ (-0.011285190200840851) [Y5 X6 X11 Y12]
+ (-0.011285190200840851) [X5 Y6 Y11 X12]
+ (-0.009560705729135989) [Y8 Y9 X10 X11]
+ (-0.009560705729135989) [X8 X9 Y10 Y11]
+ (-0.008125251921381034) [Y1 X2 X8 Y9]
+ (-0.008125251921381034) [Y1 Y2 Y8 Y9]
+ (-0.008125251921381034) [X1 X2 X8 X9]
+ (-0.008125251921381034) [X1 Y2 Y8 X9]
+ (-0.007731425250775318) [Y0 Y1 X10 X11]
+ (-0.007731425250775318) [X0 X1 Y10 Y11]
+ (-0.007156934919856936) [Y4 Y5 X8 X9]
+ (-0.007156934919856936) [X4 X5 Y8 Y9]
+ (-0.00688819435297059) [Y0 Y1 X6 X7]
+ (-0.00688819435297059) [X0 X1 Y6 Y7]
+ (-0.0065093612011772415) [Y0 Y1 X8 X9]
+ (-0.0065093612011772415) [X0 X1 Y8 Y9]
+ (-0.005283776488402968) [Y0 Y1 X12 X13]
+ (-0.005283776488402968) [X0 X1 Y12 Y13]
+ (-0.005143391768825084) [Y3 X4 X5 Y6]
+ (-0.005143391768825084) [X3 Y4 Y5 X6]
+ (-0.004684903388155204) [Y1 X2 X6 Y7]
+ (-0.004684903388155204) [Y1 Y2 Y6 Y7]
+ (-0.004684903388155204) [X1 X2 X6 X7]
+ (-0.004684903388155204) [X1 Y2 Y6 X7]
+ (-0.004575007626639207) [Y1 X2 X12 Y13]
+ (-0.004575007626639207) [Y1 Y2 Y12 Y13]
+ (-0.004575007626639207) [X1 X2 X12 X13]
+ (-0.004575007626639207) [X1 Y2 Y12 X13]
+ (-0.004424855449441851) [Y1 X2 X4 Y5]
+ (-0.004424855449441851) [Y1 Y2 Y4 Y5]
+ (-0.004424855449441851) [X1 X2 X4 X5]
+ (-0.004424855449441851) [X1 Y2 Y4 X5]
+ (-0.003479511890334316) [Y2 Z3 Z5 Y6]
+ (-0.003479511890334316) [X2 Z3 Z5 X6]
+ (-0.003479511890334316) [Y3 Z4 Z6 Y7]
+ (-0.003479511890334316) [X3 Z4 Z6 X7]
+ (-0.002745836470186805) [Y0 Y1 X4 X5]
+ (-0.002745836470186805) [X0 X1 Y4 Y5]
+ (-0.0017992194936630181) [Y1 X2 X10 Y11]
+ (-0.0017992194936630181) [Y1 Y2 Y10 Y11]
+ (-0.0017992194936630181) [X1 X2 X10 X11]
+ (-0.0017992194936630181) [X1 Y2 Y10 X11]
+ (-0.0002921986261111044) [Y7 Y8 X9 X10]
+ (-0.0002921986261111044) [X7 X8 Y9 Y10]
+ (-8.194261372076604e-06) [Z10 Y11 Z12 Y13]
+ (-8.194261372076604e-06) [Z10 X11 Z12 X13]
+ (-7.80170750027676e-06) [Y2 Z3 Y4 Z11]
+ (-7.80170750027676e-06) [X2 Z3 X4 Z11]
+ (-7.80170750027676e-06) [Y3 Z4 Y5 Z10]
+ (-7.80170750027676e-06) [X3 Z4 X5 Z10]
+ (-4.643051068307e-06) [Y3 X4 X10 Y11]
+ (-4.643051068307e-06) [Y3 Y4 Y10 Y11]
+ (-4.643051068307e-06) [X3 X4 X10 X11]
+ (-4.643051068307e-06) [X3 Y4 Y10 X11]
+ (-4.5888551555265515e-06) [Y4 Z5 Y6 Z13]
+ (-4.5888551555265515e-06) [X4 Z5 X6 Z13]
+ (-4.5888551555265515e-06) [Y5 Z6 Y7 Z12]
+ (-4.5888551555265515e-06) [X5 Z6 X7 Z12]
+ (-4.5565692178950785e-06) [Y5 X6 X12 Y13]
+ (-4.5565692178950785e-06) [Y5 Y6 Y12 Y13]
+ (-4.5565692178950785e-06) [X5 X6 X12 X13]
+ (-4.5565692178950785e-06) [X5 Y6 Y12 X13]
+ (-3.6945132943170153e-06) [Y4 X5 X11 Y12]
+ (-3.6945132943170153e-06) [Y4 Y5 Y11 Y12]
+ (-3.6945132943170153e-06) [X4 X5 X11 X12]
+ (-3.6945132943170153e-06) [X4 Y5 Y11 X12]
+ (-3.3440815566277394e-06) [Z0 Y5 Z6 Y7]
+ (-3.3440815566277394e-06) [Z0 X5 Z6 X7]
+ (-3.3440815566277394e-06) [Z1 Y4 Z5 Y6]
+ (-3.3440815566277394e-06) [Z1 X4 Z5 X6]
+ (-3.158656431969761e-06) [Y2 Z3 Y4 Z10]
+ (-3.158656431969761e-06) [X2 Z3 X4 Z10]
+ (-3.158656431969761e-06) [Y3 Z4 Y5 Z11]
+ (-3.158656431969761e-06) [X3 Z4 X5 Z11]
+ (-3.099349243733105e-06) [Z0 Y4 Z5 Y6]
+ (-3.099349243733105e-06) [Z0 X4 Z5 X6]
+ (-3.099349243733105e-06) [Z1 Y5 Z6 Y7]
+ (-3.099349243733105e-06) [Z1 X5 Z6 X7]
+ (-2.890967881745713e-06) [Z6 Y11 Z12 Y13]
+ (-2.890967881745713e-06) [Z6 X11 Z12 X13]
+ (-2.890967881745713e-06) [Z7 Y10 Z11 Y12]
+ (-2.890967881745713e-06) [Z7 X10 Z11 X12]
+ (-2.1776646050203675e-06) [Z0 Y10 Z11 Y12]
+ (-2.1776646050203675e-06) [Z0 X10 Z11 X12]
+ (-2.1776646050203675e-06) [Z1 Y11 Z12 Y13]
+ (-2.1776646050203675e-06) [Z1 X11 Z12 X13]
+ (-1.881850183289113e-06) [Y4 Z5 Y6 Z9]
+ (-1.881850183289113e-06) [X4 Z5 X6 Z9]
+ (-1.881850183289113e-06) [Y5 Z6 Y7 Z8]
+ (-1.881850183289113e-06) [X5 Z6 X7 Z8]
+ (-1.8551201214938497e-06) [Z6 Y10 Z11 Y12]
+ (-1.8551201214938497e-06) [Z6 X10 Z11 X12]
+ (-1.8551201214938497e-06) [Z7 Y11 Z12 Y13]
+ (-1.8551201214938497e-06) [Z7 X11 Z12 X13]
+ (-1.854060858037578e-06) [Y4 Z5 Y6 Z7]
+ (-1.854060858037578e-06) [X4 Z5 X6 Z7]
+ (-1.8163031697415585e-06) [Z4 Y11 Z12 Y13]
+ (-1.8163031697415585e-06) [Z4 X11 Z12 X13]
+ (-1.8163031697415585e-06) [Z5 Y10 Z11 Y12]
+ (-1.8163031697415585e-06) [Z5 X10 Z11 X12]
+ (-1.6923978285256602e-06) [Y4 Z5 Y6 Z10]
+ (-1.6923978285256602e-06) [X4 Z5 X6 Z10]
+ (-1.6923978285256602e-06) [Y5 Z6 Y7 Z11]
+ (-1.6923978285256602e-06) [X5 Z6 X7 Z11]
+ (-1.614879413898796e-06) [Z0 Y11 Z12 Y13]
+ (-1.614879413898796e-06) [Z0 X11 Z12 X13]
+ (-1.614879413898796e-06) [Z1 Y10 Z11 Y12]
+ (-1.614879413898796e-06) [Z1 X10 Z11 X12]
+ (-1.5973171977957211e-06) [Z8 Y10 Z11 Y12]
+ (-1.5973171977957211e-06) [Z8 X10 Z11 X12]
+ (-1.5973171977957211e-06) [Z9 Y11 Z12 Y13]
+ (-1.5973171977957211e-06) [Z9 X11 Z12 X13]
+ (-1.4548424491623226e-06) [Y3 X4 X6 Y7]
+ (-1.4548424491623226e-06) [Y3 Y4 Y6 Y7]
+ (-1.4548424491623226e-06) [X3 X4 X6 X7]
+ (-1.4548424491623226e-06) [X3 Y4 Y6 X7]
+ (-1.3980449081694052e-06) [Y4 Z5 Y6 Z8]
+ (-1.3980449081694052e-06) [X4 Z5 X6 Z8]
+ (-1.3980449081694052e-06) [Y5 Z6 Y7 Z9]
+ (-1.3980449081694052e-06) [X5 Z6 X7 Z9]
+ (-1.195489010113525e-06) [Y2 Z3 Y4 Z7]
+ (-1.195489010113525e-06) [X2 Z3 X4 Z7]
+ (-1.195489010113525e-06) [Y3 Z4 Y5 Z6]
+ (-1.195489010113525e-06) [X3 Z4 X5 Z6]
+ (-1.1908508085790194e-06) [Z0 Y3 Z4 Y5]
+ (-1.1908508085790194e-06) [Z0 X3 Z4 X5]
+ (-1.1908508085790194e-06) [Z1 Y2 Z3 Y4]
+ (-1.1908508085790194e-06) [Z1 X2 Z3 X4]
+ (-1.1708301370576463e-06) [Z2 Y5 Z6 Y7]
+ (-1.1708301370576463e-06) [Z2 X5 Z6 X7]
+ (-1.1708301370576463e-06) [Z3 Y4 Z5 Y6]
+ (-1.1708301370576463e-06) [Z3 X4 Z5 X6]
+ (-1.0632283423505742e-06) [Z2 Y10 Z11 Y12]
+ (-1.0632283423505742e-06) [Z2 X10 Z11 X12]
+ (-1.0632283423505742e-06) [Z3 Y11 Z12 Y13]
+ (-1.0632283423505742e-06) [Z3 X11 Z12 X13]
+ (-1.0358477602518635e-06) [Y6 X7 X11 Y12]
+ (-1.0358477602518635e-06) [Y6 Y7 Y11 Y12]
+ (-1.0358477602518635e-06) [X6 X7 X11 X12]
+ (-1.0358477602518635e-06) [X6 Y7 Y11 X12]
+ (-9.509249752010806e-07) [Z2 Y4 Z5 Y6]
+ (-9.509249752010806e-07) [Z2 X4 Z5 X6]
+ (-9.509249752010806e-07) [Z3 Y5 Z6 Y7]
+ (-9.509249752010806e-07) [Z3 X5 Z6 X7]
+ (-9.344557776442589e-07) [Z8 Y11 Z12 Y13]
+ (-9.344557776442589e-07) [Z8 X11 Z12 X13]
+ (-9.344557776442589e-07) [Z9 Y10 Z11 Y12]
+ (-9.344557776442589e-07) [Z9 X10 Z11 X12]
+ (-8.337746756408159e-07) [Z0 Y2 Z3 Y4]
+ (-8.337746756408159e-07) [Z0 X2 Z3 X4]
+ (-8.337746756408159e-07) [Z1 Y3 Z4 Y5]
+ (-8.337746756408159e-07) [Z1 X3 Z4 X5]
+ (-7.956895373229693e-07) [Y3 X4 X8 Y9]
+ (-7.956895373229693e-07) [Y3 Y4 Y8 Y9]
+ (-7.956895373229693e-07) [X3 X4 X8 X9]
+ (-7.956895373229693e-07) [X3 Y4 Y8 X9]
+ (-7.764994119392702e-07) [Y2 Z3 Y4 Z5]
+ (-7.764994119392702e-07) [X2 Z3 X4 Z5]
+ (-5.929765816219533e-07) [Z4 Y5 Z6 Y7]
+ (-5.929765816219533e-07) [Z4 X5 Z6 X7]
+ (-5.770052996360595e-07) [Y2 Z3 Y4 Z9]
+ (-5.770052996360595e-07) [X2 Z3 X4 Z9]
+ (-5.770052996360595e-07) [Y3 Z4 Y5 Z8]
+ (-5.770052996360595e-07) [X3 Z4 X5 Z8]
+ (-5.471647744491171e-07) [Y1 Y2 X11 X12]
+ (-5.471647744491171e-07) [X1 X2 Y11 Y12]
+ (-4.838052751197076e-07) [Y5 X6 X8 Y9]
+ (-4.838052751197076e-07) [Y5 Y6 Y8 Y9]
+ (-4.838052751197076e-07) [X5 X6 X8 X9]
+ (-4.838052751197076e-07) [X5 Y6 Y8 X9]
+ (-3.5707613293820367e-07) [Y0 X1 X3 Y4]
+ (-3.5707613293820367e-07) [Y0 Y1 Y3 Y4]
+ (-3.5707613293820367e-07) [X0 X1 X3 X4]
+ (-3.5707613293820367e-07) [X0 Y1 Y3 X4]
+ (-2.4473231289463473e-07) [Y0 X1 X5 Y6]
+ (-2.4473231289463473e-07) [Y0 Y1 Y5 Y6]
+ (-2.4473231289463473e-07) [X0 X1 X5 X6]
+ (-2.4473231289463473e-07) [X0 Y1 Y5 X6]
+ (-2.1990516185656584e-07) [Y2 X3 X5 Y6]
+ (-2.1990516185656584e-07) [Y2 Y3 Y5 Y6]
+ (-2.1990516185656584e-07) [X2 X3 X5 X6]
+ (-2.1990516185656584e-07) [X2 Y3 Y5 X6]
+ (-1.9332412772300546e-07) [Y1 X2 X3 Y4]
+ (-1.9332412772300546e-07) [X1 Y2 Y3 X4]
+ (-1.2919694863497273e-07) [Y1 Z2 Z3 Y5]
+ (-1.2919694863497273e-07) [X1 Z2 Z3 X5]
+ (1.7379332624691197e-07) [Y0 Z1 Z3 Y4]
+ (1.7379332624691197e-07) [X0 Z1 Z3 X4]
+ (1.7379332624691197e-07) [Y1 Z2 Z4 Y5]
+ (1.7379332624691197e-07) [X1 Z2 Z4 X5]
+ (1.9332412772300546e-07) [Y1 Y2 X3 X4]
+ (1.9332412772300546e-07) [X1 X2 Y3 Y4]
+ (2.1868423768690985e-07) [Y2 Z3 Y4 Z8]
+ (2.1868423768690985e-07) [X2 Z3 X4 Z8]
+ (2.1868423768690985e-07) [Y3 Z4 Y5 Z9]
+ (2.1868423768690985e-07) [X3 Z4 X5 Z9]
+ (2.5935343904879755e-07) [Y2 Z3 Y4 Z6]
+ (2.5935343904879755e-07) [X2 Z3 X4 Z6]
+ (2.5935343904879755e-07) [Y3 Z4 Y5 Z7]
+ (2.5935343904879755e-07) [X3 Z4 X5 Z7]
+ (3.6060718681121934e-07) [Y0 Z1 Z2 Y4]
+ (3.6060718681121934e-07) [X0 Z1 Z2 X4]
+ (3.6060718681121934e-07) [Y1 Z3 Z4 Y5]
+ (3.6060718681121934e-07) [X1 Z3 Z4 X5]
+ (5.471647744491171e-07) [Y1 X2 X11 Y12]
+ (5.471647744491171e-07) [X1 Y2 Y11 X12]
+ (5.627851911215719e-07) [Y0 X1 X11 Y12]
+ (5.627851911215719e-07) [Y0 Y1 Y11 Y12]
+ (5.627851911215719e-07) [X0 X1 X11 X12]
+ (5.627851911215719e-07) [X0 Y1 Y11 X12]
+ (6.62861420151462e-07) [Y8 X9 X11 Y12]
+ (6.62861420151462e-07) [Y8 Y9 Y11 Y12]
+ (6.62861420151462e-07) [X8 X9 X11 X12]
+ (6.62861420151462e-07) [X8 Y9 Y11 X12]
+ (1.109440759116429e-06) [Z2 Y11 Z12 Y13]
+ (1.109440759116429e-06) [Z2 X11 Z12 X13]
+ (1.109440759116429e-06) [Z3 Y10 Z11 Y12]
+ (1.109440759116429e-06) [Z3 X10 Z11 X12]
+ (1.602116740616277e-06) [Z2 Y3 Z4 Y5]
+ (1.602116740616277e-06) [Z2 X3 Z4 X5]
+ (1.8782101245754564e-06) [Z4 Y10 Z11 Y12]
+ (1.8782101245754564e-06) [Z4 X10 Z11 X12]
+ (1.8782101245754564e-06) [Z5 Y11 Z12 Y13]
+ (1.8782101245754564e-06) [Z5 X11 Z12 X13]
+ (2.172669101467003e-06) [Y2 X3 X11 Y12]
+ (2.172669101467003e-06) [Y2 Y3 Y11 Y12]
+ (2.172669101467003e-06) [X2 X3 X11 X12]
+ (2.172669101467003e-06) [X2 Y3 Y11 X12]
+ (3.1174479463051e-06) [Y0 Z2 Z3 Y4]
+ (3.1174479463051e-06) [X0 Z2 Z3 X4]
+ (3.5390541843076334e-06) [Y2 Z3 Y4 Z12]
+ (3.5390541843076334e-06) [X2 Z3 X4 Z12]
+ (3.5390541843076334e-06) [Y3 Z4 Y5 Z13]
+ (3.5390541843076334e-06) [X3 Z4 X5 Z13]
+ (4.281913884759498e-06) [Y4 Z5 Y6 Z11]
+ (4.281913884759498e-06) [X4 Z5 X6 Z11]
+ (4.281913884759498e-06) [Y5 Z6 Y7 Z10]
+ (4.281913884759498e-06) [X5 Z6 X7 Z10]
+ (5.275883121999843e-06) [Y3 X4 X12 Y13]
+ (5.275883121999843e-06) [Y3 Y4 Y12 Y13]
+ (5.275883121999843e-06) [X3 X4 X12 X13]
+ (5.275883121999843e-06) [X3 Y4 Y12 X13]
+ (5.974311713285158e-06) [Y5 X6 X10 Y11]
+ (5.974311713285158e-06) [Y5 Y6 Y10 Y11]
+ (5.974311713285158e-06) [X5 X6 X10 X11]
+ (5.974311713285158e-06) [X5 Y6 Y10 X11]
+ (7.954413175916487e-06) [Y10 Z11 Y12 Z13]
+ (7.954413175916487e-06) [X10 Z11 X12 Z13]
+ (8.814937306307477e-06) [Y2 Z3 Y4 Z13]
+ (8.814937306307477e-06) [X2 Z3 X4 Z13]
+ (8.814937306307477e-06) [Y3 Z4 Y5 Z12]
+ (8.814937306307477e-06) [X3 Z4 X5 Z12]
+ (0.0002921986261111044) [Y7 X8 X9 Y10]
+ (0.0002921986261111044) [X7 Y8 Y9 X10]
+ (0.0004956762314916902) [Y2 Z4 Z5 Y6]
+ (0.0004956762314916902) [X2 Z4 Z5 X6]
+ (0.0011059037691895912) [Y0 Z1 Y2 Z5]
+ (0.0011059037691895912) [X0 Z1 X2 Z5]
+ (0.0011059037691895912) [Y1 Z2 Y3 Z4]
+ (0.0011059037691895912) [X1 Z2 X3 Z4]
+ (0.0016638798784907676) [Y2 Z3 Z4 Y6]
+ (0.0016638798784907676) [X2 Z3 Z4 X6]
+ (0.0016638798784907676) [Y3 Z5 Z6 Y7]
+ (0.0016638798784907676) [X3 Z5 Z6 X7]
+ (0.0017560707018411518) [Y0 Z1 Y2 Z11]
+ (0.0017560707018411518) [X0 Z1 X2 Z11]
+ (0.0017560707018411518) [Y1 Z2 Y3 Z10]
+ (0.0017560707018411518) [X1 Z2 X3 Z10]
+ (0.002326230623157997) [Y0 Z1 Y2 Z13]
+ (0.002326230623157997) [X0 Z1 X2 Z13]
+ (0.002326230623157997) [Y1 Z2 Y3 Z12]
+ (0.002326230623157997) [X1 Z2 X3 Z12]
+ (0.002745836470186805) [Y0 X1 X4 Y5]
+ (0.002745836470186805) [X0 Y1 Y4 X5]
+ (0.002929768674750931) [Y0 Z1 Y2 Z9]
+ (0.002929768674750931) [X0 Z1 X2 Z9]
+ (0.002929768674750931) [Y1 Z2 Y3 Z8]
+ (0.002929768674750931) [X1 Z2 X3 Z8]
+ (0.0032769719312315208) [Y0 Z1 Y2 Z3]
+ (0.0032769719312315208) [X0 Z1 X2 Z3]
+ (0.0033476175306660937) [Y0 Z1 Y2 Z7]
+ (0.0033476175306660937) [X0 Z1 X2 Z7]
+ (0.0033476175306660937) [Y1 Z2 Y3 Z6]
+ (0.0033476175306660937) [X1 Z2 X3 Z6]
+ (0.0035552901955041697) [Y0 Z1 Y2 Z10]
+ (0.0035552901955041697) [X0 Z1 X2 Z10]
+ (0.0035552901955041697) [Y1 Z2 Y3 Z11]
+ (0.0035552901955041697) [X1 Z2 X3 Z11]
+ (0.005143391768825084) [Y3 Y4 X5 X6]
+ (0.005143391768825084) [X3 X4 Y5 Y6]
+ (0.005283776488402968) [Y0 X1 X12 Y13]
+ (0.005283776488402968) [X0 Y1 Y12 X13]
+ (0.0055307592186314425) [Y0 Z1 Y2 Z4]
+ (0.0055307592186314425) [X0 Z1 X2 Z4]
+ (0.0055307592186314425) [Y1 Z2 Y3 Z5]
+ (0.0055307592186314425) [X1 Z2 X3 Z5]
+ (0.0065093612011772415) [Y0 X1 X8 Y9]
+ (0.0065093612011772415) [X0 Y1 Y8 X9]
+ (0.00688819435297059) [Y0 X1 X6 Y7]
+ (0.00688819435297059) [X0 Y1 Y6 X7]
+ (0.006901238249797204) [Y0 Z1 Y2 Z12]
+ (0.006901238249797204) [X0 Z1 X2 Z12]
+ (0.006901238249797204) [Y1 Z2 Y3 Z13]
+ (0.006901238249797204) [X1 Z2 X3 Z13]
+ (0.007156934919856936) [Y4 X5 X8 Y9]
+ (0.007156934919856936) [X4 Y5 Y8 X9]
+ (0.007731425250775318) [Y0 X1 X10 Y11]
+ (0.007731425250775318) [X0 Y1 Y10 X11]
+ (0.0080325209188213) [Y0 Z1 Y2 Z6]
+ (0.0080325209188213) [X0 Z1 X2 Z6]
+ (0.0080325209188213) [Y1 Z2 Y3 Z7]
+ (0.0080325209188213) [X1 Z2 X3 Z7]
+ (0.009560705729135989) [Y8 X9 X10 Y11]
+ (0.009560705729135989) [X8 Y9 Y10 X11]
+ (0.011055020596131964) [Y0 Z1 Y2 Z8]
+ (0.011055020596131964) [X0 Z1 X2 Z8]
+ (0.011055020596131964) [Y1 Z2 Y3 Z9]
+ (0.011055020596131964) [X1 Z2 X3 Z9]
+ (0.011285190200840851) [Y5 Y6 X11 X12]
+ (0.011285190200840851) [X5 X6 Y11 Y12]
+ (0.011307274008848203) [Y7 Z8 Z9 Y11]
+ (0.011307274008848203) [X7 Z8 Z9 X11]
+ (0.011982389010247911) [Y4 X5 X6 Y7]
+ (0.011982389010247911) [X4 Y5 Y6 X7]
+ (0.013873381748426145) [Y6 X7 X8 Y9]
+ (0.013873381748426145) [X6 Y7 Y8 X9]
+ (0.014583648907612587) [Y0 X1 X2 Y3]
+ (0.014583648907612587) [X0 Y1 Y2 X3]
+ (0.017366118994651375) [Y6 X7 X12 Y13]
+ (0.017366118994651375) [X6 Y7 Y12 X13]
+ (0.01768006795248153) [Y4 X5 X10 Y11]
+ (0.01768006795248153) [X4 Y5 Y10 X11]
+ (0.017825140995786342) [Y6 X7 X10 Y11]
+ (0.017825140995786342) [X6 Y7 Y10 X11]
+ (0.019028242443847362) [Y3 X4 X11 Y12]
+ (0.019028242443847362) [X3 Y4 Y11 X12]
+ (0.025384657508457482) [Y2 X3 X10 Y11]
+ (0.025384657508457482) [X2 Y3 Y10 X11]
+ (0.02868518371610592) [Y10 X11 X12 Y13]
+ (0.02868518371610592) [X10 Y11 Y12 X13]
+ (0.029812424517345663) [Y6 Z7 Z8 Y10]
+ (0.029812424517345663) [X6 Z7 Z8 X10]
+ (0.029812424517345663) [Y7 Z9 Z10 Y11]
+ (0.029812424517345663) [X7 Z9 Z10 X11]
+ (0.030104623143456764) [Y6 Z7 Z9 Y10]
+ (0.030104623143456764) [X6 Z7 Z9 X10]
+ (0.030104623143456764) [Y7 Z8 Z10 Y11]
+ (0.030104623143456764) [X7 Z8 Z10 X11]
+ (0.03078750538914387) [Y6 Z8 Z9 Y10]
+ (0.03078750538914387) [X6 Z8 Z9 X10]
+ (0.031143817988967037) [Y2 X3 X6 Y7]
+ (0.031143817988967037) [X2 Y3 Y6 X7]
+ (0.03583956795335346) [Y2 X3 X4 Y5]
+ (0.03583956795335346) [X2 Y3 Y4 X5]
+ (0.03619412355904253) [Y2 X3 X8 Y9]
+ (0.03619412355904253) [X2 Y3 Y8 X9]
+ (0.03831467029480387) [Y4 X5 X12 Y13]
+ (0.03831467029480387) [X4 Y5 Y12 X13]
+ (0.10433064780651356) [Z0 Y1 Z2 Y3]
+ (0.10433064780651356) [Z0 X1 Z2 X3]
+ (-0.12133276911042423) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133276911042423) [X3 Z4 Z5 Z6 X7]
+ (-0.12133276911042418) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042418) [X2 Z3 Z4 Z5 X6]
+ (3.20207687993282e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.20207687993282e-06) [X0 Z1 Z2 Z3 X4]
+ (3.202076879932821e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.202076879932821e-06) [X1 Z2 Z3 Z4 X5]
+ (0.228481065649187) [Y7 Z8 Z9 Z10 Y11]
+ (0.228481065649187) [X7 Z8 Z9 Z10 X11]
+ (0.22848106564918702) [Y6 Z7 Z8 Z9 Y10]
+ (0.22848106564918702) [X6 Z7 Z8 Z9 X10]
+ (-0.03276765782329049) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.03276765782329049) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.03276765782329049) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.03276765782329049) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.02711503684527315) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.02711503684527315) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.02711503684527315) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.02711503684527315) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.025996177598021312) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.025996177598021312) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.017561202409646183) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.017561202409646183) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.017561202409646183) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.017561202409646183) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.014564531231172973) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564531231172973) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564531231172973) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564531231172973) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.01221504099761387) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.01221504099761387) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.01221504099761387) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.01221504099761387) [X4 Z5 X6 X11 Z12 X13]
+ (-0.01221504099761387) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.01221504099761387) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.01221504099761387) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.01221504099761387) [X5 Z6 X7 X10 Z11 X12]
+ (-0.011756013419819276) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.011756013419819276) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.011756013419819276) [X3 Z4 Z5 X6 X8 X9]
+ (-0.011756013419819276) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.008764827575688819) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.008764827575688819) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.008764827575688819) [X2 Z3 Z4 X5 X11 X12]
+ (-0.008764827575688819) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.008764827575688819) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.008764827575688819) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.008764827575688819) [X3 X4 X10 Z11 Z12 X13]
+ (-0.008764827575688819) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.008125251921381034) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125251921381034) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.007306759928832947) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306759928832947) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306759928832947) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306759928832947) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005805188989826907) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.005805188989826907) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.005805188989826907) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.005805188989826907) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.005652620978017345) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.005652620978017345) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.005652620978017345) [X0 X1 X3 Z4 Z5 X6]
+ (-0.005652620978017345) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.005143391768825084) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.005143391768825084) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.005143391768825084) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.005143391768825084) [X2 Z3 X4 X5 Z6 X7]
+ (-0.004684903388155204) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.004684903388155204) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668620318776306) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668620318776306) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.0045750076266392065) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.0045750076266392065) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424855449441851) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424855449441851) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.004158797381840057) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.004158797381840057) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.004158797381840057) [X3 Z4 Z5 X6 X12 X13]
+ (-0.004158797381840057) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.0034937903598901963) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.0034937903598901963) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.0034937903598901963) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.0034937903598901963) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.0027790267990255735) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.0027790267990255735) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.0022939566113524584) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.0022939566113524584) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.001799219493663018) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.001799219493663018) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278753941369638) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.0017278753941369638) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298507967730207) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298507967730207) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298507967730207) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298507967730207) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533856254125547) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533856254125547) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008145313270957193) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008145313270957193) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008145313270957193) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008145313270957193) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.735036880592685e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735036880592685e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735036880592685e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735036880592685e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.774817864308792e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.774817864308792e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.774817864308792e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.774817864308792e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.518362215470406e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.518362215470406e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.518362215470406e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.518362215470406e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.444344675653489e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.444344675653489e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.444344675653489e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.444344675653489e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.524373848328739e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.524373848328739e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.524373848328739e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.524373848328739e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.290028432904171e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.290028432904171e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.290028432904171e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.290028432904171e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.974311713285157e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.974311713285157e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.275883121999843e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.275883121999843e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.643051068307e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.643051068307e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.556569217895079e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.556569217895079e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.253224225499797e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.253224225499797e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.7696594516606176e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.7696594516606176e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.6945132943170153e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.6945132943170153e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.610297130253331e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.610297130253331e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.610297130253331e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.610297130253331e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.313145500137265e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.313145500137265e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.2774831951717187e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.2774831951717187e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.2774831951717187e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.2774831951717187e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.2112283481914745e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.2112283481914745e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.2112283481914745e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.2112283481914745e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.1513463110389345e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.1513463110389345e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.0882507111416055e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.0882507111416055e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.172669101467003e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.172669101467003e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.4548424491623226e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.4548424491623226e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.3304731886553028e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.3304731886553028e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.228333782566235e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.228333782566235e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.0358477602518635e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.0358477602518635e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956895373229693e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956895373229693e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733197741857188e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733197741857188e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733197741857188e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733197741857188e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.62861420151462e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.62861420151462e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.556281914391058e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.556281914391058e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.556281914391058e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.556281914391058e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.418291574301356e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.418291574301356e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.418291574301356e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.418291574301356e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.927453082429325e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927453082429325e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927453082429325e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927453082429325e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627851911215719e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627851911215719e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.28766062437634e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.28766062437634e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.28766062437634e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.28766062437634e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.28766062437634e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.28766062437634e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.28766062437634e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.28766062437634e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.838052751197076e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.838052751197076e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.5707613293820367e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.5707613293820367e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.328139350816122e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.328139350816122e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.086826565043034e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.086826565043034e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.086826565043034e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.086826565043034e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.4473231289463473e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.4473231289463473e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.3713289480786222e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.3713289480786222e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.3713289480786222e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.3713289480786222e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.1990516185656584e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.1990516185656584e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.9332412772300546e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.9332412772300546e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.9332412772300546e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.9332412772300546e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.8394209156070853e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.8394209156070853e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.8394209156070853e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.8394209156070853e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.5510539176716055e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.5510539176716055e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.5510539176716055e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.5510539176716055e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.380778148160037e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.380778148160037e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.380778148160037e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.380778148160037e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.380778148160037e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.380778148160037e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.380778148160037e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.380778148160037e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.380778148160037e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.380778148160037e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.380778148160037e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.380778148160037e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.2919694863497273e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.2919694863497273e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.1076325599124517e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.1076325599124517e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.1076325599124517e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.1076325599124517e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.1076325599124517e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.1076325599124517e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.1076325599124517e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.1076325599124517e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.05744659427863e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.05744659427863e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.05744659427863e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.05744659427863e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649310135472782e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649310135472782e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649310135472782e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649310135472782e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.839420915607085e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.839420915607085e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.839420915607085e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.839420915607085e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.1990516185656584e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.1990516185656584e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.4473231289463473e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.4473231289463473e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.2362599616259e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.2362599616259e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.2362599616259e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.2362599616259e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.328139350816122e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.328139350816122e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.5707613293820367e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.5707613293820367e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.838052751197076e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.838052751197076e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.627851911215719e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627851911215719e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.62861420151462e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.62861420151462e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956895373229693e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956895373229693e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.306536651723591e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.306536651723591e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.306536651723591e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.306536651723591e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.0358477602518635e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.0358477602518635e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.228333782566235e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.228333782566235e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.2393363216766625e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393363216766625e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393363216766625e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393363216766625e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.3304731886553028e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.3304731886553028e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.4548424491623226e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.4548424491623226e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.172669101467003e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.172669101467003e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.0882507111416055e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.0882507111416055e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.1174479463051e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.1174479463051e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.1513463110389345e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.1513463110389345e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.313145500137265e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.313145500137265e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.3343312893247023e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.3343312893247023e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.6945132943170153e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.6945132943170153e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.1839325593826665e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.1839325593826665e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.556569217895079e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.556569217895079e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.643051068307e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.643051068307e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.275883121999843e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.275883121999843e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.974311713285157e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.974311713285157e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.0002921986261111044) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.0002921986261111044) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.0002921986261111044) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.0002921986261111044) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004956762314916902) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004956762314916902) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650070219498601) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650070219498601) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650070219498601) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650070219498601) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533856254125547) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533856254125547) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.001609531381721369) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.001609531381721369) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.001609531381721369) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.001609531381721369) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0016676041811440664) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.0016676041811440664) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.0016676041811440664) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.0016676041811440664) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.0017278753941369638) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.0017278753941369638) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.001799219493663018) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.001799219493663018) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.0022939566113524584) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.0022939566113524584) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.0024629170071339235) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.0024629170071339235) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.0024629170071339235) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.0024629170071339235) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.003961560792496524) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.003961560792496524) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.003961560792496524) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.003961560792496524) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.004424855449441851) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424855449441851) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.0045750076266392065) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.0045750076266392065) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668620318776306) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668620318776306) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.004684903388155204) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.004684903388155204) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005324835234221658) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.005324835234221658) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.005324835234221658) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.005324835234221658) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.005368659358109454) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368659358109454) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368659358109454) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368659358109454) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.008125251921381034) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125251921381034) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.008890731522694567) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.008890731522694567) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.008890731522694567) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.008890731522694567) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.010263414868158544) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.010263414868158544) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.010263414868158544) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.010263414868158544) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.010540425907671524) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.010540425907671524) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.010540425907671524) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.010540425907671524) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010960074940542523) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010960074940542523) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010960074940542523) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010960074940542523) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.0113072740088482) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.0113072740088482) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.014411099430130888) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411099430130888) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411099430130888) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411099430130888) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.015225630757226605) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.015225630757226605) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.015225630757226605) [X3 Z4 Z5 X6 X10 X11]
+ (0.015225630757226605) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.015588250102380198) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.015588250102380198) [X2 Z3 X4 X10 Z11 X12]
+ (0.015588250102380198) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.015588250102380198) [X3 Z4 X5 X11 Z12 X13]
+ (0.01826683486937547) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.01826683486937547) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.01826683486937547) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.01826683486937547) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.019020423173039928) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.019020423173039928) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.019020423173039928) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.019020423173039928) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.02017592172353542) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.02017592172353542) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.02017592172353542) [X4 Z5 Z6 X7 X11 X12]
+ (0.02017592172353542) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.02017592172353542) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.02017592172353542) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.02017592172353542) [X5 X6 X10 Z11 Z12 X13]
+ (0.02017592172353542) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.02435307767806902) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.02435307767806902) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.02435307767806902) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.02435307767806902) [X2 Z3 X4 X11 Z12 X13]
+ (0.02435307767806902) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.02435307767806902) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.02435307767806902) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.02435307767806902) [X3 Z4 X5 X10 Z11 X12]
+ (0.024389082531149377) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.024389082531149377) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.024389082531149377) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.024389082531149377) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.0251049571388445) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.0251049571388445) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.0251049571388445) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.0251049571388445) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.03078750538914387) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.03078750538914387) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.045879470781298115) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.045879470781298115) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.05600733087780741) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.05600733087780741) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.05600733087780741) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.05600733087780741) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.056084681246613345) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.056084681246613345) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.056084681246613345) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.056084681246613345) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.63127792821168e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.63127792821168e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.63127792821168e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.63127792821168e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.5950860067215506e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.5950860067215506e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.5950860067215496e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.5950860067215496e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.04274327701378314) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.04274327701378314) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (0.04274327701378314) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04274327701378314) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.03956441632289344) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.03956441632289344) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.03956441632289344) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03956441632289344) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03935916802205325) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.03935916802205325) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.03935916802205325) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.03935916802205325) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.03931805194719757) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03931805194719757) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.03931805194719757) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03931805194719757) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03560837898831262) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.03560837898831262) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.02990378951262493) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.02990378951262493) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.02990378951262493) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.02990378951262493) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.02873077955190557) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.02873077955190557) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.02873077955190557) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.02873077955190557) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.02563723829602682) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.02563723829602682) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.02563723829602682) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.02563723829602682) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.024755463292891088) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.024755463292891088) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.024755463292891088) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.024755463292891088) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.024282117354693107) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.024282117354693107) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.023145130929529092) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.023145130929529092) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.02252844019601294) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.02252844019601294) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.021433810721601068) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.021433810721601068) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.021433810721601068) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.021433810721601068) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251582) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.019257505095251582) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019028242443847362) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.019028242443847362) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.018889030304942954) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.018889030304942954) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.018889030304942954) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.018889030304942954) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.01602460368917952) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.01602460368917952) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.015225630757226605) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225630757226605) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.014603704729162158) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.014603704729162158) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564531231172973) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564531231172973) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.011756013419819276) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.011756013419819276) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.011285190200840851) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.011285190200840851) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.009841749246962617) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009841749246962617) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.009612634606847302) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612634606847302) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612634606847302) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612634606847302) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.008469978791023855) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.008469978791023855) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.007306759928832947) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.007306759928832947) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005923798336561344) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.005923798336561344) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.005652620978017345) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.005652620978017345) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368659358109454) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368659358109454) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.004158797381840057) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.004158797381840057) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0033566705638329035) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.0033566705638329035) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.0033566705638329035) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.0033566705638329035) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.003267513854423567) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.003267513854423567) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.003267513854423567) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.003267513854423567) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.0027790267990255735) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.0027790267990255735) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.00268604097780662) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.00268604097780662) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.00268604097780662) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.00268604097780662) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.0022939566113524584) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524584) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.0022939566113524584) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524584) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.0009581655836696572) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.0009581655836696572) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0009581655836696572) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.0009581655836696572) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.0009581655836696572) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.0009581655836696572) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0009581655836696572) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.0009581655836696572) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.00024636437569586624) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00024636437569586624) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303549197) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.00013840177303549197) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.00013840177303549197) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.00013840177303549197) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.735036880592685e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735036880592685e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585305177326e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585305177326e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.6103585305177326e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.6103585305177326e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808794769196e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.5316808794769196e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808794769196e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5316808794769196e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.80610277486807e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.80610277486807e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.80610277486807e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.80610277486807e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.089799467349904e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.089799467349904e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.089799467349904e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.089799467349904e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.652209668863621e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.652209668863621e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.652209668863621e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.652209668863621e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.4818518333130945e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.4818518333130945e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.4818518333130945e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.4818518333130945e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.0714807362888375e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.0714807362888375e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.0714807362888375e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.0714807362888375e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.734622038579234e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.734622038579234e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.734622038579234e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.734622038579234e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.728843147072425e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.728843147072425e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.728843147072425e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.728843147072425e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.253224225499797e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.253224225499797e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.7696594516606176e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.7696594516606176e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.5443954291769517e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.5443954291769517e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.5443954291769517e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.5443954291769517e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.5443954291769517e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.5443954291769517e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.5443954291769517e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.5443954291769517e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.3609563202774802e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563202774802e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563202774802e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.3609563202774802e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.103215604476429e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.103215604476429e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.103215604476429e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.103215604476429e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220979396417e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.0111220979396417e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220979396417e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.0111220979396417e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.942946836532957e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.942946836532957e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.942946836532957e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.942946836532957e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174768945827e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.6541174768945827e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174768945827e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.6541174768945827e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.5224930675751078e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224930675751078e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224930675751078e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224930675751078e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224930675751078e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930675751078e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930675751078e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224930675751078e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.228333782566235e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.228333782566235e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.228333782566235e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.228333782566235e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.9887702891529e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.9887702891529e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.9887702891529e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.9887702891529e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.86776510408129e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.86776510408129e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.86776510408129e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.86776510408129e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.189990974916569e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.189990974916569e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.17524620687909e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.17524620687909e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.471647744491171e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.471647744491171e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.5614471803534597e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.5614471803534597e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.5614471803534597e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.5614471803534597e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.523389677494963e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.523389677494963e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.4273231087994406e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.4273231087994406e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.4273231087994406e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.4273231087994406e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.328139350816122e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350816122e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.328139350816122e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350816122e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.086826565043034e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.086826565043034e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.8882935963837467e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935963837467e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935963837467e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.8882935963837467e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.3713289480786222e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.3713289480786222e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.839420915607085e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.839420915607085e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.05744659427863e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.05744659427863e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.5371780959447865e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.5371780959447865e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.5371780959447865e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.5371780959447865e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.05744659427863e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.05744659427863e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.209350653678716e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350653678716e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350653678716e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.209350653678716e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783555052693e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783555052693e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783555052693e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783555052693e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.839420915607085e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.839420915607085e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.3713289480786222e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.3713289480786222e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.086826565043034e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.086826565043034e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.523389677494963e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.523389677494963e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.471647744491171e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.471647744491171e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.17524620687909e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.17524620687909e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189990974916569e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.189990974916569e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.3304731886553028e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.3304731886553028e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.3304731886553028e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.3304731886553028e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288532434638273e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288532434638273e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288532434638273e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288532434638273e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.6893489514085497e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.6893489514085497e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.6893489514085497e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.6893489514085497e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.7455184002616617e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.7455184002616617e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.7455184002616617e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.7455184002616617e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.7455184002616617e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.7455184002616617e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.7455184002616617e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.7455184002616617e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.2118420189836573e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420189836573e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.2118420189836573e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420189836573e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.2118420189836573e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420189836573e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.2118420189836573e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420189836573e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.3131455001372646e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.3131455001372646e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.3131455001372646e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.3131455001372646e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.3343312893247023e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.3343312893247023e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.1839325593826665e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.1839325593826665e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.735036880592685e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735036880592685e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00024636437569586624) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00024636437569586624) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00044585351288407935) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.00044585351288407935) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.00044585351288407935) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.00044585351288407935) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940221543005158) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005158) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005158) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005158) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005158) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005158) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940221543005158) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005158) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533856254125547) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533856254125547) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533856254125547) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533856254125547) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0010435246534907495) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435246534907495) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435246534907495) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435246534907495) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.001280306097349662) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.001280306097349662) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.001280306097349662) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.001280306097349662) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.0013038004788126982) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.0013038004788126982) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.0013038004788126982) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.0013038004788126982) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.002261966062482355) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.002261966062482355) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.002261966062482355) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.002261966062482355) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.002261966062482355) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.002261966062482355) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.002261966062482355) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.002261966062482355) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.003989841456619318) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.003989841456619318) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.003989841456619318) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.003989841456619318) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.004158797381840057) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.004158797381840057) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.004311038507914316) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.004311038507914316) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.004311038507914316) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.004311038507914316) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.004636976661182566) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.004636976661182566) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.004636976661182566) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.004636976661182566) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.005114473831660385) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.005114473831660385) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.005114473831660385) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.005114473831660385) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.005114473831660385) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660385) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660385) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005114473831660385) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005262642473076821) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262642473076821) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262642473076821) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262642473076821) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368659358109454) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368659358109454) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.0053799371558393766) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.0053799371558393766) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.0053799371558393766) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.0053799371558393766) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.005652620978017345) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.005652620978017345) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.0057084959859609) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.0057084959859609) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.0057084959859609) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.0057084959859609) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.005923798336561344) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.005923798336561344) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.007306759928832947) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.007306759928832947) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469978791023855) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.008469978791023855) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.009841749246962617) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.009841749246962617) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.011285190200840851) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.011285190200840851) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.011756013419819276) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.011756013419819276) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.014564531231172973) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564531231172973) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603704729162158) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.014603704729162158) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.015225630757226605) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225630757226605) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.01602460368917952) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.01602460368917952) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.019028242443847362) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.019028242443847362) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.019257505095251582) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.019257505095251582) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.045879470781298115) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.045879470781298115) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.36937089366156295) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.36937089366156295) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.36937089366156295) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.36937089366156295) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.2816425776702311) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.2816425776702311) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.281642577670231) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.281642577670231) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0906514420703647) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0906514420703647) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0906514420703647) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0906514420703647) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863617) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684737589863617) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863617) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684737589863617) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635021950635032) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635021950635032) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635021950635032) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635021950635032) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214046) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.06752385099214046) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214046) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.06752385099214046) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03560837898831262) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.03560837898831262) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03490334337366166) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03490334337366166) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03490334337366166) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03490334337366166) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02459186088382994) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.02459186088382994) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02459186088382994) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.02459186088382994) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.024282117354693104) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.024282117354693104) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.02314513092952909) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.02314513092952909) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.02252844019601294) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.02252844019601294) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019538050311314854) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.019538050311314854) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.019538050311314854) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.019538050311314854) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.017091553155898977) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.017091553155898977) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.017091553155898977) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.017091553155898977) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.01602460368917952) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.01602460368917952) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.01602460368917952) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.01602460368917952) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.010311482489831717) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831717) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831717) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311482489831717) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.009841749246962617) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962617) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.009841749246962617) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962617) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209856) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209856) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209856) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008826368514209856) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008541996625454849) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454849) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454849) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454849) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454849) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454849) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454849) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008541996625454849) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008469978791023855) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023855) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.008469978791023855) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023855) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.004668620318776306) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668620318776306) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.0038764708993369672) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.0038764708993369672) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.00380406617172854) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00380406617172854) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00380406617172854) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.00380406617172854) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484157300217882) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484157300217882) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0033566705638329035) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.0033566705638329035) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.003267513854423567) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.003267513854423567) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0021413612231016375) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.0021413612231016375) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.0017278753941369638) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.0017278753941369638) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.0016407548553123812) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0016407548553123812) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214169297) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0014528843214169297) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214169297) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0014528843214169297) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870896771024526) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007870896771024526) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192743499487762) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192743499487762) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.00019400857029756348) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00019400857029756348) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303549197) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.00013840177303549197) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.141625221159545e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.141625221159545e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.141625221159545e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.141625221159545e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.0714807362888375e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.0714807362888375e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.1513463110389345e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.1513463110389345e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0882507111416055e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.0882507111416055e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.988511706414389e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.988511706414389e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8742990713153185e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.8742990713153185e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.3609563202774802e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.3609563202774802e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.3002946561892815e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3002946561892815e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.1468376506881022e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.1468376506881022e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.1468376506881022e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.1468376506881022e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.352332102535534e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.352332102535534e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.352332102535534e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.352332102535534e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.09163719852473e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.09163719852473e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.09163719852473e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.09163719852473e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.09163719852473e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.09163719852473e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.09163719852473e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.09163719852473e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.074305985471869e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.074305985471869e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.074305985471869e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.074305985471869e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128985851251e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128985851251e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128985851251e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128985851251e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867765104081292e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.867765104081292e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560692464338979e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692464338979e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692464338979e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692464338979e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692464338979e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692464338979e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692464338979e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560692464338979e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.997018422036595e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.997018422036595e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.997018422036595e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.997018422036595e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.997018422036595e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.997018422036595e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.997018422036595e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.997018422036595e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.568247521029768e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.568247521029768e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.568247521029768e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.568247521029768e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376739308356289e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.376739308356289e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376739308356289e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.376739308356289e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376739308356289e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.376739308356289e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.376739308356289e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.376739308356289e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.888293596383747e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.888293596383747e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6863815437219435e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.6863815437219435e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035783555052693e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7035783555052693e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.209350653678716e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.209350653678716e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773243759273e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773243759273e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773243759273e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773243759273e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773243759273e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773243759273e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773243759273e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773243759273e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253790346164e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253790346164e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9742253790346164e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9742253790346164e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0474716554814212e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.0474716554814212e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.0474716554814212e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.0474716554814212e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.209350653678716e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.209350653678716e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0717282182014286e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0717282182014286e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0717282182014286e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0717282182014286e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.2004287493327192e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.2004287493327192e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.2004287493327192e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.2004287493327192e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.7035783555052693e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7035783555052693e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.3120943050276907e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.3120943050276907e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.3120943050276907e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.3120943050276907e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.6863815437219435e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815437219435e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.888293596383747e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.888293596383747e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.0922506159874567e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506159874567e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.0922506159874567e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506159874567e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.0922506159874567e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506159874567e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.0922506159874567e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506159874567e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.4445978541793147e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.4445978541793147e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.4445978541793147e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.4445978541793147e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.6849150951225583e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.6849150951225583e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.6849150951225583e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.6849150951225583e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246974425080519e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246974425080519e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246974425080519e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246974425080519e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246974425080519e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246974425080519e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246974425080519e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246974425080519e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.867765104081292e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.867765104081292e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3002946561892815e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3002946561892815e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.3609563202774802e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.3609563202774802e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.8742990713153185e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.8742990713153185e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.88367657605125e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.88367657605125e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.9473560115612565e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9473560115612565e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.9473560115612565e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9473560115612565e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.988511706414389e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.988511706414389e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.0882507111416055e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.0882507111416055e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.1513463110389345e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.1513463110389345e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.846201671195179e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846201671195179e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846201671195179e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846201671195179e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.0714807362888375e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.0714807362888375e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.105526721902257e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.105526721902257e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.105526721902257e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.105526721902257e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.1464963273844595e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.1464963273844595e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.1464963273844595e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.1464963273844595e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159350501896937e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159350501896937e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159350501896937e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159350501896937e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427988656269131e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427988656269131e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427988656269131e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427988656269131e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.9358677179756455e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.9358677179756455e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.9358677179756455e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.9358677179756455e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273347945627e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.253273347945627e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979825793217576e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979825793217576e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979825793217576e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979825793217576e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.205548411218181e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.205548411218181e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.205548411218181e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.205548411218181e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00013840177303549197) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.00013840177303549197) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.00018787053389545132) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.00018787053389545132) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.00018787053389545132) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00018787053389545132) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00019400857029756348) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00019400857029756348) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00024636437569586624) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569586624) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00024636437569586624) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569586624) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192743499487762) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192743499487762) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156734248908573) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156734248908573) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156734248908573) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156734248908573) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024526) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007870896771024526) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.0015324835230730038) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.0015324835230730038) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.0015324835230730038) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.0015324835230730038) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.0016407548553123812) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0016407548553123812) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0017278753941369638) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.0017278753941369638) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.0024464971554158743) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.0024464971554158743) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.0024464971554158743) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.0024464971554158743) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.003267513854423567) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.003267513854423567) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.0033566705638329035) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.0033566705638329035) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.003484157300217882) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.003484157300217882) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0038764708993369672) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.0038764708993369672) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.004668620318776306) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668620318776306) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278099) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.004767272188278099) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.004767272188278099) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278099) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.005286546538226876) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.005286546538226876) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.005286546538226876) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.005286546538226876) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.005408954422409971) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.005408954422409971) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.005408954422409971) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.005408954422409971) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.005923798336561344) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561344) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.005923798336561344) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561344) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.010715508469796738) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010715508469796738) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010715508469796738) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010715508469796738) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.010757563953908918) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010757563953908918) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010757563953908918) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010757563953908918) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.014603704729162158) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.014603704729162158) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.014603704729162158) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.014603704729162158) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.019299560579363766) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363766) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363766) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363766) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363766) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363766) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.019299560579363766) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363766) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.058591988733862) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.058591988733862) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.775950527129912e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.775950527129912e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.775950527129912e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.775950527129912e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.07165035181002673) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07165035181002673) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.07165035181002678) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.07165035181002678) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251582) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019257505095251582) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010311482489831717) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311482489831717) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008826368514209856) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209856) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.00759746402977059) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00759746402977059) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.00759746402977059) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00759746402977059) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0057335697473118825) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0057335697473118825) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0057335697473118825) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0057335697473118825) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0057335697473118825) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0057335697473118825) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0057335697473118825) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0057335697473118825) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00534805158267659) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.00534805158267659) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00534805158267659) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00534805158267659) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0038040661717285403) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0038040661717285403) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219538) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.0029841661681219538) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.0029841661681219538) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.0029841661681219538) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.002446497155415874) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.002446497155415874) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.0022494124470939995) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0022494124470939995) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022494124470939995) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0022494124470939995) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0021413612231016375) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.0021413612231016375) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.0018638942824587067) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587067) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587067) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587067) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587067) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587067) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0018638942824587067) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587067) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0016407548553123814) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553123814) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0016407548553123814) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553123814) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0012223378081538422) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538422) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538422) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538422) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538422) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538422) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538422) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0012223378081538422) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0010283292378562786) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0010283292378562786) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0010283292378562786) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0010283292378562786) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1463061452488086e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1463061452488086e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.8742990713153185e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990713153185e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.8742990713153185e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990713153185e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3002946561892815e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3002946561892815e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3002946561892815e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3002946561892815e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.0444941296937912e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.0444941296937912e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.0444941296937912e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.0444941296937912e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.95607922901172e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.95607922901172e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.95607922901172e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.95607922901172e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105515036039329e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105515036039329e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105515036039329e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105515036039329e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.6613472123186e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.6613472123186e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.6613472123186e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.6613472123186e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341413513366e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341413513366e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189990974916569e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.189990974916569e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876621657388889e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876621657388889e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876621657388889e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876621657388889e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.17524620687909e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.17524620687909e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.523389677494963e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.523389677494963e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.076732531367807e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.076732531367807e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.076732531367807e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.076732531367807e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.013471458761145e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.013471458761145e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.904599883424548e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.904599883424548e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.904599883424548e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.904599883424548e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.66673175418267e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.66673175418267e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.66673175418267e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.66673175418267e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641929723918e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.8505641929723918e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309323841732e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6569309323841732e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309323841732e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6569309323841732e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.8505641929723918e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.8505641929723918e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815437219435e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815437219435e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.6863815437219435e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815437219435e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458761145e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.013471458761145e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.523389677494963e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.523389677494963e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.6704023911453177e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.6704023911453177e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.6704023911453177e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.6704023911453177e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.17524620687909e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.17524620687909e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189990974916569e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.189990974916569e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341413513366e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341413513366e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.94947648780414e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.94947648780414e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.7924939576396952e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939576396952e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939576396952e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.7924939576396952e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.8836765760512503e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.8836765760512503e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.988511706414389e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.988511706414389e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.988511706414389e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.988511706414389e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273347945627e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253273347945627e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4017109734933791e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4017109734933791e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4017109734933791e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4017109734933791e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603692573488e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.5809603692573488e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603692573488e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.5809603692573488e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192743499487764) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487764) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192743499487764) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487764) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.0007870896771024526) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024526) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024526) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024526) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0011726348316441852) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0011726348316441852) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0011726348316441852) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0011726348316441852) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012366478019245068) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.0012366478019245068) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.0012366478019245068) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.0012366478019245068) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.002200964069500464) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002200964069500464) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002200964069500464) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002200964069500464) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798027) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798027) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798027) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798027) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798027) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798027) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002394972639798027) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798027) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002446497155415874) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.002446497155415874) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.0038040661717285403) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0038040661717285403) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.003876470899336967) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.003876470899336967) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.003876470899336967) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.003876470899336967) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.00422081397004646) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.00422081397004646) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.00422081397004646) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.00422081397004646) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.008826368514209856) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.008826368514209856) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.010311482489831717) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311482489831717) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257505095251582) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019257505095251582) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.058591988733862004) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.058591988733862004) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3987009012924894e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3987009012924894e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.398700901292489e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.398700901292489e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484157300217882) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003484157300217882) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219538) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.0029841661681219538) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.00019400857029756348) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00019400857029756348) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061452488086e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061452488086e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7924939576396952e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7924939576396952e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540341413513366e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413513366e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540341413513366e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413513366e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.8505641929723918e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641929723918e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641929723918e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641929723918e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458761145e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471458761145e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458761145e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471458761145e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.94947648780414e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.94947648780414e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.7924939576396952e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7924939576396952e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756348) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756348) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0029841661681219538) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.0029841661681219538) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.003484157300217882) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.003484157300217882) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.18066792656583291) [Z6]
+ (-0.1806679265658329) [Z7]
+ (-0.1596143250180983) [Z4]
+ (-0.1596143250180983) [Z5]
+ (0.17419956155055794) [Z2]
+ (0.17419956155055813) [Z3]
+ (0.22757269005453648) [Z1]
+ (0.22757269005453656) [Z0]
+ (-8.194261373187165e-06) [Y4 Y6]
+ (-8.194261373187165e-06) [X4 X6]
+ (7.954413177012473e-06) [Y5 Y7]
+ (7.954413177012473e-06) [X5 X7]
+ (0.11270386920332222) [Z4 Z6]
+ (0.11270386920332222) [Z5 Z7]
+ (0.11952438964682678) [Z0 Z4]
+ (0.11952438964682678) [Z1 Z5]
+ (0.1340171526196371) [Z0 Z6]
+ (0.1340171526196371) [Z1 Z7]
+ (0.13734953064261318) [Z0 Z5]
+ (0.13734953064261318) [Z1 Z4]
+ (0.1376687264585258) [Z2 Z4]
+ (0.1376687264585258) [Z3 Z5]
+ (0.14138905291942805) [Z4 Z7]
+ (0.14138905291942805) [Z5 Z6]
+ (0.14722943218766177) [Z2 Z5]
+ (0.14722943218766177) [Z3 Z4]
+ (0.14926355147388898) [Z4 Z5]
+ (0.14973486803496935) [Z2 Z6]
+ (0.14973486803496935) [Z3 Z7]
+ (0.1513832716142885) [Z0 Z7]
+ (0.1513832716142885) [Z1 Z6]
+ (0.15435748657223639) [Z6 Z7]
+ (0.15582269051553121) [Z2 Z7]
+ (0.15582269051553121) [Z3 Z6]
+ (0.1675665326546127) [Z0 Z2]
+ (0.1675665326546127) [Z1 Z3]
+ (0.18143991440303883) [Z0 Z3]
+ (0.18143991440303883) [Z1 Z2]
+ (0.19392534613270213) [Z0 Z1]
+ (-7.0378875100846895e-06) [Y4 Z5 Y6]
+ (-7.0378875100846895e-06) [X4 Z5 X6]
+ (-7.0378875100846895e-06) [Y5 Z6 Y7]
+ (-7.0378875100846895e-06) [X5 Z6 X7]
+ (-0.02868518371610582) [Y4 Y5 X6 X7]
+ (-0.02868518371610582) [X4 X5 Y6 Y7]
+ (-0.017825140995786384) [Y0 Y1 X4 X5]
+ (-0.017825140995786384) [X0 X1 Y4 Y5]
+ (-0.017366118994651406) [Y0 Y1 X6 X7]
+ (-0.017366118994651406) [X0 X1 Y6 Y7]
+ (-0.013873381748426145) [Y0 Y1 X2 X3]
+ (-0.013873381748426145) [X0 X1 Y2 Y3]
+ (-0.009560705729135978) [Y2 Y3 X4 X5]
+ (-0.009560705729135978) [X2 X3 Y4 Y5]
+ (-0.006087822480561874) [Y2 Y3 X6 X7]
+ (-0.006087822480561874) [X2 X3 Y6 Y7]
+ (-0.00029219862611109957) [Y1 Y2 X3 X4]
+ (-0.00029219862611109957) [X1 X2 Y3 Y4]
+ (-8.194261373187165e-06) [Z4 Y5 Z6 Y7]
+ (-8.194261373187165e-06) [Z4 X5 Z6 X7]
+ (-2.8909678819463283e-06) [Z0 Y5 Z6 Y7]
+ (-2.8909678819463283e-06) [Z0 X5 Z6 X7]
+ (-2.8909678819463283e-06) [Z1 Y4 Z5 Y6]
+ (-2.8909678819463283e-06) [Z1 X4 Z5 X6]
+ (-1.8551201218762654e-06) [Z0 Y4 Z5 Y6]
+ (-1.8551201218762654e-06) [Z0 X4 Z5 X6]
+ (-1.8551201218762654e-06) [Z1 Y5 Z6 Y7]
+ (-1.8551201218762654e-06) [Z1 X5 Z6 X7]
+ (-1.5973171980902868e-06) [Z2 Y4 Z5 Y6]
+ (-1.5973171980902868e-06) [Z2 X4 Z5 X6]
+ (-1.5973171980902868e-06) [Z3 Y5 Z6 Y7]
+ (-1.5973171980902868e-06) [Z3 X5 Z6 X7]
+ (-1.0358477600700629e-06) [Y0 X1 X5 Y6]
+ (-1.0358477600700629e-06) [Y0 Y1 Y5 Y6]
+ (-1.0358477600700629e-06) [X0 X1 X5 X6]
+ (-1.0358477600700629e-06) [X0 Y1 Y5 X6]
+ (-9.344557778531845e-07) [Z2 Y5 Z6 Y7]
+ (-9.344557778531845e-07) [Z2 X5 Z6 X7]
+ (-9.344557778531845e-07) [Z3 Y4 Z5 Y6]
+ (-9.344557778531845e-07) [Z3 X4 Z5 X6]
+ (6.628614202371026e-07) [Y2 X3 X5 Y6]
+ (6.628614202371026e-07) [Y2 Y3 Y5 Y6]
+ (6.628614202371026e-07) [X2 X3 X5 X6]
+ (6.628614202371026e-07) [X2 Y3 Y5 X6]
+ (7.954413177012473e-06) [Y4 Z5 Y6 Z7]
+ (7.954413177012473e-06) [X4 Z5 X6 Z7]
+ (0.00029219862611109957) [Y1 X2 X3 Y4]
+ (0.00029219862611109957) [X1 Y2 Y3 X4]
+ (0.006087822480561874) [Y2 X3 X6 Y7]
+ (0.006087822480561874) [X2 Y3 Y6 X7]
+ (0.009560705729135978) [Y2 X3 X4 Y5]
+ (0.009560705729135978) [X2 Y3 Y4 X5]
+ (0.011307274008848189) [Y1 Z2 Z3 Y5]
+ (0.011307274008848189) [X1 Z2 Z3 X5]
+ (0.013873381748426145) [Y0 X1 X2 Y3]
+ (0.013873381748426145) [X0 Y1 Y2 X3]
+ (0.017366118994651406) [Y0 X1 X6 Y7]
+ (0.017366118994651406) [X0 Y1 Y6 X7]
+ (0.017825140995786384) [Y0 X1 X4 Y5]
+ (0.017825140995786384) [X0 Y1 Y4 X5]
+ (0.02868518371610582) [Y4 X5 X6 Y7]
+ (0.02868518371610582) [X4 Y5 Y6 X7]
+ (0.02981242451734567) [Y0 Z1 Z2 Y4]
+ (0.02981242451734567) [X0 Z1 Z2 X4]
+ (0.02981242451734567) [Y1 Z3 Z4 Y5]
+ (0.02981242451734567) [X1 Z3 Z4 X5]
+ (0.03010462314345677) [Y0 Z1 Z3 Y4]
+ (0.03010462314345677) [X0 Z1 Z3 X4]
+ (0.03010462314345677) [Y1 Z2 Z4 Y5]
+ (0.03010462314345677) [X1 Z2 Z4 X5]
+ (0.03078750538914387) [Y0 Z2 Z3 Y4]
+ (0.03078750538914387) [X0 Z2 Z3 X4]
+ (0.04375263801066028) [Y1 Z2 Z3 Z4 Y5]
+ (0.04375263801066028) [X1 Z2 Z3 Z4 X5]
+ (0.043752638010660296) [Y0 Z1 Z2 Z3 Y4]
+ (0.043752638010660296) [X0 Z1 Z2 Z3 X4]
+ (-0.014564531231172961) [Y1 Z2 Z3 X4 X6 Y7]
+ (-0.014564531231172961) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-0.014564531231172961) [X1 Z2 Z3 X4 X6 X7]
+ (-0.014564531231172961) [X1 Z2 Z3 Y4 Y6 X7]
+ (-6.524373849352773e-06) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (-6.524373849352773e-06) [X0 Z1 Z2 Z3 Z4 X6]
+ (-6.524373849352773e-06) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (-6.524373849352773e-06) [X1 Z2 Z3 Z5 Z6 X7]
+ (-3.7696594526245445e-06) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (-3.7696594526245445e-06) [X0 Z2 Z3 Z4 Z5 X6]
+ (-3.6102971312272304e-06) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (-3.6102971312272304e-06) [X0 Z1 Z3 Z4 Z5 X6]
+ (-3.6102971312272304e-06) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (-3.6102971312272304e-06) [X1 Z2 Z4 Z5 Z6 X7]
+ (-3.3131455004016435e-06) [Y1 Z2 Z3 Y4 X5 X6]
+ (-3.3131455004016435e-06) [X1 Z2 Z3 X4 Y5 Y6]
+ (-3.277483196167484e-06) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (-3.277483196167484e-06) [X0 Z1 Z2 Z4 Z5 X6]
+ (-3.277483196167484e-06) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (-3.277483196167484e-06) [X1 Z3 Z4 Z5 Z6 X7]
+ (-3.2112283489511305e-06) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (-3.2112283489511305e-06) [X0 Z1 Z2 Z3 Z5 X6]
+ (-3.2112283489511305e-06) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (-3.2112283489511305e-06) [X1 Z2 Z3 Z4 Z6 X7]
+ (-1.0358477600700629e-06) [Y0 Y1 X4 Z5 Z6 X7]
+ (-1.0358477600700629e-06) [X0 X1 Y4 Z5 Z6 Y7]
+ (-6.628614202371026e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (-6.628614202371026e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (-3.328139350597464e-07) [Y1 X2 X3 Z4 Z5 Y6]
+ (-3.328139350597464e-07) [X1 Y2 Y3 Z4 Z5 X6]
+ (3.328139350597464e-07) [Y1 Y2 X3 Z4 Z5 X6]
+ (3.328139350597464e-07) [X1 X2 Y3 Z4 Z5 Y6]
+ (6.628614202371026e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (6.628614202371026e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (1.0358477600700629e-06) [Y0 X1 X4 Z5 Z6 Y7]
+ (1.0358477600700629e-06) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.3131455004016435e-06) [Y1 Z2 Z3 X4 X5 Y6]
+ (3.3131455004016435e-06) [X1 Z2 Z3 Y4 Y5 X6]
+ (4.1839325596961475e-06) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (4.1839325596961475e-06) [X1 Z2 Z3 Z4 Z5 X7]
+ (0.00029219862611109957) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (0.00029219862611109957) [Y0 Z1 Y2 X3 Z4 X5]
+ (0.00029219862611109957) [X0 Z1 X2 Y3 Z4 Y5]
+ (0.00029219862611109957) [X0 Z1 X2 X3 Z4 X5]
+ (0.010540425907671552) [Y0 Z1 Z2 Z3 Y4 Z7]
+ (0.010540425907671552) [X0 Z1 Z2 Z3 X4 Z7]
+ (0.010540425907671552) [Y1 Z2 Z3 Z4 Y5 Z6]
+ (0.010540425907671552) [X1 Z2 Z3 Z4 X5 Z6]
+ (0.011307274008848189) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (0.011307274008848189) [X0 Z1 Z2 Z3 X4 Z5]
+ (0.025104957138844513) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (0.025104957138844513) [X0 Z1 Z2 Z3 X4 Z6]
+ (0.025104957138844513) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (0.025104957138844513) [X1 Z2 Z3 Z4 X5 Z7]
+ (0.03078750538914387) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (0.03078750538914387) [Z0 X1 Z2 Z3 Z4 X5]
+ (-5.10539655045301e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (-5.10539655045301e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (-5.105396550453007e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-5.105396550453007e-06) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.014564531231172963) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-0.014564531231172963) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-3.7696594526245445e-06) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-3.7696594526245445e-06) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-3.328139350597464e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-3.328139350597464e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-3.328139350597464e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-3.328139350597464e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (3.3131455004016435e-06) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (3.3131455004016435e-06) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (3.3131455004016435e-06) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (3.3131455004016435e-06) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (4.183932559696147e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (4.183932559696147e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (0.014564531231172963) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (0.014564531231172963) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
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
(-46.463906788688945+0j) [] +
(-3.5707613292001126e-07+0j) [X0 X1 Y2 Z3 Z4 Y5] +
(-0.005652620978017362+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7] +
(-0.00882636851420981+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.7924939576213785e-06+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.5707613292001126e-07+0j) [X0 X1 X3 X4] +
(-0.005652620978017362+0j) [X0 X1 X3 Z4 Z5 X6] +
(-0.00882636851420981+0j) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.7924939576213785e-06+0j) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.0027458364701868003+0j) [X0 X1 Y4 Y5] +
(-2.447323128813329e-07+0j) [X0 X1 Y4 Z5 Z6 Y7] +
(-7.867765103915637e-07+0j) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.003804066171728535+0j) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.447323128813329e-07+0j) [X0 X1 X5 X6] +
(-7.867765103915637e-07+0j) [X0 X1 X5 Z6 Z7 Z8 Z9 X10] +
(-0.003804066171728535+0j) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.006888194352970544+0j) [X0 X1 Y6 Y7] +
(-7.73503688058884e-05+0j) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11] +
(1.7035783554601653e-07+0j) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.73503688058884e-05+0j) [X0 X1 X7 Z8 Z9 X10] +
(1.7035783554601653e-07+0j) [X0 X1 X7 Z8 Z9 Z10 Z11 X12] +
(-0.006509361201177232+0j) [X0 X1 Y8 Y9] +
(-0.007731425250775249+0j) [X0 X1 Y10 Y11] +
(5.627851911228886e-07+0j) [X0 X1 Y10 Z11 Z12 Y13] +
(5.627851911228886e-07+0j) [X0 X1 X11 X12] +
(-0.005283776488402962+0j) [X0 X1 Y12 Y13] +
(3.5707613292001126e-07+0j) [X0 Y1 Y2 Z3 Z4 X5] +
(0.005652620978017362+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7] +
(0.00882636851420981+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-1.7924939576213785e-06+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.5707613292001126e-07+0j) [X0 Y1 Y3 X4] +
(-0.005652620978017362+0j) [X0 Y1 Y3 Z4 Z5 X6] +
(-0.00882636851420981+0j) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.7924939576213785e-06+0j) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0027458364701868003+0j) [X0 Y1 Y4 X5] +
(2.447323128813329e-07+0j) [X0 Y1 Y4 Z5 Z6 X7] +
(7.867765103915637e-07+0j) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.003804066171728535+0j) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.447323128813329e-07+0j) [X0 Y1 Y5 X6] +
(-7.867765103915637e-07+0j) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.003804066171728535+0j) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.006888194352970544+0j) [X0 Y1 Y6 X7] +
(7.73503688058884e-05+0j) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11] +
(-1.7035783554601653e-07+0j) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.73503688058884e-05+0j) [X0 Y1 Y7 Z8 Z9 X10] +
(1.7035783554601653e-07+0j) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12] +
(0.006509361201177232+0j) [X0 Y1 Y8 X9] +
(0.007731425250775249+0j) [X0 Y1 Y10 X11] +
(-5.627851911228886e-07+0j) [X0 Y1 Y10 Z11 Z12 X13] +
(5.627851911228886e-07+0j) [X0 Y1 Y11 X12] +
(0.005283776488402962+0j) [X0 Y1 Y12 X13] +
(0.1250703257977213+0j) [X0 Z1 X2] +
(-1.933241277190824e-07+0j) [X0 Z1 X2 X3 Z4 X5] +
(-0.00229395661135247+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 X7] +
(-0.0016407548553124176+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(3.013471458694432e-07+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.933241277190824e-07+0j) [X0 Z1 X2 Y3 Z4 Y5] +
(-0.00229395661135247+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7] +
(-0.0016407548553124176+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(3.013471458694432e-07+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0032769719312317116+0j) [X0 Z1 X2 Z3] +
(-1.5510539175794388e-07+0j) [X0 Z1 X2 X4 Z5 X6] +
(-1.1468376506795559e-06+0j) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.007597464029770612+0j) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.38077814808055e-07+0j) [X0 Z1 X2 Y4 Z5 Y6] +
(-7.900128985792138e-07+0j) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.005348051582676631+0j) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005530759218631553+0j) [X0 Z1 X2 Z4] +
(-1.38077814808055e-07+0j) [X0 Z1 X2 X5 Z6 X7] +
(-3.3767393083039356e-07+0j) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.001863894282458747+0j) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.38077814808055e-07+0j) [X0 Z1 X2 Y5 Z6 Y7] +
(-3.3767393083039356e-07+0j) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.001863894282458747+0j) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.001105903769189714+0j) [X0 Z1 X2 Z5] +
(0.005708495985960944+0j) [X0 Z1 X2 X6 Z7 Z8 Z9 X10] +
(-8.352332102613404e-07+0j) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
1.97422537940826e-08j [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005262642473076856+0j) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10] +
(-8.07430598545385e-07+0j) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.008032520918821388+0j) [X0 Z1 X2 Z6] +
(0.0005940221543005582+0j) [X0 Z1 X2 X7 Z8 Z9 Z10 X11] +
(-8.379773244239258e-08+0j) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005940221543005582+0j) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11] +
(-8.379773244239258e-08+0j) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0033476175306662095+0j) [X0 Z1 X2 Z7] +
(0.011055020596132127+0j) [X0 Z1 X2 Z8] +
(0.002929768674751102+0j) [X0 Z1 X2 Z9] +
(-6.418291574347231e-07+0j) [X0 Z1 X2 X10 Z11 X12] +
(-6.556281914259837e-07+0j) [X0 Z1 X2 Y10 Z11 Y12] +
(0.0035552901955043324+0j) [X0 Z1 X2 Z10] +
(-1.1076325599061215e-07+0j) [X0 Z1 X2 X11 Z12 X13] +
(-1.1076325599061215e-07+0j) [X0 Z1 X2 Y11 Z12 Y13] +
(0.0017560707018412765+0j) [X0 Z1 X2 Z11] +
(0.006901238249797335+0j) [X0 Z1 X2 Z12] +
(0.002326230623158119+0j) [X0 Z1 X2 Z13] +
(-3.568247521003422e-07+0j) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.0474716555290867e-08+0j) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0004458535128840876+0j) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10] +
(-1.974225379397259e-08+0j) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441839+0j) [X0 Z1 Z2 X3 Y4 Y5] +
(-4.5233896774882e-07+0j) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.003484157300217884+0j) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-8.091637198491622e-07+0j) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10] +
(-0.005733569747311866+0j) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004684903388155178+0j) [X0 Z1 Z2 X3 Y6 Y7] +
(0.004668620318776299+0j) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11] +
(-7.189990974835964e-07+0j) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005114473831660386+0j) [X0 Z1 Z2 X3 X7 Z8 Z9 X10] +
(-7.560692464383436e-07+0j) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381025+0j) [X0 Z1 Z2 X3 Y8 Y9] +
(-0.001799219493663056+0j) [X0 Z1 Z2 X3 Y10 Y11] +
(-5.471647744336253e-07+0j) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13] +
(-5.287660624458577e-07+0j) [X0 Z1 Z2 X3 X11 X12] +
(-0.004575007626639216+0j) [X0 Z1 Z2 X3 Y12 Y13] +
(0.004424855449441839+0j) [X0 Z1 Z2 Y3 Y4 X5] +
(4.5233896774882e-07+0j) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.003484157300217884+0j) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.091637198491622e-07+0j) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.005733569747311866+0j) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.004684903388155178+0j) [X0 Z1 Z2 Y3 Y6 X7] +
(-0.004668620318776299+0j) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11] +
(7.189990974835964e-07+0j) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005114473831660386+0j) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10] +
(-7.560692464383436e-07+0j) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12] +
(0.008125251921381025+0j) [X0 Z1 Z2 Y3 Y8 X9] +
(0.001799219493663056+0j) [X0 Z1 Z2 Y3 Y10 X11] +
(5.471647744336253e-07+0j) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13] +
(-5.287660624458577e-07+0j) [X0 Z1 Z2 Y3 Y11 X12] +
(0.004575007626639216+0j) [X0 Z1 Z2 Y3 Y12 X13] +
(3.202076880217695e-06+0j) [X0 Z1 Z2 Z3 X4] +
(0.0008533856254125428+0j) [X0 Z1 Z2 Z3 X4 X5 Z6 X7] +
(0.0007870896771024346+0j) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0008533856254125428+0j) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7] +
(0.0007870896771024346+0j) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.2919694862987925e-07+0j) [X0 Z1 Z2 Z3 X4 Z5] +
(4.444597853972137e-07+0j) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10] +
(0.0011726348316441902+0j) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.6849150950019763e-07+0j) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10] +
(0.0022009640695004593+0j) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.8394209154769802e-07+0j) [X0 Z1 Z2 Z3 X4 Z6] +
(4.092250615828311e-07+0j) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11] +
(0.002394972639798019+0j) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.092250615828311e-07+0j) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11] +
(0.002394972639798019+0j) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.2362599614848903e-07+0j) [X0 Z1 Z2 Z3 X4 Z8] +
(8.649310135012233e-08+0j) [X0 Z1 Z2 Z3 X4 Z9] +
(0.0013038004788126917+0j) [X0 Z1 Z2 Z3 X4 X10 Z11 X12] +
(0.003989841456619284+0j) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12] +
(-6.73319774178594e-07+0j) [X0 Z1 Z2 Z3 X4 Z10] +
(0.00226196606248233+0j) [X0 Z1 Z2 Z3 X4 X11 Z12 X13] +
(0.00226196606248233+0j) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13] +
(-5.927453082318325e-07+0j) [X0 Z1 Z2 Z3 X4 Z11] +
(1.2393363216743296e-06+0j) [X0 Z1 Z2 Z3 X4 Z12] +
(9.306536651395927e-07+0j) [X0 Z1 Z2 Z3 X4 Z13] +
(-0.0010283292378562695+0j) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.002686040977806593+0j) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12] +
(-1.8394209154769805e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7] +
(-0.00019400857029755936+0j) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0012223378081538288+0j) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-2.3713289479836672e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9] +
(8.05744659467615e-08+0j) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11] +
(0.0017278753941369542+0j) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13] +
(-0.0009581655836696389+0j) [X0 Z1 Z2 Z3 Z4 X5 X11 X12] +
(-3.0868265653473707e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13] +
(1.8394209154769805e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7] +
(0.00019400857029755936+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0012223378081538288+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(2.3713289479836672e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9] +
(-8.05744659467615e-08+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11] +
(-0.0017278753941369542+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13] +
(-0.0009581655836696389+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12] +
(3.0868265653473707e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13] +
(0.04274327701378266+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6] +
(0.0005192743499487464+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(-1.85056419273964e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005192743499487464+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(-1.85056419273964e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0027790267990255263+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7] +
(0.004636976661182555+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8] +
(0.0012803060973496662+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9] +
(2.3120943051362158e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12] +
(1.0717282182831676e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12] +
(0.0053799371558393705+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10] +
(7.246974424943887e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13] +
(7.246974424943887e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13] +
(0.005241535382803844+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11] +
(0.004311038507914309+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12] +
(0.0010435246534907542+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13] +
(1.2004287493471362e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12] +
(-0.003356670563832889+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9] +
(-0.0001384017730355267+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11] +
(-6.17524620666072e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(-4.997018421699254e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12] +
(-0.0032675138544235554+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13] +
(0.003356670563832889+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9] +
(0.0001384017730355267+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11] +
(6.17524620666072e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(-4.997018421699254e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12] +
(0.0032675138544235554+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13] +
(0.003876470899336922+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(-7.540341413199794e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.003876470899336922+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(-7.540341413199794e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(0.07165035181002515+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0021413612231016075+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(0.004220813970046441+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(0.0012366478019245248+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(-0.0029841661681219156+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(0.0029841661681219156+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-1.3987009016677274e-05+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(8.949476486310668e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-6.876621658147772e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(-7.661347212964672e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(0.001532483523073022+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10] +
(-2.9045998846426017e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(0.005408954422409944+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10] +
(-1.0444941297842397e-06+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(0.004767272188278092+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10] +
(-8.105515037052217e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(0.005286546538226838+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10] +
(-9.956079229791857e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(0.001609531381721374+0j) [X0 Z1 Z2 Z3 Z4 X6] +
(-7.141625221157021e-05+0j) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10] +
(-2.6667317551369273e-07+0j) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0024629170071339165+0j) [X0 Z1 Z2 Z3 Z5 X6] +
(0.0007156734248908642+0j) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10] +
(-3.0767325322250015e-07+0j) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.60607186806513e-07+0j) [X0 Z1 Z2 X4] +
(0.003961560792496514+0j) [X0 Z1 Z2 Z4 Z5 X6] +
(0.00018787053389549422+0j) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.656930931334479e-07+0j) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.7379332624519022e-07+0j) [X0 Z1 Z3 X4] +
(0.0016676041811440443+0j) [X0 Z1 Z3 Z4 Z5 X6] +
(-0.001452884321416923+0j) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(4.6704023900289103e-07+0j) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.10433064780651416+0j) [X0 X2] +
(3.1174479461784894e-06+0j) [X0 Z2 Z3 X4] +
(0.04587947078129803+0j) [X0 Z2 Z3 Z4 Z5 X6] +
(0.0585919887338615+0j) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-1.1463061452549042e-05+0j) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.5707613292001126e-07+0j) [Y0 X1 X2 Z3 Z4 Y5] +
(0.005652620978017362+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7] +
(0.00882636851420981+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.7924939576213785e-06+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.5707613292001126e-07+0j) [Y0 X1 X3 Y4] +
(-0.005652620978017362+0j) [Y0 X1 X3 Z4 Z5 Y6] +
(-0.00882636851420981+0j) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.7924939576213785e-06+0j) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0027458364701868003+0j) [Y0 X1 X4 Y5] +
(2.447323128813329e-07+0j) [Y0 X1 X4 Z5 Z6 Y7] +
(7.867765103915637e-07+0j) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.003804066171728535+0j) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.447323128813329e-07+0j) [Y0 X1 X5 Y6] +
(-7.867765103915637e-07+0j) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.003804066171728535+0j) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.006888194352970544+0j) [Y0 X1 X6 Y7] +
(7.73503688058884e-05+0j) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11] +
(-1.7035783554601653e-07+0j) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.73503688058884e-05+0j) [Y0 X1 X7 Z8 Z9 Y10] +
(1.7035783554601653e-07+0j) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12] +
(0.006509361201177232+0j) [Y0 X1 X8 Y9] +
(0.007731425250775249+0j) [Y0 X1 X10 Y11] +
(-5.627851911228886e-07+0j) [Y0 X1 X10 Z11 Z12 Y13] +
(5.627851911228886e-07+0j) [Y0 X1 X11 Y12] +
(0.005283776488402962+0j) [Y0 X1 X12 Y13] +
(-3.5707613292001126e-07+0j) [Y0 Y1 X2 Z3 Z4 X5] +
(-0.005652620978017362+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7] +
(-0.00882636851420981+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.7924939576213785e-06+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.5707613292001126e-07+0j) [Y0 Y1 Y3 Y4] +
(-0.005652620978017362+0j) [Y0 Y1 Y3 Z4 Z5 Y6] +
(-0.00882636851420981+0j) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.7924939576213785e-06+0j) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.0027458364701868003+0j) [Y0 Y1 X4 X5] +
(-2.447323128813329e-07+0j) [Y0 Y1 X4 Z5 Z6 X7] +
(-7.867765103915637e-07+0j) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.003804066171728535+0j) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.447323128813329e-07+0j) [Y0 Y1 Y5 Y6] +
(-7.867765103915637e-07+0j) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.003804066171728535+0j) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.006888194352970544+0j) [Y0 Y1 X6 X7] +
(-7.73503688058884e-05+0j) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11] +
(1.7035783554601653e-07+0j) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.73503688058884e-05+0j) [Y0 Y1 Y7 Z8 Z9 Y10] +
(1.7035783554601653e-07+0j) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.006509361201177232+0j) [Y0 Y1 X8 X9] +
(-0.007731425250775249+0j) [Y0 Y1 X10 X11] +
(5.627851911228886e-07+0j) [Y0 Y1 X10 Z11 Z12 X13] +
(5.627851911228886e-07+0j) [Y0 Y1 Y11 Y12] +
(-0.005283776488402962+0j) [Y0 Y1 X12 X13] +
(-3.568247521003422e-07+0j) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10] +
(0.0004458535128840876+0j) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10] +
(-1.974225379397259e-08+0j) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.0474716555290867e-08+0j) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.1250703257977213+0j) [Y0 Z1 Y2] +
(-1.933241277190824e-07+0j) [Y0 Z1 Y2 X3 Z4 X5] +
(-0.00229395661135247+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7] +
(-0.0016407548553124176+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(3.013471458694432e-07+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.933241277190824e-07+0j) [Y0 Z1 Y2 Y3 Z4 Y5] +
(-0.00229395661135247+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7] +
(-0.0016407548553124176+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(3.013471458694432e-07+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0032769719312317116+0j) [Y0 Z1 Y2 Z3] +
(-1.38077814808055e-07+0j) [Y0 Z1 Y2 X4 Z5 X6] +
(-7.900128985792138e-07+0j) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.005348051582676631+0j) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.5510539175794388e-07+0j) [Y0 Z1 Y2 Y4 Z5 Y6] +
(-1.1468376506795559e-06+0j) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.007597464029770612+0j) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005530759218631553+0j) [Y0 Z1 Y2 Z4] +
(-1.38077814808055e-07+0j) [Y0 Z1 Y2 X5 Z6 X7] +
(-3.3767393083039356e-07+0j) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.001863894282458747+0j) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.38077814808055e-07+0j) [Y0 Z1 Y2 Y5 Z6 Y7] +
(-3.3767393083039356e-07+0j) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.001863894282458747+0j) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.001105903769189714+0j) [Y0 Z1 Y2 Z5] +
(0.005262642473076856+0j) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10] +
(-8.07430598545385e-07+0j) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.005708495985960944+0j) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10] +
-1.97422537940826e-08j [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-8.352332102613404e-07+0j) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.008032520918821388+0j) [Y0 Z1 Y2 Z6] +
(0.0005940221543005582+0j) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11] +
(-8.379773244239258e-08+0j) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005940221543005582+0j) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11] +
(-8.379773244239258e-08+0j) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0033476175306662095+0j) [Y0 Z1 Y2 Z7] +
(0.011055020596132127+0j) [Y0 Z1 Y2 Z8] +
(0.002929768674751102+0j) [Y0 Z1 Y2 Z9] +
(-6.556281914259837e-07+0j) [Y0 Z1 Y2 X10 Z11 X12] +
(-6.418291574347231e-07+0j) [Y0 Z1 Y2 Y10 Z11 Y12] +
(0.0035552901955043324+0j) [Y0 Z1 Y2 Z10] +
(-1.1076325599061215e-07+0j) [Y0 Z1 Y2 X11 Z12 X13] +
(-1.1076325599061215e-07+0j) [Y0 Z1 Y2 Y11 Z12 Y13] +
(0.0017560707018412765+0j) [Y0 Z1 Y2 Z11] +
(0.006901238249797335+0j) [Y0 Z1 Y2 Z12] +
(0.002326230623158119+0j) [Y0 Z1 Y2 Z13] +
(0.004424855449441839+0j) [Y0 Z1 Z2 X3 X4 Y5] +
(4.5233896774882e-07+0j) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.003484157300217884+0j) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-8.091637198491622e-07+0j) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.005733569747311866+0j) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.004684903388155178+0j) [Y0 Z1 Z2 X3 X6 Y7] +
(-0.004668620318776299+0j) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11] +
(7.189990974835964e-07+0j) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005114473831660386+0j) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10] +
(-7.560692464383436e-07+0j) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12] +
(0.008125251921381025+0j) [Y0 Z1 Z2 X3 X8 Y9] +
(0.001799219493663056+0j) [Y0 Z1 Z2 X3 X10 Y11] +
(5.471647744336253e-07+0j) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13] +
(-5.287660624458577e-07+0j) [Y0 Z1 Z2 X3 X11 Y12] +
(0.004575007626639216+0j) [Y0 Z1 Z2 X3 X12 Y13] +
(-0.004424855449441839+0j) [Y0 Z1 Z2 Y3 X4 X5] +
(-4.5233896774882e-07+0j) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.003484157300217884+0j) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.091637198491622e-07+0j) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.005733569747311866+0j) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004684903388155178+0j) [Y0 Z1 Z2 Y3 X6 X7] +
(0.004668620318776299+0j) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11] +
(-7.189990974835964e-07+0j) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005114473831660386+0j) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10] +
(-7.560692464383436e-07+0j) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381025+0j) [Y0 Z1 Z2 Y3 X8 X9] +
(-0.001799219493663056+0j) [Y0 Z1 Z2 Y3 X10 X11] +
(-5.471647744336253e-07+0j) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13] +
(-5.287660624458577e-07+0j) [Y0 Z1 Z2 Y3 Y11 Y12] +
(-0.004575007626639216+0j) [Y0 Z1 Z2 Y3 X12 X13] +
(-0.0010283292378562695+0j) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.002686040977806593+0j) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12] +
(3.202076880217695e-06+0j) [Y0 Z1 Z2 Z3 Y4] +
(0.0008533856254125428+0j) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7] +
(0.0007870896771024346+0j) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0008533856254125428+0j) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7] +
(0.0007870896771024346+0j) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.2919694862987925e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z5] +
(4.6849150950019763e-07+0j) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10] +
(0.0022009640695004593+0j) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.444597853972137e-07+0j) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10] +
(0.0011726348316441902+0j) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.8394209154769802e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z6] +
(4.092250615828311e-07+0j) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11] +
(0.002394972639798019+0j) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.092250615828311e-07+0j) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11] +
(0.002394972639798019+0j) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.2362599614848903e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z8] +
(8.649310135012233e-08+0j) [Y0 Z1 Z2 Z3 Y4 Z9] +
(0.003989841456619284+0j) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12] +
(0.0013038004788126917+0j) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12] +
(-6.73319774178594e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z10] +
(0.00226196606248233+0j) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13] +
(0.00226196606248233+0j) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13] +
(-5.927453082318325e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z11] +
(1.2393363216743296e-06+0j) [Y0 Z1 Z2 Z3 Y4 Z12] +
(9.306536651395927e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z13] +
(1.8394209154769805e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7] +
(0.00019400857029755936+0j) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0012223378081538288+0j) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(2.3713289479836672e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9] +
(-8.05744659467615e-08+0j) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11] +
(-0.0017278753941369542+0j) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13] +
(-0.0009581655836696389+0j) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12] +
(3.0868265653473707e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13] +
(-1.8394209154769805e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7] +
(-0.00019400857029755936+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0012223378081538288+0j) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-2.3713289479836672e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9] +
(8.05744659467615e-08+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11] +
(0.0017278753941369542+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13] +
(-0.0009581655836696389+0j) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12] +
(-3.0868265653473707e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13] +
(1.2004287493471362e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12] +
(0.04274327701378266+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6] +
(0.0005192743499487464+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(-1.85056419273964e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005192743499487464+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(-1.85056419273964e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0027790267990255263+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7] +
(0.004636976661182555+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8] +
(0.0012803060973496662+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9] +
(1.0717282182831676e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12] +
(2.3120943051362158e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12] +
(0.0053799371558393705+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10] +
(7.246974424943887e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13] +
(7.246974424943887e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13] +
(0.005241535382803844+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11] +
(0.004311038507914309+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12] +
(0.0010435246534907542+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13] +
(0.003356670563832889+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9] +
(0.0001384017730355267+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11] +
(6.17524620666072e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(-4.997018421699254e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12] +
(0.0032675138544235554+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13] +
(-0.003356670563832889+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9] +
(-0.0001384017730355267+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11] +
(-6.17524620666072e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(-4.997018421699254e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12] +
(-0.0032675138544235554+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13] +
(0.003876470899336922+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(-7.540341413199794e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.003876470899336922+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(-7.540341413199794e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(0.07165035181002515+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0021413612231016075+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(0.004220813970046441+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(0.0012366478019245248+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(0.0029841661681219156+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-0.0029841661681219156+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-1.3987009016677274e-05+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(8.949476486310668e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-6.876621658147772e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(-7.661347212964672e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(0.001532483523073022+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10] +
(-2.9045998846426017e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(0.005408954422409944+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10] +
(-1.0444941297842397e-06+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(0.004767272188278092+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10] +
(-8.105515037052217e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(0.005286546538226838+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10] +
(-9.956079229791857e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.001609531381721374+0j) [Y0 Z1 Z2 Z3 Z4 Y6] +
(-7.141625221157021e-05+0j) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10] +
(-2.6667317551369273e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0024629170071339165+0j) [Y0 Z1 Z2 Z3 Z5 Y6] +
(0.0007156734248908642+0j) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10] +
(-3.0767325322250015e-07+0j) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(3.60607186806513e-07+0j) [Y0 Z1 Z2 Y4] +
(0.003961560792496514+0j) [Y0 Z1 Z2 Z4 Z5 Y6] +
(0.00018787053389549422+0j) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.656930931334479e-07+0j) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.7379332624519022e-07+0j) [Y0 Z1 Z3 Y4] +
(0.0016676041811440443+0j) [Y0 Z1 Z3 Z4 Z5 Y6] +
(-0.001452884321416923+0j) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(4.6704023900289103e-07+0j) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.10433064780651416+0j) [Y0 Y2] +
(3.1174479461784894e-06+0j) [Y0 Z2 Z3 Y4] +
(0.04587947078129803+0j) [Y0 Z2 Z3 Z4 Z5 Y6] +
(0.0585919887338615+0j) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-1.1463061452549042e-05+0j) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.10433064780651415+0j) [Z0 X1 Z2 X3] +
(3.1174479461784894e-06+0j) [Z0 X1 Z2 Z3 Z4 X5] +
(0.04587947078129803+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7] +
(0.0585919887338615+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-1.1463061452549042e-05+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.10433064780651415+0j) [Z0 Y1 Z2 Y3] +
(3.1174479461784894e-06+0j) [Z0 Y1 Z2 Z3 Z4 Y5] +
(0.04587947078129803+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7] +
(0.0585919887338615+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.1463061452549042e-05+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.1861763734860487+0j) [Z0 Z1] +
(-8.337746754710025e-07+0j) [Z0 X2 Z3 X4] +
(-0.027115036845273294+0j) [Z0 X2 Z3 Z4 Z5 X6] +
(-0.06752385099213996+0j) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.4017109734571095e-05+0j) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-8.337746754710025e-07+0j) [Z0 Y2 Z3 Y4] +
(-0.027115036845273294+0j) [Z0 Y2 Z3 Z4 Z5 Y6] +
(-0.06752385099213996+0j) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.4017109734571095e-05+0j) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.1908508083910137e-06+0j) [Z0 X3 Z4 X5] +
(-0.032767657823290657+0j) [Z0 X3 Z4 Z5 Z6 X7] +
(-0.07635021950634978+0j) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.5809603692192475e-05+0j) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.1908508083910137e-06+0j) [Z0 Y3 Z4 Y5] +
(-0.032767657823290657+0j) [Z0 Y3 Z4 Z5 Z6 Y7] +
(-0.07635021950634978+0j) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.5809603692192475e-05+0j) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.0993492436103793e-06+0j) [Z0 X4 Z5 X6] +
(-1.531680879459665e-05+0j) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-3.0993492436103793e-06+0j) [Z0 Y4 Z5 Y6] +
(-1.531680879459665e-05+0j) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(0.19661770890342117+0j) [Z0 Z4] +
(-3.3440815564917123e-06+0j) [Z0 X5 Z6 X7] +
(-1.6103585304988214e-05+0j) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-3.3440815564917123e-06+0j) [Z0 Y5 Z6 Y7] +
(-1.6103585304988214e-05+0j) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.19936354537360795+0j) [Z0 Z5] +
(0.056084681246613796+0j) [Z0 X6 Z7 Z8 Z9 X10] +
(-6.652209668939035e-06+0j) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.056084681246613796+0j) [Z0 Y6 Z7 Z8 Z9 Y10] +
(-6.652209668939035e-06+0j) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.24164663936017142+0j) [Z0 Z6] +
(0.05600733087780791+0j) [Z0 X7 Z8 Z9 Z10 X11] +
(-6.481851833393019e-06+0j) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.05600733087780791+0j) [Z0 Y7 Z8 Z9 Z10 Y11] +
(-6.481851833393019e-06+0j) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.24853483371314194+0j) [Z0 Z7] +
(-2.17766460482917e-06+0j) [Z0 X10 Z11 X12] +
(-2.17766460482917e-06+0j) [Z0 Y10 Z11 Y12] +
(0.19299723935364252+0j) [Z0 Z10] +
(-1.6148794137062814e-06+0j) [Z0 X11 Z12 X13] +
(-1.6148794137062814e-06+0j) [Z0 Y11 Z12 Y13] +
(0.20072866460441774+0j) [Z0 Z11] +
(0.21102659849791525+0j) [Z0 Z12] +
(0.21631037498631822+0j) [Z0 Z13] +
(1.933241277190824e-07+0j) [X1 X2 Y3 Y4] +
(0.00229395661135247+0j) [X1 X2 Y3 Z4 Z5 Y6] +
(0.0016407548553124176+0j) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-3.013471458694432e-07+0j) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004424855449441839+0j) [X1 X2 X4 X5] +
(-8.091637198491622e-07+0j) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.005733569747311866+0j) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-4.5233896774882e-07+0j) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.003484157300217884+0j) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004684903388155179+0j) [X1 X2 X6 X7] +
(0.005114473831660386+0j) [X1 X2 X6 Z7 Z8 Z9 Z10 X11] +
(-7.560692464383436e-07+0j) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.004668620318776299+0j) [X1 X2 Y7 Z8 Z9 Y10] +
(-7.189990974835964e-07+0j) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381025+0j) [X1 X2 X8 X9] +
(-0.001799219493663056+0j) [X1 X2 X10 X11] +
(-5.287660624458577e-07+0j) [X1 X2 X10 Z11 Z12 X13] +
(-5.471647744336253e-07+0j) [X1 X2 Y11 Y12] +
(-0.004575007626639217+0j) [X1 X2 X12 X13] +
(-1.933241277190824e-07+0j) [X1 Y2 Y3 X4] +
(-0.00229395661135247+0j) [X1 Y2 Y3 Z4 Z5 X6] +
(-0.0016407548553124176+0j) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(3.013471458694432e-07+0j) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441839+0j) [X1 Y2 Y4 X5] +
(-8.091637198491622e-07+0j) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.005733569747311866+0j) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.5233896774882e-07+0j) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10] +
(0.003484157300217884+0j) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004684903388155179+0j) [X1 Y2 Y6 X7] +
(0.005114473831660386+0j) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11] +
(-7.560692464383436e-07+0j) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.004668620318776299+0j) [X1 Y2 Y7 Z8 Z9 X10] +
(7.189990974835964e-07+0j) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381025+0j) [X1 Y2 Y8 X9] +
(-0.001799219493663056+0j) [X1 Y2 Y10 X11] +
(-5.287660624458577e-07+0j) [X1 Y2 Y10 Z11 Z12 X13] +
(5.471647744336253e-07+0j) [X1 Y2 Y11 X12] +
(-0.004575007626639217+0j) [X1 Y2 Y12 X13] +
(0.12507032579772134+0j) [X1 Z2 X3] +
(-1.38077814808055e-07+0j) [X1 Z2 X3 X4 Z5 X6] +
(-3.3767393083039356e-07+0j) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.001863894282458747+0j) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.38077814808055e-07+0j) [X1 Z2 X3 Y4 Z5 Y6] +
(-3.3767393083039356e-07+0j) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.001863894282458747+0j) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.001105903769189714+0j) [X1 Z2 X3 Z4] +
(-1.5510539175794388e-07+0j) [X1 Z2 X3 X5 Z6 X7] +
(-1.1468376506795559e-06+0j) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.007597464029770612+0j) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.38077814808055e-07+0j) [X1 Z2 X3 Y5 Z6 Y7] +
(-7.900128985792138e-07+0j) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.005348051582676631+0j) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005530759218631553+0j) [X1 Z2 X3 Z5] +
(0.0005940221543005582+0j) [X1 Z2 X3 X6 Z7 Z8 Z9 X10] +
(-8.379773244239258e-08+0j) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0005940221543005582+0j) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10] +
(-8.379773244239258e-08+0j) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0033476175306662095+0j) [X1 Z2 X3 Z6] +
(0.005708495985960944+0j) [X1 Z2 X3 X7 Z8 Z9 Z10 X11] +
(-8.352332102613404e-07+0j) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
1.97422537940826e-08j [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005262642473076856+0j) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11] +
(-8.07430598545385e-07+0j) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.008032520918821388+0j) [X1 Z2 X3 Z7] +
(0.002929768674751102+0j) [X1 Z2 X3 Z8] +
(0.011055020596132127+0j) [X1 Z2 X3 Z9] +
(-1.1076325599061215e-07+0j) [X1 Z2 X3 X10 Z11 X12] +
(-1.1076325599061215e-07+0j) [X1 Z2 X3 Y10 Z11 Y12] +
(0.0017560707018412765+0j) [X1 Z2 X3 Z10] +
(-6.418291574347231e-07+0j) [X1 Z2 X3 X11 Z12 X13] +
(-6.556281914259837e-07+0j) [X1 Z2 X3 Y11 Z12 Y13] +
(0.0035552901955043324+0j) [X1 Z2 X3 Z11] +
(0.002326230623158119+0j) [X1 Z2 X3 Z12] +
(0.006901238249797335+0j) [X1 Z2 X3 Z13] +
(-3.568247521003422e-07+0j) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.0474716555290867e-08+0j) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0004458535128840876+0j) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11] +
(-1.974225379397259e-08+0j) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0008533856254125428+0j) [X1 Z2 Z3 X4 Y5 Y6] +
(-0.0007870896771024347+0j) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10] +
(-1.83942091547698e-07+0j) [X1 Z2 Z3 X4 X6 X7] +
(-0.0012223378081538288+0j) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.00019400857029755936+0j) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12] +
(-2.371328947983667e-07+0j) [X1 Z2 Z3 X4 X8 X9] +
(8.05744659467615e-08+0j) [X1 Z2 Z3 X4 X10 X11] +
(-0.0009581655836696389+0j) [X1 Z2 Z3 X4 X10 Z11 Z12 X13] +
(0.0017278753941369542+0j) [X1 Z2 Z3 X4 Y11 Y12] +
(-3.0868265653473707e-07+0j) [X1 Z2 Z3 X4 X12 X13] +
(0.0008533856254125428+0j) [X1 Z2 Z3 Y4 Y5 X6] +
(0.0007870896771024347+0j) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10] +
(-1.83942091547698e-07+0j) [X1 Z2 Z3 Y4 Y6 X7] +
(-0.0012223378081538288+0j) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.00019400857029755936+0j) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12] +
(-2.371328947983667e-07+0j) [X1 Z2 Z3 Y4 Y8 X9] +
(8.05744659467615e-08+0j) [X1 Z2 Z3 Y4 Y10 X11] +
(-0.0009581655836696389+0j) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13] +
(-0.0017278753941369542+0j) [X1 Z2 Z3 Y4 Y11 X12] +
(-3.0868265653473707e-07+0j) [X1 Z2 Z3 Y4 Y12 X13] +
(3.202076880217695e-06+0j) [X1 Z2 Z3 Z4 X5] +
(4.092250615828311e-07+0j) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10] +
(0.002394972639798019+0j) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.092250615828311e-07+0j) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10] +
(0.002394972639798019+0j) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.444597853972137e-07+0j) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11] +
(0.0011726348316441902+0j) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.6849150950019763e-07+0j) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11] +
(0.0022009640695004593+0j) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.8394209154769802e-07+0j) [X1 Z2 Z3 Z4 X5 Z7] +
(8.649310135012233e-08+0j) [X1 Z2 Z3 Z4 X5 Z8] +
(3.2362599614848903e-07+0j) [X1 Z2 Z3 Z4 X5 Z9] +
(0.00226196606248233+0j) [X1 Z2 Z3 Z4 X5 X10 Z11 X12] +
(0.00226196606248233+0j) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12] +
(-5.927453082318325e-07+0j) [X1 Z2 Z3 Z4 X5 Z10] +
(0.0013038004788126917+0j) [X1 Z2 Z3 Z4 X5 X11 Z12 X13] +
(0.003989841456619284+0j) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13] +
(-6.73319774178594e-07+0j) [X1 Z2 Z3 Z4 X5 Z11] +
(9.306536651395927e-07+0j) [X1 Z2 Z3 Z4 X5 Z12] +
(1.2393363216743296e-06+0j) [X1 Z2 Z3 Z4 X5 Z13] +
(-0.0010283292378562695+0j) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.002686040977806593+0j) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13] +
(-0.0005192743499487464+0j) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10] +
(1.85056419273964e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.003356670563832889+0j) [X1 Z2 Z3 Z4 Z5 X6 X8 X9] +
(-0.0001384017730355267+0j) [X1 Z2 Z3 Z4 Z5 X6 X10 X11] +
(-4.997018421699254e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13] +
(-6.17524620666072e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12] +
(-0.0032675138544235554+0j) [X1 Z2 Z3 Z4 Z5 X6 X12 X13] +
(0.0005192743499487464+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10] +
(-1.85056419273964e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.003356670563832889+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9] +
(-0.0001384017730355267+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11] +
(-4.997018421699254e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13] +
(6.17524620666072e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12] +
(-0.0032675138544235554+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13] +
(0.04274327701378266+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7] +
(0.0012803060973496662+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8] +
(0.004636976661182555+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9] +
(7.246974424943887e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12] +
(7.246974424943887e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12] +
(0.005241535382803844+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10] +
(2.3120943051362158e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13] +
(1.0717282182831676e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13] +
(0.0053799371558393705+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11] +
(0.0010435246534907542+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12] +
(0.004311038507914309+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13] +
(1.2004287493471362e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13] +
(-0.003876470899336922+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10] +
(7.540341413199794e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(0.003876470899336922+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10] +
(-7.540341413199794e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(-0.0029841661681219156+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-0.0029841661681219156+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(0.07165035181002512+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0012366478019245248+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(0.004220813970046441+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(-1.3987009016677277e-05+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(8.949476486310667e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(-7.661347212964672e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(-0.0021413612231016075+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11] +
(-6.876621658147772e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(0.005408954422409944+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11] +
(-1.0444941297842397e-06+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(0.001532483523073022+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11] +
(-2.9045998846426017e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(0.005286546538226838+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11] +
(-9.956079229791857e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0027790267990255263+0j) [X1 Z2 Z3 Z4 Z5 X7] +
(0.004767272188278092+0j) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11] +
(-8.105515037052217e-07+0j) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0024629170071339165+0j) [X1 Z2 Z3 Z4 Z6 X7] +
(0.0007156734248908642+0j) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11] +
(-3.0767325322250015e-07+0j) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2919694862987925e-07+0j) [X1 Z2 Z3 X5] +
(0.001609531381721374+0j) [X1 Z2 Z3 Z5 Z6 X7] +
(-7.141625221157021e-05+0j) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-2.6667317551369273e-07+0j) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.7379332624519022e-07+0j) [X1 Z2 Z4 X5] +
(0.0016676041811440443+0j) [X1 Z2 Z4 Z5 Z6 X7] +
(-0.001452884321416923+0j) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(4.6704023900289103e-07+0j) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0032769719312317116+0j) [X1 X3] +
(3.60607186806513e-07+0j) [X1 Z3 Z4 X5] +
(0.003961560792496514+0j) [X1 Z3 Z4 Z5 Z6 X7] +
(0.00018787053389549422+0j) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.656930931334479e-07+0j) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.933241277190824e-07+0j) [Y1 X2 X3 Y4] +
(-0.00229395661135247+0j) [Y1 X2 X3 Z4 Z5 Y6] +
(-0.0016407548553124176+0j) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(3.013471458694432e-07+0j) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004424855449441839+0j) [Y1 X2 X4 Y5] +
(-8.091637198491622e-07+0j) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.005733569747311866+0j) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(4.5233896774882e-07+0j) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10] +
(0.003484157300217884+0j) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004684903388155179+0j) [Y1 X2 X6 Y7] +
(0.005114473831660386+0j) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11] +
(-7.560692464383436e-07+0j) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.004668620318776299+0j) [Y1 X2 X7 Z8 Z9 Y10] +
(7.189990974835964e-07+0j) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381025+0j) [Y1 X2 X8 Y9] +
(-0.001799219493663056+0j) [Y1 X2 X10 Y11] +
(-5.287660624458577e-07+0j) [Y1 X2 X10 Z11 Z12 Y13] +
(5.471647744336253e-07+0j) [Y1 X2 X11 Y12] +
(-0.004575007626639217+0j) [Y1 X2 X12 Y13] +
(1.933241277190824e-07+0j) [Y1 Y2 X3 X4] +
(0.00229395661135247+0j) [Y1 Y2 X3 Z4 Z5 X6] +
(0.0016407548553124176+0j) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-3.013471458694432e-07+0j) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441839+0j) [Y1 Y2 Y4 Y5] +
(-8.091637198491622e-07+0j) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.005733569747311866+0j) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-4.5233896774882e-07+0j) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10] +
(-0.003484157300217884+0j) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004684903388155179+0j) [Y1 Y2 Y6 Y7] +
(0.005114473831660386+0j) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11] +
(-7.560692464383436e-07+0j) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.004668620318776299+0j) [Y1 Y2 X7 Z8 Z9 X10] +
(-7.189990974835964e-07+0j) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381025+0j) [Y1 Y2 Y8 Y9] +
(-0.001799219493663056+0j) [Y1 Y2 Y10 Y11] +
(-5.287660624458577e-07+0j) [Y1 Y2 Y10 Z11 Z12 Y13] +
(-5.471647744336253e-07+0j) [Y1 Y2 X11 X12] +
(-0.004575007626639217+0j) [Y1 Y2 Y12 Y13] +
(-3.568247521003422e-07+0j) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.0004458535128840876+0j) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11] +
(-1.974225379397259e-08+0j) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.0474716555290867e-08+0j) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.12507032579772134+0j) [Y1 Z2 Y3] +
(-1.38077814808055e-07+0j) [Y1 Z2 Y3 X4 Z5 X6] +
(-3.3767393083039356e-07+0j) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.001863894282458747+0j) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.38077814808055e-07+0j) [Y1 Z2 Y3 Y4 Z5 Y6] +
(-3.3767393083039356e-07+0j) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.001863894282458747+0j) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.001105903769189714+0j) [Y1 Z2 Y3 Z4] +
(-1.38077814808055e-07+0j) [Y1 Z2 Y3 X5 Z6 X7] +
(-7.900128985792138e-07+0j) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.005348051582676631+0j) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.5510539175794388e-07+0j) [Y1 Z2 Y3 Y5 Z6 Y7] +
(-1.1468376506795559e-06+0j) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.007597464029770612+0j) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005530759218631553+0j) [Y1 Z2 Y3 Z5] +
(0.0005940221543005582+0j) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10] +
(-8.379773244239258e-08+0j) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0005940221543005582+0j) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10] +
(-8.379773244239258e-08+0j) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0033476175306662095+0j) [Y1 Z2 Y3 Z6] +
(0.005262642473076856+0j) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11] +
(-8.07430598545385e-07+0j) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005708495985960944+0j) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11] +
-1.97422537940826e-08j [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.352332102613404e-07+0j) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.008032520918821388+0j) [Y1 Z2 Y3 Z7] +
(0.002929768674751102+0j) [Y1 Z2 Y3 Z8] +
(0.011055020596132127+0j) [Y1 Z2 Y3 Z9] +
(-1.1076325599061215e-07+0j) [Y1 Z2 Y3 X10 Z11 X12] +
(-1.1076325599061215e-07+0j) [Y1 Z2 Y3 Y10 Z11 Y12] +
(0.0017560707018412765+0j) [Y1 Z2 Y3 Z10] +
(-6.556281914259837e-07+0j) [Y1 Z2 Y3 X11 Z12 X13] +
(-6.418291574347231e-07+0j) [Y1 Z2 Y3 Y11 Z12 Y13] +
(0.0035552901955043324+0j) [Y1 Z2 Y3 Z11] +
(0.002326230623158119+0j) [Y1 Z2 Y3 Z12] +
(0.006901238249797335+0j) [Y1 Z2 Y3 Z13] +
(0.0008533856254125428+0j) [Y1 Z2 Z3 X4 X5 Y6] +
(0.0007870896771024347+0j) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10] +
(-1.83942091547698e-07+0j) [Y1 Z2 Z3 X4 X6 Y7] +
(-0.0012223378081538288+0j) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.00019400857029755936+0j) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12] +
(-2.371328947983667e-07+0j) [Y1 Z2 Z3 X4 X8 Y9] +
(8.05744659467615e-08+0j) [Y1 Z2 Z3 X4 X10 Y11] +
(-0.0009581655836696389+0j) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13] +
(-0.0017278753941369542+0j) [Y1 Z2 Z3 X4 X11 Y12] +
(-3.0868265653473707e-07+0j) [Y1 Z2 Z3 X4 X12 Y13] +
(-0.0008533856254125428+0j) [Y1 Z2 Z3 Y4 X5 X6] +
(-0.0007870896771024347+0j) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10] +
(-1.83942091547698e-07+0j) [Y1 Z2 Z3 Y4 Y6 Y7] +
(-0.0012223378081538288+0j) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.00019400857029755936+0j) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12] +
(-2.371328947983667e-07+0j) [Y1 Z2 Z3 Y4 Y8 Y9] +
(8.05744659467615e-08+0j) [Y1 Z2 Z3 Y4 Y10 Y11] +
(-0.0009581655836696389+0j) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13] +
(0.0017278753941369542+0j) [Y1 Z2 Z3 Y4 X11 X12] +
(-3.0868265653473707e-07+0j) [Y1 Z2 Z3 Y4 Y12 Y13] +
(-0.0010283292378562695+0j) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.002686040977806593+0j) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13] +
(3.202076880217695e-06+0j) [Y1 Z2 Z3 Z4 Y5] +
(4.092250615828311e-07+0j) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10] +
(0.002394972639798019+0j) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.092250615828311e-07+0j) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10] +
(0.002394972639798019+0j) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.6849150950019763e-07+0j) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11] +
(0.0022009640695004593+0j) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.444597853972137e-07+0j) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11] +
(0.0011726348316441902+0j) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.8394209154769802e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z7] +
(8.649310135012233e-08+0j) [Y1 Z2 Z3 Z4 Y5 Z8] +
(3.2362599614848903e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z9] +
(0.00226196606248233+0j) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12] +
(0.00226196606248233+0j) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12] +
(-5.927453082318325e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z10] +
(0.003989841456619284+0j) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13] +
(0.0013038004788126917+0j) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13] +
(-6.73319774178594e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z11] +
(9.306536651395927e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z12] +
(1.2393363216743296e-06+0j) [Y1 Z2 Z3 Z4 Y5 Z13] +
(0.0005192743499487464+0j) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10] +
(-1.85056419273964e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.003356670563832889+0j) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9] +
(-0.0001384017730355267+0j) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11] +
(-4.997018421699254e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13] +
(6.17524620666072e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12] +
(-0.0032675138544235554+0j) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13] +
(-0.0005192743499487464+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10] +
(1.85056419273964e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-0.003356670563832889+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9] +
(-0.0001384017730355267+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11] +
(-4.997018421699254e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13] +
(-6.17524620666072e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12] +
(-0.0032675138544235554+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13] +
(1.2004287493471362e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13] +
(0.04274327701378266+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7] +
(0.0012803060973496662+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8] +
(0.004636976661182555+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9] +
(7.246974424943887e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12] +
(7.246974424943887e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12] +
(0.005241535382803844+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10] +
(1.0717282182831676e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13] +
(2.3120943051362158e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13] +
(0.0053799371558393705+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11] +
(0.0010435246534907542+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12] +
(0.004311038507914309+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13] +
(0.003876470899336922+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10] +
(-7.540341413199794e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-0.003876470899336922+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10] +
(7.540341413199794e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-0.0029841661681219156+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(-0.0029841661681219156+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(0.07165035181002512+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.0012366478019245248+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(0.004220813970046441+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(-1.3987009016677277e-05+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(8.949476486310667e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(-7.661347212964672e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(-0.0021413612231016075+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11] +
(-6.876621658147772e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(0.005408954422409944+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11] +
(-1.0444941297842397e-06+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(0.001532483523073022+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11] +
(-2.9045998846426017e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(0.005286546538226838+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11] +
(-9.956079229791857e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0027790267990255263+0j) [Y1 Z2 Z3 Z4 Z5 Y7] +
(0.004767272188278092+0j) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11] +
(-8.105515037052217e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0024629170071339165+0j) [Y1 Z2 Z3 Z4 Z6 Y7] +
(0.0007156734248908642+0j) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11] +
(-3.0767325322250015e-07+0j) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.2919694862987925e-07+0j) [Y1 Z2 Z3 Y5] +
(0.001609531381721374+0j) [Y1 Z2 Z3 Z5 Z6 Y7] +
(-7.141625221157021e-05+0j) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-2.6667317551369273e-07+0j) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.7379332624519022e-07+0j) [Y1 Z2 Z4 Y5] +
(0.0016676041811440443+0j) [Y1 Z2 Z4 Z5 Z6 Y7] +
(-0.001452884321416923+0j) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(4.6704023900289103e-07+0j) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0032769719312317116+0j) [Y1 Y3] +
(3.60607186806513e-07+0j) [Y1 Z3 Z4 Y5] +
(0.003961560792496514+0j) [Y1 Z3 Z4 Z5 Z6 Y7] +
(0.00018787053389549422+0j) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.656930931334479e-07+0j) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.1908508083910137e-06+0j) [Z1 X2 Z3 X4] +
(-0.032767657823290657+0j) [Z1 X2 Z3 Z4 Z5 X6] +
(-0.07635021950634978+0j) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.5809603692192475e-05+0j) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.1908508083910137e-06+0j) [Z1 Y2 Z3 Y4] +
(-0.032767657823290657+0j) [Z1 Y2 Z3 Z4 Z5 Y6] +
(-0.07635021950634978+0j) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.5809603692192475e-05+0j) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-8.337746754710025e-07+0j) [Z1 X3 Z4 X5] +
(-0.027115036845273294+0j) [Z1 X3 Z4 Z5 Z6 X7] +
(-0.06752385099213996+0j) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.4017109734571095e-05+0j) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.337746754710025e-07+0j) [Z1 Y3 Z4 Y5] +
(-0.027115036845273294+0j) [Z1 Y3 Z4 Z5 Z6 Y7] +
(-0.06752385099213996+0j) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.4017109734571095e-05+0j) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.3440815564917123e-06+0j) [Z1 X4 Z5 X6] +
(-1.6103585304988214e-05+0j) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-3.3440815564917123e-06+0j) [Z1 Y4 Z5 Y6] +
(-1.6103585304988214e-05+0j) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(0.19936354537360795+0j) [Z1 Z4] +
(-3.0993492436103793e-06+0j) [Z1 X5 Z6 X7] +
(-1.531680879459665e-05+0j) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-3.0993492436103793e-06+0j) [Z1 Y5 Z6 Y7] +
(-1.531680879459665e-05+0j) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.19661770890342117+0j) [Z1 Z5] +
(0.05600733087780791+0j) [Z1 X6 Z7 Z8 Z9 X10] +
(-6.481851833393019e-06+0j) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.05600733087780791+0j) [Z1 Y6 Z7 Z8 Z9 Y10] +
(-6.481851833393019e-06+0j) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.24853483371314194+0j) [Z1 Z6] +
(0.056084681246613796+0j) [Z1 X7 Z8 Z9 Z10 X11] +
(-6.652209668939035e-06+0j) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.056084681246613796+0j) [Z1 Y7 Z8 Z9 Z10 Y11] +
(-6.652209668939035e-06+0j) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.24164663936017142+0j) [Z1 Z7] +
(-1.6148794137062814e-06+0j) [Z1 X10 Z11 X12] +
(-1.6148794137062814e-06+0j) [Z1 Y10 Z11 Y12] +
(0.20072866460441774+0j) [Z1 Z10] +
(-2.17766460482917e-06+0j) [Z1 X11 Z12 X13] +
(-2.17766460482917e-06+0j) [Z1 Y11 Z12 Y13] +
(0.19299723935364252+0j) [Z1 Z11] +
(0.21631037498631822+0j) [Z1 Z12] +
(0.21102659849791525+0j) [Z1 Z13] +
(-0.035839567953353364+0j) [X2 X3 Y4 Y5] +
(-2.1990516183257419e-07+0j) [X2 X3 Y4 Z5 Z6 Y7] +
(-2.3609563202103618e-06+0j) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.010311482489831882+0j) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.1990516183257419e-07+0j) [X2 X3 X5 X6] +
(-2.3609563202103618e-06+0j) [X2 X3 X5 Z6 Z7 Z8 Z9 X10] +
(-0.010311482489831882+0j) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.031143817988967086+0j) [X2 X3 Y6 Y7] +
(0.005368659358109605+0j) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11] +
(9.209350651005051e-08+0j) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005368659358109605+0j) [X2 X3 X7 Z8 Z9 X10] +
(9.209350651005051e-08+0j) [X2 X3 X7 Z8 Z9 Z10 Z11 X12] +
(-0.03619412355904263+0j) [X2 X3 Y8 Y9] +
(-0.025384657508457344+0j) [X2 X3 Y10 Y11] +
(2.172669101416568e-06+0j) [X2 X3 Y10 Z11 Z12 Y13] +
(2.1726691014165674e-06+0j) [X2 X3 X11 X12] +
(-0.01557720806397646+0j) [X2 X3 Y12 Y13] +
(0.035839567953353364+0j) [X2 Y3 Y4 X5] +
(2.1990516183257419e-07+0j) [X2 Y3 Y4 Z5 Z6 X7] +
(2.3609563202103618e-06+0j) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.010311482489831882+0j) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.1990516183257419e-07+0j) [X2 Y3 Y5 X6] +
(-2.3609563202103618e-06+0j) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.010311482489831882+0j) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.031143817988967086+0j) [X2 Y3 Y6 X7] +
(-0.005368659358109605+0j) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11] +
(-9.209350651005051e-08+0j) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005368659358109605+0j) [X2 Y3 Y7 Z8 Z9 X10] +
(9.209350651005051e-08+0j) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12] +
(0.03619412355904263+0j) [X2 Y3 Y8 X9] +
(0.025384657508457344+0j) [X2 Y3 Y10 X11] +
(-2.172669101416568e-06+0j) [X2 Y3 Y10 Z11 Z12 X13] +
(2.1726691014165674e-06+0j) [X2 Y3 Y11 X12] +
(0.01557720806397646+0j) [X2 Y3 Y12 X13] +
(-3.887051673487614e-06+0j) [X2 Z3 X4] +
(-0.005143391768825178+0j) [X2 Z3 X4 X5 Z6 X7] +
(-0.009841749246962636+0j) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.988511706263365e-06+0j) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.005143391768825178+0j) [X2 Z3 X4 Y5 Z6 Y7] +
(-0.009841749246962636+0j) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.988511706263365e-06+0j) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.764994119377161e-07+0j) [X2 Z3 X4 Z5] +
(1.6893489513559773e-06+0j) [X2 Z3 X4 X6 Z7 Z8 Z9 X10] +
(0.010757563953908976+0j) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.5371780943628625e-08+0j) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10] +
(4.2055484112183544e-05+0j) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.5935343911031914e-07+0j) [X2 Z3 X4 Z6] +
(3.211842018880629e-06+0j) [X2 Z3 X4 X7 Z8 Z9 Z10 X11] +
(0.019299560579363797+0j) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.211842018880629e-06+0j) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11] +
(0.019299560579363797+0j) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.195489009919614e-06+0j) [X2 Z3 X4 Z7] +
(2.186842377549152e-07+0j) [X2 Z3 X4 Z8] +
(-5.770052995237359e-07+0j) [X2 Z3 X4 Z9] +
(0.015588250102380154+0j) [X2 Z3 X4 X10 Z11 X12] +
(0.005324835234221671+0j) [X2 Z3 X4 Y10 Z11 Y12] +
(-3.158656431838776e-06+0j) [X2 Z3 X4 Z10] +
(0.024353077678068828+0j) [X2 Z3 X4 X11 Z12 X13] +
(0.024353077678068828+0j) [X2 Z3 X4 Y11 Z12 Y13] +
(-7.801707500084142e-06+0j) [X2 Z3 X4 Z11] +
(3.5390541842629702e-06+0j) [X2 Z3 X4 Z12] +
(8.814937306085744e-06+0j) [X2 Z3 X4 Z13] +
(1.6288532434309747e-06+0j) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10] +
(0.010715508469796796+0j) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.010263414868158481+0j) [X2 Z3 Y4 Y10 Z11 X12] +
(-1.454842449029933e-06+0j) [X2 Z3 Z4 X5 Y6 Y7] +
(-3.151346310955626e-06+0j) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11] +
(-0.019257505095251613+0j) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.5224930675246511e-06+0j) [X2 Z3 Z4 X5 X7 Z8 Z9 X10] +
(-0.00854199662545482+0j) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-7.956895372786512e-07+0j) [X2 Z3 Z4 X5 Y8 Y9] +
(-4.6430510682453656e-06+0j) [X2 Z3 Z4 X5 Y10 Y11] +
(-0.019028242443847158+0j) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13] +
(-0.008764827575688678+0j) [X2 Z3 Z4 X5 X11 X12] +
(5.275883121822774e-06+0j) [X2 Z3 Z4 X5 Y12 Y13] +
(1.454842449029933e-06+0j) [X2 Z3 Z4 Y5 Y6 X7] +
(3.151346310955626e-06+0j) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11] +
(0.019257505095251613+0j) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.5224930675246511e-06+0j) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10] +
(-0.00854199662545482+0j) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(7.956895372786512e-07+0j) [X2 Z3 Z4 Y5 Y8 X9] +
(4.6430510682453656e-06+0j) [X2 Z3 Z4 Y5 Y10 X11] +
(0.019028242443847158+0j) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13] +
(-0.008764827575688678+0j) [X2 Z3 Z4 Y5 Y11 X12] +
(-5.275883121822774e-06+0j) [X2 Z3 Z4 Y5 Y12 X13] +
(-0.12133276911042379+0j) [X2 Z3 Z4 Z5 X6] +
(-0.008469978791024112+0j) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(2.686381545025142e-07+0j) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.008469978791024112+0j) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(2.686381545025142e-07+0j) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.02599617759802112+0j) [X2 Z3 Z4 Z5 X6 Z7] +
(-0.005805188989826982+0j) [X2 Z3 Z4 Z5 X6 Z8] +
(-0.017561202409646245+0j) [X2 Z3 Z4 Z5 X6 Z9] +
(-7.988770288532228e-07+0j) [X2 Z3 Z4 Z5 X6 X10 Z11 X12] +
(-3.4273231087055004e-07+0j) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12] +
(-0.0008145313270957492+0j) [X2 Z3 Z4 Z5 X6 Z10] +
(2.745518400245925e-06+0j) [X2 Z3 Z4 Z5 X6 X11 Z12 X13] +
(2.745518400245925e-06+0j) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13] +
(0.014411099430130773+0j) [X2 Z3 Z4 Z5 X6 Z11] +
(0.0006650070219498492+0j) [X2 Z3 Z4 Z5 X6 Z12] +
(-0.0034937903598902496+0j) [X2 Z3 Z4 Z5 X6 Z13] +
(-4.561447179826729e-07+0j) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12] +
(-0.011756013419819269+0j) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9] +
(0.015225630757226523+0j) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11] +
(-3.0882507111164753e-06+0j) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(-3.5443954290991475e-06+0j) [X2 Z3 Z4 Z5 Z6 X7 X11 X12] +
(-0.004158797381840098+0j) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13] +
(0.011756013419819269+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9] +
(-0.015225630757226523+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11] +
(3.0882507111164753e-06+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(-3.5443954290991475e-06+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12] +
(0.004158797381840098+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13] +
(0.014603704729162031+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(-2.8742990712069643e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.014603704729162031+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(-2.8742990712069643e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(-0.28164257767022755+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-1.3002946561682863e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(-1.3002946561682863e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(-0.02428211735469301+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(-0.019538050311314635+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(-0.017091553155898838+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(0.0024464971554158015+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(-0.0024464971554158015+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(5.77595052690911e-05+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(2.8836765759243042e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(5.146496327254752e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(3.8462016710864655e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(-0.0393591680220529+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10] +
(7.979825792981884e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(-0.02475546329289087+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10] +
(5.1055267217749194e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(-0.021433810721600634+0j) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10] +
(5.159350501701728e-06+0j) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(-0.029903789512624745+0j) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10] +
(5.427988656204242e-06+0j) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0016638798784907682+0j) [X2 Z3 Z4 X6] +
(-0.018889030304942808+0j) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10] +
(2.947356011509977e-06+0j) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.00347951189033441+0j) [X2 Z3 Z5 X6] +
(-0.028730779551905443+0j) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10] +
(5.935867717773343e-06+0j) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.6021167406624548e-06+0j) [X2 X4] +
(0.0004956762314915059+0j) [X2 Z4 Z5 X6] +
(-0.035608378988312456+0j) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(7.253273347778515e-06+0j) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.035839567953353364+0j) [Y2 X3 X4 Y5] +
(2.1990516183257419e-07+0j) [Y2 X3 X4 Z5 Z6 Y7] +
(2.3609563202103618e-06+0j) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.010311482489831882+0j) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.1990516183257419e-07+0j) [Y2 X3 X5 Y6] +
(-2.3609563202103618e-06+0j) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.010311482489831882+0j) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.031143817988967086+0j) [Y2 X3 X6 Y7] +
(-0.005368659358109605+0j) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11] +
(-9.209350651005051e-08+0j) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005368659358109605+0j) [Y2 X3 X7 Z8 Z9 Y10] +
(9.209350651005051e-08+0j) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12] +
(0.03619412355904263+0j) [Y2 X3 X8 Y9] +
(0.025384657508457344+0j) [Y2 X3 X10 Y11] +
(-2.172669101416568e-06+0j) [Y2 X3 X10 Z11 Z12 Y13] +
(2.1726691014165674e-06+0j) [Y2 X3 X11 Y12] +
(0.01557720806397646+0j) [Y2 X3 X12 Y13] +
(-0.035839567953353364+0j) [Y2 Y3 X4 X5] +
(-2.1990516183257419e-07+0j) [Y2 Y3 X4 Z5 Z6 X7] +
(-2.3609563202103618e-06+0j) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.010311482489831882+0j) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.1990516183257419e-07+0j) [Y2 Y3 Y5 Y6] +
(-2.3609563202103618e-06+0j) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.010311482489831882+0j) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.031143817988967086+0j) [Y2 Y3 X6 X7] +
(0.005368659358109605+0j) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11] +
(9.209350651005051e-08+0j) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005368659358109605+0j) [Y2 Y3 Y7 Z8 Z9 Y10] +
(9.209350651005051e-08+0j) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.03619412355904263+0j) [Y2 Y3 X8 X9] +
(-0.025384657508457344+0j) [Y2 Y3 X10 X11] +
(2.172669101416568e-06+0j) [Y2 Y3 X10 Z11 Z12 X13] +
(2.1726691014165674e-06+0j) [Y2 Y3 Y11 Y12] +
(-0.01557720806397646+0j) [Y2 Y3 X12 X13] +
(1.6288532434309747e-06+0j) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10] +
(0.010715508469796796+0j) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.010263414868158481+0j) [Y2 Z3 X4 X10 Z11 Y12] +
(-3.887051673487614e-06+0j) [Y2 Z3 Y4] +
(-0.005143391768825178+0j) [Y2 Z3 Y4 X5 Z6 X7] +
(-0.009841749246962636+0j) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.988511706263365e-06+0j) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.005143391768825178+0j) [Y2 Z3 Y4 Y5 Z6 Y7] +
(-0.009841749246962636+0j) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.988511706263365e-06+0j) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.764994119377161e-07+0j) [Y2 Z3 Y4 Z5] +
(4.5371780943628625e-08+0j) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10] +
(4.2055484112183544e-05+0j) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.6893489513559773e-06+0j) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10] +
(0.010757563953908976+0j) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.5935343911031914e-07+0j) [Y2 Z3 Y4 Z6] +
(3.211842018880629e-06+0j) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11] +
(0.019299560579363797+0j) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.211842018880629e-06+0j) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11] +
(0.019299560579363797+0j) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.195489009919614e-06+0j) [Y2 Z3 Y4 Z7] +
(2.186842377549152e-07+0j) [Y2 Z3 Y4 Z8] +
(-5.770052995237359e-07+0j) [Y2 Z3 Y4 Z9] +
(0.005324835234221671+0j) [Y2 Z3 Y4 X10 Z11 X12] +
(0.015588250102380154+0j) [Y2 Z3 Y4 Y10 Z11 Y12] +
(-3.158656431838776e-06+0j) [Y2 Z3 Y4 Z10] +
(0.024353077678068828+0j) [Y2 Z3 Y4 X11 Z12 X13] +
(0.024353077678068828+0j) [Y2 Z3 Y4 Y11 Z12 Y13] +
(-7.801707500084142e-06+0j) [Y2 Z3 Y4 Z11] +
(3.5390541842629702e-06+0j) [Y2 Z3 Y4 Z12] +
(8.814937306085744e-06+0j) [Y2 Z3 Y4 Z13] +
(1.454842449029933e-06+0j) [Y2 Z3 Z4 X5 X6 Y7] +
(3.151346310955626e-06+0j) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11] +
(0.019257505095251613+0j) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.5224930675246511e-06+0j) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10] +
(-0.00854199662545482+0j) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(7.956895372786512e-07+0j) [Y2 Z3 Z4 X5 X8 Y9] +
(4.6430510682453656e-06+0j) [Y2 Z3 Z4 X5 X10 Y11] +
(0.019028242443847158+0j) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13] +
(-0.008764827575688678+0j) [Y2 Z3 Z4 X5 X11 Y12] +
(-5.275883121822774e-06+0j) [Y2 Z3 Z4 X5 X12 Y13] +
(-1.454842449029933e-06+0j) [Y2 Z3 Z4 Y5 X6 X7] +
(-3.151346310955626e-06+0j) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11] +
(-0.019257505095251613+0j) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.5224930675246511e-06+0j) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10] +
(-0.00854199662545482+0j) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895372786512e-07+0j) [Y2 Z3 Z4 Y5 X8 X9] +
(-4.6430510682453656e-06+0j) [Y2 Z3 Z4 Y5 X10 X11] +
(-0.019028242443847158+0j) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13] +
(-0.008764827575688678+0j) [Y2 Z3 Z4 Y5 Y11 Y12] +
(5.275883121822774e-06+0j) [Y2 Z3 Z4 Y5 X12 X13] +
(-4.561447179826729e-07+0j) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12] +
(-0.12133276911042379+0j) [Y2 Z3 Z4 Z5 Y6] +
(-0.008469978791024112+0j) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(2.686381545025142e-07+0j) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.008469978791024112+0j) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(2.686381545025142e-07+0j) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.02599617759802112+0j) [Y2 Z3 Z4 Z5 Y6 Z7] +
(-0.005805188989826982+0j) [Y2 Z3 Z4 Z5 Y6 Z8] +
(-0.017561202409646245+0j) [Y2 Z3 Z4 Z5 Y6 Z9] +
(-3.4273231087055004e-07+0j) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12] +
(-7.988770288532228e-07+0j) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12] +
(-0.0008145313270957492+0j) [Y2 Z3 Z4 Z5 Y6 Z10] +
(2.745518400245925e-06+0j) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13] +
(2.745518400245925e-06+0j) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13] +
(0.014411099430130773+0j) [Y2 Z3 Z4 Z5 Y6 Z11] +
(0.0006650070219498492+0j) [Y2 Z3 Z4 Z5 Y6 Z12] +
(-0.0034937903598902496+0j) [Y2 Z3 Z4 Z5 Y6 Z13] +
(0.011756013419819269+0j) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9] +
(-0.015225630757226523+0j) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11] +
(3.0882507111164753e-06+0j) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(-3.5443954290991475e-06+0j) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12] +
(0.004158797381840098+0j) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13] +
(-0.011756013419819269+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9] +
(0.015225630757226523+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11] +
(-3.0882507111164753e-06+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(-3.5443954290991475e-06+0j) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12] +
(-0.004158797381840098+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13] +
(0.014603704729162031+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(-2.8742990712069643e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.014603704729162031+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(-2.8742990712069643e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(-0.28164257767022755+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-1.3002946561682863e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(-1.3002946561682863e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(-0.02428211735469301+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(-0.019538050311314635+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(-0.017091553155898838+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(-0.0024464971554158015+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(0.0024464971554158015+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(5.77595052690911e-05+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.8836765759243042e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(5.146496327254752e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(3.8462016710864655e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(-0.0393591680220529+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10] +
(7.979825792981884e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(-0.02475546329289087+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10] +
(5.1055267217749194e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(-0.021433810721600634+0j) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10] +
(5.159350501701728e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(-0.029903789512624745+0j) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10] +
(5.427988656204242e-06+0j) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0016638798784907682+0j) [Y2 Z3 Z4 Y6] +
(-0.018889030304942808+0j) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10] +
(2.947356011509977e-06+0j) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.00347951189033441+0j) [Y2 Z3 Z5 Y6] +
(-0.028730779551905443+0j) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10] +
(5.935867717773343e-06+0j) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.6021167406624548e-06+0j) [Y2 Y4] +
(0.0004956762314915059+0j) [Y2 Z4 Z5 Y6] +
(-0.035608378988312456+0j) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(7.253273347778515e-06+0j) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.6538942226831734+0j) [Z2] +
(1.6021167406624548e-06+0j) [Z2 X3 Z4 X5] +
(0.0004956762314915059+0j) [Z2 X3 Z4 Z5 Z6 X7] +
(-0.03560837898831246+0j) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(7.253273347778515e-06+0j) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.6021167406624548e-06+0j) [Z2 Y3 Z4 Y5] +
(0.0004956762314915059+0j) [Z2 Y3 Z4 Z5 Z6 Y7] +
(-0.03560837898831246+0j) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(7.253273347778515e-06+0j) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.1818908579075134+0j) [Z2 Z3] +
(-9.509249752210598e-07+0j) [Z2 X4 Z5 X6] +
(-4.728843147063187e-06+0j) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.024591860883829995+0j) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-9.509249752210598e-07+0j) [Z2 Y4 Z5 Y6] +
(-4.728843147063187e-06+0j) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.024591860883829995+0j) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.12495807739503204+0j) [Z2 Z4] +
(-1.170830137053634e-06+0j) [Z2 X5 Z6 X7] +
(-7.089799467273549e-06+0j) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.034903343373661876+0j) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.170830137053634e-06+0j) [Z2 Y5 Z6 Y7] +
(-7.089799467273549e-06+0j) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.034903343373661876+0j) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.16079764534838537+0j) [Z2 Z5] +
(0.019020423173040035+0j) [Z2 X6 Z7 Z8 Z9 X10] +
(-2.103215604448088e-06+0j) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.019020423173040035+0j) [Z2 Y6 Z7 Z8 Z9 Y10] +
(-2.103215604448088e-06+0j) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.137391047626832+0j) [Z2 Z6] +
(0.024389082531149645+0j) [Z2 X7 Z8 Z9 Z10 X11] +
(-2.0111220979380374e-06+0j) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.024389082531149645+0j) [Z2 Y7 Z8 Z9 Z10 Y11] +
(-2.0111220979380374e-06+0j) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.16853486561579908+0j) [Z2 Z7] +
(0.15071408121008284+0j) [Z2 Z8] +
(0.18690820476912545+0j) [Z2 Z9] +
(-1.0632283422212445e-06+0j) [Z2 X10 Z11 X12] +
(-1.0632283422212445e-06+0j) [Z2 Y10 Z11 Y12] +
(1.1094407591953231e-06+0j) [Z2 X11 Z12 X13] +
(1.1094407591953231e-06+0j) [Z2 Y11 Z12 Y13] +
(0.14011289865354803+0j) [Z2 Z12] +
(0.15569010671752448+0j) [Z2 Z13] +
(0.005143391768825179+0j) [X3 X4 Y5 Y6] +
(0.009841749246962635+0j) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10] +
(-2.988511706263365e-06+0j) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.454842449029933e-06+0j) [X3 X4 X6 X7] +
(-1.5224930675246511e-06+0j) [X3 X4 X6 Z7 Z8 Z9 Z10 X11] +
(-0.00854199662545482+0j) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.151346310955626e-06+0j) [X3 X4 Y7 Z8 Z9 Y10] +
(-0.019257505095251613+0j) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895372786512e-07+0j) [X3 X4 X8 X9] +
(-4.6430510682453656e-06+0j) [X3 X4 X10 X11] +
(-0.008764827575688678+0j) [X3 X4 X10 Z11 Z12 X13] +
(-0.019028242443847158+0j) [X3 X4 Y11 Y12] +
(5.275883121822774e-06+0j) [X3 X4 X12 X13] +
(-0.005143391768825179+0j) [X3 Y4 Y5 X6] +
(-0.009841749246962635+0j) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10] +
(2.988511706263365e-06+0j) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.454842449029933e-06+0j) [X3 Y4 Y6 X7] +
(-1.5224930675246511e-06+0j) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11] +
(-0.00854199662545482+0j) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.151346310955626e-06+0j) [X3 Y4 Y7 Z8 Z9 X10] +
(0.019257505095251613+0j) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12] +
(-7.956895372786512e-07+0j) [X3 Y4 Y8 X9] +
(-4.6430510682453656e-06+0j) [X3 Y4 Y10 X11] +
(-0.008764827575688678+0j) [X3 Y4 Y10 Z11 Z12 X13] +
(0.019028242443847158+0j) [X3 Y4 Y11 X12] +
(5.275883121822774e-06+0j) [X3 Y4 Y12 X13] +
(-3.887051673487612e-06+0j) [X3 Z4 X5] +
(3.211842018880629e-06+0j) [X3 Z4 X5 X6 Z7 Z8 Z9 X10] +
(0.019299560579363797+0j) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.211842018880629e-06+0j) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10] +
(0.019299560579363797+0j) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.195489009919614e-06+0j) [X3 Z4 X5 Z6] +
(1.6893489513559773e-06+0j) [X3 Z4 X5 X7 Z8 Z9 Z10 X11] +
(0.010757563953908976+0j) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.5371780943628625e-08+0j) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11] +
(4.2055484112183544e-05+0j) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.5935343911031914e-07+0j) [X3 Z4 X5 Z7] +
(-5.770052995237359e-07+0j) [X3 Z4 X5 Z8] +
(2.186842377549152e-07+0j) [X3 Z4 X5 Z9] +
(0.024353077678068828+0j) [X3 Z4 X5 X10 Z11 X12] +
(0.024353077678068828+0j) [X3 Z4 X5 Y10 Z11 Y12] +
(-7.801707500084142e-06+0j) [X3 Z4 X5 Z10] +
(0.015588250102380154+0j) [X3 Z4 X5 X11 Z12 X13] +
(0.005324835234221671+0j) [X3 Z4 X5 Y11 Z12 Y13] +
(-3.158656431838776e-06+0j) [X3 Z4 X5 Z11] +
(8.814937306085744e-06+0j) [X3 Z4 X5 Z12] +
(3.5390541842629702e-06+0j) [X3 Z4 X5 Z13] +
(1.6288532434309747e-06+0j) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11] +
(0.010715508469796796+0j) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.010263414868158481+0j) [X3 Z4 Y5 Y11 Z12 X13] +
(0.008469978791024112+0j) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10] +
(-2.686381545025142e-07+0j) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.011756013419819269+0j) [X3 Z4 Z5 X6 X8 X9] +
(0.015225630757226521+0j) [X3 Z4 Z5 X6 X10 X11] +
(-3.5443954290991475e-06+0j) [X3 Z4 Z5 X6 X10 Z11 Z12 X13] +
(-3.0882507111164753e-06+0j) [X3 Z4 Z5 X6 Y11 Y12] +
(-0.004158797381840098+0j) [X3 Z4 Z5 X6 X12 X13] +
(-0.008469978791024112+0j) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10] +
(2.686381545025142e-07+0j) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.011756013419819269+0j) [X3 Z4 Z5 Y6 Y8 X9] +
(0.015225630757226521+0j) [X3 Z4 Z5 Y6 Y10 X11] +
(-3.5443954290991475e-06+0j) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13] +
(3.0882507111164753e-06+0j) [X3 Z4 Z5 Y6 Y11 X12] +
(-0.004158797381840098+0j) [X3 Z4 Z5 Y6 Y12 X13] +
(-0.12133276911042376+0j) [X3 Z4 Z5 Z6 X7] +
(-0.017561202409646245+0j) [X3 Z4 Z5 Z6 X7 Z8] +
(-0.005805188989826982+0j) [X3 Z4 Z5 Z6 X7 Z9] +
(2.745518400245925e-06+0j) [X3 Z4 Z5 Z6 X7 X10 Z11 X12] +
(2.745518400245925e-06+0j) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12] +
(0.014411099430130773+0j) [X3 Z4 Z5 Z6 X7 Z10] +
(-7.988770288532228e-07+0j) [X3 Z4 Z5 Z6 X7 X11 Z12 X13] +
(-3.4273231087055004e-07+0j) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13] +
(-0.0008145313270957492+0j) [X3 Z4 Z5 Z6 X7 Z11] +
(-0.0034937903598902496+0j) [X3 Z4 Z5 Z6 X7 Z12] +
(0.0006650070219498492+0j) [X3 Z4 Z5 Z6 X7 Z13] +
(-4.561447179826729e-07+0j) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13] +
(-0.014603704729162031+0j) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10] +
(2.8742990712069643e-06+0j) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(0.014603704729162031+0j) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10] +
(-2.8742990712069643e-06+0j) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(1.3002946561682863e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12] +
(0.0024464971554158015+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-1.3002946561682863e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12] +
(0.0024464971554158015+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(-0.2816425776702274+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.017091553155898838+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(-0.019538050311314635+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(5.77595052690911e-05+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(2.883676575924304e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(3.8462016710864655e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(-0.02428211735469301+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11] +
(5.146496327254752e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(-0.02475546329289087+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11] +
(5.1055267217749194e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(-0.0393591680220529+0j) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11] +
(7.979825792981884e-06+0j) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(-0.029903789512624745+0j) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11] +
(5.427988656204242e-06+0j) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.02599617759802112+0j) [X3 Z4 Z5 X7] +
(-0.021433810721600634+0j) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11] +
(5.159350501701728e-06+0j) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.00347951189033441+0j) [X3 Z4 Z6 X7] +
(-0.028730779551905443+0j) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11] +
(5.935867717773343e-06+0j) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.764994119377162e-07+0j) [X3 X5] +
(0.0016638798784907682+0j) [X3 Z5 Z6 X7] +
(-0.018889030304942808+0j) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.947356011509977e-06+0j) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.005143391768825179+0j) [Y3 X4 X5 Y6] +
(-0.009841749246962635+0j) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10] +
(2.988511706263365e-06+0j) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.454842449029933e-06+0j) [Y3 X4 X6 Y7] +
(-1.5224930675246511e-06+0j) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11] +
(-0.00854199662545482+0j) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.151346310955626e-06+0j) [Y3 X4 X7 Z8 Z9 Y10] +
(0.019257505095251613+0j) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895372786512e-07+0j) [Y3 X4 X8 Y9] +
(-4.6430510682453656e-06+0j) [Y3 X4 X10 Y11] +
(-0.008764827575688678+0j) [Y3 X4 X10 Z11 Z12 Y13] +
(0.019028242443847158+0j) [Y3 X4 X11 Y12] +
(5.275883121822774e-06+0j) [Y3 X4 X12 Y13] +
(0.005143391768825179+0j) [Y3 Y4 X5 X6] +
(0.009841749246962635+0j) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10] +
(-2.988511706263365e-06+0j) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.454842449029933e-06+0j) [Y3 Y4 Y6 Y7] +
(-1.5224930675246511e-06+0j) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11] +
(-0.00854199662545482+0j) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.151346310955626e-06+0j) [Y3 Y4 X7 Z8 Z9 X10] +
(-0.019257505095251613+0j) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12] +
(-7.956895372786512e-07+0j) [Y3 Y4 Y8 Y9] +
(-4.6430510682453656e-06+0j) [Y3 Y4 Y10 Y11] +
(-0.008764827575688678+0j) [Y3 Y4 Y10 Z11 Z12 Y13] +
(-0.019028242443847158+0j) [Y3 Y4 X11 X12] +
(5.275883121822774e-06+0j) [Y3 Y4 Y12 Y13] +
(1.6288532434309747e-06+0j) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11] +
(0.010715508469796796+0j) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.010263414868158481+0j) [Y3 Z4 X5 X11 Z12 Y13] +
(-3.887051673487612e-06+0j) [Y3 Z4 Y5] +
(3.211842018880629e-06+0j) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10] +
(0.019299560579363797+0j) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.211842018880629e-06+0j) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10] +
(0.019299560579363797+0j) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.195489009919614e-06+0j) [Y3 Z4 Y5 Z6] +
(4.5371780943628625e-08+0j) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11] +
(4.2055484112183544e-05+0j) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.6893489513559773e-06+0j) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11] +
(0.010757563953908976+0j) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.5935343911031914e-07+0j) [Y3 Z4 Y5 Z7] +
(-5.770052995237359e-07+0j) [Y3 Z4 Y5 Z8] +
(2.186842377549152e-07+0j) [Y3 Z4 Y5 Z9] +
(0.024353077678068828+0j) [Y3 Z4 Y5 X10 Z11 X12] +
(0.024353077678068828+0j) [Y3 Z4 Y5 Y10 Z11 Y12] +
(-7.801707500084142e-06+0j) [Y3 Z4 Y5 Z10] +
(0.005324835234221671+0j) [Y3 Z4 Y5 X11 Z12 X13] +
(0.015588250102380154+0j) [Y3 Z4 Y5 Y11 Z12 Y13] +
(-3.158656431838776e-06+0j) [Y3 Z4 Y5 Z11] +
(8.814937306085744e-06+0j) [Y3 Z4 Y5 Z12] +
(3.5390541842629702e-06+0j) [Y3 Z4 Y5 Z13] +
(-0.008469978791024112+0j) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10] +
(2.686381545025142e-07+0j) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.011756013419819269+0j) [Y3 Z4 Z5 X6 X8 Y9] +
(0.015225630757226521+0j) [Y3 Z4 Z5 X6 X10 Y11] +
(-3.5443954290991475e-06+0j) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13] +
(3.0882507111164753e-06+0j) [Y3 Z4 Z5 X6 X11 Y12] +
(-0.004158797381840098+0j) [Y3 Z4 Z5 X6 X12 Y13] +
(0.008469978791024112+0j) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10] +
(-2.686381545025142e-07+0j) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-0.011756013419819269+0j) [Y3 Z4 Z5 Y6 Y8 Y9] +
(0.015225630757226521+0j) [Y3 Z4 Z5 Y6 Y10 Y11] +
(-3.5443954290991475e-06+0j) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13] +
(-3.0882507111164753e-06+0j) [Y3 Z4 Z5 Y6 X11 X12] +
(-0.004158797381840098+0j) [Y3 Z4 Z5 Y6 Y12 Y13] +
(-4.561447179826729e-07+0j) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13] +
(-0.12133276911042376+0j) [Y3 Z4 Z5 Z6 Y7] +
(-0.017561202409646245+0j) [Y3 Z4 Z5 Z6 Y7 Z8] +
(-0.005805188989826982+0j) [Y3 Z4 Z5 Z6 Y7 Z9] +
(2.745518400245925e-06+0j) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12] +
(2.745518400245925e-06+0j) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12] +
(0.014411099430130773+0j) [Y3 Z4 Z5 Z6 Y7 Z10] +
(-3.4273231087055004e-07+0j) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13] +
(-7.988770288532228e-07+0j) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13] +
(-0.0008145313270957492+0j) [Y3 Z4 Z5 Z6 Y7 Z11] +
(-0.0034937903598902496+0j) [Y3 Z4 Z5 Z6 Y7 Z12] +
(0.0006650070219498492+0j) [Y3 Z4 Z5 Z6 Y7 Z13] +
(0.014603704729162031+0j) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10] +
(-2.8742990712069643e-06+0j) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-0.014603704729162031+0j) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10] +
(2.8742990712069643e-06+0j) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-1.3002946561682863e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12] +
(0.0024464971554158015+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(1.3002946561682863e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12] +
(0.0024464971554158015+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(-0.2816425776702274+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.017091553155898838+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(-0.019538050311314635+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(5.77595052690911e-05+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.883676575924304e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(3.8462016710864655e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(-0.02428211735469301+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11] +
(5.146496327254752e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(-0.02475546329289087+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11] +
(5.1055267217749194e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(-0.0393591680220529+0j) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11] +
(7.979825792981884e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(-0.029903789512624745+0j) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11] +
(5.427988656204242e-06+0j) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.02599617759802112+0j) [Y3 Z4 Z5 Y7] +
(-0.021433810721600634+0j) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11] +
(5.159350501701728e-06+0j) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.00347951189033441+0j) [Y3 Z4 Z6 Y7] +
(-0.028730779551905443+0j) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11] +
(5.935867717773343e-06+0j) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.764994119377162e-07+0j) [Y3 Y5] +
(0.0016638798784907682+0j) [Y3 Z5 Z6 Y7] +
(-0.018889030304942808+0j) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.947356011509977e-06+0j) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.6538942226831732+0j) [Z3] +
(-1.170830137053634e-06+0j) [Z3 X4 Z5 X6] +
(-7.089799467273549e-06+0j) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.034903343373661876+0j) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.170830137053634e-06+0j) [Z3 Y4 Z5 Y6] +
(-7.089799467273549e-06+0j) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.034903343373661876+0j) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.16079764534838537+0j) [Z3 Z4] +
(-9.509249752210598e-07+0j) [Z3 X5 Z6 X7] +
(-4.728843147063187e-06+0j) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.024591860883829995+0j) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-9.509249752210598e-07+0j) [Z3 Y5 Z6 Y7] +
(-4.728843147063187e-06+0j) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.024591860883829995+0j) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.12495807739503204+0j) [Z3 Z5] +
(0.024389082531149645+0j) [Z3 X6 Z7 Z8 Z9 X10] +
(-2.0111220979380374e-06+0j) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.024389082531149645+0j) [Z3 Y6 Z7 Z8 Z9 Y10] +
(-2.0111220979380374e-06+0j) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.16853486561579908+0j) [Z3 Z6] +
(0.019020423173040035+0j) [Z3 X7 Z8 Z9 Z10 X11] +
(-2.103215604448088e-06+0j) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.019020423173040035+0j) [Z3 Y7 Z8 Z9 Z10 Y11] +
(-2.103215604448088e-06+0j) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.137391047626832+0j) [Z3 Z7] +
(0.18690820476912545+0j) [Z3 Z8] +
(0.15071408121008284+0j) [Z3 Z9] +
(1.1094407591953231e-06+0j) [Z3 X10 Z11 X12] +
(1.1094407591953231e-06+0j) [Z3 Y10 Z11 Y12] +
(-1.0632283422212445e-06+0j) [Z3 X11 Z12 X13] +
(-1.0632283422212445e-06+0j) [Z3 Y11 Z12 Y13] +
(0.15569010671752448+0j) [Z3 Z12] +
(0.14011289865354803+0j) [Z3 Z13] +
(-0.011982389010247996+0j) [X4 X5 Y6 Y7] +
(-0.007306759928833017+0j) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11] +
(-2.8882935958657654e-07+0j) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.007306759928833017+0j) [X4 X5 X7 Z8 Z9 X10] +
(-2.8882935958657654e-07+0j) [X4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-0.007156934919856931+0j) [X4 X5 Y8 Y9] +
(-3.6945132942993415e-06+0j) [X4 X5 Y10 Z11 Z12 Y13] +
(-3.6945132942993415e-06+0j) [X4 X5 X11 X12] +
(-0.03831467029480389+0j) [X4 X5 Y12 Y13] +
(0.011982389010247996+0j) [X4 Y5 Y6 X7] +
(0.007306759928833017+0j) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11] +
(2.8882935958657654e-07+0j) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.007306759928833017+0j) [X4 Y5 Y7 Z8 Z9 X10] +
(-2.8882935958657654e-07+0j) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(0.007156934919856931+0j) [X4 Y5 Y8 X9] +
(3.6945132942993415e-06+0j) [X4 Y5 Y10 Z11 Z12 X13] +
(-3.6945132942993415e-06+0j) [X4 Y5 Y11 X12] +
(0.03831467029480389+0j) [X4 Y5 Y12 X13] +
(-1.2260484989253838e-05+0j) [X4 Z5 X6] +
(-1.2283337824505935e-06+0j) [X4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(0.00024636437569569585+0j) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2283337824505935e-06+0j) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(0.00024636437569569585+0j) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.8540608580179013e-06+0j) [X4 Z5 X6 Z7] +
(-1.3980449081509555e-06+0j) [X4 Z5 X6 Z8] +
(-1.88185018324629e-06+0j) [X4 Z5 X6 Z9] +
(0.007960880725921623+0j) [X4 Z5 X6 X10 Z11 X12] +
(-0.0009298507967730396+0j) [X4 Z5 X6 Y10 Z11 Y12] +
(-1.6923978285447468e-06+0j) [X4 Z5 X6 Z10] +
(-0.012215040997614023+0j) [X4 Z5 X6 X11 Z12 X13] +
(-0.012215040997614023+0j) [X4 Z5 X6 Y11 Z12 Y13] +
(4.281913884560978e-06+0j) [X4 Z5 X6 Z11] +
(-4.588855155513191e-06+0j) [X4 Z5 X6 Z13] +
(0.008890731522694663+0j) [X4 Z5 Y6 Y10 Z11 X12] +
(-4.838052750953346e-07+0j) [X4 Z5 Z6 X7 Y8 Y9] +
(5.974311713105724e-06+0j) [X4 Z5 Z6 X7 Y10 Y11] +
(0.011285190200840983+0j) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(0.020175921723535644+0j) [X4 Z5 Z6 X7 X11 X12] +
(-4.55656921792131e-06+0j) [X4 Z5 Z6 X7 Y12 Y13] +
(4.838052750953346e-07+0j) [X4 Z5 Z6 Y7 Y8 X9] +
(-5.974311713105724e-06+0j) [X4 Z5 Z6 Y7 Y10 X11] +
(-0.011285190200840983+0j) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(0.020175921723535644+0j) [X4 Z5 Z6 Y7 Y11 X12] +
(4.55656921792131e-06+0j) [X4 Z5 Z6 Y7 Y12 X13] +
(1.3304731886286619e-06+0j) [X4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(0.005923798336561347+0j) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(1.3304731886286619e-06+0j) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(0.005923798336561347+0j) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(-6.631277928141312e-05+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.01602460368917951+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(-0.01602460368917951+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(3.3343312891297336e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(-4.7346220385483565e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(-9.806102774707355e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(-5.071480736158998e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(5.071480736158998e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-0.36937089366156306+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.02314513092952919+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-0.009612634606847421+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(-0.025637238296026932+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(-8.774817864219894e-06+0j) [X4 Z5 Z6 Z7 Z8 X10] +
(-0.04764261217638316+0j) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(-7.444344675591232e-06+0j) [X4 Z5 Z6 Z7 Z9 X10] +
(-0.04171881383982181+0j) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(-6.290028432931232e-06+0j) [X4 Z5 Z6 Z8 Z9 X10] +
(-0.0395644163228932+0j) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(-7.518362215381825e-06+0j) [X4 Z5 Z7 Z8 Z9 X10] +
(-0.03931805194719751+0j) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(-5.929765816403469e-07+0j) [X4 X6] +
(-4.253224225428714e-06+0j) [X4 Z6 Z7 Z8 Z9 X10] +
(-0.02252844019601298+0j) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.011982389010247996+0j) [Y4 X5 X6 Y7] +
(0.007306759928833017+0j) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11] +
(2.8882935958657654e-07+0j) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.007306759928833017+0j) [Y4 X5 X7 Z8 Z9 Y10] +
(-2.8882935958657654e-07+0j) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(0.007156934919856931+0j) [Y4 X5 X8 Y9] +
(3.6945132942993415e-06+0j) [Y4 X5 X10 Z11 Z12 Y13] +
(-3.6945132942993415e-06+0j) [Y4 X5 X11 Y12] +
(0.03831467029480389+0j) [Y4 X5 X12 Y13] +
(-0.011982389010247996+0j) [Y4 Y5 X6 X7] +
(-0.007306759928833017+0j) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11] +
(-2.8882935958657654e-07+0j) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.007306759928833017+0j) [Y4 Y5 Y7 Z8 Z9 Y10] +
(-2.8882935958657654e-07+0j) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.007156934919856931+0j) [Y4 Y5 X8 X9] +
(-3.6945132942993415e-06+0j) [Y4 Y5 X10 Z11 Z12 X13] +
(-3.6945132942993415e-06+0j) [Y4 Y5 Y11 Y12] +
(-0.03831467029480389+0j) [Y4 Y5 X12 X13] +
(0.008890731522694663+0j) [Y4 Z5 X6 X10 Z11 Y12] +
(-1.2260484989253838e-05+0j) [Y4 Z5 Y6] +
(-1.2283337824505935e-06+0j) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(0.00024636437569569585+0j) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2283337824505935e-06+0j) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(0.00024636437569569585+0j) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.8540608580179013e-06+0j) [Y4 Z5 Y6 Z7] +
(-1.3980449081509555e-06+0j) [Y4 Z5 Y6 Z8] +
(-1.88185018324629e-06+0j) [Y4 Z5 Y6 Z9] +
(-0.0009298507967730396+0j) [Y4 Z5 Y6 X10 Z11 X12] +
(0.007960880725921623+0j) [Y4 Z5 Y6 Y10 Z11 Y12] +
(-1.6923978285447468e-06+0j) [Y4 Z5 Y6 Z10] +
(-0.012215040997614023+0j) [Y4 Z5 Y6 X11 Z12 X13] +
(-0.012215040997614023+0j) [Y4 Z5 Y6 Y11 Z12 Y13] +
(4.281913884560978e-06+0j) [Y4 Z5 Y6 Z11] +
(-4.588855155513191e-06+0j) [Y4 Z5 Y6 Z13] +
(4.838052750953346e-07+0j) [Y4 Z5 Z6 X7 X8 Y9] +
(-5.974311713105724e-06+0j) [Y4 Z5 Z6 X7 X10 Y11] +
(-0.011285190200840983+0j) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(0.020175921723535644+0j) [Y4 Z5 Z6 X7 X11 Y12] +
(4.55656921792131e-06+0j) [Y4 Z5 Z6 X7 X12 Y13] +
(-4.838052750953346e-07+0j) [Y4 Z5 Z6 Y7 X8 X9] +
(5.974311713105724e-06+0j) [Y4 Z5 Z6 Y7 X10 X11] +
(0.011285190200840983+0j) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(0.020175921723535644+0j) [Y4 Z5 Z6 Y7 Y11 Y12] +
(-4.55656921792131e-06+0j) [Y4 Z5 Z6 Y7 X12 X13] +
(1.3304731886286619e-06+0j) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(0.005923798336561347+0j) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(1.3304731886286619e-06+0j) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(0.005923798336561347+0j) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(-6.631277928141312e-05+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.01602460368917951+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(-0.01602460368917951+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(3.3343312891297336e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(-4.7346220385483565e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(-9.806102774707355e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(5.071480736158998e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-5.071480736158998e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-0.36937089366156306+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.02314513092952919+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-0.009612634606847421+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(-0.025637238296026932+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(-8.774817864219894e-06+0j) [Y4 Z5 Z6 Z7 Z8 Y10] +
(-0.04764261217638316+0j) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(-7.444344675591232e-06+0j) [Y4 Z5 Z6 Z7 Z9 Y10] +
(-0.04171881383982181+0j) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(-6.290028432931232e-06+0j) [Y4 Z5 Z6 Z8 Z9 Y10] +
(-0.0395644163228932+0j) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(-7.518362215381825e-06+0j) [Y4 Z5 Z7 Z8 Z9 Y10] +
(-0.03931805194719751+0j) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(-5.929765816403469e-07+0j) [Y4 Y6] +
(-4.253224225428714e-06+0j) [Y4 Z6 Z7 Z8 Z9 Y10] +
(-0.02252844019601298+0j) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.2034402289145638+0j) [Z4] +
(-5.929765816403469e-07+0j) [Z4 X5 Z6 X7] +
(-4.253224225428714e-06+0j) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.02252844019601298+0j) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-5.929765816403469e-07+0j) [Z4 Y5 Z6 Y7] +
(-4.253224225428714e-06+0j) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.02252844019601298+0j) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.15755314797985653+0j) [Z4 Z5] +
(0.018266834869375623+0j) [Z4 X6 Z7 Z8 Z9 X10] +
(-1.6541174769287691e-06+0j) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.018266834869375623+0j) [Z4 Y6 Z7 Z8 Z9 Y10] +
(-1.6541174769287691e-06+0j) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.13701191674040722+0j) [Z4 Z6] +
(0.010960074940542606+0j) [Z4 X7 Z8 Z9 Z10 X11] +
(-1.9429468365153456e-06+0j) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.010960074940542606+0j) [Z4 Y7 Z8 Z9 Z10 Y11] +
(-1.9429468365153456e-06+0j) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.1489943057506552+0j) [Z4 Z7] +
(0.14960702684445287+0j) [Z4 Z8] +
(0.15676396176430982+0j) [Z4 Z9] +
(1.8782101246335006e-06+0j) [Z4 X10 Z11 X12] +
(1.8782101246335006e-06+0j) [Z4 Y10 Z11 Y12] +
(0.12489990917237599+0j) [Z4 Z10] +
(-1.8163031696658408e-06+0j) [Z4 X11 Z12 X13] +
(-1.8163031696658408e-06+0j) [Z4 Y11 Z12 Y13] +
(0.1425799771248575+0j) [Z4 Z11] +
(0.11383573679388653+0j) [Z4 Z12] +
(0.15215040708869043+0j) [Z4 Z13] +
(1.2283337824505933e-06+0j) [X5 X6 Y7 Z8 Z9 Y10] +
(-0.00024636437569569585+0j) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-4.838052750953346e-07+0j) [X5 X6 X8 X9] +
(5.974311713105723e-06+0j) [X5 X6 X10 X11] +
(0.020175921723535644+0j) [X5 X6 X10 Z11 Z12 X13] +
(0.011285190200840983+0j) [X5 X6 Y11 Y12] +
(-4.556569217921312e-06+0j) [X5 X6 X12 X13] +
(-1.2283337824505933e-06+0j) [X5 Y6 Y7 Z8 Z9 X10] +
(0.00024636437569569585+0j) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-4.838052750953346e-07+0j) [X5 Y6 Y8 X9] +
(5.974311713105723e-06+0j) [X5 Y6 Y10 X11] +
(0.020175921723535644+0j) [X5 Y6 Y10 Z11 Z12 X13] +
(-0.011285190200840983+0j) [X5 Y6 Y11 X12] +
(-4.556569217921312e-06+0j) [X5 Y6 Y12 X13] +
(-1.226048498925384e-05+0j) [X5 Z6 X7] +
(-1.88185018324629e-06+0j) [X5 Z6 X7 Z8] +
(-1.3980449081509555e-06+0j) [X5 Z6 X7 Z9] +
(-0.012215040997614023+0j) [X5 Z6 X7 X10 Z11 X12] +
(-0.012215040997614023+0j) [X5 Z6 X7 Y10 Z11 Y12] +
(4.281913884560978e-06+0j) [X5 Z6 X7 Z10] +
(0.007960880725921623+0j) [X5 Z6 X7 X11 Z12 X13] +
(-0.0009298507967730396+0j) [X5 Z6 X7 Y11 Z12 Y13] +
(-1.6923978285447468e-06+0j) [X5 Z6 X7 Z11] +
(-4.588855155513191e-06+0j) [X5 Z6 X7 Z12] +
(0.008890731522694663+0j) [X5 Z6 Y7 Y11 Z12 X13] +
(-1.3304731886286619e-06+0j) [X5 Z6 Z7 X8 Y9 Y10] +
(-0.005923798336561347+0j) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(1.3304731886286619e-06+0j) [X5 Z6 Z7 Y8 Y9 X10] +
(0.005923798336561347+0j) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(0.01602460368917951+0j) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12] +
(-5.0714807361589975e-06+0j) [X5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-0.01602460368917951+0j) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12] +
(-5.0714807361589975e-06+0j) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(-6.631277928141315e-05+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-9.806102774707355e-06+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(-4.7346220385483565e-06+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(-0.36937089366156306+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.023145130929529186+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(-0.025637238296026932+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(3.334331289129734e-06+0j) [X5 Z6 Z7 Z8 Z9 X11] +
(-0.009612634606847421+0j) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(-7.444344675591232e-06+0j) [X5 Z6 Z7 Z8 Z10 X11] +
(-0.04171881383982181+0j) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(-8.774817864219894e-06+0j) [X5 Z6 Z7 Z9 Z10 X11] +
(-0.04764261217638316+0j) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(-7.518362215381825e-06+0j) [X5 Z6 Z8 Z9 Z10 X11] +
(-0.03931805194719751+0j) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.8540608580179015e-06+0j) [X5 X7] +
(-6.290028432931232e-06+0j) [X5 Z7 Z8 Z9 Z10 X11] +
(-0.0395644163228932+0j) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2283337824505933e-06+0j) [Y5 X6 X7 Z8 Z9 Y10] +
(0.00024636437569569585+0j) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-4.838052750953346e-07+0j) [Y5 X6 X8 Y9] +
(5.974311713105723e-06+0j) [Y5 X6 X10 Y11] +
(0.020175921723535644+0j) [Y5 X6 X10 Z11 Z12 Y13] +
(-0.011285190200840983+0j) [Y5 X6 X11 Y12] +
(-4.556569217921312e-06+0j) [Y5 X6 X12 Y13] +
(1.2283337824505933e-06+0j) [Y5 Y6 X7 Z8 Z9 X10] +
(-0.00024636437569569585+0j) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-4.838052750953346e-07+0j) [Y5 Y6 Y8 Y9] +
(5.974311713105723e-06+0j) [Y5 Y6 Y10 Y11] +
(0.020175921723535644+0j) [Y5 Y6 Y10 Z11 Z12 Y13] +
(0.011285190200840983+0j) [Y5 Y6 X11 X12] +
(-4.556569217921312e-06+0j) [Y5 Y6 Y12 Y13] +
(0.008890731522694663+0j) [Y5 Z6 X7 X11 Z12 Y13] +
(-1.226048498925384e-05+0j) [Y5 Z6 Y7] +
(-1.88185018324629e-06+0j) [Y5 Z6 Y7 Z8] +
(-1.3980449081509555e-06+0j) [Y5 Z6 Y7 Z9] +
(-0.012215040997614023+0j) [Y5 Z6 Y7 X10 Z11 X12] +
(-0.012215040997614023+0j) [Y5 Z6 Y7 Y10 Z11 Y12] +
(4.281913884560978e-06+0j) [Y5 Z6 Y7 Z10] +
(-0.0009298507967730396+0j) [Y5 Z6 Y7 X11 Z12 X13] +
(0.007960880725921623+0j) [Y5 Z6 Y7 Y11 Z12 Y13] +
(-1.6923978285447468e-06+0j) [Y5 Z6 Y7 Z11] +
(-4.588855155513191e-06+0j) [Y5 Z6 Y7 Z12] +
(1.3304731886286619e-06+0j) [Y5 Z6 Z7 X8 X9 Y10] +
(0.005923798336561347+0j) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-1.3304731886286619e-06+0j) [Y5 Z6 Z7 Y8 X9 X10] +
(-0.005923798336561347+0j) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-0.01602460368917951+0j) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12] +
(-5.0714807361589975e-06+0j) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(0.01602460368917951+0j) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12] +
(-5.0714807361589975e-06+0j) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(-6.631277928141315e-05+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-9.806102774707355e-06+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(-4.7346220385483565e-06+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(-0.36937089366156306+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.023145130929529186+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(-0.025637238296026932+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(3.334331289129734e-06+0j) [Y5 Z6 Z7 Z8 Z9 Y11] +
(-0.009612634606847421+0j) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(-7.444344675591232e-06+0j) [Y5 Z6 Z7 Z8 Z10 Y11] +
(-0.04171881383982181+0j) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(-8.774817864219894e-06+0j) [Y5 Z6 Z7 Z9 Z10 Y11] +
(-0.04764261217638316+0j) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(-7.518362215381825e-06+0j) [Y5 Z6 Z8 Z9 Z10 Y11] +
(-0.03931805194719751+0j) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.8540608580179015e-06+0j) [Y5 Y7] +
(-6.290028432931232e-06+0j) [Y5 Z7 Z8 Z9 Z10 Y11] +
(-0.0395644163228932+0j) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.2034402289145638+0j) [Z5] +
(0.010960074940542606+0j) [Z5 X6 Z7 Z8 Z9 X10] +
(-1.9429468365153456e-06+0j) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.010960074940542606+0j) [Z5 Y6 Z7 Z8 Z9 Y10] +
(-1.9429468365153456e-06+0j) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.1489943057506552+0j) [Z5 Z6] +
(0.018266834869375623+0j) [Z5 X7 Z8 Z9 Z10 X11] +
(-1.6541174769287691e-06+0j) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.018266834869375623+0j) [Z5 Y7 Z8 Z9 Z10 Y11] +
(-1.6541174769287691e-06+0j) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.13701191674040722+0j) [Z5 Z7] +
(0.15676396176430982+0j) [Z5 Z8] +
(0.14960702684445287+0j) [Z5 Z9] +
(-1.8163031696658408e-06+0j) [Z5 X10 Z11 X12] +
(-1.8163031696658408e-06+0j) [Z5 Y10 Z11 Y12] +
(0.1425799771248575+0j) [Z5 Z10] +
(1.8782101246335006e-06+0j) [Z5 X11 Z12 X13] +
(1.8782101246335006e-06+0j) [Z5 Y11 Z12 Y13] +
(0.12489990917237599+0j) [Z5 Z11] +
(0.15215040708869043+0j) [Z5 Z12] +
(0.11383573679388653+0j) [Z5 Z13] +
(-0.013873381748426073+0j) [X6 X7 Y8 Y9] +
(-0.017825140995786633+0j) [X6 X7 Y10 Y11] +
(-1.0358477600827157e-06+0j) [X6 X7 Y10 Z11 Z12 Y13] +
(-1.0358477600827157e-06+0j) [X6 X7 X11 X12] +
(-0.017366118994651455+0j) [X6 X7 Y12 Y13] +
(0.013873381748426073+0j) [X6 Y7 Y8 X9] +
(0.017825140995786633+0j) [X6 Y7 Y10 X11] +
(1.0358477600827157e-06+0j) [X6 Y7 Y10 Z11 Z12 X13] +
(-1.0358477600827157e-06+0j) [X6 Y7 Y11 X12] +
(0.017366118994651455+0j) [X6 Y7 Y12 X13] +
(0.0002921986261110143+0j) [X6 Z7 X8 X9 Z10 X11] +
(-3.3281393503820405e-07+0j) [X6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.0002921986261110143+0j) [X6 Z7 X8 Y9 Z10 Y11] +
(-3.3281393503820405e-07+0j) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(0.22848106564919027+0j) [X6 Z7 Z8 Z9 X10] +
(3.3131454999697855e-06+0j) [X6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(3.3131454999697855e-06+0j) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(0.01130727400884826+0j) [X6 Z7 Z8 Z9 X10 Z11] +
(0.025104957138844634+0j) [X6 Z7 Z8 Z9 X10 Z12] +
(0.010540425907671644+0j) [X6 Z7 Z8 Z9 X10 Z13] +
(-0.014564531231172989+0j) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(0.014564531231172989+0j) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-2.595086006814132e-05+0j) [X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.1839325592297085e-06+0j) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-6.524373848116864e-06+0j) [X6 Z7 Z8 Z9 Z10 X12] +
(-3.211228348147078e-06+0j) [X6 Z7 Z8 Z9 Z11 X12] +
(0.02981242451734597+0j) [X6 Z7 Z8 X10] +
(-3.277483195236087e-06+0j) [X6 Z7 Z8 Z10 Z11 X12] +
(0.030104623143456983+0j) [X6 Z7 Z9 X10] +
(-3.610297130274291e-06+0j) [X6 Z7 Z9 Z10 Z11 X12] +
(0.03078750538914404+0j) [X6 Z8 Z9 X10] +
(-3.769659451670834e-06+0j) [X6 Z8 Z9 Z10 Z11 X12] +
(0.013873381748426073+0j) [Y6 X7 X8 Y9] +
(0.017825140995786633+0j) [Y6 X7 X10 Y11] +
(1.0358477600827157e-06+0j) [Y6 X7 X10 Z11 Z12 Y13] +
(-1.0358477600827157e-06+0j) [Y6 X7 X11 Y12] +
(0.017366118994651455+0j) [Y6 X7 X12 Y13] +
(-0.013873381748426073+0j) [Y6 Y7 X8 X9] +
(-0.017825140995786633+0j) [Y6 Y7 X10 X11] +
(-1.0358477600827157e-06+0j) [Y6 Y7 X10 Z11 Z12 X13] +
(-1.0358477600827157e-06+0j) [Y6 Y7 Y11 Y12] +
(-0.017366118994651455+0j) [Y6 Y7 X12 X13] +
(0.0002921986261110143+0j) [Y6 Z7 Y8 X9 Z10 X11] +
(-3.3281393503820405e-07+0j) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.0002921986261110143+0j) [Y6 Z7 Y8 Y9 Z10 Y11] +
(-3.3281393503820405e-07+0j) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(0.22848106564919027+0j) [Y6 Z7 Z8 Z9 Y10] +
(3.3131454999697855e-06+0j) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(3.3131454999697855e-06+0j) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(0.01130727400884826+0j) [Y6 Z7 Z8 Z9 Y10 Z11] +
(0.025104957138844634+0j) [Y6 Z7 Z8 Z9 Y10 Z12] +
(0.010540425907671644+0j) [Y6 Z7 Z8 Z9 Y10 Z13] +
(0.014564531231172989+0j) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-0.014564531231172989+0j) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-2.595086006814132e-05+0j) [Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.1839325592297085e-06+0j) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-6.524373848116864e-06+0j) [Y6 Z7 Z8 Z9 Z10 Y12] +
(-3.211228348147078e-06+0j) [Y6 Z7 Z8 Z9 Z11 Y12] +
(0.02981242451734597+0j) [Y6 Z7 Z8 Y10] +
(-3.277483195236087e-06+0j) [Y6 Z7 Z8 Z10 Z11 Y12] +
(0.030104623143456983+0j) [Y6 Z7 Z9 Y10] +
(-3.610297130274291e-06+0j) [Y6 Z7 Z9 Z10 Z11 Y12] +
(0.03078750538914404+0j) [Y6 Z8 Z9 Y10] +
(-3.769659451670834e-06+0j) [Y6 Z8 Z9 Z10 Z11 Y12] +
(1.3096862988615405+0j) [Z6] +
(0.03078750538914404+0j) [Z6 X7 Z8 Z9 Z10 X11] +
(-3.769659451670834e-06+0j) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.03078750538914404+0j) [Z6 Y7 Z8 Z9 Z10 Y11] +
(-3.769659451670834e-06+0j) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.19392534613270135+0j) [Z6 Z7] +
(0.16756653265461238+0j) [Z6 Z8] +
(0.18143991440303847+0j) [Z6 Z9] +
(-1.8551201213878374e-06+0j) [Z6 X10 Z11 X12] +
(-1.8551201213878374e-06+0j) [Z6 Y10 Z11 Y12] +
(0.11952438964682657+0j) [Z6 Z10] +
(-2.8909678814705533e-06+0j) [Z6 X11 Z12 X13] +
(-2.8909678814705533e-06+0j) [Z6 Y11 Z12 Y13] +
(0.13734953064261324+0j) [Z6 Z11] +
(0.13401715261963687+0j) [Z6 Z12] +
(0.15138327161428833+0j) [Z6 Z13] +
(-0.00029219862611101435+0j) [X7 X8 Y9 Y10] +
(3.3281393503820405e-07+0j) [X7 X8 Y9 Z10 Z11 Y12] +
(0.00029219862611101435+0j) [X7 Y8 Y9 X10] +
(-3.3281393503820405e-07+0j) [X7 Y8 Y9 Z10 Z11 X12] +
(-3.3131454999697855e-06+0j) [X7 Z8 Z9 X10 Y11 Y12] +
(-0.014564531231172989+0j) [X7 Z8 Z9 X10 X12 X13] +
(3.3131454999697855e-06+0j) [X7 Z8 Z9 Y10 Y11 X12] +
(-0.014564531231172989+0j) [X7 Z8 Z9 Y10 Y12 X13] +
(0.22848106564919032+0j) [X7 Z8 Z9 Z10 X11] +
(0.010540425907671644+0j) [X7 Z8 Z9 Z10 X11 Z12] +
(0.025104957138844634+0j) [X7 Z8 Z9 Z10 X11 Z13] +
(-2.5950860068141337e-05+0j) [X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.1839325592297085e-06+0j) [X7 Z8 Z9 Z10 Z11 X13] +
(-3.211228348147078e-06+0j) [X7 Z8 Z9 Z10 Z12 X13] +
(0.011307274008848262+0j) [X7 Z8 Z9 X11] +
(-6.524373848116864e-06+0j) [X7 Z8 Z9 Z11 Z12 X13] +
(0.030104623143456983+0j) [X7 Z8 Z10 X11] +
(-3.610297130274291e-06+0j) [X7 Z8 Z10 Z11 Z12 X13] +
(0.02981242451734597+0j) [X7 Z9 Z10 X11] +
(-3.277483195236087e-06+0j) [X7 Z9 Z10 Z11 Z12 X13] +
(0.00029219862611101435+0j) [Y7 X8 X9 Y10] +
(-3.3281393503820405e-07+0j) [Y7 X8 X9 Z10 Z11 Y12] +
(-0.00029219862611101435+0j) [Y7 Y8 X9 X10] +
(3.3281393503820405e-07+0j) [Y7 Y8 X9 Z10 Z11 X12] +
(3.3131454999697855e-06+0j) [Y7 Z8 Z9 X10 X11 Y12] +
(-0.014564531231172989+0j) [Y7 Z8 Z9 X10 X12 Y13] +
(-3.3131454999697855e-06+0j) [Y7 Z8 Z9 Y10 X11 X12] +
(-0.014564531231172989+0j) [Y7 Z8 Z9 Y10 Y12 Y13] +
(0.22848106564919032+0j) [Y7 Z8 Z9 Z10 Y11] +
(0.010540425907671644+0j) [Y7 Z8 Z9 Z10 Y11 Z12] +
(0.025104957138844634+0j) [Y7 Z8 Z9 Z10 Y11 Z13] +
(-2.5950860068141337e-05+0j) [Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(4.1839325592297085e-06+0j) [Y7 Z8 Z9 Z10 Z11 Y13] +
(-3.211228348147078e-06+0j) [Y7 Z8 Z9 Z10 Z12 Y13] +
(0.011307274008848262+0j) [Y7 Z8 Z9 Y11] +
(-6.524373848116864e-06+0j) [Y7 Z8 Z9 Z11 Z12 Y13] +
(0.030104623143456983+0j) [Y7 Z8 Z10 Y11] +
(-3.610297130274291e-06+0j) [Y7 Z8 Z10 Z11 Z12 Y13] +
(0.02981242451734597+0j) [Y7 Z9 Z10 Y11] +
(-3.277483195236087e-06+0j) [Y7 Z9 Z10 Z11 Z12 Y13] +
(1.3096862988615405+0j) [Z7] +
(0.18143991440303847+0j) [Z7 Z8] +
(0.16756653265461238+0j) [Z7 Z9] +
(-2.8909678814705533e-06+0j) [Z7 X10 Z11 X12] +
(-2.8909678814705533e-06+0j) [Z7 Y10 Z11 Y12] +
(0.13734953064261324+0j) [Z7 Z10] +
(-1.8551201213878374e-06+0j) [Z7 X11 Z12 X13] +
(-1.8551201213878374e-06+0j) [Z7 Y11 Z12 Y13] +
(0.11952438964682657+0j) [Z7 Z11] +
(0.15138327161428833+0j) [Z7 Z12] +
(0.13401715261963687+0j) [Z7 Z13] +
(-0.009560705729135907+0j) [X8 X9 Y10 Y11] +
(6.628614201220374e-07+0j) [X8 X9 Y10 Z11 Z12 Y13] +
(6.628614201220374e-07+0j) [X8 X9 X11 X12] +
(-0.006087822480561871+0j) [X8 X9 Y12 Y13] +
(0.009560705729135907+0j) [X8 Y9 Y10 X11] +
(-6.628614201220374e-07+0j) [X8 Y9 Y10 Z11 Z12 X13] +
(6.628614201220374e-07+0j) [X8 Y9 Y11 X12] +
(0.006087822480561871+0j) [X8 Y9 Y12 X13] +
(0.009560705729135907+0j) [Y8 X9 X10 Y11] +
(-6.628614201220374e-07+0j) [Y8 X9 X10 Z11 Z12 Y13] +
(6.628614201220374e-07+0j) [Y8 X9 X11 Y12] +
(0.006087822480561871+0j) [Y8 X9 X12 Y13] +
(-0.009560705729135907+0j) [Y8 Y9 X10 X11] +
(6.628614201220374e-07+0j) [Y8 Y9 X10 Z11 Z12 X13] +
(6.628614201220374e-07+0j) [Y8 Y9 Y11 Y12] +
(-0.006087822480561871+0j) [Y8 Y9 X12 X13] +
(1.3693525634718184+0j) [Z8] +
(0.2200397733437609+0j) [Z8 Z9] +
(-1.5973171976657734e-06+0j) [Z8 X10 Z11 X12] +
(-1.5973171976657734e-06+0j) [Z8 Y10 Z11 Y12] +
(0.1376687264585259+0j) [Z8 Z10] +
(-9.344557775437362e-07+0j) [Z8 X11 Z12 X13] +
(-9.344557775437362e-07+0j) [Z8 Y11 Z12 Y13] +
(0.14722943218766182+0j) [Z8 Z11] +
(0.1497348680349693+0j) [Z8 Z12] +
(0.15582269051553116+0j) [Z8 Z13] +
(-9.344557775437362e-07+0j) [Z9 X10 Z11 X12] +
(-9.344557775437362e-07+0j) [Z9 Y10 Z11 Y12] +
(0.14722943218766182+0j) [Z9 Z10] +
(-1.5973171976657734e-06+0j) [Z9 X11 Z12 X13] +
(-1.5973171976657734e-06+0j) [Z9 Y11 Z12 Y13] +
(0.1376687264585259+0j) [Z9 Z11] +
(0.15582269051553116+0j) [Z9 Z12] +
(0.1497348680349693+0j) [Z9 Z13] +
(-0.028685183716105785+0j) [X10 X11 Y12 Y13] +
(0.028685183716105785+0j) [X10 Y11 Y12 X13] +
(-1.0722312157392696e-05+0j) [X10 Z11 X12] +
(7.954413175805647e-06+0j) [X10 Z11 X12 Z13] +
(-8.19426137174296e-06+0j) [X10 X12] +
(0.028685183716105785+0j) [Y10 X11 X12 Y13] +
(-0.028685183716105785+0j) [Y10 Y11 X12 X13] +
(-1.0722312157392696e-05+0j) [Y10 Z11 Y12] +
(7.954413175805647e-06+0j) [Y10 Z11 Y12 Z13] +
(-8.19426137174296e-06+0j) [Y10 Y12] +
(0.7829661725950204+0j) [Z10] +
(-8.19426137174296e-06+0j) [Z10 X11 Z12 X13] +
(-8.19426137174296e-06+0j) [Z10 Y11 Z12 Y13] +
(0.14926355147388898+0j) [Z10 Z11] +
(0.11270386920332229+0j) [Z10 Z12] +
(0.14138905291942808+0j) [Z10 Z13] +
(-1.0722312157392696e-05+0j) [X11 Z12 X13] +
(7.954413175805647e-06+0j) [X11 X13] +
(-1.0722312157392696e-05+0j) [Y11 Z12 Y13] +
(7.954413175805647e-06+0j) [Y11 Y13] +
(0.7829661725950203+0j) [Z11] +
(0.14138905291942808+0j) [Z11 Z12] +
(0.11270386920332229+0j) [Z11 Z13] +
(0.8084581961720483+0j) [Z12] +
(0.15435748657223633+0j) [Z12 Z13] +
(0.8084581961720483+0j) [Z13]
  (-46.46390678868893) [I0]
+ (0.7829661725950172) [Z11]
+ (0.7829661725950187) [Z10]
+ (0.8084581961720488) [Z12]
+ (0.808458196172049) [Z13]
+ (1.2034402289145623) [Z4]
+ (1.203440228914563) [Z5]
+ (1.3096862988615423) [Z6]
+ (1.3096862988615423) [Z7]
+ (1.3693525634718169) [Z8]
+ (1.3693525634718173) [Z9]
+ (1.6538942226831694) [Z2]
+ (1.6538942226831697) [Z3]
+ (12.412630742111766) [Z0]
+ (12.412630742111766) [Z1]
+ (-8.194261372454245e-06) [Y10 Y12]
+ (-8.194261372454245e-06) [X10 X12]
+ (-1.8540608579131116e-06) [Y5 Y7]
+ (-1.8540608579131116e-06) [X5 X7]
+ (-7.764994119155435e-07) [Y3 Y5]
+ (-7.764994119155435e-07) [X3 X5]
+ (-5.929765814984023e-07) [Y4 Y6]
+ (-5.929765814984023e-07) [X4 X6]
+ (1.6021167407252466e-06) [Y2 Y4]
+ (1.6021167407252466e-06) [X2 X4]
+ (7.954413176457233e-06) [Y11 Y13]
+ (7.954413176457233e-06) [X11 X13]
+ (0.0032769719312317224) [Y1 Y3]
+ (0.0032769719312317224) [X1 X3]
+ (0.10433064780651428) [Y0 Y2]
+ (0.10433064780651428) [X0 X2]
+ (0.11270386920332233) [Z10 Z12]
+ (0.11270386920332233) [Z11 Z13]
+ (0.11383573679388677) [Z4 Z12]
+ (0.11383573679388677) [Z5 Z13]
+ (0.11952438964682696) [Z6 Z10]
+ (0.11952438964682696) [Z7 Z11]
+ (0.1248999091723761) [Z4 Z10]
+ (0.1248999091723761) [Z5 Z11]
+ (0.12495807739503227) [Z2 Z4]
+ (0.12495807739503227) [Z3 Z5]
+ (0.1279950249246842) [Z2 Z10]
+ (0.1279950249246842) [Z3 Z11]
+ (0.13401715261963743) [Z6 Z12]
+ (0.13401715261963743) [Z7 Z13]
+ (0.13701191674040772) [Z4 Z6]
+ (0.13701191674040772) [Z5 Z7]
+ (0.13734953064261338) [Z6 Z11]
+ (0.13734953064261338) [Z7 Z10]
+ (0.13739104762683246) [Z2 Z6]
+ (0.13739104762683246) [Z3 Z7]
+ (0.13766872645852585) [Z8 Z10]
+ (0.13766872645852585) [Z9 Z11]
+ (0.14011289865354834) [Z2 Z12]
+ (0.14011289865354834) [Z3 Z13]
+ (0.14138905291942835) [Z10 Z13]
+ (0.14138905291942835) [Z11 Z12]
+ (0.1425799771248577) [Z4 Z11]
+ (0.1425799771248577) [Z5 Z10]
+ (0.14722943218766177) [Z8 Z11]
+ (0.14722943218766177) [Z9 Z10]
+ (0.14899430575065567) [Z4 Z7]
+ (0.14899430575065567) [Z5 Z6]
+ (0.1492635514738893) [Z10 Z11]
+ (0.14960702684445304) [Z4 Z8]
+ (0.14960702684445304) [Z5 Z9]
+ (0.1497348680349694) [Z8 Z12]
+ (0.1497348680349694) [Z9 Z13]
+ (0.150714081210083) [Z2 Z8]
+ (0.150714081210083) [Z3 Z9]
+ (0.15138327161428883) [Z6 Z13]
+ (0.15138327161428883) [Z7 Z12]
+ (0.1521504070886907) [Z4 Z13]
+ (0.1521504070886907) [Z5 Z12]
+ (0.15337968243314168) [Z2 Z11]
+ (0.15337968243314168) [Z3 Z10]
+ (0.15435748657223666) [Z12 Z13]
+ (0.1556901067175248) [Z2 Z13]
+ (0.1556901067175248) [Z3 Z12]
+ (0.15582269051553127) [Z8 Z13]
+ (0.15582269051553127) [Z9 Z12]
+ (0.15676396176430996) [Z4 Z9]
+ (0.15676396176430996) [Z5 Z8]
+ (0.15755314797985678) [Z4 Z5]
+ (0.16079764534838575) [Z2 Z5]
+ (0.16079764534838575) [Z3 Z4]
+ (0.16756653265461294) [Z6 Z8]
+ (0.16756653265461294) [Z7 Z9]
+ (0.1685348656157996) [Z2 Z7]
+ (0.1685348656157996) [Z3 Z6]
+ (0.18143991440303908) [Z6 Z9]
+ (0.18143991440303908) [Z7 Z8]
+ (0.1818908579075138) [Z2 Z3]
+ (0.18690820476912562) [Z2 Z9]
+ (0.18690820476912562) [Z3 Z8]
+ (0.19299723935364232) [Z0 Z10]
+ (0.19299723935364232) [Z1 Z11]
+ (0.19392534613270265) [Z6 Z7]
+ (0.19661770890342145) [Z0 Z4]
+ (0.19661770890342145) [Z1 Z5]
+ (0.19936354537360823) [Z0 Z5]
+ (0.19936354537360823) [Z1 Z4]
+ (0.20072866460441763) [Z0 Z11]
+ (0.20072866460441763) [Z1 Z10]
+ (0.21102659849791539) [Z0 Z12]
+ (0.21102659849791539) [Z1 Z13]
+ (0.21631037498631833) [Z0 Z13]
+ (0.21631037498631833) [Z1 Z12]
+ (0.2367108078383042) [Z0 Z2]
+ (0.2367108078383042) [Z1 Z3]
+ (0.24164663936017242) [Z0 Z6]
+ (0.24164663936017242) [Z1 Z7]
+ (0.248534833713143) [Z0 Z7]
+ (0.248534833713143) [Z1 Z6]
+ (0.2512944567459169) [Z0 Z3]
+ (0.2512944567459169) [Z1 Z2]
+ (0.2723251830660568) [Z0 Z8]
+ (0.2723251830660568) [Z1 Z9]
+ (0.278834544267234) [Z0 Z9]
+ (0.278834544267234) [Z1 Z8]
+ (1.1861763734860495) [Z0 Z1]
+ (-1.2260484988791041e-05) [Y5 Z6 Y7]
+ (-1.2260484988791041e-05) [X5 Z6 X7]
+ (-1.2260484988791038e-05) [Y4 Z5 Y6]
+ (-1.2260484988791038e-05) [X4 Z5 X6]
+ (-1.0722312158016717e-05) [Y10 Z11 Y12]
+ (-1.0722312158016717e-05) [X10 Z11 X12]
+ (-1.0722312158016712e-05) [Y11 Z12 Y13]
+ (-1.0722312158016712e-05) [X11 Z12 X13]
+ (-3.887051672440412e-06) [Y2 Z3 Y4]
+ (-3.887051672440412e-06) [X2 Z3 X4]
+ (-3.887051672440412e-06) [Y3 Z4 Y5]
+ (-3.887051672440412e-06) [X3 Z4 X5]
+ (0.12507032579772262) [Y0 Z1 Y2]
+ (0.12507032579772262) [X0 Z1 X2]
+ (0.1250703257977227) [Y1 Z2 Y3]
+ (0.1250703257977227) [X1 Z2 X3]
+ (-0.03831467029480392) [Y4 Y5 X12 X13]
+ (-0.03831467029480392) [X4 X5 Y12 Y13]
+ (-0.036194123559042606) [Y2 Y3 X8 X9]
+ (-0.036194123559042606) [X2 X3 Y8 Y9]
+ (-0.03583956795335349) [Y2 Y3 X4 X5]
+ (-0.03583956795335349) [X2 X3 Y4 Y5]
+ (-0.03114381798896712) [Y2 Y3 X6 X7]
+ (-0.03114381798896712) [X2 X3 Y6 Y7]
+ (-0.028685183716106035) [Y10 Y11 X12 X13]
+ (-0.028685183716106035) [X10 X11 Y12 Y13]
+ (-0.02599617759802128) [Y3 Z4 Z5 Y7]
+ (-0.02599617759802128) [X3 Z4 Z5 X7]
+ (-0.02538465750845745) [Y2 Y3 X10 X11]
+ (-0.02538465750845745) [X2 X3 Y10 Y11]
+ (-0.019028242443847345) [Y3 Y4 X11 X12]
+ (-0.019028242443847345) [X3 X4 Y11 Y12]
+ (-0.01782514099578641) [Y6 Y7 X10 X11]
+ (-0.01782514099578641) [X6 X7 Y10 Y11]
+ (-0.017680067952481598) [Y4 Y5 X10 X11]
+ (-0.017680067952481598) [X4 X5 Y10 Y11]
+ (-0.017366118994651403) [Y6 Y7 X12 X13]
+ (-0.017366118994651403) [X6 X7 Y12 Y13]
+ (-0.01458364890761271) [Y0 Y1 X2 X3]
+ (-0.01458364890761271) [X0 X1 Y2 Y3]
+ (-0.013873381748426143) [Y6 Y7 X8 X9]
+ (-0.013873381748426143) [X6 X7 Y8 Y9]
+ (-0.011982389010247941) [Y4 Y5 X6 X7]
+ (-0.011982389010247941) [X4 X5 Y6 Y7]
+ (-0.011285190200840917) [Y5 X6 X11 Y12]
+ (-0.011285190200840917) [X5 Y6 Y11 X12]
+ (-0.009560705729135942) [Y8 Y9 X10 X11]
+ (-0.009560705729135942) [X8 X9 Y10 Y11]
+ (-0.008125251921381017) [Y1 X2 X8 Y9]
+ (-0.008125251921381017) [Y1 Y2 Y8 Y9]
+ (-0.008125251921381017) [X1 X2 X8 X9]
+ (-0.008125251921381017) [X1 Y2 Y8 X9]
+ (-0.007731425250775277) [Y0 Y1 X10 X11]
+ (-0.007731425250775277) [X0 X1 Y10 Y11]
+ (-0.0071569349198569426) [Y4 Y5 X8 X9]
+ (-0.0071569349198569426) [X4 X5 Y8 Y9]
+ (-0.0068881943529705905) [Y0 Y1 X6 X7]
+ (-0.0068881943529705905) [X0 X1 Y6 Y7]
+ (-0.006509361201177232) [Y0 Y1 X8 X9]
+ (-0.006509361201177232) [X0 X1 Y8 Y9]
+ (-0.005283776488402962) [Y0 Y1 X12 X13]
+ (-0.005283776488402962) [X0 X1 Y12 Y13]
+ (-0.005143391768825092) [Y3 X4 X5 Y6]
+ (-0.005143391768825092) [X3 Y4 Y5 X6]
+ (-0.0046849033881552074) [Y1 X2 X6 Y7]
+ (-0.0046849033881552074) [Y1 Y2 Y6 Y7]
+ (-0.0046849033881552074) [X1 X2 X6 X7]
+ (-0.0046849033881552074) [X1 Y2 Y6 X7]
+ (-0.004575007626639206) [Y1 X2 X12 Y13]
+ (-0.004575007626639206) [Y1 Y2 Y12 Y13]
+ (-0.004575007626639206) [X1 X2 X12 X13]
+ (-0.004575007626639206) [X1 Y2 Y12 X13]
+ (-0.004424855449441849) [Y1 X2 X4 Y5]
+ (-0.004424855449441849) [Y1 Y2 Y4 Y5]
+ (-0.004424855449441849) [X1 X2 X4 X5]
+ (-0.004424855449441849) [X1 Y2 Y4 X5]
+ (-0.0034795118903342844) [Y2 Z3 Z5 Y6]
+ (-0.0034795118903342844) [X2 Z3 Z5 X6]
+ (-0.0034795118903342844) [Y3 Z4 Z6 Y7]
+ (-0.0034795118903342844) [X3 Z4 Z6 X7]
+ (-0.0027458364701868046) [Y0 Y1 X4 X5]
+ (-0.0027458364701868046) [X0 X1 Y4 Y5]
+ (-0.001799219493663005) [Y1 X2 X10 Y11]
+ (-0.001799219493663005) [Y1 Y2 Y10 Y11]
+ (-0.001799219493663005) [X1 X2 X10 X11]
+ (-0.001799219493663005) [X1 Y2 Y10 X11]
+ (-0.00029219862611107924) [Y7 Y8 X9 X10]
+ (-0.00029219862611107924) [X7 X8 Y9 Y10]
+ (-8.194261372454245e-06) [Z10 Y11 Z12 Y13]
+ (-8.194261372454245e-06) [Z10 X11 Z12 X13]
+ (-7.801707500746608e-06) [Y2 Z3 Y4 Z11]
+ (-7.801707500746608e-06) [X2 Z3 X4 Z11]
+ (-7.801707500746608e-06) [Y3 Z4 Y5 Z10]
+ (-7.801707500746608e-06) [X3 Z4 X5 Z10]
+ (-4.643051068662327e-06) [Y3 X4 X10 Y11]
+ (-4.643051068662327e-06) [Y3 Y4 Y10 Y11]
+ (-4.643051068662327e-06) [X3 X4 X10 X11]
+ (-4.643051068662327e-06) [X3 Y4 Y10 X11]
+ (-4.588855155791022e-06) [Y4 Z5 Y6 Z13]
+ (-4.588855155791022e-06) [X4 Z5 X6 Z13]
+ (-4.588855155791022e-06) [Y5 Z6 Y7 Z12]
+ (-4.588855155791022e-06) [X5 Z6 X7 Z12]
+ (-4.556569218356426e-06) [Y5 X6 X12 Y13]
+ (-4.556569218356426e-06) [Y5 Y6 Y12 Y13]
+ (-4.556569218356426e-06) [X5 X6 X12 X13]
+ (-4.556569218356426e-06) [X5 Y6 Y12 X13]
+ (-3.6945132947056688e-06) [Y4 X5 X11 Y12]
+ (-3.6945132947056688e-06) [Y4 Y5 Y11 Y12]
+ (-3.6945132947056688e-06) [X4 X5 X11 X12]
+ (-3.6945132947056688e-06) [X4 Y5 Y11 X12]
+ (-3.3440815563768127e-06) [Z0 Y5 Z6 Y7]
+ (-3.3440815563768127e-06) [Z0 X5 Z6 X7]
+ (-3.3440815563768127e-06) [Z1 Y4 Z5 Y6]
+ (-3.3440815563768127e-06) [Z1 X4 Z5 X6]
+ (-3.1586564320842818e-06) [Y2 Z3 Y4 Z10]
+ (-3.1586564320842818e-06) [X2 Z3 X4 Z10]
+ (-3.1586564320842818e-06) [Y3 Z4 Y5 Z11]
+ (-3.1586564320842818e-06) [X3 Z4 X5 Z11]
+ (-3.099349243501555e-06) [Z0 Y4 Z5 Y6]
+ (-3.099349243501555e-06) [Z0 X4 Z5 X6]
+ (-3.099349243501555e-06) [Z1 Y5 Z6 Y7]
+ (-3.099349243501555e-06) [Z1 X5 Z6 X7]
+ (-2.8909678817140503e-06) [Z6 Y11 Z12 Y13]
+ (-2.8909678817140503e-06) [Z6 X11 Z12 X13]
+ (-2.8909678817140503e-06) [Z7 Y10 Z11 Y12]
+ (-2.8909678817140503e-06) [Z7 X10 Z11 X12]
+ (-2.1776646052089496e-06) [Z0 Y10 Z11 Y12]
+ (-2.1776646052089496e-06) [Z0 X10 Z11 X12]
+ (-2.1776646052089496e-06) [Z1 Y11 Z12 Y13]
+ (-2.1776646052089496e-06) [Z1 X11 Z12 X13]
+ (-1.8818501831512258e-06) [Y4 Z5 Y6 Z9]
+ (-1.8818501831512258e-06) [X4 Z5 X6 Z9]
+ (-1.8818501831512258e-06) [Y5 Z6 Y7 Z8]
+ (-1.8818501831512258e-06) [X5 Z6 X7 Z8]
+ (-1.8551201216625774e-06) [Z6 Y10 Z11 Y12]
+ (-1.8551201216625774e-06) [Z6 X10 Z11 X12]
+ (-1.8551201216625774e-06) [Z7 Y11 Z12 Y13]
+ (-1.8551201216625774e-06) [Z7 X11 Z12 X13]
+ (-1.8540608579131114e-06) [Y4 Z5 Y6 Z7]
+ (-1.8540608579131114e-06) [X4 Z5 X6 Z7]
+ (-1.8163031699368767e-06) [Z4 Y11 Z12 Y13]
+ (-1.8163031699368767e-06) [Z4 X11 Z12 X13]
+ (-1.8163031699368767e-06) [Z5 Y10 Z11 Y12]
+ (-1.8163031699368767e-06) [Z5 X10 Z11 X12]
+ (-1.692397828628194e-06) [Y4 Z5 Y6 Z10]
+ (-1.692397828628194e-06) [X4 Z5 X6 Z10]
+ (-1.692397828628194e-06) [Y5 Z6 Y7 Z11]
+ (-1.692397828628194e-06) [X5 Z6 X7 Z11]
+ (-1.614879414052561e-06) [Z0 Y11 Z12 Y13]
+ (-1.614879414052561e-06) [Z0 X11 Z12 X13]
+ (-1.614879414052561e-06) [Z1 Y10 Z11 Y12]
+ (-1.614879414052561e-06) [Z1 X10 Z11 X12]
+ (-1.5973171979018738e-06) [Z8 Y10 Z11 Y12]
+ (-1.5973171979018738e-06) [Z8 X10 Z11 X12]
+ (-1.5973171979018738e-06) [Z9 Y11 Z12 Y13]
+ (-1.5973171979018738e-06) [Z9 X11 Z12 X13]
+ (-1.454842448974006e-06) [Y3 X4 X6 Y7]
+ (-1.454842448974006e-06) [Y3 Y4 Y6 Y7]
+ (-1.454842448974006e-06) [X3 X4 X6 X7]
+ (-1.454842448974006e-06) [X3 Y4 Y6 X7]
+ (-1.3980449080777676e-06) [Y4 Z5 Y6 Z8]
+ (-1.3980449080777676e-06) [X4 Z5 X6 Z8]
+ (-1.3980449080777676e-06) [Y5 Z6 Y7 Z9]
+ (-1.3980449080777676e-06) [X5 Z6 X7 Z9]
+ (-1.1954890098126153e-06) [Y2 Z3 Y4 Z7]
+ (-1.1954890098126153e-06) [X2 Z3 X4 Z7]
+ (-1.1954890098126153e-06) [Y3 Z4 Y5 Z6]
+ (-1.1954890098126153e-06) [X3 Z4 X5 Z6]
+ (-1.1908508081829136e-06) [Z0 Y3 Z4 Y5]
+ (-1.1908508081829136e-06) [Z0 X3 Z4 X5]
+ (-1.1908508081829136e-06) [Z1 Y2 Z3 Y4]
+ (-1.1908508081829136e-06) [Z1 X2 Z3 X4]
+ (-1.1708301369988116e-06) [Z2 Y5 Z6 Y7]
+ (-1.1708301369988116e-06) [Z2 X5 Z6 X7]
+ (-1.1708301369988116e-06) [Z3 Y4 Z5 Y6]
+ (-1.1708301369988116e-06) [Z3 X4 Z5 X6]
+ (-1.063228342433968e-06) [Z2 Y10 Z11 Y12]
+ (-1.063228342433968e-06) [Z2 X10 Z11 X12]
+ (-1.063228342433968e-06) [Z3 Y11 Z12 Y13]
+ (-1.063228342433968e-06) [Z3 X11 Z12 X13]
+ (-1.0358477600514726e-06) [Y6 X7 X11 Y12]
+ (-1.0358477600514726e-06) [Y6 Y7 Y11 Y12]
+ (-1.0358477600514726e-06) [X6 X7 X11 X12]
+ (-1.0358477600514726e-06) [X6 Y7 Y11 X12]
+ (-9.509249751380598e-07) [Z2 Y4 Z5 Y6]
+ (-9.509249751380598e-07) [Z2 X4 Z5 X6]
+ (-9.509249751380598e-07) [Z3 Y5 Z6 Y7]
+ (-9.509249751380598e-07) [Z3 X5 Z6 X7]
+ (-9.344557777244257e-07) [Z8 Y11 Z12 Y13]
+ (-9.344557777244257e-07) [Z8 X11 Z12 X13]
+ (-9.344557777244257e-07) [Z9 Y10 Z11 Y12]
+ (-9.344557777244257e-07) [Z9 X10 Z11 X12]
+ (-8.337746752815771e-07) [Z0 Y2 Z3 Y4]
+ (-8.337746752815771e-07) [Z0 X2 Z3 X4]
+ (-8.337746752815771e-07) [Z1 Y3 Z4 Y5]
+ (-8.337746752815771e-07) [Z1 X3 Z4 X5]
+ (-7.956895372163085e-07) [Y3 X4 X8 Y9]
+ (-7.956895372163085e-07) [Y3 Y4 Y8 Y9]
+ (-7.956895372163085e-07) [X3 X4 X8 X9]
+ (-7.956895372163085e-07) [X3 Y4 Y8 X9]
+ (-7.764994119155435e-07) [Y2 Z3 Y4 Z5]
+ (-7.764994119155435e-07) [X2 Z3 X4 Z5]
+ (-5.929765814984023e-07) [Z4 Y5 Z6 Y7]
+ (-5.929765814984023e-07) [Z4 X5 Z6 X7]
+ (-5.770052993949372e-07) [Y2 Z3 Y4 Z9]
+ (-5.770052993949372e-07) [X2 Z3 X4 Z9]
+ (-5.770052993949372e-07) [Y3 Z4 Y5 Z8]
+ (-5.770052993949372e-07) [X3 Z4 X5 Z8]
+ (-5.471647744820939e-07) [Y1 Y2 X11 X12]
+ (-5.471647744820939e-07) [X1 X2 Y11 Y12]
+ (-4.838052750734583e-07) [Y5 X6 X8 Y9]
+ (-4.838052750734583e-07) [Y5 Y6 Y8 Y9]
+ (-4.838052750734583e-07) [X5 X6 X8 X9]
+ (-4.838052750734583e-07) [X5 Y6 Y8 X9]
+ (-3.570761329013366e-07) [Y0 X1 X3 Y4]
+ (-3.570761329013366e-07) [Y0 Y1 Y3 Y4]
+ (-3.570761329013366e-07) [X0 X1 X3 X4]
+ (-3.570761329013366e-07) [X0 Y1 Y3 X4]
+ (-2.447323128752577e-07) [Y0 X1 X5 Y6]
+ (-2.447323128752577e-07) [Y0 Y1 Y5 Y6]
+ (-2.447323128752577e-07) [X0 X1 X5 X6]
+ (-2.447323128752577e-07) [X0 Y1 Y5 X6]
+ (-2.1990516186075172e-07) [Y2 X3 X5 Y6]
+ (-2.1990516186075172e-07) [Y2 Y3 Y5 Y6]
+ (-2.1990516186075172e-07) [X2 X3 X5 X6]
+ (-2.1990516186075172e-07) [X2 Y3 Y5 X6]
+ (-1.9332412770431303e-07) [Y1 X2 X3 Y4]
+ (-1.9332412770431303e-07) [X1 Y2 Y3 X4]
+ (-1.2919694861145907e-07) [Y1 Z2 Z3 Y5]
+ (-1.2919694861145907e-07) [X1 Z2 Z3 X5]
+ (1.7379332625094263e-07) [Y0 Z1 Z3 Y4]
+ (1.7379332625094263e-07) [X0 Z1 Z3 X4]
+ (1.7379332625094263e-07) [Y1 Z2 Z4 Y5]
+ (1.7379332625094263e-07) [X1 Z2 Z4 X5]
+ (1.9332412770431303e-07) [Y1 Y2 X3 X4]
+ (1.9332412770431303e-07) [X1 X2 Y3 Y4]
+ (2.1868423782137136e-07) [Y2 Z3 Y4 Z8]
+ (2.1868423782137136e-07) [X2 Z3 X4 Z8]
+ (2.1868423782137136e-07) [Y3 Z4 Y5 Z9]
+ (2.1868423782137136e-07) [X3 Z4 X5 Z9]
+ (2.5935343916139047e-07) [Y2 Z3 Y4 Z6]
+ (2.5935343916139047e-07) [X2 Z3 X4 Z6]
+ (2.5935343916139047e-07) [Y3 Z4 Y5 Z7]
+ (2.5935343916139047e-07) [X3 Z4 X5 Z7]
+ (3.6060718680436896e-07) [Y0 Z1 Z2 Y4]
+ (3.6060718680436896e-07) [X0 Z1 Z2 X4]
+ (3.6060718680436896e-07) [Y1 Z3 Z4 Y5]
+ (3.6060718680436896e-07) [X1 Z3 Z4 X5]
+ (5.471647744820939e-07) [Y1 X2 X11 Y12]
+ (5.471647744820939e-07) [X1 Y2 Y11 X12]
+ (5.627851911563885e-07) [Y0 X1 X11 Y12]
+ (5.627851911563885e-07) [Y0 Y1 Y11 Y12]
+ (5.627851911563885e-07) [X0 X1 X11 X12]
+ (5.627851911563885e-07) [X0 Y1 Y11 X12]
+ (6.628614201774483e-07) [Y8 X9 X11 Y12]
+ (6.628614201774483e-07) [Y8 Y9 Y11 Y12]
+ (6.628614201774483e-07) [X8 X9 X11 X12]
+ (6.628614201774483e-07) [X8 Y9 Y11 X12]
+ (1.1094407591286987e-06) [Z2 Y11 Z12 Y13]
+ (1.1094407591286987e-06) [Z2 X11 Z12 X13]
+ (1.1094407591286987e-06) [Z3 Y10 Z11 Y12]
+ (1.1094407591286987e-06) [Z3 X10 Z11 X12]
+ (1.6021167407252466e-06) [Z2 Y3 Z4 Y5]
+ (1.6021167407252466e-06) [Z2 X3 Z4 X5]
+ (1.8782101247687923e-06) [Z4 Y10 Z11 Y12]
+ (1.8782101247687923e-06) [Z4 X10 Z11 X12]
+ (1.8782101247687923e-06) [Z5 Y11 Z12 Y13]
+ (1.8782101247687923e-06) [Z5 X11 Z12 X13]
+ (2.1726691015626666e-06) [Y2 X3 X11 Y12]
+ (2.1726691015626666e-06) [Y2 Y3 Y11 Y12]
+ (2.1726691015626666e-06) [X2 X3 X11 X12]
+ (2.1726691015626666e-06) [X2 Y3 Y11 X12]
+ (3.11744794603745e-06) [Y0 Z2 Z3 Y4]
+ (3.11744794603745e-06) [X0 Z2 Z3 X4]
+ (3.5390541846291337e-06) [Y2 Z3 Y4 Z12]
+ (3.5390541846291337e-06) [X2 Z3 X4 Z12]
+ (3.5390541846291337e-06) [Y3 Z4 Y5 Z13]
+ (3.5390541846291337e-06) [X3 Z4 X5 Z13]
+ (4.28191388495378e-06) [Y4 Z5 Y6 Z11]
+ (4.28191388495378e-06) [X4 Z5 X6 Z11]
+ (4.28191388495378e-06) [Y5 Z6 Y7 Z10]
+ (4.28191388495378e-06) [X5 Z6 X7 Z10]
+ (5.275883122295677e-06) [Y3 X4 X12 Y13]
+ (5.275883122295677e-06) [Y3 Y4 Y12 Y13]
+ (5.275883122295677e-06) [X3 X4 X12 X13]
+ (5.275883122295677e-06) [X3 Y4 Y12 X13]
+ (5.9743117135819745e-06) [Y5 X6 X10 Y11]
+ (5.9743117135819745e-06) [Y5 Y6 Y10 Y11]
+ (5.9743117135819745e-06) [X5 X6 X10 X11]
+ (5.9743117135819745e-06) [X5 Y6 Y10 X11]
+ (7.954413176457233e-06) [Y10 Z11 Y12 Z13]
+ (7.954413176457233e-06) [X10 Z11 X12 Z13]
+ (8.814937306924811e-06) [Y2 Z3 Y4 Z13]
+ (8.814937306924811e-06) [X2 Z3 X4 Z13]
+ (8.814937306924811e-06) [Y3 Z4 Y5 Z12]
+ (8.814937306924811e-06) [X3 Z4 X5 Z12]
+ (0.00029219862611107924) [Y7 X8 X9 Y10]
+ (0.00029219862611107924) [X7 Y8 Y9 X10]
+ (0.0004956762314917412) [Y2 Z4 Z5 Y6]
+ (0.0004956762314917412) [X2 Z4 Z5 X6]
+ (0.0011059037691897337) [Y0 Z1 Y2 Z5]
+ (0.0011059037691897337) [X0 Z1 X2 Z5]
+ (0.0011059037691897337) [Y1 Z2 Y3 Z4]
+ (0.0011059037691897337) [X1 Z2 X3 Z4]
+ (0.0016638798784908073) [Y2 Z3 Z4 Y6]
+ (0.0016638798784908073) [X2 Z3 Z4 X6]
+ (0.0016638798784908073) [Y3 Z5 Z6 Y7]
+ (0.0016638798784908073) [X3 Z5 Z6 X7]
+ (0.0017560707018412875) [Y0 Z1 Y2 Z11]
+ (0.0017560707018412875) [X0 Z1 X2 Z11]
+ (0.0017560707018412875) [Y1 Z2 Y3 Z10]
+ (0.0017560707018412875) [X1 Z2 X3 Z10]
+ (0.0023262306231581417) [Y0 Z1 Y2 Z13]
+ (0.0023262306231581417) [X0 Z1 X2 Z13]
+ (0.0023262306231581417) [Y1 Z2 Y3 Z12]
+ (0.0023262306231581417) [X1 Z2 X3 Z12]
+ (0.0027458364701868046) [Y0 X1 X4 Y5]
+ (0.0027458364701868046) [X0 Y1 Y4 X5]
+ (0.00292976867475113) [Y0 Z1 Y2 Z9]
+ (0.00292976867475113) [X0 Z1 X2 Z9]
+ (0.00292976867475113) [Y1 Z2 Y3 Z8]
+ (0.00292976867475113) [X1 Z2 X3 Z8]
+ (0.003276971931231722) [Y0 Z1 Y2 Z3]
+ (0.003276971931231722) [X0 Z1 X2 Z3]
+ (0.0033476175306662607) [Y0 Z1 Y2 Z7]
+ (0.0033476175306662607) [X0 Z1 X2 Z7]
+ (0.0033476175306662607) [Y1 Z2 Y3 Z6]
+ (0.0033476175306662607) [X1 Z2 X3 Z6]
+ (0.003555290195504292) [Y0 Z1 Y2 Z10]
+ (0.003555290195504292) [X0 Z1 X2 Z10]
+ (0.003555290195504292) [Y1 Z2 Y3 Z11]
+ (0.003555290195504292) [X1 Z2 X3 Z11]
+ (0.005143391768825092) [Y3 Y4 X5 X6]
+ (0.005143391768825092) [X3 X4 Y5 Y6]
+ (0.005283776488402962) [Y0 X1 X12 Y13]
+ (0.005283776488402962) [X0 Y1 Y12 X13]
+ (0.005530759218631583) [Y0 Z1 Y2 Z4]
+ (0.005530759218631583) [X0 Z1 X2 Z4]
+ (0.005530759218631583) [Y1 Z2 Y3 Z5]
+ (0.005530759218631583) [X1 Z2 X3 Z5]
+ (0.006509361201177232) [Y0 X1 X8 Y9]
+ (0.006509361201177232) [X0 Y1 Y8 X9]
+ (0.0068881943529705905) [Y0 X1 X6 Y7]
+ (0.0068881943529705905) [X0 Y1 Y6 X7]
+ (0.0069012382497973465) [Y0 Z1 Y2 Z12]
+ (0.0069012382497973465) [X0 Z1 X2 Z12]
+ (0.0069012382497973465) [Y1 Z2 Y3 Z13]
+ (0.0069012382497973465) [X1 Z2 X3 Z13]
+ (0.0071569349198569426) [Y4 X5 X8 Y9]
+ (0.0071569349198569426) [X4 Y5 Y8 X9]
+ (0.007731425250775277) [Y0 X1 X10 Y11]
+ (0.007731425250775277) [X0 Y1 Y10 X11]
+ (0.008032520918821468) [Y0 Z1 Y2 Z6]
+ (0.008032520918821468) [X0 Z1 X2 Z6]
+ (0.008032520918821468) [Y1 Z2 Y3 Z7]
+ (0.008032520918821468) [X1 Z2 X3 Z7]
+ (0.009560705729135942) [Y8 X9 X10 Y11]
+ (0.009560705729135942) [X8 Y9 Y10 X11]
+ (0.011055020596132148) [Y0 Z1 Y2 Z8]
+ (0.011055020596132148) [X0 Z1 X2 Z8]
+ (0.011055020596132148) [Y1 Z2 Y3 Z9]
+ (0.011055020596132148) [X1 Z2 X3 Z9]
+ (0.011285190200840917) [Y5 Y6 X11 X12]
+ (0.011285190200840917) [X5 X6 Y11 Y12]
+ (0.011307274008848067) [Y7 Z8 Z9 Y11]
+ (0.011307274008848067) [X7 Z8 Z9 X11]
+ (0.011982389010247941) [Y4 X5 X6 Y7]
+ (0.011982389010247941) [X4 Y5 Y6 X7]
+ (0.013873381748426143) [Y6 X7 X8 Y9]
+ (0.013873381748426143) [X6 Y7 Y8 X9]
+ (0.01458364890761271) [Y0 X1 X2 Y3]
+ (0.01458364890761271) [X0 Y1 Y2 X3]
+ (0.017366118994651403) [Y6 X7 X12 Y13]
+ (0.017366118994651403) [X6 Y7 Y12 X13]
+ (0.017680067952481598) [Y4 X5 X10 Y11]
+ (0.017680067952481598) [X4 Y5 Y10 X11]
+ (0.01782514099578641) [Y6 X7 X10 Y11]
+ (0.01782514099578641) [X6 Y7 Y10 X11]
+ (0.019028242443847345) [Y3 X4 X11 Y12]
+ (0.019028242443847345) [X3 Y4 Y11 X12]
+ (0.02538465750845745) [Y2 X3 X10 Y11]
+ (0.02538465750845745) [X2 Y3 Y10 X11]
+ (0.028685183716106035) [Y10 X11 X12 Y13]
+ (0.028685183716106035) [X10 Y11 Y12 X13]
+ (0.029812424517345677) [Y6 Z7 Z8 Y10]
+ (0.029812424517345677) [X6 Z7 Z8 X10]
+ (0.029812424517345677) [Y7 Z9 Z10 Y11]
+ (0.029812424517345677) [X7 Z9 Z10 X11]
+ (0.030104623143456757) [Y6 Z7 Z9 Y10]
+ (0.030104623143456757) [X6 Z7 Z9 X10]
+ (0.030104623143456757) [Y7 Z8 Z10 Y11]
+ (0.030104623143456757) [X7 Z8 Z10 X11]
+ (0.030787505389143925) [Y6 Z8 Z9 Y10]
+ (0.030787505389143925) [X6 Z8 Z9 X10]
+ (0.03114381798896712) [Y2 X3 X6 Y7]
+ (0.03114381798896712) [X2 Y3 Y6 X7]
+ (0.03583956795335349) [Y2 X3 X4 Y5]
+ (0.03583956795335349) [X2 Y3 Y4 X5]
+ (0.036194123559042606) [Y2 X3 X8 Y9]
+ (0.036194123559042606) [X2 Y3 Y8 X9]
+ (0.03831467029480392) [Y4 X5 X12 Y13]
+ (0.03831467029480392) [X4 Y5 Y12 X13]
+ (0.10433064780651428) [Z0 Y1 Z2 Y3]
+ (0.10433064780651428) [Z0 X1 Z2 X3]
+ (-0.12133276911042341) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133276911042341) [X3 Z4 Z5 Z6 X7]
+ (-0.1213327691104234) [Y2 Z3 Z4 Z5 Y6]
+ (-0.1213327691104234) [X2 Z3 Z4 Z5 X6]
+ (3.2020768810808507e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.2020768810808507e-06) [X1 Z2 Z3 Z4 X5]
+ (3.2020768810808515e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.2020768810808515e-06) [X0 Z1 Z2 Z3 X4]
+ (0.22848106564918694) [Y6 Z7 Z8 Z9 Y10]
+ (0.22848106564918694) [X6 Z7 Z8 Z9 X10]
+ (0.22848106564918694) [Y7 Z8 Z9 Z10 Y11]
+ (0.22848106564918694) [X7 Z8 Z9 Z10 X11]
+ (-0.03276765782329043) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.03276765782329043) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.03276765782329043) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.03276765782329043) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273062) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.027115036845273062) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273062) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.027115036845273062) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.02599617759802128) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.02599617759802128) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.017561202409646134) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.017561202409646134) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.017561202409646134) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.017561202409646134) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.014564531231173013) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564531231173013) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564531231173013) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564531231173013) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.012215040997613932) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.012215040997613932) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.012215040997613932) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.012215040997613932) [X4 Z5 X6 X11 Z12 X13]
+ (-0.012215040997613932) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.012215040997613932) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.012215040997613932) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.012215040997613932) [X5 Z6 X7 X10 Z11 X12]
+ (-0.011756013419819269) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.011756013419819269) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.011756013419819269) [X3 Z4 Z5 X6 X8 X9]
+ (-0.011756013419819269) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.008764827575688795) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.008764827575688795) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.008764827575688795) [X2 Z3 Z4 X5 X11 X12]
+ (-0.008764827575688795) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.008764827575688795) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.008764827575688795) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.008764827575688795) [X3 X4 X10 Z11 Z12 X13]
+ (-0.008764827575688795) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.008125251921381017) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125251921381017) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.007306759928832979) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306759928832979) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306759928832979) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306759928832979) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005805188989826866) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.005805188989826866) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.005805188989826866) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.005805188989826866) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.0056526209780173664) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.0056526209780173664) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.0056526209780173664) [X0 X1 X3 Z4 Z5 X6]
+ (-0.0056526209780173664) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.005143391768825092) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.005143391768825092) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.005143391768825092) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.005143391768825092) [X2 Z3 X4 X5 Z6 X7]
+ (-0.004684903388155206) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.004684903388155206) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668620318776303) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668620318776303) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.004575007626639206) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.004575007626639206) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424855449441849) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424855449441849) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.004158797381840071) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.004158797381840071) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.004158797381840071) [X3 Z4 Z5 X6 X12 X13]
+ (-0.004158797381840071) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.0034937903598901516) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.0034937903598901516) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.0034937903598901516) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.0034937903598901516) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.002779026799025543) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.002779026799025543) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.0022939566113524814) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.0022939566113524814) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.001799219493663005) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.001799219493663005) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278753941369603) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.0017278753941369603) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298507967730161) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298507967730161) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298507967730161) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298507967730161) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533856254125506) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533856254125506) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008145313270956425) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008145313270956425) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008145313270956425) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008145313270956425) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.735036880591797e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735036880591797e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735036880591797e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735036880591797e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.774817864940294e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.774817864940294e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.774817864940294e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.774817864940294e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.51836221596692e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.51836221596692e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.51836221596692e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.51836221596692e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.444344676238104e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.444344676238104e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.444344676238104e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.444344676238104e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.5243738487661216e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.5243738487661216e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.5243738487661216e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.5243738487661216e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.290028433556012e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.290028433556012e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.290028433556012e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.290028433556012e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.9743117135819745e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.9743117135819745e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.275883122295677e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.275883122295677e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.643051068662326e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.643051068662326e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.556569218356426e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.556569218356426e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.253224225664189e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.253224225664189e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.7696594522402892e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.7696594522402892e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.6945132947056683e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.6945132947056683e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.6102971308346565e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.6102971308346565e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.6102971308346565e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.6102971308346565e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.3131455001497706e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.3131455001497706e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.2774831957911555e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.2774831957911555e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.2774831957911555e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.2774831957911555e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.2112283486163505e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.2112283486163505e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.2112283486163505e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.2112283486163505e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.1513463112588878e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.1513463112588878e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.0882507114597616e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.0882507114597616e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.1726691015626666e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.1726691015626666e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.454842448974006e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.454842448974006e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.33047318870219e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.33047318870219e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2283337824109085e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2283337824109085e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.0358477600514726e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.0358477600514726e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956895372163085e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956895372163085e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733197742605319e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733197742605319e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733197742605319e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733197742605319e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.628614201774483e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.628614201774483e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.556281914841462e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.556281914841462e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.556281914841462e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.556281914841462e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.418291574943104e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.418291574943104e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.418291574943104e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.418291574943104e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.927453082975489e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927453082975489e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927453082975489e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927453082975489e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627851911563885e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627851911563885e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660624956771e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660624956771e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660624956771e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660624956771e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660624956771e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660624956771e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660624956771e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660624956771e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.838052750734583e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.838052750734583e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.5707613290133667e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.5707613290133667e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.328139350435009e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.328139350435009e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.0868265652400643e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.0868265652400643e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.0868265652400643e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.0868265652400643e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.447323128752577e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.447323128752577e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.371328947811605e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.371328947811605e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.371328947811605e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.371328947811605e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.1990516186075172e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.1990516186075172e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.9332412770431303e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.9332412770431303e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.9332412770431303e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.9332412770431303e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.83942091537124e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.83942091537124e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.83942091537124e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.83942091537124e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.5510539176222496e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.5510539176222496e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.5510539176222496e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.5510539176222496e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.380778148014566e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.380778148014566e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.380778148014566e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.380778148014566e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.3807781480145658e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.3807781480145658e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.3807781480145658e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.3807781480145658e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.3807781480145658e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.3807781480145658e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.3807781480145658e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.3807781480145658e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.2919694861145907e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.2919694861145907e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.1076325600034288e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.1076325600034288e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.1076325600034288e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.1076325600034288e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.1076325600034288e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.1076325600034288e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.1076325600034288e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.1076325600034288e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.057446596298295e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057446596298295e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057446596298295e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057446596298295e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649310136046995e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649310136046995e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649310136046995e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649310136046995e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.8394209153712403e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.8394209153712403e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.8394209153712403e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.8394209153712403e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.1990516186075172e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.1990516186075172e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.447323128752577e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.447323128752577e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.236259961416304e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.236259961416304e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.236259961416304e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.236259961416304e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.328139350435009e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.328139350435009e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.5707613290133667e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.5707613290133667e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.838052750734583e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.838052750734583e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.627851911563885e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627851911563885e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.628614201774483e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.628614201774483e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956895372163085e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956895372163085e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.306536652226529e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.306536652226529e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.306536652226529e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.306536652226529e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.0358477600514726e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.0358477600514726e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2283337824109085e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2283337824109085e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.2393363217466594e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393363217466594e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393363217466594e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393363217466594e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.33047318870219e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.33047318870219e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.454842448974006e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.454842448974006e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.1726691015626666e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.1726691015626666e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.0882507114597616e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.0882507114597616e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.11744794603745e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.11744794603745e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.1513463112588878e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.1513463112588878e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.3131455001497706e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.3131455001497706e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.3343312894575205e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.3343312894575205e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.6945132947056683e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.6945132947056683e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.1839325594337486e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.1839325594337486e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.556569218356426e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.556569218356426e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.643051068662326e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.643051068662326e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.275883122295677e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.275883122295677e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.9743117135819745e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.9743117135819745e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.00029219862611107924) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.00029219862611107924) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.00029219862611107924) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.00029219862611107924) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004956762314917412) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004956762314917412) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650070219499201) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650070219499201) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650070219499201) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650070219499201) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533856254125506) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533856254125506) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.0016095313817213938) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.0016095313817213938) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.0016095313817213938) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.0016095313817213938) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0016676041811440803) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.0016676041811440803) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.0016676041811440803) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.0016676041811440803) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.0017278753941369603) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.0017278753941369603) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.001799219493663005) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.001799219493663005) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.0022939566113524814) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.0022939566113524814) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.0024629170071339443) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.0024629170071339443) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.0024629170071339443) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.0024629170071339443) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.003961560792496562) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.003961560792496562) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.003961560792496562) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.003961560792496562) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.004424855449441849) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424855449441849) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004575007626639206) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.004575007626639206) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668620318776303) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668620318776303) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.004684903388155206) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.004684903388155206) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005324835234221695) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.005324835234221695) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.005324835234221695) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.005324835234221695) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.005368659358109507) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368659358109507) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368659358109507) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368659358109507) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.008125251921381017) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125251921381017) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.00889073152269456) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.00889073152269456) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.00889073152269456) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.00889073152269456) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.010263414868158547) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.010263414868158547) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.010263414868158547) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.010263414868158547) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.01054042590767148) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.01054042590767148) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.01054042590767148) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.01054042590767148) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010960074940542517) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010960074940542517) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010960074940542517) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010960074940542517) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307274008848067) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307274008848067) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.014411099430130955) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411099430130955) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411099430130955) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411099430130955) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.015225630757226598) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.015225630757226598) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.015225630757226598) [X3 Z4 Z5 X6 X10 X11]
+ (0.015225630757226598) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.015588250102380243) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.015588250102380243) [X2 Z3 X4 X10 Z11 X12]
+ (0.015588250102380243) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.015588250102380243) [X3 Z4 X5 X11 Z12 X13]
+ (0.018266834869375498) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.018266834869375498) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.018266834869375498) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.018266834869375498) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.019020423173039924) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.019020423173039924) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.019020423173039924) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.019020423173039924) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.020175921723535477) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.020175921723535477) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.020175921723535477) [X4 Z5 Z6 X7 X11 X12]
+ (0.020175921723535477) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.020175921723535477) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.020175921723535477) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.020175921723535477) [X5 X6 X10 Z11 Z12 X13]
+ (0.020175921723535477) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.02435307767806904) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.02435307767806904) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.02435307767806904) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.02435307767806904) [X2 Z3 X4 X11 Z12 X13]
+ (0.02435307767806904) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.02435307767806904) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.02435307767806904) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.02435307767806904) [X3 Z4 X5 X10 Z11 X12]
+ (0.024389082531149433) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.024389082531149433) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.024389082531149433) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.024389082531149433) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.025104957138844496) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.025104957138844496) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.025104957138844496) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.025104957138844496) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.030787505389143925) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.030787505389143925) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.04587947078129816) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.04587947078129816) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.056007330877807494) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.056007330877807494) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.056007330877807494) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.056007330877807494) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661341) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.05608468124661341) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661341) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.05608468124661341) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.631277928700954e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631277928700954e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.631277928700953e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631277928700953e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.595086007184588e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.595086007184588e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.5950860071845866e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.5950860071845866e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.04274327701378391) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.04274327701378391) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (0.04274327701378391) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04274327701378391) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.03956441632289347) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.03956441632289347) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.03956441632289347) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03956441632289347) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03935916802205314) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.03935916802205314) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.03935916802205314) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.03935916802205314) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.03931805194719762) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03931805194719762) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.03931805194719762) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03931805194719762) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03560837898831261) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.03560837898831261) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.029903789512624866) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.029903789512624866) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.029903789512624866) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.029903789512624866) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.02873077955190556) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.02873077955190556) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.02873077955190556) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.02873077955190556) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.025637238296026824) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.025637238296026824) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.025637238296026824) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.025637238296026824) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.02475546329289101) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.02475546329289101) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.02475546329289101) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.02475546329289101) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.024282117354692948) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.024282117354692948) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.023145130929529065) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.023145130929529065) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.02252844019601295) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.02252844019601295) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.021433810721601006) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.021433810721601006) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.021433810721601006) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.021433810721601006) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251592) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.019257505095251592) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019028242443847345) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.019028242443847345) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.01888903030494289) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.01888903030494289) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.01888903030494289) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.01888903030494289) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.016024603689179642) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.016024603689179642) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.015225630757226598) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225630757226598) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.014603704729162122) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.014603704729162122) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564531231173013) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564531231173013) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.01175601341981927) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.01175601341981927) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.011285190200840917) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.011285190200840917) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.00984174924696267) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.00984174924696267) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.009612634606847185) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612634606847185) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612634606847185) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612634606847185) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.00846997879102387) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.00846997879102387) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.007306759928832978) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.007306759928832978) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0059237983365613475) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.0059237983365613475) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.0056526209780173664) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.0056526209780173664) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368659358109507) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368659358109507) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.004158797381840071) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.004158797381840071) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0033566705638328927) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.0033566705638328927) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.0033566705638328927) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.0033566705638328927) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.003267513854423554) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.003267513854423554) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.003267513854423554) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.003267513854423554) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.002779026799025543) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.002779026799025543) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.0026860409778066132) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.0026860409778066132) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.0026860409778066132) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.0026860409778066132) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.0022939566113524814) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524814) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.0022939566113524814) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524814) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.0009581655836696526) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.0009581655836696526) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0009581655836696526) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.0009581655836696526) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.0009581655836696526) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.0009581655836696526) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0009581655836696526) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.0009581655836696526) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.00024636437569584726) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00024636437569584726) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303549712) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.00013840177303549712) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.00013840177303549712) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.00013840177303549712) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.735036880591797e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735036880591797e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.610358530637906e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.610358530637906e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.610358530637906e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.610358530637906e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808795935526e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.5316808795935526e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808795935526e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5316808795935526e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.806102775500193e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.806102775500193e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.806102775500193e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.806102775500193e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.089799467802333e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.089799467802333e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.089799467802333e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.089799467802333e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.652209669977058e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.652209669977058e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.652209669977058e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.652209669977058e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851834439452e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.481851834439452e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851834439452e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.481851834439452e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.071480736554009e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.071480736554009e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.071480736554009e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.071480736554009e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.734622038946184e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.734622038946184e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.734622038946184e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.734622038946184e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.728843147417908e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.728843147417908e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.728843147417908e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.728843147417908e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.253224225664189e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.253224225664189e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.7696594522402892e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.7696594522402892e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.544395429423524e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.544395429423524e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.544395429423524e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.544395429423524e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.544395429423524e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.544395429423524e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.544395429423524e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.544395429423524e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.3609563203844245e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563203844245e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563203844245e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.3609563203844245e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.103215604843574e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.103215604843574e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.103215604843574e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.103215604843574e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.011122098419903e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.011122098419903e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.011122098419903e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.011122098419903e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.942946836802792e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.942946836802792e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.942946836802792e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.942946836802792e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174773247215e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.6541174773247215e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174773247215e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.6541174773247215e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.5224930676683058e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224930676683058e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224930676683058e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224930676683058e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224930676683058e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930676683058e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930676683058e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224930676683058e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.2283337824109085e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337824109085e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.2283337824109085e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337824109085e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.988770288320358e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988770288320358e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988770288320358e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988770288320358e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.867765104435319e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104435319e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104435319e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.867765104435319e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.189990975626834e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.189990975626834e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246207177152e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246207177152e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.471647744820939e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.471647744820939e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.5614471796376247e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.5614471796376247e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.5614471796376247e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.5614471796376247e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.523389678118293e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.523389678118293e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.4273231086827337e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.4273231086827337e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.4273231086827337e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.4273231086827337e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.328139350435009e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350435009e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.328139350435009e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350435009e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.0868265652400643e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.0868265652400643e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.8882935947807045e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935947807045e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935947807045e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.8882935947807045e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.371328947811605e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.371328947811605e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.8394209153712403e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.8394209153712403e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.057446596298295e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057446596298295e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.5371780962314595e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.5371780962314595e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.5371780962314595e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.5371780962314595e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.057446596298295e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057446596298295e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.209350642367092e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350642367092e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350642367092e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.209350642367092e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783553760623e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783553760623e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783553760623e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783553760623e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.8394209153712403e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.8394209153712403e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.371328947811605e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.371328947811605e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.0868265652400643e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.0868265652400643e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.523389678118293e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.523389678118293e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.471647744820939e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.471647744820939e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246207177152e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246207177152e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189990975626834e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.189990975626834e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.33047318870219e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.33047318870219e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.33047318870219e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.33047318870219e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.628853243590582e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.628853243590582e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.628853243590582e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.628853243590582e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.6893489515400427e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.6893489515400427e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.6893489515400427e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.6893489515400427e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.7455184005914887e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.7455184005914887e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.7455184005914887e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.7455184005914887e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.7455184005914887e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.7455184005914887e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.7455184005914887e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.7455184005914887e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.211842019208348e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.211842019208348e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.211842019208348e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.211842019208348e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.211842019208348e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.211842019208348e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.211842019208348e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.211842019208348e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.3131455001497706e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.3131455001497706e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.3131455001497706e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.3131455001497706e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.3343312894575205e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.3343312894575205e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.1839325594337486e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.1839325594337486e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.735036880591797e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735036880591797e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00024636437569584726) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00024636437569584726) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00044585351288407415) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.00044585351288407415) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.00044585351288407415) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.00044585351288407415) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940221543005511) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005511) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005511) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005511) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005511) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005511) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940221543005511) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005511) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533856254125504) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533856254125504) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533856254125504) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533856254125504) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0010435246534907761) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435246534907761) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435246534907761) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435246534907761) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.0012803060973496886) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.0012803060973496886) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.0012803060973496886) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.0012803060973496886) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.0013038004788126993) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.0013038004788126993) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.0013038004788126993) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.0013038004788126993) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.0022619660624823516) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.0022619660624823516) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.0022619660624823516) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.0022619660624823516) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.0022619660624823516) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.0022619660624823516) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.0022619660624823516) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.0022619660624823516) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.003989841456619311) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.003989841456619311) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.003989841456619311) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.003989841456619311) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.004158797381840071) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.004158797381840071) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.00431103850791433) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.00431103850791433) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.00431103850791433) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.00431103850791433) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.00463697666118258) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.00463697666118258) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.00463697666118258) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.00463697666118258) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.005114473831660377) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.005114473831660377) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.005114473831660377) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.005114473831660377) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.005114473831660377) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660377) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660377) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005114473831660377) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005262642473076853) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262642473076853) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262642473076853) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262642473076853) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368659358109507) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368659358109507) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005379937155839381) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.005379937155839381) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.005379937155839381) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.005379937155839381) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.0056526209780173664) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.0056526209780173664) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.005708495985960927) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708495985960927) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708495985960927) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708495985960927) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.0059237983365613475) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.0059237983365613475) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.007306759928832978) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.007306759928832978) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00846997879102387) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.00846997879102387) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.00984174924696267) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.00984174924696267) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.011285190200840917) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.011285190200840917) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.01175601341981927) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.01175601341981927) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.014564531231173013) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564531231173013) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603704729162122) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.014603704729162122) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.015225630757226598) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225630757226598) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.016024603689179642) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.016024603689179642) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.019028242443847345) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.019028242443847345) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.019257505095251592) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.019257505095251592) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.045879470781298164) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.045879470781298164) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.3693708936615616) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.3693708936615616) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.3693708936615616) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.3693708936615616) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.2816425776702291) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.2816425776702291) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.281642577670229) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.281642577670229) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.09065144207036475) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.09065144207036475) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.09065144207036475) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.09065144207036475) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863623) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684737589863623) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863623) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684737589863623) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635021950635021) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635021950635021) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635021950635021) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635021950635021) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214034) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.06752385099214034) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214034) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.06752385099214034) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03560837898831261) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.03560837898831261) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.034903343373661744) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.034903343373661744) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.034903343373661744) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.034903343373661744) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883829964) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.024591860883829964) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883829964) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.024591860883829964) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.02428211735469295) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.02428211735469295) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.02314513092952906) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.02314513092952906) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.02252844019601295) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.02252844019601295) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019538050311314767) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.019538050311314767) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.019538050311314767) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.019538050311314767) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.017091553155898886) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.017091553155898886) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.017091553155898886) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.017091553155898886) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.016024603689179642) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.016024603689179642) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.016024603689179642) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.016024603689179642) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.010311482489831783) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831783) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831783) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311482489831783) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.00984174924696267) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.00984174924696267) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.00984174924696267) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.00984174924696267) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209862) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209862) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209862) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008826368514209862) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008541996625454839) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454839) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454839) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454839) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454839) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454839) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454839) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008541996625454839) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00846997879102387) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.00846997879102387) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.00846997879102387) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.00846997879102387) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.004668620318776303) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668620318776303) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.003876470899336946) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.003876470899336946) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.003804066171728537) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728537) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728537) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003804066171728537) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484157300217879) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484157300217879) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0033566705638328927) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.0033566705638328927) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.003267513854423554) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.003267513854423554) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0021413612231015694) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.0021413612231015694) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.0017278753941369603) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.0017278753941369603) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.0016407548553124367) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0016407548553124367) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214169213) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0014528843214169213) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214169213) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0014528843214169213) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870896771024421) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007870896771024421) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192743499487794) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192743499487794) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.00019400857029755881) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00019400857029755881) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303549712) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.00013840177303549712) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.141625221154273e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.141625221154273e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.141625221154273e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.141625221154273e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.071480736554009e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.071480736554009e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.1513463112588878e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.1513463112588878e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0882507114597616e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.0882507114597616e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.988511706224744e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.988511706224744e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8742990714502686e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.8742990714502686e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.3609563203844245e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.3609563203844245e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.3002946562663052e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3002946562663052e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.1468376508070932e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.1468376508070932e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.1468376508070932e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.1468376508070932e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.352332103590471e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.352332103590471e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.352332103590471e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.352332103590471e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091637199516312e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637199516312e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637199516312e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637199516312e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637199516312e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637199516312e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637199516312e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091637199516312e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.074305986352681e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.074305986352681e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.074305986352681e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.074305986352681e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128986672915e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128986672915e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128986672915e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128986672915e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867765104435319e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.867765104435319e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560692465281139e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692465281139e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692465281139e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692465281139e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692465281139e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692465281139e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692465281139e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560692465281139e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.997018422236147e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.997018422236147e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.997018422236147e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.997018422236147e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.997018422236147e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.997018422236147e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.997018422236147e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.997018422236147e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.568247521398017e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.568247521398017e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.568247521398017e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.568247521398017e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393085546214e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393085546214e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393085546214e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393085546214e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393085546214e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393085546214e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.3767393085546214e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393085546214e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.8882935947807045e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.8882935947807045e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6863815468884714e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.6863815468884714e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035783553760626e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7035783553760626e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.209350642367092e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.209350642367092e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773245175891e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773245175891e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773245175891e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773245175891e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773245175891e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773245175891e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773245175891e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773245175891e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.974225379764584e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.974225379764584e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.974225379764584e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.974225379764584e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.047471655652236e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.047471655652236e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.047471655652236e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.047471655652236e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.209350642367092e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.209350642367092e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0717282185093111e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0717282185093111e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0717282185093111e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0717282185093111e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.2004287494347943e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.2004287494347943e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.2004287494347943e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.2004287494347943e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.7035783553760626e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7035783553760626e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.3120943053566103e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.3120943053566103e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.3120943053566103e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.3120943053566103e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.6863815468884714e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815468884714e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.8882935947807045e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.8882935947807045e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.092250616235365e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.092250616235365e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.092250616235365e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.092250616235365e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.092250616235365e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.092250616235365e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.092250616235365e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.092250616235365e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.444597854137022e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.444597854137022e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.444597854137022e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.444597854137022e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.6849150953243406e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.6849150953243406e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.6849150953243406e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.6849150953243406e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246974425686464e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246974425686464e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246974425686464e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246974425686464e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246974425686464e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246974425686464e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246974425686464e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246974425686464e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.867765104435319e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.867765104435319e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3002946562663052e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3002946562663052e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.3609563203844245e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.3609563203844245e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.8742990714502686e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.8742990714502686e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.883676576139899e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.883676576139899e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.94735601187036e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.94735601187036e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.94735601187036e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.94735601187036e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.988511706224744e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.988511706224744e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.0882507114597616e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.0882507114597616e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.1513463112588878e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.1513463112588878e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.846201671409681e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846201671409681e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846201671409681e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846201671409681e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.071480736554009e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.071480736554009e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.105526722162071e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.105526722162071e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.105526722162071e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.105526722162071e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146496327675988e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146496327675988e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146496327675988e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146496327675988e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159350502016612e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159350502016612e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159350502016612e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159350502016612e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427988656705459e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427988656705459e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427988656705459e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427988656705459e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935867718095103e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935867718095103e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935867718095103e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935867718095103e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348318091e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.253273348318091e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.97982579361234e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.97982579361234e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.97982579361234e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.97982579361234e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.205548411218745e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.205548411218745e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.205548411218745e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.205548411218745e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00013840177303549712) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.00013840177303549712) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.00018787053389551555) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.00018787053389551555) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.00018787053389551555) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00018787053389551555) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00019400857029755881) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00019400857029755881) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00024636437569584726) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569584726) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00024636437569584726) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569584726) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192743499487794) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192743499487794) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156734248908993) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156734248908993) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156734248908993) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156734248908993) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024421) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007870896771024421) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.0015324835230730552) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.0015324835230730552) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.0015324835230730552) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.0015324835230730552) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.0016407548553124367) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0016407548553124367) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0017278753941369603) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.0017278753941369603) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.002446497155415882) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.002446497155415882) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.002446497155415882) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.002446497155415882) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.003267513854423554) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.003267513854423554) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.0033566705638328927) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.0033566705638328927) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.003484157300217879) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.003484157300217879) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.003876470899336946) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.003876470899336946) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.004668620318776303) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668620318776303) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278136) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.004767272188278136) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.004767272188278136) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278136) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.005286546538226915) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.005286546538226915) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.005286546538226915) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.005286546538226915) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.005408954422410001) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.005408954422410001) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.005408954422410001) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.005408954422410001) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.0059237983365613475) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.0059237983365613475) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.0059237983365613475) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.0059237983365613475) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.01071550846979675) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01071550846979675) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.01071550846979675) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01071550846979675) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.010757563953908937) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010757563953908937) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010757563953908937) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010757563953908937) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.014603704729162123) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.014603704729162123) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.014603704729162123) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.014603704729162123) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.019299560579363776) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363776) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363776) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363776) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363776) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363776) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.019299560579363776) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363776) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0585919887338619) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0585919887338619) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.775950527398006e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.775950527398006e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.7759505273980086e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.7759505273980086e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0716503518100258) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0716503518100258) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.07165035181002585) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07165035181002585) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.019257505095251592) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019257505095251592) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010311482489831783) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311482489831783) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008826368514209862) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209862) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.00759746402977063) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00759746402977063) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.00759746402977063) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00759746402977063) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311868) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311868) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311868) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311868) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311868) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311868) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311868) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311868) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676642) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005348051582676642) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005348051582676642) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676642) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.003804066171728537) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003804066171728537) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984166168121934) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.002984166168121934) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.002984166168121934) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.002984166168121934) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.002446497155415882) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.002446497155415882) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.002249412447093989) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.002249412447093989) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.002249412447093989) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002249412447093989) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0021413612231015694) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.0021413612231015694) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.001863894282458761) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.001863894282458761) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.001863894282458761) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.001863894282458761) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.001863894282458761) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.001863894282458761) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.001863894282458761) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.001863894282458761) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0016407548553124369) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124369) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0016407548553124369) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124369) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0012223378081538411) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538411) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538411) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538411) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538411) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538411) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538411) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0012223378081538411) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0010283292378562825) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0010283292378562825) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0010283292378562825) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0010283292378562825) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1463061453460747e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1463061453460747e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.8742990714502686e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990714502686e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.8742990714502686e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990714502686e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3002946562663052e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3002946562663052e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3002946562663052e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3002946562663052e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.044494129879619e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.044494129879619e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.044494129879619e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.044494129879619e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.956079230707415e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.956079230707415e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.956079230707415e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.956079230707415e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105515037949221e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105515037949221e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105515037949221e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105515037949221e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.66134721372531e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.66134721372531e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.66134721372531e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.66134721372531e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341413838855e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341413838855e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189990975626834e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.189990975626834e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876621658878703e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876621658878703e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876621658878703e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876621658878703e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246207177152e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246207177152e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.523389678118293e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.523389678118293e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.076732532482142e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.076732532482142e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.076732532482142e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.076732532482142e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.0134714590111076e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0134714590111076e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.904599884957336e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.904599884957336e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.904599884957336e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.904599884957336e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.666731755118129e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.666731755118129e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.666731755118129e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.666731755118129e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641927581952e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.8505641927581952e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309311218813e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6569309311218813e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309311218813e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6569309311218813e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.8505641927581952e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.8505641927581952e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815468884714e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815468884714e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.6863815468884714e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815468884714e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714590111076e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.0134714590111076e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.523389678118293e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.523389678118293e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.670402390132989e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.670402390132989e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.670402390132989e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.670402390132989e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246207177152e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246207177152e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189990975626834e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.189990975626834e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341413838855e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341413838855e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949476487021924e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949476487021924e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.7924939577540272e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939577540272e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939577540272e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.7924939577540272e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.883676576139899e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.883676576139899e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.988511706224744e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.988511706224744e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.988511706224744e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.988511706224744e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348318091e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253273348318091e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4017109735623567e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4017109735623567e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4017109735623567e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4017109735623567e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603693377593e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.5809603693377593e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603693377593e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.5809603693377593e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192743499487794) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487794) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192743499487794) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487794) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.0007870896771024421) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024421) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024421) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024421) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0011726348316441822) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0011726348316441822) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0011726348316441822) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0011726348316441822) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012366478019245491) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.0012366478019245491) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.0012366478019245491) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.0012366478019245491) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.002200964069500465) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002200964069500465) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002200964069500465) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002200964069500465) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798023) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798023) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798023) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798023) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798023) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798023) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002394972639798023) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798023) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002446497155415882) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.002446497155415882) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.003804066171728537) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.003804066171728537) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.003876470899336946) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.003876470899336946) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.003876470899336946) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.003876470899336946) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.004220813970046483) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.004220813970046483) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.004220813970046483) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.004220813970046483) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.008826368514209862) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.008826368514209862) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.010311482489831783) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311482489831783) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257505095251592) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019257505095251592) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0585919887338619) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0585919887338619) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3987009017082345e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3987009017082345e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.3987009017082342e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.3987009017082342e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484157300217879) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003484157300217879) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984166168121934) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.002984166168121934) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.00019400857029755881) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00019400857029755881) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453460746e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453460746e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7924939577540272e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7924939577540272e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540341413838855e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413838855e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540341413838855e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413838855e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.8505641927581952e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641927581952e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641927581952e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641927581952e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714590111076e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0134714590111076e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714590111076e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0134714590111076e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949476487021924e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949476487021924e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.7924939577540272e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7924939577540272e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029755881) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029755881) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002984166168121934) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.002984166168121934) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.003484157300217879) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.003484157300217879) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.18066792656583303) [Z7]
+ (-0.18066792656583283) [Z6]
+ (-0.1596143250180998) [Z4]
+ (-0.1596143250180998) [Z5]
+ (0.1741995615505566) [Z2]
+ (0.1741995615505567) [Z3]
+ (0.22757269005453395) [Z0]
+ (0.22757269005453415) [Z1]
+ (-8.194261371970758e-06) [Y4 Y6]
+ (-8.194261371970758e-06) [X4 X6]
+ (7.954413175861321e-06) [Y5 Y7]
+ (7.954413175861321e-06) [X5 X7]
+ (0.11270386920332233) [Z4 Z6]
+ (0.11270386920332233) [Z5 Z7]
+ (0.1195243896468268) [Z0 Z4]
+ (0.1195243896468268) [Z1 Z5]
+ (0.1340171526196369) [Z0 Z6]
+ (0.1340171526196369) [Z1 Z7]
+ (0.13734953064261332) [Z0 Z5]
+ (0.13734953064261332) [Z1 Z4]
+ (0.13766872645852596) [Z2 Z4]
+ (0.13766872645852596) [Z3 Z5]
+ (0.1413890529194282) [Z4 Z7]
+ (0.1413890529194282) [Z5 Z6]
+ (0.14722943218766193) [Z2 Z5]
+ (0.14722943218766193) [Z3 Z4]
+ (0.1492635514738893) [Z4 Z5]
+ (0.14973486803496924) [Z2 Z6]
+ (0.14973486803496924) [Z3 Z7]
+ (0.15138327161428833) [Z0 Z7]
+ (0.15138327161428833) [Z1 Z6]
+ (0.1543574865722363) [Z6 Z7]
+ (0.15582269051553113) [Z2 Z7]
+ (0.15582269051553113) [Z3 Z6]
+ (0.16756653265461247) [Z0 Z2]
+ (0.16756653265461247) [Z1 Z3]
+ (0.18143991440303858) [Z0 Z3]
+ (0.18143991440303858) [Z1 Z2]
+ (0.1939253461327016) [Z0 Z1]
+ (-7.037887510690209e-06) [Y5 Z6 Y7]
+ (-7.037887510690209e-06) [X5 Z6 X7]
+ (-7.037887510690206e-06) [Y4 Z5 Y6]
+ (-7.037887510690206e-06) [X4 Z5 X6]
+ (-0.028685183716105865) [Y4 Y5 X6 X7]
+ (-0.028685183716105865) [X4 X5 Y6 Y7]
+ (-0.01782514099578653) [Y0 Y1 X4 X5]
+ (-0.01782514099578653) [X0 X1 Y4 Y5]
+ (-0.017366118994651427) [Y0 Y1 X6 X7]
+ (-0.017366118994651427) [X0 X1 Y6 Y7]
+ (-0.013873381748426089) [Y0 Y1 X2 X3]
+ (-0.013873381748426089) [X0 X1 Y2 Y3]
+ (-0.00956070572913595) [Y2 Y3 X4 X5]
+ (-0.00956070572913595) [X2 X3 Y4 Y5]
+ (-0.0060878224805618626) [Y2 Y3 X6 X7]
+ (-0.0060878224805618626) [X2 X3 Y6 Y7]
+ (-0.0002921986261110504) [Y1 Y2 X3 X4]
+ (-0.0002921986261110504) [X1 X2 Y3 Y4]
+ (-8.194261371970758e-06) [Z4 Y5 Z6 Y7]
+ (-8.194261371970758e-06) [Z4 X5 Z6 X7]
+ (-2.89096788160794e-06) [Z0 Y5 Z6 Y7]
+ (-2.89096788160794e-06) [Z0 X5 Z6 X7]
+ (-2.89096788160794e-06) [Z1 Y4 Z5 Y6]
+ (-2.89096788160794e-06) [Z1 X4 Z5 X6]
+ (-1.855120121421823e-06) [Z0 Y4 Z5 Y6]
+ (-1.855120121421823e-06) [Z0 X4 Z5 X6]
+ (-1.855120121421823e-06) [Z1 Y5 Z6 Y7]
+ (-1.855120121421823e-06) [Z1 X5 Z6 X7]
+ (-1.5973171977560995e-06) [Z2 Y4 Z5 Y6]
+ (-1.5973171977560995e-06) [Z2 X4 Z5 X6]
+ (-1.5973171977560995e-06) [Z3 Y5 Z6 Y7]
+ (-1.5973171977560995e-06) [Z3 X5 Z6 X7]
+ (-1.0358477601861174e-06) [Y0 X1 X5 Y6]
+ (-1.0358477601861174e-06) [Y0 Y1 Y5 Y6]
+ (-1.0358477601861174e-06) [X0 X1 X5 X6]
+ (-1.0358477601861174e-06) [X0 Y1 Y5 X6]
+ (-9.34455777611724e-07) [Z2 Y5 Z6 Y7]
+ (-9.34455777611724e-07) [Z2 X5 Z6 X7]
+ (-9.34455777611724e-07) [Z3 Y4 Z5 Y6]
+ (-9.34455777611724e-07) [Z3 X4 Z5 X6]
+ (6.628614201443755e-07) [Y2 X3 X5 Y6]
+ (6.628614201443755e-07) [Y2 Y3 Y5 Y6]
+ (6.628614201443755e-07) [X2 X3 X5 X6]
+ (6.628614201443755e-07) [X2 Y3 Y5 X6]
+ (7.954413175861321e-06) [Y4 Z5 Y6 Z7]
+ (7.954413175861321e-06) [X4 Z5 X6 Z7]
+ (0.0002921986261110504) [Y1 X2 X3 Y4]
+ (0.0002921986261110504) [X1 Y2 Y3 X4]
+ (0.0060878224805618626) [Y2 X3 X6 Y7]
+ (0.0060878224805618626) [X2 Y3 Y6 X7]
+ (0.00956070572913595) [Y2 X3 X4 Y5]
+ (0.00956070572913595) [X2 Y3 Y4 X5]
+ (0.01130727400884822) [Y1 Z2 Z3 Y5]
+ (0.01130727400884822) [X1 Z2 Z3 X5]
+ (0.013873381748426089) [Y0 X1 X2 Y3]
+ (0.013873381748426089) [X0 Y1 Y2 X3]
+ (0.017366118994651427) [Y0 X1 X6 Y7]
+ (0.017366118994651427) [X0 Y1 Y6 X7]
+ (0.01782514099578653) [Y0 X1 X4 Y5]
+ (0.01782514099578653) [X0 Y1 Y4 X5]
+ (0.028685183716105865) [Y4 X5 X6 Y7]
+ (0.028685183716105865) [X4 Y5 Y6 X7]
+ (0.029812424517345823) [Y0 Z1 Z2 Y4]
+ (0.029812424517345823) [X0 Z1 Z2 X4]
+ (0.029812424517345823) [Y1 Z3 Z4 Y5]
+ (0.029812424517345823) [X1 Z3 Z4 X5]
+ (0.030104623143456872) [Y0 Z1 Z3 Y4]
+ (0.030104623143456872) [X0 Z1 Z3 X4]
+ (0.030104623143456872) [Y1 Z2 Z4 Y5]
+ (0.030104623143456872) [X1 Z2 Z4 X5]
+ (0.03078750538914393) [Y0 Z2 Z3 Y4]
+ (0.03078750538914393) [X0 Z2 Z3 X4]
+ (0.043752638010661024) [Y1 Z2 Z3 Z4 Y5]
+ (0.043752638010661024) [X1 Z2 Z3 Z4 X5]
+ (0.04375263801066103) [Y0 Z1 Z2 Z3 Y4]
+ (0.04375263801066103) [X0 Z1 Z2 Z3 X4]
+ (-0.014564531231172998) [Y1 Z2 Z3 X4 X6 Y7]
+ (-0.014564531231172998) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-0.014564531231172998) [X1 Z2 Z3 X4 X6 X7]
+ (-0.014564531231172998) [X1 Z2 Z3 Y4 Y6 X7]
+ (-6.5243738482908e-06) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (-6.5243738482908e-06) [X0 Z1 Z2 Z3 Z4 X6]
+ (-6.5243738482908e-06) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (-6.5243738482908e-06) [X1 Z2 Z3 Z5 Z6 X7]
+ (-3.7696594516682726e-06) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (-3.7696594516682726e-06) [X0 Z2 Z3 Z4 Z5 X6]
+ (-3.6102971302908054e-06) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (-3.6102971302908054e-06) [X0 Z1 Z3 Z4 Z5 X6]
+ (-3.6102971302908054e-06) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (-3.6102971302908054e-06) [X1 Z2 Z4 Z5 Z6 X7]
+ (-3.313145500095508e-06) [Y1 Z2 Z3 Y4 X5 X6]
+ (-3.313145500095508e-06) [X1 Z2 Z3 X4 Y5 Y6]
+ (-3.2774831952253325e-06) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (-3.2774831952253325e-06) [X0 Z1 Z2 Z4 Z5 X6]
+ (-3.2774831952253325e-06) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (-3.2774831952253325e-06) [X1 Z3 Z4 Z5 Z6 X7]
+ (-3.211228348195291e-06) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (-3.211228348195291e-06) [X0 Z1 Z2 Z3 Z5 X6]
+ (-3.211228348195291e-06) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (-3.211228348195291e-06) [X1 Z2 Z3 Z4 Z6 X7]
+ (-1.0358477601861174e-06) [Y0 Y1 X4 Z5 Z6 X7]
+ (-1.0358477601861174e-06) [X0 X1 Y4 Z5 Z6 Y7]
+ (-6.628614201443755e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (-6.628614201443755e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (-3.3281393506547364e-07) [Y1 X2 X3 Z4 Z5 Y6]
+ (-3.3281393506547364e-07) [X1 Y2 Y3 Z4 Z5 X6]
+ (3.3281393506547364e-07) [Y1 Y2 X3 Z4 Z5 X6]
+ (3.3281393506547364e-07) [X1 X2 Y3 Z4 Z5 Y6]
+ (6.628614201443755e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (6.628614201443755e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (1.0358477601861174e-06) [Y0 X1 X4 Z5 Z6 Y7]
+ (1.0358477601861174e-06) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.313145500095508e-06) [Y1 Z2 Z3 X4 X5 Y6]
+ (3.313145500095508e-06) [X1 Z2 Z3 Y4 Y5 X6]
+ (4.1839325592891744e-06) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (4.1839325592891744e-06) [X1 Z2 Z3 Z4 Z5 X7]
+ (0.0002921986261110504) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (0.0002921986261110504) [Y0 Z1 Y2 X3 Z4 X5]
+ (0.0002921986261110504) [X0 Z1 X2 Y3 Z4 Y5]
+ (0.0002921986261110504) [X0 Z1 X2 X3 Z4 X5]
+ (0.010540425907671548) [Y0 Z1 Z2 Z3 Y4 Z7]
+ (0.010540425907671548) [X0 Z1 Z2 Z3 X4 Z7]
+ (0.010540425907671548) [Y1 Z2 Z3 Z4 Y5 Z6]
+ (0.010540425907671548) [X1 Z2 Z3 Z4 X5 Z6]
+ (0.01130727400884822) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (0.01130727400884822) [X0 Z1 Z2 Z3 X4 Z5]
+ (0.025104957138844548) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (0.025104957138844548) [X0 Z1 Z2 Z3 X4 Z6]
+ (0.025104957138844548) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (0.025104957138844548) [X1 Z2 Z3 Z4 X5 Z7]
+ (0.03078750538914393) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (0.03078750538914393) [Z0 X1 Z2 Z3 Z4 X5]
+ (-5.10539654948289e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (-5.10539654948289e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (-5.105396549482889e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-5.105396549482889e-06) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.014564531231172998) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-0.014564531231172998) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-3.7696594516682726e-06) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-3.7696594516682726e-06) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-3.3281393506547364e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-3.3281393506547364e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-3.3281393506547364e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-3.3281393506547364e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (3.313145500095508e-06) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (3.313145500095508e-06) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (3.313145500095508e-06) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (3.313145500095508e-06) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (4.1839325592891744e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (4.1839325592891744e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (0.014564531231172998) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (0.014564531231172998) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
 </code>
 </pre>
 </details>

---

## 22. tutorial_barren_plateaus.html <a name="demo21"></a>

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

## 23. tutorial_falqon.html <a name="demo22"></a>

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

## 24. tutorial_qgrnn.html <a name="demo23"></a>

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

## 25. tutorial_error_mitigation.html <a name="demo24"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_error_mitigation.html):

```
2: ──RY(4.05)──╭C──RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)──╰Z──RY(-3.32)──╭C──╭C──────────╭C──RY(-4.05)──┤
3: ──RY(3.51)──╰Z──RY(3.66)───RY(-3.66)──────────────────────────────────────╰Z──╰Z──────────╰Z──RY(-3.51)──┤
0: ──RY(4.56)──╭C──RY(5.93)───RY(-5.93)──────────────────────────────────────╭C──RY(-4.56)─────────────────────┤
1: ──RY(3.6)───╰Z──RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z──RY(-3.6)───RY(3.6)──RY(-3.6)──┤
2: ──RY(4.05)──╭C──RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)──╰Z──RY(-3.32)──╭C──RY(-4.05)─────────────────────┤
3: ──RY(3.51)──╰Z──RY(3.66)───RY(-3.66)──────────────────────────────────────╰Z──RY(-3.51)─────────────────────┤
0.8985196547410969
0.9589759497509437
ZNE result: 0.8985196547410969
0.9732463333204999
0: ──RY(4.56)──╭C──RY(5.93)───RY(-5.93)──────────────────────────────────────────────╭C──RY(-4.56)──┤
1: ──RY(3.6)───╰Z──RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)──╭C──╭C──╭C──RY(-5.9)───╰Z──RY(-3.6)───┤
2: ──RY(4.05)──╭C──RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)──╰Z──╰Z──╰Z──RY(-3.32)──╭C──RY(-4.05)──┤
3: ──RY(3.51)──╰Z──RY(3.66)───RY(-3.66)──────────────────────────────────────────────╰Z──RY(-3.51)──┤
0: ──RY(4.56)──╭C──RY(5.93)───RY(-5.93)──────────────────────────────────────╭C───RY(-4.56)─────────────────┤
1: ──RY(3.6)───╰Z──RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z───RY(-3.6)──────────────────┤
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_error_mitigation.html):

```
2: ──RY(4.05)──╭C──RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)──╰Z──RY(-3.32)──╭C──RY(-4.05)──RY(4.05)──RY(-4.05)──┤
3: ──RY(3.51)──╰Z──RY(3.66)───RY(-3.66)──────────────────────────────────────╰Z──RY(-3.51)───────────────────────┤
0: ──RY(4.56)──╭C──RY(5.93)───RY(-5.93)───────────────────────────────────────────────────────────╭C──RY(-4.56)──┤
1: ──RY(3.6)───╰Z──RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)───────────────────────╭C──RY(-5.9)───╰Z──RY(-3.6)───┤
2: ──RY(4.05)──╭C──RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)──RY(1.07)──RY(-1.07)──╰Z──RY(-3.32)──╭C──RY(-4.05)──┤
3: ──RY(3.51)──╰Z──RY(3.66)───RY(-3.66)───────────────────────────────────────────────────────────╰Z──RY(-3.51)──┤
0.8985196547410973
0.9589759451472933
ZNE result: 0.8985196547410973
0.9581053690316875
0: ──RY(4.56)──╭C──RY(5.93)───RY(-5.93)───────────────────────────────────────────────────────────╭C──RY(-4.56)──┤
1: ──RY(3.6)───╰Z──RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)──RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z──RY(-3.6)───┤
2: ──RY(4.05)──╭C──RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)───────────────────────╰Z──RY(-3.32)──╭C──RY(-4.05)──┤
3: ──RY(3.51)──╰Z──RY(3.66)───RY(-3.66)───────────────────────────────────────────────────────────╰Z──RY(-3.51)──┤
0: ──RY(4.56)──╭C──RY(5.93)───RY(-5.93)──────────────────────────────────────╭C──RY(-4.56)───────────────────────┤
1: ──RY(3.6)───╰Z──RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z──RY(-3.6)────────────────────────┤
```

---

## 26. tutorial_ensemble_multi_qpu.html <a name="demo25"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_ensemble_multi_qpu.html):

```
Training accuracy (ensemble): 0.808
Training accuracy (QPU0):  0.64
Choices: [0 0 1 1 0 0 1 0 0 0 1 0 0 0 0 0 0 1 1 0 0 1 0 1 1 0 0 0 1 0 0 1 0 1 0 0 0
 0 1 1 0 0 0 0 0 0 0 1 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 1 0 0 0
 0 0 0 0 0 1 1 1 1 1 0 0 1 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0
 0 0 0 0 1 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 1 1 0 1 0 0
Choices counts: Counter({0: 112, 1: 38})
Counter({2: 57, 0: 55})
Counter({1: 35, 0: 3})
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_ensemble_multi_qpu.html):

```
Training accuracy (ensemble): 0.824
Training accuracy (QPU0):  0.648
Choices: [0 0 1 1 0 0 1 0 0 0 1 0 0 0 0 0 0 1 1 0 0 1 0 1 1 0 0 0 1 0 0 1 0 1 0 0 1
 0 1 1 0 0 0 0 0 0 0 1 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 1 1 0 0 0 0 1 0 0 0
 0 0 0 0 0 1 1 1 1 0 0 0 1 0 1 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0
 1 0 0 0 1 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 1 1 0 1 0 0
Choices counts: Counter({0: 109, 1: 41})
Counter({0: 55, 2: 54})
Counter({1: 37, 0: 4})
```

---

## 27. tutorial_quanvolution.html <a name="demo26"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quanvolution.html):

```
   16384/11490434 [..............................] - ETA: 0s
 2162688/11490434 [====>.........................] - ETA: 0s
 5169152/11490434 [============>.................] - ETA: 0s
13/13 - 0s - loss: 3.0160 - accuracy: 0.1000 - val_loss: 2.0646 - val_accuracy: 0.2000
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_quanvolution.html):

```
   16384/11490434 [..............................] - ETA: 2s
 2498560/11490434 [=====>........................] - ETA: 0s
10936320/11490434 [===========================>..] - ETA: 0s
13/13 - 1s - loss: 3.0160 - accuracy: 0.1000 - val_loss: 2.0646 - val_accuracy: 0.2000
```

---

## 28. tutorial_qnn_module_tf.html <a name="demo27"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_qnn_module_tf.html):

```
30/30 - 10s - loss: 0.4997 - accuracy: 0.5000 - val_loss: 0.5081 - val_accuracy: 0.4400
30/30 - 10s - loss: 0.4673 - accuracy: 0.6200 - val_loss: 0.4488 - val_accuracy: 0.6200
30/30 - 9s - loss: 0.3230 - accuracy: 0.8267 - val_loss: 0.2562 - val_accuracy: 0.8400
30/30 - 10s - loss: 0.2124 - accuracy: 0.8867 - val_loss: 0.1997 - val_accuracy: 0.8400
30/30 - 10s - loss: 0.1800 - accuracy: 0.8933 - val_loss: 0.1841 - val_accuracy: 0.8400
30/30 - 9s - loss: 0.1593 - accuracy: 0.8667 - val_loss: 0.2177 - val_accuracy: 0.8400
30/30 - 19s - loss: 0.5189 - accuracy: 0.4000 - val_loss: 0.4945 - val_accuracy: 0.5400
30/30 - 19s - loss: 0.4822 - accuracy: 0.6200 - val_loss: 0.4412 - val_accuracy: 0.7200
30/30 - 19s - loss: 0.3850 - accuracy: 0.7133 - val_loss: 0.2898 - val_accuracy: 0.7800
30/30 - 19s - loss: 0.2720 - accuracy: 0.7867 - val_loss: 0.2185 - val_accuracy: 0.8200
30/30 - 19s - loss: 0.2056 - accuracy: 0.8400 - val_loss: 0.1893 - val_accuracy: 0.8400
30/30 - 19s - loss: 0.1753 - accuracy: 0.8400 - val_loss: 0.1856 - val_accuracy: 0.8400
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_qnn_module_tf.html):

```
30/30 - 17s - loss: 0.4997 - accuracy: 0.5000 - val_loss: 0.5081 - val_accuracy: 0.4400
30/30 - 17s - loss: 0.4673 - accuracy: 0.6200 - val_loss: 0.4488 - val_accuracy: 0.6200
30/30 - 18s - loss: 0.3230 - accuracy: 0.8267 - val_loss: 0.2562 - val_accuracy: 0.8400
30/30 - 17s - loss: 0.2124 - accuracy: 0.8867 - val_loss: 0.1997 - val_accuracy: 0.8400
30/30 - 17s - loss: 0.1800 - accuracy: 0.8933 - val_loss: 0.1841 - val_accuracy: 0.8400
30/30 - 18s - loss: 0.1593 - accuracy: 0.8667 - val_loss: 0.2177 - val_accuracy: 0.8400
30/30 - 34s - loss: 0.5189 - accuracy: 0.4000 - val_loss: 0.4945 - val_accuracy: 0.5400
30/30 - 34s - loss: 0.4822 - accuracy: 0.6200 - val_loss: 0.4412 - val_accuracy: 0.7200
30/30 - 35s - loss: 0.3850 - accuracy: 0.7133 - val_loss: 0.2898 - val_accuracy: 0.7800
30/30 - 34s - loss: 0.2720 - accuracy: 0.7867 - val_loss: 0.2185 - val_accuracy: 0.8200
30/30 - 34s - loss: 0.2056 - accuracy: 0.8400 - val_loss: 0.1893 - val_accuracy: 0.8400
30/30 - 34s - loss: 0.1753 - accuracy: 0.8400 - val_loss: 0.1856 - val_accuracy: 0.8400
```

---

## 29. tutorial_backprop.html <a name="demo28"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_backprop.html):

```
Expectation value: -0.1197136570687156
-0.06518877224958129
-0.06518877224958129
[[-6.51887722e-02 -2.72891905e-02  1.38777878e-17 -9.33934621e-02
Forward pass (best of 3): 0.011303524899994954 sec per loop
Gradient computation (best of 3): 4.232258381200017 sec per loop
4.069268963998184
0.9358535378025422
Forward pass (best of 3): 0.049072440100007955 sec per loop
Backward pass (best of 3): 0.1080437159000212 sec per loop
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_backprop.html):

```
Expectation value: -0.11971365706871569
-0.0651887722495813
-0.0651887722495813
[[-6.51887722e-02 -2.72891905e-02  0.00000000e+00 -9.33934621e-02
Forward pass (best of 3): 0.02690380849999201 sec per loop
Gradient computation (best of 3): 7.7745595074999985 sec per loop
9.685371059997124
0.9358535378025427
Forward pass (best of 3): 0.07328153320004276 sec per loop
Backward pass (best of 3): 0.13813745730003574 sec per loop
```

---

## 30. tutorial_adjoint_diff.html <a name="demo29"></a>

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

## 31. tutorial_doubly_stochastic.html <a name="demo30"></a>

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

## 32. tutorial_unitary_designs.html <a name="demo31"></a>

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

