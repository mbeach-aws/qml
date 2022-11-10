Last update: 2022-11-10  14:16:10 (All times shown in Eastern time)
# List of differences in demonstration outputs

# Table of contents

1. [tutorial_noisy_circuit_optimization.html](#demo0)
2. [tutorial_qubit_tapering.html](#demo1)
3. [tutorial_tn_circuits.html](#demo2)
4. [tutorial_local_cost_functions.html](#demo3)
5. [tutorial_qnn_module_tf.html](#demo4)
6. [tutorial_quantum_transfer_learning.html](#demo5)
7. [tutorial_quanvolution.html](#demo6)
8. [tutorial_adaptive_circuits.html](#demo7)
9. [tutorial_qft_arithmetics.html](#demo8)
10. [tutorial_quantum_circuit_cutting.html](#demo9)
11. [tutorial_rotoselect.html](#demo10)
12. [tutorial_backprop.html](#demo11)
13. [tutorial_quantum_chemistry.html](#demo12)
14. [tutorial_error_mitigation.html](#demo13)
15. [tutorial_vqe_spin_sectors.html](#demo14)
16. [tutorial_jax_transformations.html](#demo15)
17. [tutorial_measurement_optimize.html](#demo16)


Number of demos different/all demos: 17/71

## 1. tutorial_noisy_circuit_optimization.html <a name="demo0"></a>

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

## 2. tutorial_qubit_tapering.html <a name="demo1"></a>

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

## 3. tutorial_tn_circuits.html <a name="demo2"></a>

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

## 4. tutorial_local_cost_functions.html <a name="demo3"></a>

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

## 5. tutorial_qnn_module_tf.html <a name="demo4"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_qnn_module_tf.html):

```
30/30 - 11s - loss: 0.4997 - accuracy: 0.5000 - val_loss: 0.5081 - val_accuracy: 0.4400 - 11s/epoch - 383ms/step
30/30 - 11s - loss: 0.4673 - accuracy: 0.6200 - val_loss: 0.4488 - val_accuracy: 0.6200 - 11s/epoch - 383ms/step
30/30 - 12s - loss: 0.3230 - accuracy: 0.8267 - val_loss: 0.2562 - val_accuracy: 0.8400 - 12s/epoch - 393ms/step
30/30 - 12s - loss: 0.2124 - accuracy: 0.8867 - val_loss: 0.1997 - val_accuracy: 0.8400 - 12s/epoch - 414ms/step
30/30 - 12s - loss: 0.1800 - accuracy: 0.8933 - val_loss: 0.1841 - val_accuracy: 0.8400 - 12s/epoch - 385ms/step
30/30 - 11s - loss: 0.1593 - accuracy: 0.8667 - val_loss: 0.2177 - val_accuracy: 0.8400 - 11s/epoch - 383ms/step
30/30 - 23s - loss: 0.5189 - accuracy: 0.4000 - val_loss: 0.4945 - val_accuracy: 0.5400 - 23s/epoch - 769ms/step
30/30 - 23s - loss: 0.4822 - accuracy: 0.6200 - val_loss: 0.4412 - val_accuracy: 0.7200 - 23s/epoch - 764ms/step
30/30 - 23s - loss: 0.3850 - accuracy: 0.7133 - val_loss: 0.2898 - val_accuracy: 0.7800 - 23s/epoch - 762ms/step
30/30 - 23s - loss: 0.2720 - accuracy: 0.7867 - val_loss: 0.2185 - val_accuracy: 0.8200 - 23s/epoch - 751ms/step
30/30 - 23s - loss: 0.2056 - accuracy: 0.8400 - val_loss: 0.1893 - val_accuracy: 0.8400 - 23s/epoch - 756ms/step
30/30 - 23s - loss: 0.1753 - accuracy: 0.8400 - val_loss: 0.1856 - val_accuracy: 0.8400 - 23s/epoch - 755ms/step
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_qnn_module_tf.html):

```
30/30 - 9s - loss: 0.4997 - accuracy: 0.5000 - val_loss: 0.5081 - val_accuracy: 0.4400 - 9s/epoch - 303ms/step
30/30 - 9s - loss: 0.4673 - accuracy: 0.6200 - val_loss: 0.4488 - val_accuracy: 0.6200 - 9s/epoch - 300ms/step
30/30 - 9s - loss: 0.3230 - accuracy: 0.8267 - val_loss: 0.2562 - val_accuracy: 0.8400 - 9s/epoch - 302ms/step
30/30 - 9s - loss: 0.2124 - accuracy: 0.8867 - val_loss: 0.1997 - val_accuracy: 0.8400 - 9s/epoch - 304ms/step
30/30 - 9s - loss: 0.1800 - accuracy: 0.8933 - val_loss: 0.1841 - val_accuracy: 0.8400 - 9s/epoch - 300ms/step
30/30 - 9s - loss: 0.1593 - accuracy: 0.8667 - val_loss: 0.2177 - val_accuracy: 0.8400 - 9s/epoch - 300ms/step
30/30 - 18s - loss: 0.5189 - accuracy: 0.4000 - val_loss: 0.4945 - val_accuracy: 0.5400 - 18s/epoch - 599ms/step
30/30 - 18s - loss: 0.4822 - accuracy: 0.6200 - val_loss: 0.4412 - val_accuracy: 0.7200 - 18s/epoch - 600ms/step
30/30 - 18s - loss: 0.3850 - accuracy: 0.7133 - val_loss: 0.2898 - val_accuracy: 0.7800 - 18s/epoch - 606ms/step
30/30 - 18s - loss: 0.2720 - accuracy: 0.7867 - val_loss: 0.2185 - val_accuracy: 0.8200 - 18s/epoch - 607ms/step
30/30 - 18s - loss: 0.2056 - accuracy: 0.8400 - val_loss: 0.1893 - val_accuracy: 0.8400 - 18s/epoch - 601ms/step
30/30 - 18s - loss: 0.1753 - accuracy: 0.8400 - val_loss: 0.1856 - val_accuracy: 0.8400 - 18s/epoch - 609ms/step
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
 58%|#####7    | 25.8M/44.7M [00:00<00:00, 270MB/s]
100%|##########| 44.7M/44.7M [00:00<00:00, 259MB/s]
Phase: train Epoch: 1/3 Iter: 1/62 Batch time: 0.3600
Phase: train Epoch: 1/3 Iter: 2/62 Batch time: 0.3208
Phase: train Epoch: 1/3 Iter: 3/62 Batch time: 0.3152
Phase: train Epoch: 1/3 Iter: 4/62 Batch time: 0.3181
Phase: train Epoch: 1/3 Iter: 5/62 Batch time: 0.3298
Phase: train Epoch: 1/3 Iter: 6/62 Batch time: 0.3278
Phase: train Epoch: 1/3 Iter: 7/62 Batch time: 0.3266
Phase: train Epoch: 1/3 Iter: 8/62 Batch time: 0.3192
Phase: train Epoch: 1/3 Iter: 9/62 Batch time: 0.3179
Phase: train Epoch: 1/3 Iter: 10/62 Batch time: 0.3264
Phase: train Epoch: 1/3 Iter: 11/62 Batch time: 0.3176
Phase: train Epoch: 1/3 Iter: 12/62 Batch time: 0.3178
Phase: train Epoch: 1/3 Iter: 13/62 Batch time: 0.3229
Phase: train Epoch: 1/3 Iter: 14/62 Batch time: 0.3171
Phase: train Epoch: 1/3 Iter: 15/62 Batch time: 0.3137
Phase: train Epoch: 1/3 Iter: 16/62 Batch time: 0.3259
Phase: train Epoch: 1/3 Iter: 17/62 Batch time: 0.3259
Phase: train Epoch: 1/3 Iter: 18/62 Batch time: 0.3245
Phase: train Epoch: 1/3 Iter: 19/62 Batch time: 0.3223
Phase: train Epoch: 1/3 Iter: 20/62 Batch time: 0.3234
Phase: train Epoch: 1/3 Iter: 21/62 Batch time: 0.3206
Phase: train Epoch: 1/3 Iter: 22/62 Batch time: 0.3229
Phase: train Epoch: 1/3 Iter: 23/62 Batch time: 0.3258
Phase: train Epoch: 1/3 Iter: 24/62 Batch time: 0.3221
Phase: train Epoch: 1/3 Iter: 25/62 Batch time: 0.3218
Phase: train Epoch: 1/3 Iter: 26/62 Batch time: 0.3262
Phase: train Epoch: 1/3 Iter: 27/62 Batch time: 0.3321
Phase: train Epoch: 1/3 Iter: 28/62 Batch time: 0.3281
Phase: train Epoch: 1/3 Iter: 29/62 Batch time: 0.3228
Phase: train Epoch: 1/3 Iter: 30/62 Batch time: 0.3277
Phase: train Epoch: 1/3 Iter: 31/62 Batch time: 0.3232
Phase: train Epoch: 1/3 Iter: 32/62 Batch time: 0.3194
Phase: train Epoch: 1/3 Iter: 33/62 Batch time: 0.3282
Phase: train Epoch: 1/3 Iter: 34/62 Batch time: 0.3351
Phase: train Epoch: 1/3 Iter: 35/62 Batch time: 0.3317
Phase: train Epoch: 1/3 Iter: 36/62 Batch time: 0.3247
Phase: train Epoch: 1/3 Iter: 37/62 Batch time: 0.3201
Phase: train Epoch: 1/3 Iter: 38/62 Batch time: 0.3173
Phase: train Epoch: 1/3 Iter: 39/62 Batch time: 0.3227
Phase: train Epoch: 1/3 Iter: 40/62 Batch time: 0.3202
Phase: train Epoch: 1/3 Iter: 41/62 Batch time: 0.3271
Phase: train Epoch: 1/3 Iter: 42/62 Batch time: 0.3314
Phase: train Epoch: 1/3 Iter: 43/62 Batch time: 0.3289
Phase: train Epoch: 1/3 Iter: 44/62 Batch time: 0.3308
Phase: train Epoch: 1/3 Iter: 45/62 Batch time: 0.3271
Phase: train Epoch: 1/3 Iter: 46/62 Batch time: 0.3296
Phase: train Epoch: 1/3 Iter: 47/62 Batch time: 0.3257
Phase: train Epoch: 1/3 Iter: 48/62 Batch time: 0.3361
Phase: train Epoch: 1/3 Iter: 49/62 Batch time: 0.3273
Phase: train Epoch: 1/3 Iter: 50/62 Batch time: 0.3287
Phase: train Epoch: 1/3 Iter: 51/62 Batch time: 0.3255
Phase: train Epoch: 1/3 Iter: 52/62 Batch time: 0.3341
Phase: train Epoch: 1/3 Iter: 53/62 Batch time: 0.3248
Phase: train Epoch: 1/3 Iter: 54/62 Batch time: 0.3344
Phase: train Epoch: 1/3 Iter: 55/62 Batch time: 0.3324
Phase: train Epoch: 1/3 Iter: 56/62 Batch time: 0.3279
Phase: train Epoch: 1/3 Iter: 57/62 Batch time: 0.3294
Phase: train Epoch: 1/3 Iter: 58/62 Batch time: 0.3346
Phase: train Epoch: 1/3 Iter: 59/62 Batch time: 0.3293
Phase: train Epoch: 1/3 Iter: 60/62 Batch time: 0.3248
Phase: train Epoch: 1/3 Iter: 61/62 Batch time: 0.3159
Phase: validation Epoch: 1/3 Iter: 1/39 Batch time: 0.2433
Phase: validation Epoch: 1/3 Iter: 2/39 Batch time: 0.2419
Phase: validation Epoch: 1/3 Iter: 3/39 Batch time: 0.2366
Phase: validation Epoch: 1/3 Iter: 4/39 Batch time: 0.2406
Phase: validation Epoch: 1/3 Iter: 5/39 Batch time: 0.2478
Phase: validation Epoch: 1/3 Iter: 6/39 Batch time: 0.2419
Phase: validation Epoch: 1/3 Iter: 7/39 Batch time: 0.2416
Phase: validation Epoch: 1/3 Iter: 8/39 Batch time: 0.2490
Phase: validation Epoch: 1/3 Iter: 9/39 Batch time: 0.2416
Phase: validation Epoch: 1/3 Iter: 10/39 Batch time: 0.2469
Phase: validation Epoch: 1/3 Iter: 11/39 Batch time: 0.2357
Phase: validation Epoch: 1/3 Iter: 12/39 Batch time: 0.2446
Phase: validation Epoch: 1/3 Iter: 13/39 Batch time: 0.2474
Phase: validation Epoch: 1/3 Iter: 14/39 Batch time: 0.2454
Phase: validation Epoch: 1/3 Iter: 15/39 Batch time: 0.2416
Phase: validation Epoch: 1/3 Iter: 16/39 Batch time: 0.2410
Phase: validation Epoch: 1/3 Iter: 17/39 Batch time: 0.2466
Phase: validation Epoch: 1/3 Iter: 18/39 Batch time: 0.2433
Phase: validation Epoch: 1/3 Iter: 19/39 Batch time: 0.2418
Phase: validation Epoch: 1/3 Iter: 20/39 Batch time: 0.2412
Phase: validation Epoch: 1/3 Iter: 21/39 Batch time: 0.2404
Phase: validation Epoch: 1/3 Iter: 22/39 Batch time: 0.2470
Phase: validation Epoch: 1/3 Iter: 23/39 Batch time: 0.2449
Phase: validation Epoch: 1/3 Iter: 24/39 Batch time: 0.2362
Phase: validation Epoch: 1/3 Iter: 25/39 Batch time: 0.2374
Phase: validation Epoch: 1/3 Iter: 26/39 Batch time: 0.2319
Phase: validation Epoch: 1/3 Iter: 27/39 Batch time: 0.2456
Phase: validation Epoch: 1/3 Iter: 28/39 Batch time: 0.2474
Phase: validation Epoch: 1/3 Iter: 29/39 Batch time: 0.2438
Phase: validation Epoch: 1/3 Iter: 30/39 Batch time: 0.2415
Phase: validation Epoch: 1/3 Iter: 31/39 Batch time: 0.2376
Phase: validation Epoch: 1/3 Iter: 32/39 Batch time: 0.2449
Phase: validation Epoch: 1/3 Iter: 33/39 Batch time: 0.2424
Phase: validation Epoch: 1/3 Iter: 34/39 Batch time: 0.2409
Phase: validation Epoch: 1/3 Iter: 35/39 Batch time: 0.2392
Phase: validation Epoch: 1/3 Iter: 36/39 Batch time: 0.2338
Phase: validation Epoch: 1/3 Iter: 37/39 Batch time: 0.2355
Phase: validation Epoch: 1/3 Iter: 38/39 Batch time: 0.2400
Phase: validation Epoch: 1/3 Iter: 39/39 Batch time: 0.0798
Phase: train Epoch: 2/3 Iter: 1/62 Batch time: 0.3215
Phase: train Epoch: 2/3 Iter: 2/62 Batch time: 0.3313
Phase: train Epoch: 2/3 Iter: 3/62 Batch time: 0.3299
Phase: train Epoch: 2/3 Iter: 4/62 Batch time: 0.3234
Phase: train Epoch: 2/3 Iter: 5/62 Batch time: 0.3222
Phase: train Epoch: 2/3 Iter: 6/62 Batch time: 0.3339
Phase: train Epoch: 2/3 Iter: 7/62 Batch time: 0.3365
Phase: train Epoch: 2/3 Iter: 8/62 Batch time: 0.3376
Phase: train Epoch: 2/3 Iter: 9/62 Batch time: 0.3429
Phase: train Epoch: 2/3 Iter: 10/62 Batch time: 0.3264
Phase: train Epoch: 2/3 Iter: 11/62 Batch time: 0.3427
Phase: train Epoch: 2/3 Iter: 12/62 Batch time: 0.3393
Phase: train Epoch: 2/3 Iter: 13/62 Batch time: 0.3294
Phase: train Epoch: 2/3 Iter: 14/62 Batch time: 0.3329
Phase: train Epoch: 2/3 Iter: 15/62 Batch time: 0.3294
Phase: train Epoch: 2/3 Iter: 16/62 Batch time: 0.3378
Phase: train Epoch: 2/3 Iter: 17/62 Batch time: 0.3289
Phase: train Epoch: 2/3 Iter: 18/62 Batch time: 0.3346
Phase: train Epoch: 2/3 Iter: 19/62 Batch time: 0.3304
Phase: train Epoch: 2/3 Iter: 20/62 Batch time: 0.3362
Phase: train Epoch: 2/3 Iter: 21/62 Batch time: 0.3408
Phase: train Epoch: 2/3 Iter: 22/62 Batch time: 0.3316
Phase: train Epoch: 2/3 Iter: 23/62 Batch time: 0.3419
Phase: train Epoch: 2/3 Iter: 24/62 Batch time: 0.3416
Phase: train Epoch: 2/3 Iter: 25/62 Batch time: 0.3255
Phase: train Epoch: 2/3 Iter: 26/62 Batch time: 0.3346
Phase: train Epoch: 2/3 Iter: 27/62 Batch time: 0.3321
Phase: train Epoch: 2/3 Iter: 28/62 Batch time: 0.3267
Phase: train Epoch: 2/3 Iter: 29/62 Batch time: 0.3289
Phase: train Epoch: 2/3 Iter: 30/62 Batch time: 0.3355
Phase: train Epoch: 2/3 Iter: 31/62 Batch time: 0.3339
Phase: train Epoch: 2/3 Iter: 32/62 Batch time: 0.3332
Phase: train Epoch: 2/3 Iter: 33/62 Batch time: 0.3307
Phase: train Epoch: 2/3 Iter: 34/62 Batch time: 0.3267
Phase: train Epoch: 2/3 Iter: 35/62 Batch time: 0.3333
Phase: train Epoch: 2/3 Iter: 36/62 Batch time: 0.3257
Phase: train Epoch: 2/3 Iter: 37/62 Batch time: 0.3336
Phase: train Epoch: 2/3 Iter: 38/62 Batch time: 0.3239
Phase: train Epoch: 2/3 Iter: 39/62 Batch time: 0.3333
Phase: train Epoch: 2/3 Iter: 40/62 Batch time: 0.3349
Phase: train Epoch: 2/3 Iter: 41/62 Batch time: 0.3367
Phase: train Epoch: 2/3 Iter: 42/62 Batch time: 0.3328
Phase: train Epoch: 2/3 Iter: 43/62 Batch time: 0.3365
Phase: train Epoch: 2/3 Iter: 44/62 Batch time: 0.3286
Phase: train Epoch: 2/3 Iter: 45/62 Batch time: 0.3259
Phase: train Epoch: 2/3 Iter: 46/62 Batch time: 0.3236
Phase: train Epoch: 2/3 Iter: 47/62 Batch time: 0.3284
Phase: train Epoch: 2/3 Iter: 48/62 Batch time: 0.3210
Phase: train Epoch: 2/3 Iter: 49/62 Batch time: 0.3235
Phase: train Epoch: 2/3 Iter: 50/62 Batch time: 0.3189
Phase: train Epoch: 2/3 Iter: 51/62 Batch time: 0.3215
Phase: train Epoch: 2/3 Iter: 52/62 Batch time: 0.3192
Phase: train Epoch: 2/3 Iter: 53/62 Batch time: 0.3209
Phase: train Epoch: 2/3 Iter: 54/62 Batch time: 0.3240
Phase: train Epoch: 2/3 Iter: 55/62 Batch time: 0.3287
Phase: train Epoch: 2/3 Iter: 56/62 Batch time: 0.3209
Phase: train Epoch: 2/3 Iter: 57/62 Batch time: 0.3145
Phase: train Epoch: 2/3 Iter: 58/62 Batch time: 0.3140
Phase: train Epoch: 2/3 Iter: 59/62 Batch time: 0.3168
Phase: train Epoch: 2/3 Iter: 60/62 Batch time: 0.3206
Phase: train Epoch: 2/3 Iter: 61/62 Batch time: 0.3174
Phase: validation Epoch: 2/3 Iter: 1/39 Batch time: 0.2500
Phase: validation Epoch: 2/3 Iter: 2/39 Batch time: 0.2424
Phase: validation Epoch: 2/3 Iter: 3/39 Batch time: 0.2400
Phase: validation Epoch: 2/3 Iter: 4/39 Batch time: 0.2399
Phase: validation Epoch: 2/3 Iter: 5/39 Batch time: 0.2442
Phase: validation Epoch: 2/3 Iter: 6/39 Batch time: 0.2418
Phase: validation Epoch: 2/3 Iter: 7/39 Batch time: 0.2343
Phase: validation Epoch: 2/3 Iter: 8/39 Batch time: 0.2424
Phase: validation Epoch: 2/3 Iter: 9/39 Batch time: 0.2412
Phase: validation Epoch: 2/3 Iter: 10/39 Batch time: 0.2412
Phase: validation Epoch: 2/3 Iter: 11/39 Batch time: 0.2391
Phase: validation Epoch: 2/3 Iter: 12/39 Batch time: 0.2390
Phase: validation Epoch: 2/3 Iter: 13/39 Batch time: 0.2410
Phase: validation Epoch: 2/3 Iter: 14/39 Batch time: 0.2436
Phase: validation Epoch: 2/3 Iter: 15/39 Batch time: 0.2406
Phase: validation Epoch: 2/3 Iter: 16/39 Batch time: 0.2388
Phase: validation Epoch: 2/3 Iter: 17/39 Batch time: 0.2434
Phase: validation Epoch: 2/3 Iter: 18/39 Batch time: 0.2401
Phase: validation Epoch: 2/3 Iter: 19/39 Batch time: 0.2419
Phase: validation Epoch: 2/3 Iter: 20/39 Batch time: 0.2447
Phase: validation Epoch: 2/3 Iter: 21/39 Batch time: 0.2430
Phase: validation Epoch: 2/3 Iter: 22/39 Batch time: 0.2403
Phase: validation Epoch: 2/3 Iter: 23/39 Batch time: 0.2417
Phase: validation Epoch: 2/3 Iter: 24/39 Batch time: 0.2430
Phase: validation Epoch: 2/3 Iter: 25/39 Batch time: 0.2425
Phase: validation Epoch: 2/3 Iter: 26/39 Batch time: 0.2430
Phase: validation Epoch: 2/3 Iter: 27/39 Batch time: 0.2398
Phase: validation Epoch: 2/3 Iter: 28/39 Batch time: 0.2353
Phase: validation Epoch: 2/3 Iter: 29/39 Batch time: 0.2403
Phase: validation Epoch: 2/3 Iter: 30/39 Batch time: 0.2385
Phase: validation Epoch: 2/3 Iter: 31/39 Batch time: 0.2410
Phase: validation Epoch: 2/3 Iter: 32/39 Batch time: 0.2339
Phase: validation Epoch: 2/3 Iter: 33/39 Batch time: 0.2313
Phase: validation Epoch: 2/3 Iter: 34/39 Batch time: 0.2398
Phase: validation Epoch: 2/3 Iter: 35/39 Batch time: 0.2426
Phase: validation Epoch: 2/3 Iter: 36/39 Batch time: 0.2432
Phase: validation Epoch: 2/3 Iter: 37/39 Batch time: 0.2385
Phase: validation Epoch: 2/3 Iter: 38/39 Batch time: 0.2448
Phase: validation Epoch: 2/3 Iter: 39/39 Batch time: 0.0743
Phase: train Epoch: 3/3 Iter: 1/62 Batch time: 0.3312
Phase: train Epoch: 3/3 Iter: 2/62 Batch time: 0.3409
Phase: train Epoch: 3/3 Iter: 3/62 Batch time: 0.3415
Phase: train Epoch: 3/3 Iter: 4/62 Batch time: 0.3338
Phase: train Epoch: 3/3 Iter: 5/62 Batch time: 0.3310
Phase: train Epoch: 3/3 Iter: 6/62 Batch time: 0.3335
Phase: train Epoch: 3/3 Iter: 7/62 Batch time: 0.3271
Phase: train Epoch: 3/3 Iter: 8/62 Batch time: 0.3300
Phase: train Epoch: 3/3 Iter: 9/62 Batch time: 0.3208
Phase: train Epoch: 3/3 Iter: 10/62 Batch time: 0.3237
Phase: train Epoch: 3/3 Iter: 11/62 Batch time: 0.3271
Phase: train Epoch: 3/3 Iter: 12/62 Batch time: 0.3195
Phase: train Epoch: 3/3 Iter: 13/62 Batch time: 0.3213
Phase: train Epoch: 3/3 Iter: 14/62 Batch time: 0.3266
Phase: train Epoch: 3/3 Iter: 15/62 Batch time: 0.3241
Phase: train Epoch: 3/3 Iter: 16/62 Batch time: 0.3216
Phase: train Epoch: 3/3 Iter: 17/62 Batch time: 0.3281
Phase: train Epoch: 3/3 Iter: 18/62 Batch time: 0.3263
Phase: train Epoch: 3/3 Iter: 19/62 Batch time: 0.3262
Phase: train Epoch: 3/3 Iter: 20/62 Batch time: 0.3277
Phase: train Epoch: 3/3 Iter: 21/62 Batch time: 0.3245
Phase: train Epoch: 3/3 Iter: 22/62 Batch time: 0.3243
Phase: train Epoch: 3/3 Iter: 23/62 Batch time: 0.3283
Phase: train Epoch: 3/3 Iter: 24/62 Batch time: 0.3297
Phase: train Epoch: 3/3 Iter: 25/62 Batch time: 0.3257
Phase: train Epoch: 3/3 Iter: 26/62 Batch time: 0.3335
Phase: train Epoch: 3/3 Iter: 27/62 Batch time: 0.3301
Phase: train Epoch: 3/3 Iter: 28/62 Batch time: 0.3234
Phase: train Epoch: 3/3 Iter: 29/62 Batch time: 0.3223
Phase: train Epoch: 3/3 Iter: 30/62 Batch time: 0.3162
Phase: train Epoch: 3/3 Iter: 31/62 Batch time: 0.3284
Phase: train Epoch: 3/3 Iter: 32/62 Batch time: 0.3341
Phase: train Epoch: 3/3 Iter: 33/62 Batch time: 0.3287
Phase: train Epoch: 3/3 Iter: 34/62 Batch time: 0.3239
Phase: train Epoch: 3/3 Iter: 35/62 Batch time: 0.3164
Phase: train Epoch: 3/3 Iter: 36/62 Batch time: 0.3152
Phase: train Epoch: 3/3 Iter: 37/62 Batch time: 0.3155
Phase: train Epoch: 3/3 Iter: 38/62 Batch time: 0.3166
Phase: train Epoch: 3/3 Iter: 39/62 Batch time: 0.3192
Phase: train Epoch: 3/3 Iter: 40/62 Batch time: 0.3267
Phase: train Epoch: 3/3 Iter: 41/62 Batch time: 0.3207
Phase: train Epoch: 3/3 Iter: 42/62 Batch time: 0.3244
Phase: train Epoch: 3/3 Iter: 43/62 Batch time: 0.3263
Phase: train Epoch: 3/3 Iter: 44/62 Batch time: 0.3182
Phase: train Epoch: 3/3 Iter: 45/62 Batch time: 0.3254
Phase: train Epoch: 3/3 Iter: 46/62 Batch time: 0.3329
Phase: train Epoch: 3/3 Iter: 47/62 Batch time: 0.3194
Phase: train Epoch: 3/3 Iter: 48/62 Batch time: 0.3215
Phase: train Epoch: 3/3 Iter: 49/62 Batch time: 0.3276
Phase: train Epoch: 3/3 Iter: 50/62 Batch time: 0.3214
Phase: train Epoch: 3/3 Iter: 51/62 Batch time: 0.3264
Phase: train Epoch: 3/3 Iter: 52/62 Batch time: 0.3263
Phase: train Epoch: 3/3 Iter: 53/62 Batch time: 0.3179
Phase: train Epoch: 3/3 Iter: 54/62 Batch time: 0.3233
Phase: train Epoch: 3/3 Iter: 55/62 Batch time: 0.3246
Phase: train Epoch: 3/3 Iter: 56/62 Batch time: 0.3237
Phase: train Epoch: 3/3 Iter: 57/62 Batch time: 0.3268
Phase: train Epoch: 3/3 Iter: 58/62 Batch time: 0.3284
Phase: train Epoch: 3/3 Iter: 59/62 Batch time: 0.3209
Phase: train Epoch: 3/3 Iter: 60/62 Batch time: 0.3227
Phase: train Epoch: 3/3 Iter: 61/62 Batch time: 0.3233
Phase: validation Epoch: 3/3 Iter: 1/39 Batch time: 0.2427
Phase: validation Epoch: 3/3 Iter: 2/39 Batch time: 0.2368
Phase: validation Epoch: 3/3 Iter: 3/39 Batch time: 0.2412
Phase: validation Epoch: 3/3 Iter: 4/39 Batch time: 0.2352
Phase: validation Epoch: 3/3 Iter: 5/39 Batch time: 0.2376
Phase: validation Epoch: 3/3 Iter: 6/39 Batch time: 0.2322
Phase: validation Epoch: 3/3 Iter: 7/39 Batch time: 0.2331
Phase: validation Epoch: 3/3 Iter: 8/39 Batch time: 0.2330
Phase: validation Epoch: 3/3 Iter: 9/39 Batch time: 0.2396
Phase: validation Epoch: 3/3 Iter: 10/39 Batch time: 0.2410
Phase: validation Epoch: 3/3 Iter: 11/39 Batch time: 0.2412
Phase: validation Epoch: 3/3 Iter: 12/39 Batch time: 0.2406
Phase: validation Epoch: 3/3 Iter: 13/39 Batch time: 0.2334
Phase: validation Epoch: 3/3 Iter: 14/39 Batch time: 0.2359
Phase: validation Epoch: 3/3 Iter: 15/39 Batch time: 0.2383
Phase: validation Epoch: 3/3 Iter: 16/39 Batch time: 0.2320
Phase: validation Epoch: 3/3 Iter: 17/39 Batch time: 0.2294
Phase: validation Epoch: 3/3 Iter: 18/39 Batch time: 0.2323
Phase: validation Epoch: 3/3 Iter: 19/39 Batch time: 0.2294
Phase: validation Epoch: 3/3 Iter: 20/39 Batch time: 0.2415
Phase: validation Epoch: 3/3 Iter: 21/39 Batch time: 0.2373
Phase: validation Epoch: 3/3 Iter: 22/39 Batch time: 0.2337
Phase: validation Epoch: 3/3 Iter: 23/39 Batch time: 0.2337
Phase: validation Epoch: 3/3 Iter: 24/39 Batch time: 0.2367
Phase: validation Epoch: 3/3 Iter: 25/39 Batch time: 0.2363
Phase: validation Epoch: 3/3 Iter: 26/39 Batch time: 0.2377
Phase: validation Epoch: 3/3 Iter: 27/39 Batch time: 0.2361
Phase: validation Epoch: 3/3 Iter: 28/39 Batch time: 0.2397
Phase: validation Epoch: 3/3 Iter: 29/39 Batch time: 0.2418
Phase: validation Epoch: 3/3 Iter: 30/39 Batch time: 0.2399
Phase: validation Epoch: 3/3 Iter: 31/39 Batch time: 0.2369
Phase: validation Epoch: 3/3 Iter: 32/39 Batch time: 0.2417
Phase: validation Epoch: 3/3 Iter: 33/39 Batch time: 0.2389
Phase: validation Epoch: 3/3 Iter: 34/39 Batch time: 0.2366
Phase: validation Epoch: 3/3 Iter: 35/39 Batch time: 0.2387
Phase: validation Epoch: 3/3 Iter: 36/39 Batch time: 0.2412
Phase: validation Epoch: 3/3 Iter: 37/39 Batch time: 0.2413
Phase: validation Epoch: 3/3 Iter: 38/39 Batch time: 0.2420
Phase: validation Epoch: 3/3 Iter: 39/39 Batch time: 0.0695
Training completed in 1m 35s
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
 40%|####      | 18.0M/44.7M [00:00<00:00, 189MB/s]
100%|##########| 44.7M/44.7M [00:00<00:00, 238MB/s]
Phase: train Epoch: 1/3 Iter: 1/62 Batch time: 0.3636
Phase: train Epoch: 1/3 Iter: 2/62 Batch time: 0.3416
Phase: train Epoch: 1/3 Iter: 3/62 Batch time: 0.3416
Phase: train Epoch: 1/3 Iter: 4/62 Batch time: 0.3412
Phase: train Epoch: 1/3 Iter: 5/62 Batch time: 0.3421
Phase: train Epoch: 1/3 Iter: 6/62 Batch time: 0.3417
Phase: train Epoch: 1/3 Iter: 7/62 Batch time: 0.3422
Phase: train Epoch: 1/3 Iter: 8/62 Batch time: 0.3434
Phase: train Epoch: 1/3 Iter: 9/62 Batch time: 0.3427
Phase: train Epoch: 1/3 Iter: 10/62 Batch time: 0.3407
Phase: train Epoch: 1/3 Iter: 11/62 Batch time: 0.3419
Phase: train Epoch: 1/3 Iter: 12/62 Batch time: 0.3416
Phase: train Epoch: 1/3 Iter: 13/62 Batch time: 0.3418
Phase: train Epoch: 1/3 Iter: 14/62 Batch time: 0.3429
Phase: train Epoch: 1/3 Iter: 15/62 Batch time: 0.3429
Phase: train Epoch: 1/3 Iter: 16/62 Batch time: 0.3439
Phase: train Epoch: 1/3 Iter: 17/62 Batch time: 0.3425
Phase: train Epoch: 1/3 Iter: 18/62 Batch time: 0.3424
Phase: train Epoch: 1/3 Iter: 19/62 Batch time: 0.3420
Phase: train Epoch: 1/3 Iter: 20/62 Batch time: 0.3420
Phase: train Epoch: 1/3 Iter: 21/62 Batch time: 0.3417
Phase: train Epoch: 1/3 Iter: 22/62 Batch time: 0.3424
Phase: train Epoch: 1/3 Iter: 23/62 Batch time: 0.3436
Phase: train Epoch: 1/3 Iter: 24/62 Batch time: 0.3434
Phase: train Epoch: 1/3 Iter: 25/62 Batch time: 0.3438
Phase: train Epoch: 1/3 Iter: 26/62 Batch time: 0.3431
Phase: train Epoch: 1/3 Iter: 27/62 Batch time: 0.3439
Phase: train Epoch: 1/3 Iter: 28/62 Batch time: 0.3417
Phase: train Epoch: 1/3 Iter: 29/62 Batch time: 0.3410
Phase: train Epoch: 1/3 Iter: 30/62 Batch time: 0.3416
Phase: train Epoch: 1/3 Iter: 31/62 Batch time: 0.3420
Phase: train Epoch: 1/3 Iter: 32/62 Batch time: 0.3428
Phase: train Epoch: 1/3 Iter: 33/62 Batch time: 0.3427
Phase: train Epoch: 1/3 Iter: 34/62 Batch time: 0.3427
Phase: train Epoch: 1/3 Iter: 35/62 Batch time: 0.3421
Phase: train Epoch: 1/3 Iter: 36/62 Batch time: 0.3421
Phase: train Epoch: 1/3 Iter: 37/62 Batch time: 0.3434
Phase: train Epoch: 1/3 Iter: 38/62 Batch time: 0.3418
Phase: train Epoch: 1/3 Iter: 39/62 Batch time: 0.3446
Phase: train Epoch: 1/3 Iter: 40/62 Batch time: 0.3417
Phase: train Epoch: 1/3 Iter: 41/62 Batch time: 0.3417
Phase: train Epoch: 1/3 Iter: 42/62 Batch time: 0.3421
Phase: train Epoch: 1/3 Iter: 43/62 Batch time: 0.3406
Phase: train Epoch: 1/3 Iter: 44/62 Batch time: 0.3407
Phase: train Epoch: 1/3 Iter: 45/62 Batch time: 0.3420
Phase: train Epoch: 1/3 Iter: 46/62 Batch time: 0.3427
Phase: train Epoch: 1/3 Iter: 47/62 Batch time: 0.3428
Phase: train Epoch: 1/3 Iter: 48/62 Batch time: 0.3427
Phase: train Epoch: 1/3 Iter: 49/62 Batch time: 0.3405
Phase: train Epoch: 1/3 Iter: 50/62 Batch time: 0.3408
Phase: train Epoch: 1/3 Iter: 51/62 Batch time: 0.3409
Phase: train Epoch: 1/3 Iter: 52/62 Batch time: 0.3429
Phase: train Epoch: 1/3 Iter: 53/62 Batch time: 0.3420
Phase: train Epoch: 1/3 Iter: 54/62 Batch time: 0.3415
Phase: train Epoch: 1/3 Iter: 55/62 Batch time: 0.3414
Phase: train Epoch: 1/3 Iter: 56/62 Batch time: 0.3418
Phase: train Epoch: 1/3 Iter: 57/62 Batch time: 0.3412
Phase: train Epoch: 1/3 Iter: 58/62 Batch time: 0.3421
Phase: train Epoch: 1/3 Iter: 59/62 Batch time: 0.3409
Phase: train Epoch: 1/3 Iter: 60/62 Batch time: 0.3408
Phase: train Epoch: 1/3 Iter: 61/62 Batch time: 0.3429
Phase: validation Epoch: 1/3 Iter: 1/39 Batch time: 0.2711
Phase: validation Epoch: 1/3 Iter: 2/39 Batch time: 0.2683
Phase: validation Epoch: 1/3 Iter: 3/39 Batch time: 0.2684
Phase: validation Epoch: 1/3 Iter: 4/39 Batch time: 0.2688
Phase: validation Epoch: 1/3 Iter: 5/39 Batch time: 0.2680
Phase: validation Epoch: 1/3 Iter: 6/39 Batch time: 0.2691
Phase: validation Epoch: 1/3 Iter: 7/39 Batch time: 0.2680
Phase: validation Epoch: 1/3 Iter: 8/39 Batch time: 0.2684
Phase: validation Epoch: 1/3 Iter: 9/39 Batch time: 0.2677
Phase: validation Epoch: 1/3 Iter: 10/39 Batch time: 0.2684
Phase: validation Epoch: 1/3 Iter: 11/39 Batch time: 0.2679
Phase: validation Epoch: 1/3 Iter: 12/39 Batch time: 0.2685
Phase: validation Epoch: 1/3 Iter: 13/39 Batch time: 0.2685
Phase: validation Epoch: 1/3 Iter: 14/39 Batch time: 0.2689
Phase: validation Epoch: 1/3 Iter: 15/39 Batch time: 0.2685
Phase: validation Epoch: 1/3 Iter: 16/39 Batch time: 0.2685
Phase: validation Epoch: 1/3 Iter: 17/39 Batch time: 0.2681
Phase: validation Epoch: 1/3 Iter: 18/39 Batch time: 0.2685
Phase: validation Epoch: 1/3 Iter: 19/39 Batch time: 0.2685
Phase: validation Epoch: 1/3 Iter: 20/39 Batch time: 0.2676
Phase: validation Epoch: 1/3 Iter: 21/39 Batch time: 0.2679
Phase: validation Epoch: 1/3 Iter: 22/39 Batch time: 0.2677
Phase: validation Epoch: 1/3 Iter: 23/39 Batch time: 0.2681
Phase: validation Epoch: 1/3 Iter: 24/39 Batch time: 0.2685
Phase: validation Epoch: 1/3 Iter: 25/39 Batch time: 0.2678
Phase: validation Epoch: 1/3 Iter: 26/39 Batch time: 0.2687
Phase: validation Epoch: 1/3 Iter: 27/39 Batch time: 0.2684
Phase: validation Epoch: 1/3 Iter: 28/39 Batch time: 0.2683
Phase: validation Epoch: 1/3 Iter: 29/39 Batch time: 0.2682
Phase: validation Epoch: 1/3 Iter: 30/39 Batch time: 0.2680
Phase: validation Epoch: 1/3 Iter: 31/39 Batch time: 0.2684
Phase: validation Epoch: 1/3 Iter: 32/39 Batch time: 0.2681
Phase: validation Epoch: 1/3 Iter: 33/39 Batch time: 0.2681
Phase: validation Epoch: 1/3 Iter: 34/39 Batch time: 0.2681
Phase: validation Epoch: 1/3 Iter: 35/39 Batch time: 0.2680
Phase: validation Epoch: 1/3 Iter: 36/39 Batch time: 0.2680
Phase: validation Epoch: 1/3 Iter: 37/39 Batch time: 0.2678
Phase: validation Epoch: 1/3 Iter: 38/39 Batch time: 0.2677
Phase: validation Epoch: 1/3 Iter: 39/39 Batch time: 0.0825
Phase: train Epoch: 2/3 Iter: 1/62 Batch time: 0.3390
Phase: train Epoch: 2/3 Iter: 2/62 Batch time: 0.3407
Phase: train Epoch: 2/3 Iter: 3/62 Batch time: 0.3416
Phase: train Epoch: 2/3 Iter: 4/62 Batch time: 0.3425
Phase: train Epoch: 2/3 Iter: 5/62 Batch time: 0.3420
Phase: train Epoch: 2/3 Iter: 6/62 Batch time: 0.3420
Phase: train Epoch: 2/3 Iter: 7/62 Batch time: 0.3434
Phase: train Epoch: 2/3 Iter: 8/62 Batch time: 0.3431
Phase: train Epoch: 2/3 Iter: 9/62 Batch time: 0.3430
Phase: train Epoch: 2/3 Iter: 10/62 Batch time: 0.3425
Phase: train Epoch: 2/3 Iter: 11/62 Batch time: 0.3428
Phase: train Epoch: 2/3 Iter: 12/62 Batch time: 0.3431
Phase: train Epoch: 2/3 Iter: 13/62 Batch time: 0.3428
Phase: train Epoch: 2/3 Iter: 14/62 Batch time: 0.3432
Phase: train Epoch: 2/3 Iter: 15/62 Batch time: 0.3436
Phase: train Epoch: 2/3 Iter: 16/62 Batch time: 0.3429
Phase: train Epoch: 2/3 Iter: 17/62 Batch time: 0.3429
Phase: train Epoch: 2/3 Iter: 18/62 Batch time: 0.3429
Phase: train Epoch: 2/3 Iter: 19/62 Batch time: 0.3426
Phase: train Epoch: 2/3 Iter: 20/62 Batch time: 0.3431
Phase: train Epoch: 2/3 Iter: 21/62 Batch time: 0.3426
Phase: train Epoch: 2/3 Iter: 22/62 Batch time: 0.3433
Phase: train Epoch: 2/3 Iter: 23/62 Batch time: 0.3424
Phase: train Epoch: 2/3 Iter: 24/62 Batch time: 0.3433
Phase: train Epoch: 2/3 Iter: 25/62 Batch time: 0.3426
Phase: train Epoch: 2/3 Iter: 26/62 Batch time: 0.3423
Phase: train Epoch: 2/3 Iter: 27/62 Batch time: 0.3406
Phase: train Epoch: 2/3 Iter: 28/62 Batch time: 0.3428
Phase: train Epoch: 2/3 Iter: 29/62 Batch time: 0.3428
Phase: train Epoch: 2/3 Iter: 30/62 Batch time: 0.3424
Phase: train Epoch: 2/3 Iter: 31/62 Batch time: 0.3443
Phase: train Epoch: 2/3 Iter: 32/62 Batch time: 0.3427
Phase: train Epoch: 2/3 Iter: 33/62 Batch time: 0.3433
Phase: train Epoch: 2/3 Iter: 34/62 Batch time: 0.3428
Phase: train Epoch: 2/3 Iter: 35/62 Batch time: 0.3426
Phase: train Epoch: 2/3 Iter: 36/62 Batch time: 0.3428
Phase: train Epoch: 2/3 Iter: 37/62 Batch time: 0.3446
Phase: train Epoch: 2/3 Iter: 38/62 Batch time: 0.3432
Phase: train Epoch: 2/3 Iter: 39/62 Batch time: 0.3432
Phase: train Epoch: 2/3 Iter: 40/62 Batch time: 0.3431
Phase: train Epoch: 2/3 Iter: 41/62 Batch time: 0.3426
Phase: train Epoch: 2/3 Iter: 42/62 Batch time: 0.3423
Phase: train Epoch: 2/3 Iter: 43/62 Batch time: 0.3433
Phase: train Epoch: 2/3 Iter: 44/62 Batch time: 0.3472
Phase: train Epoch: 2/3 Iter: 45/62 Batch time: 0.3430
Phase: train Epoch: 2/3 Iter: 46/62 Batch time: 0.3431
Phase: train Epoch: 2/3 Iter: 47/62 Batch time: 0.3433
Phase: train Epoch: 2/3 Iter: 48/62 Batch time: 0.3442
Phase: train Epoch: 2/3 Iter: 49/62 Batch time: 0.3464
Phase: train Epoch: 2/3 Iter: 50/62 Batch time: 0.3429
Phase: train Epoch: 2/3 Iter: 51/62 Batch time: 0.3430
Phase: train Epoch: 2/3 Iter: 52/62 Batch time: 0.3430
Phase: train Epoch: 2/3 Iter: 53/62 Batch time: 0.3428
Phase: train Epoch: 2/3 Iter: 54/62 Batch time: 0.3431
Phase: train Epoch: 2/3 Iter: 55/62 Batch time: 0.3430
Phase: train Epoch: 2/3 Iter: 56/62 Batch time: 0.3436
Phase: train Epoch: 2/3 Iter: 57/62 Batch time: 0.3428
Phase: train Epoch: 2/3 Iter: 58/62 Batch time: 0.3421
Phase: train Epoch: 2/3 Iter: 59/62 Batch time: 0.3439
Phase: train Epoch: 2/3 Iter: 60/62 Batch time: 0.3435
Phase: train Epoch: 2/3 Iter: 61/62 Batch time: 0.3430
Phase: validation Epoch: 2/3 Iter: 1/39 Batch time: 0.2727
Phase: validation Epoch: 2/3 Iter: 2/39 Batch time: 0.2693
Phase: validation Epoch: 2/3 Iter: 3/39 Batch time: 0.2695
Phase: validation Epoch: 2/3 Iter: 4/39 Batch time: 0.2694
Phase: validation Epoch: 2/3 Iter: 5/39 Batch time: 0.2697
Phase: validation Epoch: 2/3 Iter: 6/39 Batch time: 0.2689
Phase: validation Epoch: 2/3 Iter: 7/39 Batch time: 0.2706
Phase: validation Epoch: 2/3 Iter: 8/39 Batch time: 0.2699
Phase: validation Epoch: 2/3 Iter: 9/39 Batch time: 0.2713
Phase: validation Epoch: 2/3 Iter: 10/39 Batch time: 0.2690
Phase: validation Epoch: 2/3 Iter: 11/39 Batch time: 0.2697
Phase: validation Epoch: 2/3 Iter: 12/39 Batch time: 0.2698
Phase: validation Epoch: 2/3 Iter: 13/39 Batch time: 0.2690
Phase: validation Epoch: 2/3 Iter: 14/39 Batch time: 0.2695
Phase: validation Epoch: 2/3 Iter: 15/39 Batch time: 0.2703
Phase: validation Epoch: 2/3 Iter: 16/39 Batch time: 0.2693
Phase: validation Epoch: 2/3 Iter: 17/39 Batch time: 0.2698
Phase: validation Epoch: 2/3 Iter: 18/39 Batch time: 0.2687
Phase: validation Epoch: 2/3 Iter: 19/39 Batch time: 0.2693
Phase: validation Epoch: 2/3 Iter: 20/39 Batch time: 0.2692
Phase: validation Epoch: 2/3 Iter: 21/39 Batch time: 0.2697
Phase: validation Epoch: 2/3 Iter: 22/39 Batch time: 0.2685
Phase: validation Epoch: 2/3 Iter: 23/39 Batch time: 0.2697
Phase: validation Epoch: 2/3 Iter: 24/39 Batch time: 0.2689
Phase: validation Epoch: 2/3 Iter: 25/39 Batch time: 0.2699
Phase: validation Epoch: 2/3 Iter: 26/39 Batch time: 0.2692
Phase: validation Epoch: 2/3 Iter: 27/39 Batch time: 0.2701
Phase: validation Epoch: 2/3 Iter: 28/39 Batch time: 0.2687
Phase: validation Epoch: 2/3 Iter: 29/39 Batch time: 0.2701
Phase: validation Epoch: 2/3 Iter: 30/39 Batch time: 0.2689
Phase: validation Epoch: 2/3 Iter: 31/39 Batch time: 0.2703
Phase: validation Epoch: 2/3 Iter: 32/39 Batch time: 0.2692
Phase: validation Epoch: 2/3 Iter: 33/39 Batch time: 0.2702
Phase: validation Epoch: 2/3 Iter: 34/39 Batch time: 0.2693
Phase: validation Epoch: 2/3 Iter: 35/39 Batch time: 0.2696
Phase: validation Epoch: 2/3 Iter: 36/39 Batch time: 0.2697
Phase: validation Epoch: 2/3 Iter: 37/39 Batch time: 0.2695
Phase: validation Epoch: 2/3 Iter: 38/39 Batch time: 0.2698
Phase: validation Epoch: 2/3 Iter: 39/39 Batch time: 0.0787
Phase: train Epoch: 3/3 Iter: 1/62 Batch time: 0.3408
Phase: train Epoch: 3/3 Iter: 2/62 Batch time: 0.3413
Phase: train Epoch: 3/3 Iter: 3/62 Batch time: 0.3418
Phase: train Epoch: 3/3 Iter: 4/62 Batch time: 0.3422
Phase: train Epoch: 3/3 Iter: 5/62 Batch time: 0.3436
Phase: train Epoch: 3/3 Iter: 6/62 Batch time: 0.3438
Phase: train Epoch: 3/3 Iter: 7/62 Batch time: 0.3433
Phase: train Epoch: 3/3 Iter: 8/62 Batch time: 0.3412
Phase: train Epoch: 3/3 Iter: 9/62 Batch time: 0.3424
Phase: train Epoch: 3/3 Iter: 10/62 Batch time: 0.3430
Phase: train Epoch: 3/3 Iter: 11/62 Batch time: 0.3426
Phase: train Epoch: 3/3 Iter: 12/62 Batch time: 0.3428
Phase: train Epoch: 3/3 Iter: 13/62 Batch time: 0.3439
Phase: train Epoch: 3/3 Iter: 14/62 Batch time: 0.3433
Phase: train Epoch: 3/3 Iter: 15/62 Batch time: 0.3427
Phase: train Epoch: 3/3 Iter: 16/62 Batch time: 0.3432
Phase: train Epoch: 3/3 Iter: 17/62 Batch time: 0.3430
Phase: train Epoch: 3/3 Iter: 18/62 Batch time: 0.3425
Phase: train Epoch: 3/3 Iter: 19/62 Batch time: 0.3433
Phase: train Epoch: 3/3 Iter: 20/62 Batch time: 0.3440
Phase: train Epoch: 3/3 Iter: 21/62 Batch time: 0.3447
Phase: train Epoch: 3/3 Iter: 22/62 Batch time: 0.3436
Phase: train Epoch: 3/3 Iter: 23/62 Batch time: 0.3428
Phase: train Epoch: 3/3 Iter: 24/62 Batch time: 0.3430
Phase: train Epoch: 3/3 Iter: 25/62 Batch time: 0.3428
Phase: train Epoch: 3/3 Iter: 26/62 Batch time: 0.3436
Phase: train Epoch: 3/3 Iter: 27/62 Batch time: 0.3447
Phase: train Epoch: 3/3 Iter: 28/62 Batch time: 0.3436
Phase: train Epoch: 3/3 Iter: 29/62 Batch time: 0.3436
Phase: train Epoch: 3/3 Iter: 30/62 Batch time: 0.3430
Phase: train Epoch: 3/3 Iter: 31/62 Batch time: 0.3432
Phase: train Epoch: 3/3 Iter: 32/62 Batch time: 0.3424
Phase: train Epoch: 3/3 Iter: 33/62 Batch time: 0.3418
Phase: train Epoch: 3/3 Iter: 34/62 Batch time: 0.3474
Phase: train Epoch: 3/3 Iter: 35/62 Batch time: 0.3436
Phase: train Epoch: 3/3 Iter: 36/62 Batch time: 0.3432
Phase: train Epoch: 3/3 Iter: 37/62 Batch time: 0.3435
Phase: train Epoch: 3/3 Iter: 38/62 Batch time: 0.3432
Phase: train Epoch: 3/3 Iter: 39/62 Batch time: 0.3434
Phase: train Epoch: 3/3 Iter: 40/62 Batch time: 0.3422
Phase: train Epoch: 3/3 Iter: 41/62 Batch time: 0.3429
Phase: train Epoch: 3/3 Iter: 42/62 Batch time: 0.3436
Phase: train Epoch: 3/3 Iter: 43/62 Batch time: 0.3430
Phase: train Epoch: 3/3 Iter: 44/62 Batch time: 0.3433
Phase: train Epoch: 3/3 Iter: 45/62 Batch time: 0.3429
Phase: train Epoch: 3/3 Iter: 46/62 Batch time: 0.3433
Phase: train Epoch: 3/3 Iter: 47/62 Batch time: 0.3437
Phase: train Epoch: 3/3 Iter: 48/62 Batch time: 0.3431
Phase: train Epoch: 3/3 Iter: 49/62 Batch time: 0.3429
Phase: train Epoch: 3/3 Iter: 50/62 Batch time: 0.3440
Phase: train Epoch: 3/3 Iter: 51/62 Batch time: 0.3434
Phase: train Epoch: 3/3 Iter: 52/62 Batch time: 0.3434
Phase: train Epoch: 3/3 Iter: 53/62 Batch time: 0.3433
Phase: train Epoch: 3/3 Iter: 54/62 Batch time: 0.3431
Phase: train Epoch: 3/3 Iter: 55/62 Batch time: 0.3430
Phase: train Epoch: 3/3 Iter: 56/62 Batch time: 0.3439
Phase: train Epoch: 3/3 Iter: 57/62 Batch time: 0.3431
Phase: train Epoch: 3/3 Iter: 58/62 Batch time: 0.3442
Phase: train Epoch: 3/3 Iter: 59/62 Batch time: 0.3428
Phase: train Epoch: 3/3 Iter: 60/62 Batch time: 0.3426
Phase: train Epoch: 3/3 Iter: 61/62 Batch time: 0.3436
Phase: validation Epoch: 3/3 Iter: 1/39 Batch time: 0.2727
Phase: validation Epoch: 3/3 Iter: 2/39 Batch time: 0.2699
Phase: validation Epoch: 3/3 Iter: 3/39 Batch time: 0.2703
Phase: validation Epoch: 3/3 Iter: 4/39 Batch time: 0.2702
Phase: validation Epoch: 3/3 Iter: 5/39 Batch time: 0.2694
Phase: validation Epoch: 3/3 Iter: 6/39 Batch time: 0.2702
Phase: validation Epoch: 3/3 Iter: 7/39 Batch time: 0.2695
Phase: validation Epoch: 3/3 Iter: 8/39 Batch time: 0.2704
Phase: validation Epoch: 3/3 Iter: 9/39 Batch time: 0.2700
Phase: validation Epoch: 3/3 Iter: 10/39 Batch time: 0.2695
Phase: validation Epoch: 3/3 Iter: 11/39 Batch time: 0.2696
Phase: validation Epoch: 3/3 Iter: 12/39 Batch time: 0.2699
Phase: validation Epoch: 3/3 Iter: 13/39 Batch time: 0.2708
Phase: validation Epoch: 3/3 Iter: 14/39 Batch time: 0.2696
Phase: validation Epoch: 3/3 Iter: 15/39 Batch time: 0.2698
Phase: validation Epoch: 3/3 Iter: 16/39 Batch time: 0.2698
Phase: validation Epoch: 3/3 Iter: 17/39 Batch time: 0.2701
Phase: validation Epoch: 3/3 Iter: 18/39 Batch time: 0.2698
Phase: validation Epoch: 3/3 Iter: 19/39 Batch time: 0.2701
Phase: validation Epoch: 3/3 Iter: 20/39 Batch time: 0.2705
Phase: validation Epoch: 3/3 Iter: 21/39 Batch time: 0.2712
Phase: validation Epoch: 3/3 Iter: 22/39 Batch time: 0.2702
Phase: validation Epoch: 3/3 Iter: 23/39 Batch time: 0.2705
Phase: validation Epoch: 3/3 Iter: 24/39 Batch time: 0.2710
Phase: validation Epoch: 3/3 Iter: 25/39 Batch time: 0.2702
Phase: validation Epoch: 3/3 Iter: 26/39 Batch time: 0.2692
Phase: validation Epoch: 3/3 Iter: 27/39 Batch time: 0.2702
Phase: validation Epoch: 3/3 Iter: 28/39 Batch time: 0.2696
Phase: validation Epoch: 3/3 Iter: 29/39 Batch time: 0.2699
Phase: validation Epoch: 3/3 Iter: 30/39 Batch time: 0.2704
Phase: validation Epoch: 3/3 Iter: 31/39 Batch time: 0.2703
Phase: validation Epoch: 3/3 Iter: 32/39 Batch time: 0.2697
Phase: validation Epoch: 3/3 Iter: 33/39 Batch time: 0.2705
Phase: validation Epoch: 3/3 Iter: 34/39 Batch time: 0.2694
Phase: validation Epoch: 3/3 Iter: 35/39 Batch time: 0.2706
Phase: validation Epoch: 3/3 Iter: 36/39 Batch time: 0.2692
Phase: validation Epoch: 3/3 Iter: 37/39 Batch time: 0.2698
Phase: validation Epoch: 3/3 Iter: 38/39 Batch time: 0.2701
Phase: validation Epoch: 3/3 Iter: 39/39 Batch time: 0.0788
Training completed in 1m 40s
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
13/13 - 0s - loss: 3.0160 - accuracy: 0.1000 - val_loss: 2.0646 - val_accuracy: 0.2000 - 472ms/epoch - 36ms/step
Epoch 2/30
13/13 - 0s - loss: 2.2510 - accuracy: 0.1800 - val_loss: 1.9801 - val_accuracy: 0.3333 - 51ms/epoch - 4ms/step
Epoch 3/30
13/13 - 0s - loss: 1.7851 - accuracy: 0.4000 - val_loss: 1.8177 - val_accuracy: 0.2667 - 37ms/epoch - 3ms/step
Epoch 4/30
13/13 - 0s - loss: 1.3652 - accuracy: 0.5400 - val_loss: 1.6107 - val_accuracy: 0.4667 - 36ms/epoch - 3ms/step
Epoch 5/30
13/13 - 0s - loss: 1.1317 - accuracy: 0.7800 - val_loss: 1.4723 - val_accuracy: 0.6000 - 50ms/epoch - 4ms/step
Epoch 6/30
13/13 - 0s - loss: 0.9360 - accuracy: 0.8600 - val_loss: 1.4686 - val_accuracy: 0.5333 - 38ms/epoch - 3ms/step
Epoch 7/30
13/13 - 0s - loss: 0.7383 - accuracy: 0.9400 - val_loss: 1.3536 - val_accuracy: 0.5667 - 37ms/epoch - 3ms/step
Epoch 8/30
13/13 - 0s - loss: 0.5846 - accuracy: 0.9800 - val_loss: 1.2785 - val_accuracy: 0.6667 - 37ms/epoch - 3ms/step
Epoch 9/30
13/13 - 0s - loss: 0.4987 - accuracy: 0.9800 - val_loss: 1.2253 - val_accuracy: 0.6333 - 50ms/epoch - 4ms/step
Epoch 10/30
13/13 - 0s - loss: 0.3921 - accuracy: 1.0000 - val_loss: 1.2655 - val_accuracy: 0.6333 - 52ms/epoch - 4ms/step
Epoch 11/30
13/13 - 0s - loss: 0.3617 - accuracy: 1.0000 - val_loss: 1.1555 - val_accuracy: 0.7000 - 36ms/epoch - 3ms/step
Epoch 12/30
13/13 - 0s - loss: 0.3078 - accuracy: 1.0000 - val_loss: 1.2107 - val_accuracy: 0.6667 - 49ms/epoch - 4ms/step
Epoch 13/30
13/13 - 0s - loss: 0.2618 - accuracy: 1.0000 - val_loss: 1.1166 - val_accuracy: 0.7333 - 51ms/epoch - 4ms/step
Epoch 14/30
13/13 - 0s - loss: 0.2463 - accuracy: 1.0000 - val_loss: 1.0624 - val_accuracy: 0.7000 - 37ms/epoch - 3ms/step
Epoch 15/30
13/13 - 0s - loss: 0.2033 - accuracy: 1.0000 - val_loss: 1.0904 - val_accuracy: 0.7333 - 35ms/epoch - 3ms/step
Epoch 16/30
13/13 - 0s - loss: 0.1799 - accuracy: 1.0000 - val_loss: 1.0865 - val_accuracy: 0.7333 - 49ms/epoch - 4ms/step
Epoch 17/30
13/13 - 0s - loss: 0.1682 - accuracy: 1.0000 - val_loss: 1.0385 - val_accuracy: 0.7333 - 49ms/epoch - 4ms/step
Epoch 18/30
13/13 - 0s - loss: 0.1484 - accuracy: 1.0000 - val_loss: 1.0676 - val_accuracy: 0.7000 - 51ms/epoch - 4ms/step
Epoch 19/30
13/13 - 0s - loss: 0.1349 - accuracy: 1.0000 - val_loss: 1.0447 - val_accuracy: 0.7000 - 50ms/epoch - 4ms/step
Epoch 20/30
13/13 - 0s - loss: 0.1255 - accuracy: 1.0000 - val_loss: 0.9935 - val_accuracy: 0.7333 - 37ms/epoch - 3ms/step
Epoch 21/30
13/13 - 0s - loss: 0.1135 - accuracy: 1.0000 - val_loss: 1.0451 - val_accuracy: 0.7333 - 50ms/epoch - 4ms/step
Epoch 22/30
13/13 - 0s - loss: 0.1041 - accuracy: 1.0000 - val_loss: 1.0142 - val_accuracy: 0.7333 - 37ms/epoch - 3ms/step
Epoch 23/30
13/13 - 0s - loss: 0.0983 - accuracy: 1.0000 - val_loss: 0.9893 - val_accuracy: 0.7333 - 36ms/epoch - 3ms/step
Epoch 24/30
13/13 - 0s - loss: 0.0913 - accuracy: 1.0000 - val_loss: 0.9807 - val_accuracy: 0.7000 - 36ms/epoch - 3ms/step
Epoch 25/30
13/13 - 0s - loss: 0.0868 - accuracy: 1.0000 - val_loss: 0.9715 - val_accuracy: 0.7333 - 37ms/epoch - 3ms/step
Epoch 26/30
13/13 - 0s - loss: 0.0788 - accuracy: 1.0000 - val_loss: 0.9850 - val_accuracy: 0.7333 - 38ms/epoch - 3ms/step
Epoch 27/30
13/13 - 0s - loss: 0.0749 - accuracy: 1.0000 - val_loss: 0.9750 - val_accuracy: 0.7333 - 51ms/epoch - 4ms/step
Epoch 28/30
13/13 - 0s - loss: 0.0730 - accuracy: 1.0000 - val_loss: 0.9570 - val_accuracy: 0.7667 - 51ms/epoch - 4ms/step
Epoch 29/30
13/13 - 0s - loss: 0.0681 - accuracy: 1.0000 - val_loss: 0.9895 - val_accuracy: 0.7333 - 37ms/epoch - 3ms/step
Epoch 30/30
13/13 - 0s - loss: 0.0635 - accuracy: 1.0000 - val_loss: 0.9560 - val_accuracy: 0.7333 - 50ms/epoch - 4ms/step
Epoch 1/30
13/13 - 0s - loss: 2.3619 - accuracy: 0.1400 - val_loss: 2.0567 - val_accuracy: 0.3667 - 407ms/epoch - 31ms/step
Epoch 2/30
13/13 - 0s - loss: 1.9696 - accuracy: 0.4200 - val_loss: 1.9381 - val_accuracy: 0.4667 - 51ms/epoch - 4ms/step
Epoch 3/30
13/13 - 0s - loss: 1.6671 - accuracy: 0.6400 - val_loss: 1.8300 - val_accuracy: 0.4333 - 38ms/epoch - 3ms/step
Epoch 4/30
13/13 - 0s - loss: 1.4340 - accuracy: 0.7400 - val_loss: 1.7113 - val_accuracy: 0.4333 - 50ms/epoch - 4ms/step
Epoch 5/30
13/13 - 0s - loss: 1.2342 - accuracy: 0.7600 - val_loss: 1.6044 - val_accuracy: 0.5000 - 38ms/epoch - 3ms/step
Epoch 6/30
13/13 - 0s - loss: 1.0721 - accuracy: 0.8600 - val_loss: 1.5232 - val_accuracy: 0.5333 - 37ms/epoch - 3ms/step
Epoch 7/30
13/13 - 0s - loss: 0.9348 - accuracy: 0.9000 - val_loss: 1.4596 - val_accuracy: 0.5667 - 50ms/epoch - 4ms/step
Epoch 8/30
13/13 - 0s - loss: 0.8178 - accuracy: 0.9200 - val_loss: 1.3921 - val_accuracy: 0.6000 - 50ms/epoch - 4ms/step
Epoch 9/30
13/13 - 0s - loss: 0.7223 - accuracy: 0.9400 - val_loss: 1.3404 - val_accuracy: 0.6333 - 37ms/epoch - 3ms/step
Epoch 10/30
13/13 - 0s - loss: 0.6404 - accuracy: 0.9600 - val_loss: 1.3065 - val_accuracy: 0.6667 - 51ms/epoch - 4ms/step
Epoch 11/30
13/13 - 0s - loss: 0.5772 - accuracy: 1.0000 - val_loss: 1.2644 - val_accuracy: 0.6333 - 52ms/epoch - 4ms/step
Epoch 12/30
13/13 - 0s - loss: 0.5199 - accuracy: 1.0000 - val_loss: 1.2558 - val_accuracy: 0.6667 - 50ms/epoch - 4ms/step
Epoch 13/30
13/13 - 0s - loss: 0.4695 - accuracy: 1.0000 - val_loss: 1.2258 - val_accuracy: 0.6667 - 36ms/epoch - 3ms/step
Epoch 14/30
13/13 - 0s - loss: 0.4238 - accuracy: 1.0000 - val_loss: 1.1897 - val_accuracy: 0.6667 - 36ms/epoch - 3ms/step
Epoch 15/30
13/13 - 0s - loss: 0.3848 - accuracy: 1.0000 - val_loss: 1.1651 - val_accuracy: 0.6667 - 52ms/epoch - 4ms/step
Epoch 16/30
13/13 - 0s - loss: 0.3525 - accuracy: 1.0000 - val_loss: 1.1503 - val_accuracy: 0.7333 - 38ms/epoch - 3ms/step
Epoch 17/30
13/13 - 0s - loss: 0.3245 - accuracy: 1.0000 - val_loss: 1.1374 - val_accuracy: 0.7000 - 37ms/epoch - 3ms/step
Epoch 18/30
13/13 - 0s - loss: 0.2992 - accuracy: 1.0000 - val_loss: 1.1174 - val_accuracy: 0.6667 - 36ms/epoch - 3ms/step
Epoch 19/30
13/13 - 0s - loss: 0.2745 - accuracy: 1.0000 - val_loss: 1.1119 - val_accuracy: 0.6667 - 37ms/epoch - 3ms/step
Epoch 20/30
13/13 - 0s - loss: 0.2551 - accuracy: 1.0000 - val_loss: 1.0903 - val_accuracy: 0.7000 - 38ms/epoch - 3ms/step
Epoch 21/30
13/13 - 0s - loss: 0.2370 - accuracy: 1.0000 - val_loss: 1.0877 - val_accuracy: 0.6667 - 50ms/epoch - 4ms/step
Epoch 22/30
13/13 - 0s - loss: 0.2199 - accuracy: 1.0000 - val_loss: 1.0776 - val_accuracy: 0.6667 - 51ms/epoch - 4ms/step
Epoch 23/30
13/13 - 0s - loss: 0.2050 - accuracy: 1.0000 - val_loss: 1.0675 - val_accuracy: 0.7000 - 53ms/epoch - 4ms/step
Epoch 24/30
13/13 - 0s - loss: 0.1919 - accuracy: 1.0000 - val_loss: 1.0592 - val_accuracy: 0.7000 - 52ms/epoch - 4ms/step
Epoch 25/30
13/13 - 0s - loss: 0.1812 - accuracy: 1.0000 - val_loss: 1.0554 - val_accuracy: 0.6667 - 51ms/epoch - 4ms/step
Epoch 26/30
13/13 - 0s - loss: 0.1691 - accuracy: 1.0000 - val_loss: 1.0477 - val_accuracy: 0.7000 - 52ms/epoch - 4ms/step
Epoch 27/30
13/13 - 0s - loss: 0.1599 - accuracy: 1.0000 - val_loss: 1.0377 - val_accuracy: 0.7000 - 53ms/epoch - 4ms/step
Epoch 28/30
13/13 - 0s - loss: 0.1515 - accuracy: 1.0000 - val_loss: 1.0341 - val_accuracy: 0.6667 - 38ms/epoch - 3ms/step
Epoch 29/30
13/13 - 0s - loss: 0.1426 - accuracy: 1.0000 - val_loss: 1.0291 - val_accuracy: 0.7000 - 51ms/epoch - 4ms/step
Epoch 30/30
13/13 - 0s - loss: 0.1344 - accuracy: 1.0000 - val_loss: 1.0264 - val_accuracy: 0.7000 - 51ms/epoch - 4ms/step
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
 4014080/11490434 [=========>....................] - ETA: 0s
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
13/13 - 0s - loss: 3.0160 - accuracy: 0.1000 - val_loss: 2.0646 - val_accuracy: 0.2000 - 400ms/epoch - 31ms/step
Epoch 2/30
13/13 - 0s - loss: 2.2510 - accuracy: 0.1800 - val_loss: 1.9801 - val_accuracy: 0.3333 - 46ms/epoch - 4ms/step
Epoch 3/30
13/13 - 0s - loss: 1.7851 - accuracy: 0.4000 - val_loss: 1.8177 - val_accuracy: 0.2667 - 46ms/epoch - 4ms/step
Epoch 4/30
13/13 - 0s - loss: 1.3652 - accuracy: 0.5400 - val_loss: 1.6107 - val_accuracy: 0.4667 - 31ms/epoch - 2ms/step
Epoch 5/30
13/13 - 0s - loss: 1.1317 - accuracy: 0.7800 - val_loss: 1.4723 - val_accuracy: 0.6000 - 46ms/epoch - 4ms/step
Epoch 6/30
13/13 - 0s - loss: 0.9360 - accuracy: 0.8600 - val_loss: 1.4686 - val_accuracy: 0.5333 - 45ms/epoch - 3ms/step
Epoch 7/30
13/13 - 0s - loss: 0.7383 - accuracy: 0.9400 - val_loss: 1.3536 - val_accuracy: 0.5667 - 31ms/epoch - 2ms/step
Epoch 8/30
13/13 - 0s - loss: 0.5846 - accuracy: 0.9800 - val_loss: 1.2785 - val_accuracy: 0.6667 - 46ms/epoch - 4ms/step
Epoch 9/30
13/13 - 0s - loss: 0.4987 - accuracy: 0.9800 - val_loss: 1.2253 - val_accuracy: 0.6333 - 46ms/epoch - 4ms/step
Epoch 10/30
13/13 - 0s - loss: 0.3921 - accuracy: 1.0000 - val_loss: 1.2655 - val_accuracy: 0.6333 - 31ms/epoch - 2ms/step
Epoch 11/30
13/13 - 0s - loss: 0.3617 - accuracy: 1.0000 - val_loss: 1.1555 - val_accuracy: 0.7000 - 31ms/epoch - 2ms/step
Epoch 12/30
13/13 - 0s - loss: 0.3078 - accuracy: 1.0000 - val_loss: 1.2107 - val_accuracy: 0.6667 - 32ms/epoch - 2ms/step
Epoch 13/30
13/13 - 0s - loss: 0.2618 - accuracy: 1.0000 - val_loss: 1.1166 - val_accuracy: 0.7333 - 31ms/epoch - 2ms/step
Epoch 14/30
13/13 - 0s - loss: 0.2463 - accuracy: 1.0000 - val_loss: 1.0624 - val_accuracy: 0.7000 - 46ms/epoch - 4ms/step
Epoch 15/30
13/13 - 0s - loss: 0.2033 - accuracy: 1.0000 - val_loss: 1.0904 - val_accuracy: 0.7333 - 47ms/epoch - 4ms/step
Epoch 16/30
13/13 - 0s - loss: 0.1799 - accuracy: 1.0000 - val_loss: 1.0865 - val_accuracy: 0.7333 - 46ms/epoch - 4ms/step
Epoch 17/30
13/13 - 0s - loss: 0.1682 - accuracy: 1.0000 - val_loss: 1.0385 - val_accuracy: 0.7333 - 31ms/epoch - 2ms/step
Epoch 18/30
13/13 - 0s - loss: 0.1484 - accuracy: 1.0000 - val_loss: 1.0676 - val_accuracy: 0.7000 - 31ms/epoch - 2ms/step
Epoch 19/30
13/13 - 0s - loss: 0.1349 - accuracy: 1.0000 - val_loss: 1.0447 - val_accuracy: 0.7000 - 46ms/epoch - 4ms/step
Epoch 20/30
13/13 - 0s - loss: 0.1255 - accuracy: 1.0000 - val_loss: 0.9935 - val_accuracy: 0.7333 - 31ms/epoch - 2ms/step
Epoch 21/30
13/13 - 0s - loss: 0.1135 - accuracy: 1.0000 - val_loss: 1.0451 - val_accuracy: 0.7333 - 31ms/epoch - 2ms/step
Epoch 22/30
13/13 - 0s - loss: 0.1041 - accuracy: 1.0000 - val_loss: 1.0142 - val_accuracy: 0.7333 - 31ms/epoch - 2ms/step
Epoch 23/30
13/13 - 0s - loss: 0.0983 - accuracy: 1.0000 - val_loss: 0.9893 - val_accuracy: 0.7333 - 46ms/epoch - 4ms/step
Epoch 24/30
13/13 - 0s - loss: 0.0913 - accuracy: 1.0000 - val_loss: 0.9807 - val_accuracy: 0.7000 - 46ms/epoch - 4ms/step
Epoch 25/30
13/13 - 0s - loss: 0.0868 - accuracy: 1.0000 - val_loss: 0.9715 - val_accuracy: 0.7333 - 32ms/epoch - 2ms/step
Epoch 26/30
13/13 - 0s - loss: 0.0788 - accuracy: 1.0000 - val_loss: 0.9850 - val_accuracy: 0.7333 - 32ms/epoch - 2ms/step
Epoch 27/30
13/13 - 0s - loss: 0.0749 - accuracy: 1.0000 - val_loss: 0.9750 - val_accuracy: 0.7333 - 31ms/epoch - 2ms/step
Epoch 28/30
13/13 - 0s - loss: 0.0730 - accuracy: 1.0000 - val_loss: 0.9570 - val_accuracy: 0.7667 - 30ms/epoch - 2ms/step
Epoch 29/30
13/13 - 0s - loss: 0.0681 - accuracy: 1.0000 - val_loss: 0.9895 - val_accuracy: 0.7333 - 46ms/epoch - 4ms/step
Epoch 30/30
13/13 - 0s - loss: 0.0635 - accuracy: 1.0000 - val_loss: 0.9560 - val_accuracy: 0.7333 - 31ms/epoch - 2ms/step
Epoch 1/30
13/13 - 0s - loss: 2.3619 - accuracy: 0.1400 - val_loss: 2.0567 - val_accuracy: 0.3667 - 340ms/epoch - 26ms/step
Epoch 2/30
13/13 - 0s - loss: 1.9696 - accuracy: 0.4200 - val_loss: 1.9381 - val_accuracy: 0.4667 - 46ms/epoch - 4ms/step
Epoch 3/30
13/13 - 0s - loss: 1.6671 - accuracy: 0.6400 - val_loss: 1.8300 - val_accuracy: 0.4333 - 31ms/epoch - 2ms/step
Epoch 4/30
13/13 - 0s - loss: 1.4340 - accuracy: 0.7400 - val_loss: 1.7113 - val_accuracy: 0.4333 - 46ms/epoch - 4ms/step
Epoch 5/30
13/13 - 0s - loss: 1.2342 - accuracy: 0.7600 - val_loss: 1.6044 - val_accuracy: 0.5000 - 47ms/epoch - 4ms/step
Epoch 6/30
13/13 - 0s - loss: 1.0721 - accuracy: 0.8600 - val_loss: 1.5232 - val_accuracy: 0.5333 - 31ms/epoch - 2ms/step
Epoch 7/30
13/13 - 0s - loss: 0.9348 - accuracy: 0.9000 - val_loss: 1.4596 - val_accuracy: 0.5667 - 44ms/epoch - 3ms/step
Epoch 8/30
13/13 - 0s - loss: 0.8178 - accuracy: 0.9200 - val_loss: 1.3921 - val_accuracy: 0.6000 - 45ms/epoch - 3ms/step
Epoch 9/30
13/13 - 0s - loss: 0.7223 - accuracy: 0.9400 - val_loss: 1.3404 - val_accuracy: 0.6333 - 45ms/epoch - 3ms/step
Epoch 10/30
13/13 - 0s - loss: 0.6404 - accuracy: 0.9600 - val_loss: 1.3065 - val_accuracy: 0.6667 - 30ms/epoch - 2ms/step
Epoch 11/30
13/13 - 0s - loss: 0.5772 - accuracy: 1.0000 - val_loss: 1.2644 - val_accuracy: 0.6333 - 30ms/epoch - 2ms/step
Epoch 12/30
13/13 - 0s - loss: 0.5199 - accuracy: 1.0000 - val_loss: 1.2558 - val_accuracy: 0.6667 - 31ms/epoch - 2ms/step
Epoch 13/30
13/13 - 0s - loss: 0.4695 - accuracy: 1.0000 - val_loss: 1.2258 - val_accuracy: 0.6667 - 45ms/epoch - 3ms/step
Epoch 14/30
13/13 - 0s - loss: 0.4238 - accuracy: 1.0000 - val_loss: 1.1897 - val_accuracy: 0.6667 - 47ms/epoch - 4ms/step
Epoch 15/30
13/13 - 0s - loss: 0.3848 - accuracy: 1.0000 - val_loss: 1.1651 - val_accuracy: 0.6667 - 31ms/epoch - 2ms/step
Epoch 16/30
13/13 - 0s - loss: 0.3525 - accuracy: 1.0000 - val_loss: 1.1503 - val_accuracy: 0.7333 - 31ms/epoch - 2ms/step
Epoch 17/30
13/13 - 0s - loss: 0.3245 - accuracy: 1.0000 - val_loss: 1.1374 - val_accuracy: 0.7000 - 30ms/epoch - 2ms/step
Epoch 18/30
13/13 - 0s - loss: 0.2992 - accuracy: 1.0000 - val_loss: 1.1174 - val_accuracy: 0.6667 - 30ms/epoch - 2ms/step
Epoch 19/30
13/13 - 0s - loss: 0.2745 - accuracy: 1.0000 - val_loss: 1.1119 - val_accuracy: 0.6667 - 46ms/epoch - 4ms/step
Epoch 20/30
13/13 - 0s - loss: 0.2551 - accuracy: 1.0000 - val_loss: 1.0903 - val_accuracy: 0.7000 - 47ms/epoch - 4ms/step
Epoch 21/30
13/13 - 0s - loss: 0.2370 - accuracy: 1.0000 - val_loss: 1.0877 - val_accuracy: 0.6667 - 31ms/epoch - 2ms/step
Epoch 22/30
13/13 - 0s - loss: 0.2199 - accuracy: 1.0000 - val_loss: 1.0776 - val_accuracy: 0.6667 - 31ms/epoch - 2ms/step
Epoch 23/30
13/13 - 0s - loss: 0.2050 - accuracy: 1.0000 - val_loss: 1.0675 - val_accuracy: 0.7000 - 46ms/epoch - 4ms/step
Epoch 24/30
13/13 - 0s - loss: 0.1919 - accuracy: 1.0000 - val_loss: 1.0592 - val_accuracy: 0.7000 - 32ms/epoch - 2ms/step
Epoch 25/30
13/13 - 0s - loss: 0.1812 - accuracy: 1.0000 - val_loss: 1.0554 - val_accuracy: 0.6667 - 30ms/epoch - 2ms/step
Epoch 26/30
13/13 - 0s - loss: 0.1691 - accuracy: 1.0000 - val_loss: 1.0477 - val_accuracy: 0.7000 - 31ms/epoch - 2ms/step
Epoch 27/30
13/13 - 0s - loss: 0.1599 - accuracy: 1.0000 - val_loss: 1.0377 - val_accuracy: 0.7000 - 45ms/epoch - 3ms/step
Epoch 28/30
13/13 - 0s - loss: 0.1515 - accuracy: 1.0000 - val_loss: 1.0341 - val_accuracy: 0.6667 - 31ms/epoch - 2ms/step
Epoch 29/30
13/13 - 0s - loss: 0.1426 - accuracy: 1.0000 - val_loss: 1.0291 - val_accuracy: 0.7000 - 45ms/epoch - 3ms/step
Epoch 30/30
 </code>
 </pre>
 </details>

---

## 8. tutorial_adaptive_circuits.html <a name="demo7"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_adaptive_circuits.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
n = 0,  E = -7.86266588 H, t = 0.78 s
n = 1,  E = -7.87094622 H, t = 0.78 s
n = 2,  E = -7.87563101 H, t = 0.78 s
n = 3,  E = -7.87829148 H, t = 0.78 s
n = 4,  E = -7.87981707 H, t = 1.06 s
n = 5,  E = -7.88070478 H, t = 0.78 s
n = 6,  E = -7.88123144 H, t = 0.78 s
n = 7,  E = -7.88155162 H, t = 0.78 s
n = 8,  E = -7.88175219 H, t = 0.78 s
n = 9,  E = -7.88188238 H, t = 1.05 s
n = 10,  E = -7.88197042 H, t = 0.78 s
n = 11,  E = -7.88203269 H, t = 0.78 s
n = 12,  E = -7.88207881 H, t = 0.78 s
n = 13,  E = -7.88211453 H, t = 0.78 s
n = 14,  E = -7.88214336 H, t = 1.06 s
n = 15,  E = -7.88216745 H, t = 0.78 s
n = 16,  E = -7.88218815 H, t = 0.78 s
n = 17,  E = -7.88220635 H, t = 0.78 s
n = 18,  E = -7.88222262 H, t = 1.07 s
n = 19,  E = -7.88223735 H, t = 0.78 s
n = 0,  E = -7.86266588 H, t = 0.15 s
n = 1,  E = -7.87094622 H, t = 0.15 s
n = 2,  E = -7.87563101 H, t = 0.16 s
n = 3,  E = -7.87829148 H, t = 0.15 s
n = 4,  E = -7.87981707 H, t = 0.15 s
n = 5,  E = -7.88070478 H, t = 0.15 s
n = 6,  E = -7.88123144 H, t = 0.15 s
n = 7,  E = -7.88155162 H, t = 0.15 s
n = 8,  E = -7.88175219 H, t = 0.15 s
n = 10,  E = -7.88197042 H, t = 0.16 s
n = 11,  E = -7.88203269 H, t = 0.16 s
n = 12,  E = -7.88207881 H, t = 0.15 s
n = 13,  E = -7.88211453 H, t = 0.15 s
n = 14,  E = -7.88214336 H, t = 0.15 s
n = 15,  E = -7.88216745 H, t = 0.15 s
n = 16,  E = -7.88218815 H, t = 0.16 s
n = 17,  E = -7.88220635 H, t = 0.15 s
n = 18,  E = -7.88222262 H, t = 0.15 s
n = 19,  E = -7.88223735 H, t = 0.15 s
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
n = 0,  E = -7.86266588 H, t = 0.72 s
n = 1,  E = -7.87094622 H, t = 0.72 s
n = 2,  E = -7.87563101 H, t = 0.72 s
n = 3,  E = -7.87829148 H, t = 0.88 s
n = 4,  E = -7.87981707 H, t = 0.72 s
n = 5,  E = -7.88070478 H, t = 0.72 s
n = 6,  E = -7.88123144 H, t = 0.72 s
n = 7,  E = -7.88155162 H, t = 0.88 s
n = 8,  E = -7.88175219 H, t = 0.72 s
n = 9,  E = -7.88188238 H, t = 0.72 s
n = 10,  E = -7.88197042 H, t = 0.72 s
n = 11,  E = -7.88203269 H, t = 0.72 s
n = 12,  E = -7.88207881 H, t = 0.89 s
n = 13,  E = -7.88211453 H, t = 0.72 s
n = 14,  E = -7.88214336 H, t = 0.73 s
n = 15,  E = -7.88216745 H, t = 0.72 s
n = 16,  E = -7.88218815 H, t = 0.72 s
n = 17,  E = -7.88220635 H, t = 0.89 s
n = 18,  E = -7.88222262 H, t = 0.72 s
n = 19,  E = -7.88223735 H, t = 0.72 s
n = 0,  E = -7.86266588 H, t = 0.14 s
n = 1,  E = -7.87094622 H, t = 0.14 s
n = 2,  E = -7.87563101 H, t = 0.14 s
n = 3,  E = -7.87829148 H, t = 0.14 s
n = 4,  E = -7.87981707 H, t = 0.14 s
n = 5,  E = -7.88070478 H, t = 0.14 s
n = 6,  E = -7.88123144 H, t = 0.14 s
n = 7,  E = -7.88155162 H, t = 0.14 s
n = 8,  E = -7.88175219 H, t = 0.14 s
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

---

## 9. tutorial_qft_arithmetics.html <a name="demo8"></a>

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

## 10. tutorial_quantum_circuit_cutting.html <a name="demo9"></a>

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

## 11. tutorial_rotoselect.html <a name="demo10"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_rotoselect.html):

```
/home/runner/work/qml/qml/demonstrations/tutorial_rotoselect.py:269: MatplotlibDeprecationWarning: Calling gca() with keyword arguments was deprecated in Matplotlib 3.4. Starting two minor releases later, gca() will take no keyword arguments. The gca() function should only be used to get the current axes, or if no axes exist, create new axes with default keyword arguments. To create a new axes with non-default arguments, use plt.axes() or plt.subplot().
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_rotoselect.html):

```
Optimal generators are: ['Y', 'X']
```

---

## 12. tutorial_backprop.html <a name="demo11"></a>

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
Forward pass (best of 3): 0.029129625000041414 sec per loop
Gradient computation (best of 3): 11.854861354500008 sec per loop
10.48666500001491
0.9358535378025424
Forward pass (best of 3): 0.06344520779994127 sec per loop
Backward pass (best of 3): 0.17767611139997824 sec per loop
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

## 13. tutorial_quantum_chemistry.html <a name="demo12"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quantum_chemistry.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
  (-46.46390691340926) [I0]
+ (0.7829652070488278) [Z10]
+ (0.7829652070488278) [Z11]
+ (0.80845910051847) [Z12]
+ (0.8084591005184703) [Z13]
+ (1.203439339131281) [Z4]
+ (1.203439339131281) [Z5]
+ (1.309687661862992) [Z6]
+ (1.309687661862992) [Z7]
+ (1.3693525711695045) [Z8]
+ (1.369352571169505) [Z9]
+ (1.6538938305474364) [Z2]
+ (1.6538938305474367) [Z3]
+ (12.412630714438524) [Z0]
+ (12.412630714438524) [Z1]
+ (-8.194104900192262e-06) [Y10 Y12]
+ (-8.194104900192262e-06) [X10 X12]
+ (-1.6021751068218132e-06) [Y2 Y4]
+ (-1.6021751068218132e-06) [X2 X4]
+ (5.929280274693215e-07) [Y4 Y6]
+ (5.929280274693215e-07) [X4 X6]
+ (7.76508243338441e-07) [Y3 Y5]
+ (7.76508243338441e-07) [X3 X5]
+ (1.8540565154096748e-06) [Y5 Y7]
+ (1.8540565154096748e-06) [X5 X7]
+ (7.954224387346498e-06) [Y11 Y13]
+ (7.954224387346498e-06) [X11 X13]
+ (0.0032769650657560132) [Y1 Y3]
+ (0.0032769650657560132) [X1 X3]
+ (0.10433061485313576) [Y0 Y2]
+ (0.10433061485313576) [X0 X2]
+ (0.11270381859118761) [Z10 Z12]
+ (0.11270381859118761) [Z11 Z13]
+ (0.11383573685298291) [Z4 Z12]
+ (0.11383573685298291) [Z5 Z13]
+ (0.11952441016895111) [Z6 Z10]
+ (0.11952441016895111) [Z7 Z11]
+ (0.12489977362990852) [Z4 Z10]
+ (0.12489977362990852) [Z5 Z11]
+ (0.12495799328197955) [Z2 Z4]
+ (0.12495799328197955) [Z3 Z5]
+ (0.12799492801398263) [Z2 Z10]
+ (0.12799492801398263) [Z3 Z11]
+ (0.13401737372652434) [Z6 Z12]
+ (0.13401737372652434) [Z7 Z13]
+ (0.13701191913060837) [Z4 Z6]
+ (0.13701191913060837) [Z5 Z7]
+ (0.13734942210158252) [Z6 Z11]
+ (0.13734942210158252) [Z7 Z10]
+ (0.13739112375017043) [Z2 Z6]
+ (0.13739112375017043) [Z3 Z7]
+ (0.13766859133611747) [Z8 Z10]
+ (0.13766859133611747) [Z9 Z11]
+ (0.1401129474971318) [Z2 Z12]
+ (0.1401129474971318) [Z3 Z13]
+ (0.1413890359014588) [Z10 Z13]
+ (0.1413890359014588) [Z11 Z12]
+ (0.14257991128702385) [Z4 Z11]
+ (0.14257991128702385) [Z5 Z10]
+ (0.14722930783255977) [Z8 Z11]
+ (0.14722930783255977) [Z9 Z10]
+ (0.1489942617138656) [Z4 Z7]
+ (0.1489942617138656) [Z5 Z6]
+ (0.14926347060041678) [Z10 Z11]
+ (0.1496069255724247) [Z4 Z8]
+ (0.1496069255724247) [Z5 Z9]
+ (0.1497349700542438) [Z8 Z12]
+ (0.1497349700542438) [Z9 Z13]
+ (0.15071405482288697) [Z2 Z8]
+ (0.15071405482288697) [Z3 Z9]
+ (0.1513834269911447) [Z6 Z13]
+ (0.1513834269911447) [Z7 Z12]
+ (0.15215040622643458) [Z4 Z13]
+ (0.15215040622643458) [Z5 Z12]
+ (0.15337959171064824) [Z2 Z11]
+ (0.15337959171064824) [Z3 Z10]
+ (0.15435760065297766) [Z12 Z13]
+ (0.15569017457259154) [Z2 Z13]
+ (0.15569017457259154) [Z3 Z12]
+ (0.15582280685847377) [Z8 Z13]
+ (0.15582280685847377) [Z9 Z12]
+ (0.15676384610161093) [Z4 Z9]
+ (0.15676384610161093) [Z5 Z8]
+ (0.15755303804352117) [Z4 Z5]
+ (0.1607975504670137) [Z2 Z5]
+ (0.1607975504670137) [Z3 Z4]
+ (0.16756669356165232) [Z6 Z8]
+ (0.16756669356165232) [Z7 Z9]
+ (0.16853492794654543) [Z2 Z7]
+ (0.16853492794654543) [Z3 Z6]
+ (0.18144009362927455) [Z6 Z9]
+ (0.18144009362927455) [Z7 Z8]
+ (0.1818908124374862) [Z2 Z3]
+ (0.18690814831038954) [Z2 Z9]
+ (0.18690814831038954) [Z3 Z8]
+ (0.19299700269862527) [Z0 Z10]
+ (0.19299700269862527) [Z1 Z11]
+ (0.19392574334980825) [Z6 Z7]
+ (0.1966174995972808) [Z0 Z4]
+ (0.1966174995972808) [Z1 Z5]
+ (0.1993633269127053) [Z0 Z5]
+ (0.1993633269127053) [Z1 Z4]
+ (0.2007284355459833) [Z0 Z11]
+ (0.2007284355459833) [Z1 Z10]
+ (0.21102681234280887) [Z0 Z12]
+ (0.21102681234280887) [Z1 Z13]
+ (0.21631059809663616) [Z0 Z13]
+ (0.21631059809663616) [Z1 Z12]
+ (0.2200397724025619) [Z8 Z9]
+ (0.2367107174042757) [Z0 Z2]
+ (0.2367107174042757) [Z1 Z3]
+ (0.24164696831746305) [Z0 Z6]
+ (0.24164696831746305) [Z1 Z7]
+ (0.24853517285978088) [Z0 Z7]
+ (0.24853517285978088) [Z1 Z6]
+ (0.2512943557312974) [Z0 Z3]
+ (0.2512943557312974) [Z1 Z2]
+ (0.2723251845036176) [Z0 Z8]
+ (0.2723251845036176) [Z1 Z9]
+ (0.2788345457376928) [Z0 Z9]
+ (0.2788345457376928) [Z1 Z8]
+ (1.186176448413044) [Z0 Z1]
+ (-1.07227486880895e-05) [Y10 Z11 Y12]
+ (-1.07227486880895e-05) [X10 Z11 X12]
+ (-1.07227486880895e-05) [Y11 Z12 Y13]
+ (-1.07227486880895e-05) [X11 Z12 X13]
+ (3.8866394178250045e-06) [Y2 Z3 Y4]
+ (3.8866394178250045e-06) [X2 Z3 X4]
+ (3.8866394178250045e-06) [Y3 Z4 Y5]
+ (3.8866394178250045e-06) [X3 Z4 X5]
+ (1.2260276800747645e-05) [Y4 Z5 Y6]
+ (1.2260276800747645e-05) [X4 Z5 X6]
+ (1.2260276800747645e-05) [Y5 Z6 Y7]
+ (1.2260276800747645e-05) [X5 Z6 X7]
+ (0.12507036883980113) [Y0 Z1 Y2]
+ (0.12507036883980113) [X0 Z1 X2]
+ (0.12507036883980116) [Y1 Z2 Y3]
+ (0.12507036883980116) [X1 Z2 X3]
+ (-0.038314669373451654) [Y4 Y5 X12 X13]
+ (-0.038314669373451654) [X4 X5 Y12 Y13]
+ (-0.03619409348750257) [Y2 Y3 X8 X9]
+ (-0.03619409348750257) [X2 X3 Y8 Y9]
+ (-0.03583955718503415) [Y2 Y3 X4 X5]
+ (-0.03583955718503415) [X2 X3 Y4 Y5]
+ (-0.031143804196374976) [Y2 Y3 X6 X7]
+ (-0.031143804196374976) [X2 X3 Y6 Y7]
+ (-0.030787440718598236) [Y6 Z8 Z9 Y10]
+ (-0.030787440718598236) [X6 Z8 Z9 X10]
+ (-0.030104525273566524) [Y6 Z7 Z9 Y10]
+ (-0.030104525273566524) [X6 Z7 Z9 X10]
+ (-0.030104525273566524) [Y7 Z8 Z10 Y11]
+ (-0.030104525273566524) [X7 Z8 Z10 X11]
+ (-0.029812299601116466) [Y6 Z7 Z8 Y10]
+ (-0.029812299601116466) [X6 Z7 Z8 X10]
+ (-0.029812299601116466) [Y7 Z9 Z10 Y11]
+ (-0.029812299601116466) [X7 Z9 Z10 X11]
+ (-0.02868521731027118) [Y10 Y11 X12 X13]
+ (-0.02868521731027118) [X10 X11 Y12 Y13]
+ (-0.025996206267197022) [Y3 Z4 Z5 Y7]
+ (-0.025996206267197022) [X3 Z4 Z5 X7]
+ (-0.025384663696665626) [Y2 Y3 X10 X11]
+ (-0.025384663696665626) [X2 X3 Y10 Y11]
+ (-0.019028318718288723) [Y3 X4 X11 Y12]
+ (-0.019028318718288723) [X3 Y4 Y11 X12]
+ (-0.01782501193263139) [Y6 Y7 X10 X11]
+ (-0.01782501193263139) [X6 X7 Y10 Y11]
+ (-0.017680137657115326) [Y4 Y5 X10 X11]
+ (-0.017680137657115326) [X4 X5 Y10 Y11]
+ (-0.017366053264620398) [Y6 Y7 X12 X13]
+ (-0.017366053264620398) [X6 X7 Y12 Y13]
+ (-0.01557722707545975) [Y2 Y3 X12 X13]
+ (-0.01557722707545975) [X2 X3 Y12 Y13]
+ (-0.014583638327021652) [Y0 Y1 X2 X3]
+ (-0.014583638327021652) [X0 X1 Y2 Y3]
+ (-0.013873400067622195) [Y6 Y7 X8 X9]
+ (-0.013873400067622195) [X6 X7 Y8 Y9]
+ (-0.011982342583257208) [Y4 Y5 X6 X7]
+ (-0.011982342583257208) [X4 X5 Y6 Y7]
+ (-0.011307208030035727) [Y7 Z8 Z9 Y11]
+ (-0.011307208030035727) [X7 Z8 Z9 X11]
+ (-0.011285144618313647) [Y5 Y6 X11 X12]
+ (-0.011285144618313647) [X5 X6 Y11 Y12]
+ (-0.009560716496442287) [Y8 Y9 X10 X11]
+ (-0.009560716496442287) [X8 X9 Y10 Y11]
+ (-0.00812524841012268) [Y1 X2 X8 Y9]
+ (-0.00812524841012268) [Y1 Y2 Y8 Y9]
+ (-0.00812524841012268) [X1 X2 X8 X9]
+ (-0.00812524841012268) [X1 Y2 Y8 X9]
+ (-0.007731432847358059) [Y0 Y1 X10 X11]
+ (-0.007731432847358059) [X0 X1 Y10 Y11]
+ (-0.007156920529186221) [Y4 Y5 X8 X9]
+ (-0.007156920529186221) [X4 X5 Y8 Y9]
+ (-0.0068882045423178445) [Y0 Y1 X6 X7]
+ (-0.0068882045423178445) [X0 X1 Y6 Y7]
+ (-0.006509361234075182) [Y0 Y1 X8 X9]
+ (-0.006509361234075182) [X0 X1 Y8 Y9]
+ (-0.006087836804229986) [Y8 Y9 X12 X13]
+ (-0.006087836804229986) [X8 X9 Y12 Y13]
+ (-0.00528378575382731) [Y0 Y1 X12 X13]
+ (-0.00528378575382731) [X0 X1 Y12 Y13]
+ (-0.005143382387689708) [Y3 X4 X5 Y6]
+ (-0.005143382387689708) [X3 Y4 Y5 X6]
+ (-0.004684920226872947) [Y1 X2 X6 Y7]
+ (-0.004684920226872947) [Y1 Y2 Y6 Y7]
+ (-0.004684920226872947) [X1 X2 X6 X7]
+ (-0.004684920226872947) [X1 Y2 Y6 X7]
+ (-0.004575015188895431) [Y1 X2 X12 Y13]
+ (-0.004575015188895431) [Y1 Y2 Y12 Y13]
+ (-0.004575015188895431) [X1 X2 X12 X13]
+ (-0.004575015188895431) [X1 Y2 Y12 X13]
+ (-0.004424843668499424) [Y1 X2 X4 Y5]
+ (-0.004424843668499424) [Y1 Y2 Y4 Y5]
+ (-0.004424843668499424) [X1 X2 X4 X5]
+ (-0.004424843668499424) [X1 Y2 Y4 X5]
+ (-0.0034794217293031443) [Y2 Z3 Z5 Y6]
+ (-0.0034794217293031443) [X2 Z3 Z5 X6]
+ (-0.0034794217293031443) [Y3 Z4 Z6 Y7]
+ (-0.0034794217293031443) [X3 Z4 Z6 X7]
+ (-0.002745827315424492) [Y0 Y1 X4 X5]
+ (-0.002745827315424492) [X0 X1 Y4 Y5]
+ (-0.0017991930083859425) [Y1 X2 X10 Y11]
+ (-0.0017991930083859425) [Y1 Y2 Y10 Y11]
+ (-0.0017991930083859425) [X1 X2 X10 X11]
+ (-0.0017991930083859425) [X1 Y2 Y10 X11]
+ (-0.00029222567245005795) [Y7 X8 X9 Y10]
+ (-0.00029222567245005795) [X7 Y8 Y9 X10]
+ (-8.814793238697168e-06) [Y2 Z3 Y4 Z13]
+ (-8.814793238697168e-06) [X2 Z3 X4 Z13]
+ (-8.814793238697168e-06) [Y3 Z4 Y5 Z12]
+ (-8.814793238697168e-06) [X3 Z4 X5 Z12]
+ (-8.194104900192262e-06) [Z10 Y11 Z12 Y13]
+ (-8.194104900192262e-06) [Z10 X11 Z12 X13]
+ (-5.974176776677234e-06) [Y5 X6 X10 Y11]
+ (-5.974176776677234e-06) [Y5 Y6 Y10 Y11]
+ (-5.974176776677234e-06) [X5 X6 X10 X11]
+ (-5.974176776677234e-06) [X5 Y6 Y10 X11]
+ (-5.275783339901089e-06) [Y3 X4 X12 Y13]
+ (-5.275783339901089e-06) [Y3 Y4 Y12 Y13]
+ (-5.275783339901089e-06) [X3 X4 X12 X13]
+ (-5.275783339901089e-06) [X3 Y4 Y12 X13]
+ (-4.281811976833503e-06) [Y4 Z5 Y6 Z11]
+ (-4.281811976833503e-06) [X4 Z5 X6 Z11]
+ (-4.281811976833503e-06) [Y5 Z6 Y7 Z10]
+ (-4.281811976833503e-06) [X5 Z6 X7 Z10]
+ (-3.6945168651967425e-06) [Y4 X5 X11 Y12]
+ (-3.6945168651967425e-06) [Y4 Y5 Y11 Y12]
+ (-3.6945168651967425e-06) [X4 X5 X11 X12]
+ (-3.6945168651967425e-06) [X4 Y5 Y11 X12]
+ (-3.53900989879391e-06) [Y2 Z3 Y4 Z12]
+ (-3.53900989879391e-06) [X2 Z3 X4 Z12]
+ (-3.53900989879391e-06) [Y3 Z4 Y5 Z13]
+ (-3.53900989879391e-06) [X3 Z4 X5 Z13]
+ (-3.117366364509771e-06) [Y0 Z2 Z3 Y4]
+ (-3.117366364509771e-06) [X0 Z2 Z3 X4]
+ (-2.890929966562275e-06) [Z6 Y11 Z12 Y13]
+ (-2.890929966562275e-06) [Z6 X11 Z12 X13]
+ (-2.890929966562275e-06) [Z7 Y10 Z11 Y12]
+ (-2.890929966562275e-06) [Z7 X10 Z11 X12]
+ (-2.177733027297624e-06) [Z0 Y10 Z11 Y12]
+ (-2.177733027297624e-06) [Z0 X10 Z11 X12]
+ (-2.177733027297624e-06) [Z1 Y11 Z12 Y13]
+ (-2.177733027297624e-06) [Z1 X11 Z12 X13]
+ (-1.8551374931044008e-06) [Z6 Y10 Z11 Y12]
+ (-1.8551374931044008e-06) [Z6 X10 Z11 X12]
+ (-1.8551374931044008e-06) [Z7 Y11 Z12 Y13]
+ (-1.8551374931044008e-06) [Z7 X11 Z12 X13]
+ (-1.8163674404079316e-06) [Z4 Y11 Z12 Y13]
+ (-1.8163674404079316e-06) [Z4 X11 Z12 X13]
+ (-1.8163674404079316e-06) [Z5 Y10 Z11 Y12]
+ (-1.8163674404079316e-06) [Z5 X10 Z11 X12]
+ (-1.6149608367971127e-06) [Z0 Y11 Z12 Y13]
+ (-1.6149608367971127e-06) [Z0 X11 Z12 X13]
+ (-1.6149608367971127e-06) [Z1 Y10 Z11 Y12]
+ (-1.6149608367971127e-06) [Z1 X10 Z11 X12]
+ (-1.6021751068218132e-06) [Z2 Y3 Z4 Y5]
+ (-1.6021751068218132e-06) [Z2 X3 Z4 X5]
+ (-1.5973397553423413e-06) [Z8 Y10 Z11 Y12]
+ (-1.5973397553423413e-06) [Z8 X10 Z11 X12]
+ (-1.5973397553423413e-06) [Z9 Y11 Z12 Y13]
+ (-1.5973397553423413e-06) [Z9 X11 Z12 X13]
+ (-1.063225548045125e-06) [Z2 Y10 Z11 Y12]
+ (-1.063225548045125e-06) [Z2 X10 Z11 X12]
+ (-1.063225548045125e-06) [Z3 Y11 Z12 Y13]
+ (-1.063225548045125e-06) [Z3 X11 Z12 X13]
+ (-1.0357924734557059e-06) [Y6 X7 X11 Y12]
+ (-1.0357924734557059e-06) [Y6 Y7 Y11 Y12]
+ (-1.0357924734557059e-06) [X6 X7 X11 X12]
+ (-1.0357924734557059e-06) [X6 Y7 Y11 X12]
+ (-9.344970449999546e-07) [Z8 Y11 Z12 Y13]
+ (-9.344970449999546e-07) [Z8 X11 Z12 X13]
+ (-9.344970449999546e-07) [Z9 Y10 Z11 Y12]
+ (-9.344970449999546e-07) [Z9 X10 Z11 X12]
+ (-5.471605954255281e-07) [Y1 Y2 X11 X12]
+ (-5.471605954255281e-07) [X1 X2 Y11 Y12]
+ (-3.6060692683102775e-07) [Y0 Z1 Z2 Y4]
+ (-3.6060692683102775e-07) [X0 Z1 Z2 X4]
+ (-3.6060692683102775e-07) [Y1 Z3 Z4 Y5]
+ (-3.6060692683102775e-07) [X1 Z3 Z4 X5]
+ (-2.594146163552584e-07) [Y2 Z3 Y4 Z6]
+ (-2.594146163552584e-07) [X2 Z3 X4 Z6]
+ (-2.594146163552584e-07) [Y3 Z4 Y5 Z7]
+ (-2.594146163552584e-07) [X3 Z4 X5 Z7]
+ (-2.1872304297918498e-07) [Y2 Z3 Y4 Z8]
+ (-2.1872304297918498e-07) [X2 Z3 X4 Z8]
+ (-2.1872304297918498e-07) [Y3 Z4 Y5 Z9]
+ (-2.1872304297918498e-07) [X3 Z4 X5 Z9]
+ (-1.9332121226875444e-07) [Y1 Y2 X3 X4]
+ (-1.9332121226875444e-07) [X1 X2 Y3 Y4]
+ (-1.6728571456220046e-07) [Y0 Z1 Z3 Y4]
+ (-1.6728571456220046e-07) [X0 Z1 Z3 X4]
+ (-1.6728571456220046e-07) [Y1 Z2 Z4 Y5]
+ (-1.6728571456220046e-07) [X1 Z2 Z4 X5]
+ (-3.226104922950881e-09) [Y1 Y2 X5 X6]
+ (-3.226104922950881e-09) [X1 X2 Y5 Y6]
+ (3.226104922950881e-09) [Y1 X2 X5 Y6]
+ (3.226104922950881e-09) [X1 Y2 Y5 X6]
+ (3.228404306387933e-08) [Y4 Z5 Y6 Z12]
+ (3.228404306387933e-08) [X4 Z5 X6 Z12]
+ (3.228404306387933e-08) [Y5 Z6 Y7 Z13]
+ (3.228404306387933e-08) [X5 Z6 X7 Z13]
+ (1.2919458222413466e-07) [Y1 Z2 Z3 Y5]
+ (1.2919458222413466e-07) [X1 Z2 Z3 X5]
+ (1.9332121226875444e-07) [Y1 X2 X3 Y4]
+ (1.9332121226875444e-07) [X1 Y2 Y3 X4]
+ (2.1989637275917294e-07) [Y2 X3 X5 Y6]
+ (2.1989637275917294e-07) [Y2 Y3 Y5 Y6]
+ (2.1989637275917294e-07) [X2 X3 X5 X6]
+ (2.1989637275917294e-07) [X2 Y3 Y5 X6]
+ (2.4472647913193653e-07) [Y0 X1 X5 Y6]
+ (2.4472647913193653e-07) [Y0 Y1 Y5 Y6]
+ (2.4472647913193653e-07) [X0 X1 X5 X6]
+ (2.4472647913193653e-07) [X0 Y1 Y5 X6]
+ (3.5706354530173e-07) [Y0 X1 X3 Y4]
+ (3.5706354530173e-07) [Y0 Y1 Y3 Y4]
+ (3.5706354530173e-07) [X0 X1 X3 X4]
+ (3.5706354530173e-07) [X0 Y1 Y3 X4]
+ (4.83795330133202e-07) [Y5 X6 X8 Y9]
+ (4.83795330133202e-07) [Y5 Y6 Y8 Y9]
+ (4.83795330133202e-07) [X5 X6 X8 X9]
+ (4.83795330133202e-07) [X5 Y6 Y8 X9]
+ (5.471605954255281e-07) [Y1 X2 X11 Y12]
+ (5.471605954255281e-07) [X1 Y2 Y11 X12]
+ (5.627721905005504e-07) [Y0 X1 X11 Y12]
+ (5.627721905005504e-07) [Y0 Y1 Y11 Y12]
+ (5.627721905005504e-07) [X0 X1 X11 X12]
+ (5.627721905005504e-07) [X0 Y1 Y11 X12]
+ (5.769436453423836e-07) [Y2 Z3 Y4 Z9]
+ (5.769436453423836e-07) [X2 Z3 X4 Z9]
+ (5.769436453423836e-07) [Y3 Z4 Y5 Z8]
+ (5.769436453423836e-07) [X3 Z4 X5 Z8]
+ (5.929280274693215e-07) [Z4 Y5 Z6 Y7]
+ (5.929280274693215e-07) [Z4 X5 Z6 X7]
+ (6.628427103423868e-07) [Y8 X9 X11 Y12]
+ (6.628427103423868e-07) [Y8 Y9 Y11 Y12]
+ (6.628427103423868e-07) [X8 X9 X11 X12]
+ (6.628427103423868e-07) [X8 Y9 Y11 X12]
+ (7.76508243338441e-07) [Y2 Z3 Y4 Z5]
+ (7.76508243338441e-07) [X2 Z3 X4 Z5]
+ (7.956666883215686e-07) [Y3 X4 X8 Y9]
+ (7.956666883215686e-07) [Y3 Y4 Y8 Y9]
+ (7.956666883215686e-07) [X3 X4 X8 X9]
+ (7.956666883215686e-07) [X3 Y4 Y8 X9]
+ (8.336695325256247e-07) [Z0 Y2 Z3 Y4]
+ (8.336695325256247e-07) [Z0 X2 Z3 X4]
+ (8.336695325256247e-07) [Z1 Y3 Z4 Y5]
+ (8.336695325256247e-07) [Z1 X3 Z4 X5]
+ (9.50913435596336e-07) [Z2 Y4 Z5 Y6]
+ (9.50913435596336e-07) [Z2 X4 Z5 X6]
+ (9.50913435596336e-07) [Z3 Y5 Z6 Y7]
+ (9.50913435596336e-07) [Z3 X5 Z6 X7]
+ (1.1094124245842902e-06) [Z2 Y11 Z12 Y13]
+ (1.1094124245842902e-06) [Z2 X11 Z12 X13]
+ (1.1094124245842902e-06) [Z3 Y10 Z11 Y12]
+ (1.1094124245842902e-06) [Z3 X10 Z11 X12]
+ (1.170809808356051e-06) [Z2 Y5 Z6 Y7]
+ (1.170809808356051e-06) [Z2 X5 Z6 X7]
+ (1.170809808356051e-06) [Z3 Y4 Z5 Y6]
+ (1.170809808356051e-06) [Z3 X4 Z5 X6]
+ (1.1907330778276229e-06) [Z0 Y3 Z4 Y5]
+ (1.1907330778276229e-06) [Z0 X3 Z4 X5]
+ (1.1907330778276229e-06) [Z1 Y2 Z3 Y4]
+ (1.1907330778276229e-06) [Z1 X2 Z3 X4]
+ (1.1953920586082935e-06) [Y2 Z3 Y4 Z7]
+ (1.1953920586082935e-06) [X2 Z3 X4 Z7]
+ (1.1953920586082935e-06) [Y3 Z4 Y5 Z6]
+ (1.1953920586082935e-06) [X3 Z4 X5 Z6]
+ (1.3980242739493353e-06) [Y4 Z5 Y6 Z8]
+ (1.3980242739493353e-06) [X4 Z5 X6 Z8]
+ (1.3980242739493353e-06) [Y5 Z6 Y7 Z9]
+ (1.3980242739493353e-06) [X5 Z6 X7 Z9]
+ (1.4548066749628472e-06) [Y3 X4 X6 Y7]
+ (1.4548066749628472e-06) [Y3 Y4 Y6 Y7]
+ (1.4548066749628472e-06) [X3 X4 X6 X7]
+ (1.4548066749628472e-06) [X3 Y4 Y6 X7]
+ (1.6923647998476343e-06) [Y4 Z5 Y6 Z10]
+ (1.6923647998476343e-06) [X4 Z5 X6 Z10]
+ (1.6923647998476343e-06) [Y5 Z6 Y7 Z11]
+ (1.6923647998476343e-06) [X5 Z6 X7 Z11]
+ (1.8540565154096748e-06) [Y4 Z5 Y6 Z7]
+ (1.8540565154096748e-06) [X4 Z5 X6 Z7]
+ (1.8781494247905456e-06) [Z4 Y10 Z11 Y12]
+ (1.8781494247905456e-06) [Z4 X10 Z11 X12]
+ (1.8781494247905456e-06) [Z5 Y11 Z12 Y13]
+ (1.8781494247905456e-06) [Z5 X11 Z12 X13]
+ (1.8818196040825373e-06) [Y4 Z5 Y6 Z9]
+ (1.8818196040825373e-06) [X4 Z5 X6 Z9]
+ (1.8818196040825373e-06) [Y5 Z6 Y7 Z8]
+ (1.8818196040825373e-06) [X5 Z6 X7 Z8]
+ (2.1726379726304994e-06) [Y2 X3 X11 Y12]
+ (2.1726379726304994e-06) [Y2 Y3 Y11 Y12]
+ (2.1726379726304994e-06) [X2 X3 X11 X12]
+ (2.1726379726304994e-06) [X2 Y3 Y11 X12]
+ (3.0992966114932563e-06) [Z0 Y4 Z5 Y6]
+ (3.0992966114932563e-06) [Z0 X4 Z5 X6]
+ (3.0992966114932563e-06) [Z1 Y5 Z6 Y7]
+ (3.0992966114932563e-06) [Z1 X5 Z6 X7]
+ (3.15855931302695e-06) [Y2 Z3 Y4 Z10]
+ (3.15855931302695e-06) [X2 Z3 X4 Z10]
+ (3.15855931302695e-06) [Y3 Z4 Y5 Z11]
+ (3.15855931302695e-06) [X3 Z4 X5 Z11]
+ (3.3440230906252316e-06) [Z0 Y5 Z6 Y7]
+ (3.3440230906252316e-06) [Z0 X5 Z6 X7]
+ (3.3440230906252316e-06) [Z1 Y4 Z5 Y6]
+ (3.3440230906252316e-06) [Z1 X4 Z5 X6]
+ (4.556473649697766e-06) [Y5 X6 X12 Y13]
+ (4.556473649697766e-06) [Y5 Y6 Y12 Y13]
+ (4.556473649697766e-06) [X5 X6 X12 X13]
+ (4.556473649697766e-06) [X5 Y6 Y12 X13]
+ (4.588757692765982e-06) [Y4 Z5 Y6 Z13]
+ (4.588757692765982e-06) [X4 Z5 X6 Z13]
+ (4.588757692765982e-06) [Y5 Z6 Y7 Z12]
+ (4.588757692765982e-06) [X5 Z6 X7 Z12]
+ (4.642978893765106e-06) [Y3 X4 X10 Y11]
+ (4.642978893765106e-06) [Y3 Y4 Y10 Y11]
+ (4.642978893765106e-06) [X3 X4 X10 X11]
+ (4.642978893765106e-06) [X3 Y4 Y10 X11]
+ (7.801538206789454e-06) [Y2 Z3 Y4 Z11]
+ (7.801538206789454e-06) [X2 Z3 X4 Z11]
+ (7.801538206789454e-06) [Y3 Z4 Y5 Z10]
+ (7.801538206789454e-06) [X3 Z4 X5 Z10]
+ (7.954224387346498e-06) [Y10 Z11 Y12 Z13]
+ (7.954224387346498e-06) [X10 Z11 X12 Z13]
+ (0.00029222567245005795) [Y7 Y8 X9 X10]
+ (0.00029222567245005795) [X7 X8 Y9 Y10]
+ (0.0004957972885868317) [Y2 Z4 Z5 Y6]
+ (0.0004957972885868317) [X2 Z4 Z5 X6]
+ (0.001105898480902465) [Y0 Z1 Y2 Z5]
+ (0.001105898480902465) [X0 Z1 X2 Z5]
+ (0.001105898480902465) [Y1 Z2 Y3 Z4]
+ (0.001105898480902465) [X1 Z2 X3 Z4]
+ (0.0016639606583865618) [Y2 Z3 Z4 Y6]
+ (0.0016639606583865618) [X2 Z3 Z4 X6]
+ (0.0016639606583865618) [Y3 Z5 Z6 Y7]
+ (0.0016639606583865618) [X3 Z5 Z6 X7]
+ (0.0017560659628809054) [Y0 Z1 Y2 Z11]
+ (0.0017560659628809054) [X0 Z1 X2 Z11]
+ (0.0017560659628809054) [Y1 Z2 Y3 Z10]
+ (0.0017560659628809054) [X1 Z2 X3 Z10]
+ (0.002326234847613936) [Y0 Z1 Y2 Z13]
+ (0.002326234847613936) [X0 Z1 X2 Z13]
+ (0.002326234847613936) [Y1 Z2 Y3 Z12]
+ (0.002326234847613936) [X1 Z2 X3 Z12]
+ (0.002745827315424492) [Y0 X1 X4 Y5]
+ (0.002745827315424492) [X0 Y1 Y4 X5]
+ (0.002929768278587699) [Y0 Z1 Y2 Z9]
+ (0.002929768278587699) [X0 Z1 X2 Z9]
+ (0.002929768278587699) [Y1 Z2 Y3 Z8]
+ (0.002929768278587699) [X1 Z2 X3 Z8]
+ (0.0032769650657560132) [Y0 Z1 Y2 Z3]
+ (0.0032769650657560132) [X0 Z1 X2 Z3]
+ (0.0033476264706955136) [Y0 Z1 Y2 Z7]
+ (0.0033476264706955136) [X0 Z1 X2 Z7]
+ (0.0033476264706955136) [Y1 Z2 Y3 Z6]
+ (0.0033476264706955136) [X1 Z2 X3 Z6]
+ (0.0035552589712668486) [Y0 Z1 Y2 Z10]
+ (0.0035552589712668486) [X0 Z1 X2 Z10]
+ (0.0035552589712668486) [Y1 Z2 Y3 Z11]
+ (0.0035552589712668486) [X1 Z2 X3 Z11]
+ (0.005143382387689708) [Y3 Y4 X5 X6]
+ (0.005143382387689708) [X3 X4 Y5 Y6]
+ (0.00528378575382731) [Y0 X1 X12 Y13]
+ (0.00528378575382731) [X0 Y1 Y12 X13]
+ (0.005530742149401888) [Y0 Z1 Y2 Z4]
+ (0.005530742149401888) [X0 Z1 X2 Z4]
+ (0.005530742149401888) [Y1 Z2 Y3 Z5]
+ (0.005530742149401888) [X1 Z2 X3 Z5]
+ (0.006087836804229986) [Y8 X9 X12 Y13]
+ (0.006087836804229986) [X8 Y9 Y12 X13]
+ (0.006509361234075182) [Y0 X1 X8 Y9]
+ (0.006509361234075182) [X0 Y1 Y8 X9]
+ (0.0068882045423178445) [Y0 X1 X6 Y7]
+ (0.0068882045423178445) [X0 Y1 Y6 X7]
+ (0.006901250036509368) [Y0 Z1 Y2 Z12]
+ (0.006901250036509368) [X0 Z1 X2 Z12]
+ (0.006901250036509368) [Y1 Z2 Y3 Z13]
+ (0.006901250036509368) [X1 Z2 X3 Z13]
+ (0.007156920529186221) [Y4 X5 X8 Y9]
+ (0.007156920529186221) [X4 Y5 Y8 X9]
+ (0.007731432847358059) [Y0 X1 X10 Y11]
+ (0.007731432847358059) [X0 Y1 Y10 X11]
+ (0.00803254669756846) [Y0 Z1 Y2 Z6]
+ (0.00803254669756846) [X0 Z1 X2 Z6]
+ (0.00803254669756846) [Y1 Z2 Y3 Z7]
+ (0.00803254669756846) [X1 Z2 X3 Z7]
+ (0.009560716496442287) [Y8 X9 X10 Y11]
+ (0.009560716496442287) [X8 Y9 Y10 X11]
+ (0.01105501668871038) [Y0 Z1 Y2 Z8]
+ (0.01105501668871038) [X0 Z1 X2 Z8]
+ (0.01105501668871038) [Y1 Z2 Y3 Z9]
+ (0.01105501668871038) [X1 Z2 X3 Z9]
+ (0.011285144618313647) [Y5 X6 X11 Y12]
+ (0.011285144618313647) [X5 Y6 Y11 X12]
+ (0.011982342583257208) [Y4 X5 X6 Y7]
+ (0.011982342583257208) [X4 Y5 Y6 X7]
+ (0.013873400067622195) [Y6 X7 X8 Y9]
+ (0.013873400067622195) [X6 Y7 Y8 X9]
+ (0.014583638327021652) [Y0 X1 X2 Y3]
+ (0.014583638327021652) [X0 Y1 Y2 X3]
+ (0.01557722707545975) [Y2 X3 X12 Y13]
+ (0.01557722707545975) [X2 Y3 Y12 X13]
+ (0.017366053264620398) [Y6 X7 X12 Y13]
+ (0.017366053264620398) [X6 Y7 Y12 X13]
+ (0.017680137657115326) [Y4 X5 X10 Y11]
+ (0.017680137657115326) [X4 Y5 Y10 X11]
+ (0.01782501193263139) [Y6 X7 X10 Y11]
+ (0.01782501193263139) [X6 Y7 Y10 X11]
+ (0.019028318718288723) [Y3 Y4 X11 X12]
+ (0.019028318718288723) [X3 X4 Y11 Y12]
+ (0.025384663696665626) [Y2 X3 X10 Y11]
+ (0.025384663696665626) [X2 Y3 Y10 X11]
+ (0.02868521731027118) [Y10 X11 X12 Y13]
+ (0.02868521731027118) [X10 Y11 Y12 X13]
+ (0.031143804196374976) [Y2 X3 X6 Y7]
+ (0.031143804196374976) [X2 Y3 Y6 X7]
+ (0.03583955718503415) [Y2 X3 X4 Y5]
+ (0.03583955718503415) [X2 Y3 Y4 X5]
+ (0.03619409348750257) [Y2 X3 X8 Y9]
+ (0.03619409348750257) [X2 Y3 Y8 X9]
+ (0.038314669373451654) [Y4 X5 X12 Y13]
+ (0.038314669373451654) [X4 Y5 Y12 X13]
+ (0.10433061485313576) [Z0 Y1 Z2 Y3]
+ (0.10433061485313576) [Z0 X1 Z2 X3]
+ (-0.22847946311050038) [Y6 Z7 Z8 Z9 Y10]
+ (-0.22847946311050038) [X6 Z7 Z8 Z9 X10]
+ (-0.22847946311050038) [Y7 Z8 Z9 Z10 Y11]
+ (-0.22847946311050038) [X7 Z8 Z9 Z10 X11]
+ (-0.1213324224836876) [Y3 Z4 Z5 Z6 Y7]
+ (-0.1213324224836876) [X3 Z4 Z5 Z6 X7]
+ (-0.12133242248368756) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133242248368756) [X2 Z3 Z4 Z5 X6]
+ (-3.2041422145415035e-06) [Y0 Z1 Z2 Z3 Y4]
+ (-3.2041422145415035e-06) [X0 Z1 Z2 Z3 X4]
+ (-3.2041422145415035e-06) [Y1 Z2 Z3 Z4 Y5]
+ (-3.2041422145415035e-06) [X1 Z2 Z3 Z4 X5]
+ (-0.05608449432298653) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (-0.05608449432298653) [Z0 X6 Z7 Z8 Z9 X10]
+ (-0.05608449432298653) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (-0.05608449432298653) [Z1 X7 Z8 Z9 Z10 X11]
+ (-0.05600713561692995) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (-0.05600713561692995) [Z0 X7 Z8 Z9 Z10 X11]
+ (-0.05600713561692995) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (-0.05600713561692995) [Z1 X6 Z7 Z8 Z9 X10]
+ (-0.03276748589567796) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.03276748589567796) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.03276748589567796) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.03276748589567796) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.030787440718598236) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (-0.030787440718598236) [Z6 X7 Z8 Z9 Z10 X11]
+ (-0.02711487858045474) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.02711487858045474) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.02711487858045474) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.02711487858045474) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.025996206267197022) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.025996206267197022) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.02510490797004607) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (-0.02510490797004607) [X6 Z7 Z8 Z9 X10 Z12]
+ (-0.02510490797004607) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (-0.02510490797004607) [X7 Z8 Z9 Z10 X11 Z13]
+ (-0.024388989986592072) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (-0.024388989986592072) [Z2 X7 Z8 Z9 Z10 X11]
+ (-0.024388989986592072) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (-0.024388989986592072) [Z3 X6 Z7 Z8 Z9 X10]
+ (-0.02435313608449561) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (-0.02435313608449561) [Y2 Z3 Y4 X11 Z12 X13]
+ (-0.02435313608449561) [X2 Z3 X4 Y11 Z12 Y13]
+ (-0.02435313608449561) [X2 Z3 X4 X11 Z12 X13]
+ (-0.02435313608449561) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (-0.02435313608449561) [Y3 Z4 Y5 X10 Z11 X12]
+ (-0.02435313608449561) [X3 Z4 X5 Y10 Z11 Y12]
+ (-0.02435313608449561) [X3 Z4 X5 X10 Z11 X12]
+ (-0.020175824956992678) [Y4 Z5 Z6 X7 X11 Y12]
+ (-0.020175824956992678) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (-0.020175824956992678) [X4 Z5 Z6 X7 X11 X12]
+ (-0.020175824956992678) [X4 Z5 Z6 Y7 Y11 X12]
+ (-0.020175824956992678) [Y5 X6 X10 Z11 Z12 Y13]
+ (-0.020175824956992678) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (-0.020175824956992678) [X5 X6 X10 Z11 Z12 X13]
+ (-0.020175824956992678) [X5 Y6 Y10 Z11 Z12 X13]
+ (-0.019020373875159764) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (-0.019020373875159764) [Z2 X6 Z7 Z8 Z9 X10]
+ (-0.019020373875159764) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (-0.019020373875159764) [Z3 X7 Z8 Z9 Z10 X11]
+ (-0.018266758578547277) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (-0.018266758578547277) [Z4 X6 Z7 Z8 Z9 X10]
+ (-0.018266758578547277) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (-0.018266758578547277) [Z5 X7 Z8 Z9 Z10 X11]
+ (-0.017561116452745242) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.017561116452745242) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.017561116452745242) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.017561116452745242) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.015588277865110519) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (-0.015588277865110519) [X2 Z3 X4 X10 Z11 X12]
+ (-0.015588277865110519) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (-0.015588277865110519) [X3 Z4 X5 X11 Z12 X13]
+ (-0.011755995240384579) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.011755995240384579) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.011755995240384579) [X3 Z4 Z5 X6 X8 X9]
+ (-0.011755995240384579) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.011307208030035727) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (-0.011307208030035727) [X6 Z7 Z8 Z9 X10 Z11]
+ (-0.010959994608948455) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (-0.010959994608948455) [Z4 X7 Z8 Z9 Z10 X11]
+ (-0.010959994608948455) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (-0.010959994608948455) [Z5 X6 Z7 Z8 Z9 X10]
+ (-0.010540434329241349) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (-0.010540434329241349) [X6 Z7 Z8 Z9 X10 Z13]
+ (-0.010540434329241349) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (-0.010540434329241349) [X7 Z8 Z9 Z10 X11 Z12]
+ (-0.01026346049890363) [Y2 Z3 X4 X10 Z11 Y12]
+ (-0.01026346049890363) [X2 Z3 Y4 Y10 Z11 X12]
+ (-0.01026346049890363) [Y3 Z4 X5 X11 Z12 Y13]
+ (-0.01026346049890363) [X3 Z4 Y5 Y11 Z12 X13]
+ (-0.00889068033867903) [Y4 Z5 X6 X10 Z11 Y12]
+ (-0.00889068033867903) [X4 Z5 Y6 Y10 Z11 X12]
+ (-0.00889068033867903) [Y5 Z6 X7 X11 Z12 Y13]
+ (-0.00889068033867903) [X5 Z6 Y7 Y11 Z12 X13]
+ (-0.00812524841012268) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.00812524841012268) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.00796083963423634) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (-0.00796083963423634) [X4 Z5 X6 X10 Z11 X12]
+ (-0.00796083963423634) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (-0.00796083963423634) [X5 Z6 X7 X11 Z12 X13]
+ (-0.0058051212123606645) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.0058051212123606645) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.0058051212123606645) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.0058051212123606645) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.005652607315223221) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.005652607315223221) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.005652607315223221) [X0 X1 X3 Z4 Z5 X6]
+ (-0.005652607315223221) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.005368616111432313) [Y2 X3 X7 Z8 Z9 Y10]
+ (-0.005368616111432313) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (-0.005368616111432313) [X2 X3 X7 Z8 Z9 X10]
+ (-0.005368616111432313) [X2 Y3 Y7 Z8 Z9 X10]
+ (-0.005324817366206887) [Y2 Z3 Y4 X10 Z11 X12]
+ (-0.005324817366206887) [X2 Z3 X4 Y10 Z11 Y12]
+ (-0.005324817366206887) [Y3 Z4 Y5 X11 Z12 X13]
+ (-0.005324817366206887) [X3 Z4 X5 Y11 Z12 Y13]
+ (-0.005143382387689707) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.005143382387689707) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.005143382387689707) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.005143382387689707) [X2 Z3 X4 X5 Z6 X7]
+ (-0.004684920226872946) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.004684920226872946) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668615266021896) [Y1 Y2 X7 Z8 Z9 X10]
+ (-0.004668615266021896) [X1 X2 Y7 Z8 Z9 Y10]
+ (-0.004575015188895431) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.004575015188895431) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424843668499424) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424843668499424) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.004158830716415749) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.004158830716415749) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.004158830716415749) [X3 Z4 Z5 X6 X12 X13]
+ (-0.004158830716415749) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.0034938003715194203) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.0034938003715194203) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.0034938003715194203) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.0034938003715194203) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.0027790407628801314) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.0027790407628801314) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.0022939556230661133) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.0022939556230661133) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.0017991930083859425) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.0017991930083859425) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278745823321438) [Y1 Z2 Z3 Y4 X11 X12]
+ (-0.0017278745823321438) [X1 Z2 Z3 X4 Y11 Y12]
+ (-0.0008533831051004429) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533831051004429) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008144692870617194) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008144692870617194) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008144692870617194) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008144692870617194) [X3 Z4 Z5 Z6 X7 Z11]
+ (-0.00029222567245005795) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (-0.00029222567245005795) [Y6 Z7 Y8 X9 Z10 X11]
+ (-0.00029222567245005795) [X6 Z7 X8 Y9 Z10 Y11]
+ (-0.00029222567245005795) [X6 Z7 X8 X9 Z10 X11]
+ (-8.774724085487337e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.774724085487337e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.774724085487337e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.774724085487337e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.518288657857794e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.518288657857794e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.518288657857794e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.518288657857794e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.444267252861665e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.444267252861665e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.444267252861665e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.444267252861665e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.290019543956889e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.290019543956889e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.290019543956889e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.290019543956889e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.974176776677234e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (-5.974176776677234e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (-5.275783339901089e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (-5.275783339901089e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (-4.642978893765106e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (-4.642978893765106e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (-4.556473649697766e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (-4.556473649697766e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (-4.253118613963339e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.253118613963339e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-4.183808698706576e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (-4.183808698706576e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (-3.6945168651967425e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.6945168651967425e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.3130170220248625e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (-3.3130170220248625e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (-3.151295909154233e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.151295909154233e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.117366364509771e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (-3.117366364509771e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (-3.0882456497679373e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.0882456497679373e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.1726379726304994e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.1726379726304994e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.4548066749628472e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (-1.4548066749628472e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (-1.3304568326256722e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.3304568326256722e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2393113626030244e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (-1.2393113626030244e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (-1.2393113626030244e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (-1.2393113626030244e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (-1.2282691138991702e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2282691138991702e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.0357924734557059e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.0357924734557059e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-9.306342775622551e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (-9.306342775622551e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (-9.306342775622551e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (-9.306342775622551e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (-7.956666883215686e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (-7.956666883215686e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (-6.628427103423868e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.628427103423868e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.57925885973245e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.57925885973245e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.57925885973245e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.57925885973245e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.395302273956491e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.395302273956491e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.395302273956491e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.395302273956491e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.627721905005504e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627721905005504e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287649368477358e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287649368477358e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287649368477358e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287649368477358e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287649368477358e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287649368477358e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287649368477358e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287649368477358e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.83795330133202e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (-4.83795330133202e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (-3.5706354530173007e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (-3.5706354530173007e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (-3.328039610517807e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (-3.328039610517807e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (-3.2361833909523604e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (-3.2361833909523604e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (-3.2361833909523604e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (-3.2361833909523604e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (-2.4472647913193653e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (-2.4472647913193653e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (-2.1989637275917294e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (-2.1989637275917294e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (-1.8290428356989896e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (-1.8290428356989896e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (-1.8290428356989896e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (-1.8290428356989896e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (-1.1076529054800146e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.1076529054800146e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.1076529054800146e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.1076529054800146e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.1076529054800146e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.1076529054800146e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.1076529054800146e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.1076529054800146e-07) [X1 Z2 X3 X10 Z11 X12]
+ (-8.649129520281549e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (-8.649129520281549e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (-8.649129520281549e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (-8.649129520281549e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (-8.057465113507991e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (-8.057465113507991e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (-8.057465113507991e-08) [X1 Z2 Z3 X4 X10 X11]
+ (-8.057465113507991e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (1.0351503222726343e-09) [Y0 Z1 Z2 Z3 Y4 Z7]
+ (1.0351503222726343e-09) [X0 Z1 Z2 Z3 X4 Z7]
+ (1.0351503222726343e-09) [Y1 Z2 Z3 Z4 Y5 Z6]
+ (1.0351503222726343e-09) [X1 Z2 Z3 Z4 X5 Z6]
+ (1.8395658576552277e-08) [Y0 Z1 X2 X10 Z11 Y12]
+ (1.8395658576552277e-08) [X0 Z1 Y2 Y10 Z11 X12]
+ (1.8395658576552277e-08) [Y1 Z2 X3 X11 Z12 Y13]
+ (1.8395658576552277e-08) [X1 Z2 Y3 Y11 Z12 X13]
+ (2.2702083810474844e-08) [Y0 Z1 Z2 X3 X5 Y6]
+ (2.2702083810474844e-08) [Y0 Z1 Z2 Y3 Y5 Y6]
+ (2.2702083810474844e-08) [X0 Z1 Z2 X3 X5 X6]
+ (2.2702083810474844e-08) [X0 Z1 Z2 Y3 Y5 X6]
+ (2.2702083810474844e-08) [Y1 X2 X4 Z5 Z6 Y7]
+ (2.2702083810474844e-08) [Y1 Y2 Y4 Z5 Z6 Y7]
+ (2.2702083810474844e-08) [X1 X2 X4 Z5 Z6 X7]
+ (2.2702083810474844e-08) [X1 Y2 Y4 Z5 Z6 X7]
+ (2.5928188733517204e-08) [Y0 Z1 X2 X4 Z5 Y6]
+ (2.5928188733517204e-08) [X0 Z1 Y2 Y4 Z5 X6]
+ (2.5928188733517204e-08) [Y1 Z2 X3 X5 Z6 Y7]
+ (2.5928188733517204e-08) [X1 Z2 Y3 Y5 Z6 X7]
+ (1.2919458222413466e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (1.2919458222413466e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (1.3484968815284265e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (1.3484968815284265e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (1.3484968815284265e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (1.3484968815284265e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (1.3807579307583588e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (1.3807579307583588e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (1.3807579307583588e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (1.3807579307583588e-07) [X0 Z1 X2 X5 Z6 X7]
+ (1.3807579307583588e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (1.3807579307583588e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (1.3807579307583588e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (1.3807579307583588e-07) [X1 Z2 X3 X4 Z5 X6]
+ (1.6077787688630734e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (1.6077787688630734e-07) [X0 Z1 X2 X4 Z5 X6]
+ (1.6077787688630734e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (1.6077787688630734e-07) [X1 Z2 X3 X5 Z6 X7]
+ (1.8393943389216143e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (1.8393943389216143e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (1.8393943389216143e-07) [X1 Z2 Z3 X4 X6 X7]
+ (1.8393943389216143e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (1.9332121226875444e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (1.9332121226875444e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (1.9332121226875444e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (1.9332121226875444e-07) [X0 Z1 X2 X3 Z4 X5]
+ (2.1989637275917294e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (2.1989637275917294e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (2.3712704389242055e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (2.3712704389242055e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (2.3712704389242055e-07) [X1 Z2 Z3 X4 X8 X9]
+ (2.3712704389242055e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (2.4472647913193653e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (2.4472647913193653e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (3.086770850409861e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (3.086770850409861e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (3.086770850409861e-07) [X1 Z2 Z3 X4 X12 X13]
+ (3.086770850409861e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (3.328039610517807e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (3.328039610517807e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (3.5706354530173007e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (3.5706354530173007e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (4.83795330133202e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (4.83795330133202e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (5.627721905005504e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627721905005504e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (5.927350093986651e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (5.927350093986651e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (5.927350093986651e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (5.927350093986651e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (6.628427103423868e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.628427103423868e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (6.733096605341651e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (6.733096605341651e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (6.733096605341651e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (6.733096605341651e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (7.956666883215686e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (7.956666883215686e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (1.0357924734557059e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.0357924734557059e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2282691138991702e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2282691138991702e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.3304568326256722e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.3304568326256722e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.4548066749628472e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (1.4548066749628472e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (2.1726379726304994e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.1726379726304994e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.0882456497679373e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.0882456497679373e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.151295909154233e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.151295909154233e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.2111873859499984e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (3.2111873859499984e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (3.2111873859499984e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (3.2111873859499984e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (3.2774382175649805e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (3.2774382175649805e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (3.2774382175649805e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (3.2774382175649805e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (3.3130170220248625e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (3.3130170220248625e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (3.3342618386174305e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.3342618386174305e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.610242178616761e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (3.610242178616761e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (3.610242178616761e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (3.610242178616761e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (3.6945168651967425e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.6945168651967425e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (3.769583539967253e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (3.769583539967253e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (4.556473649697766e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (4.556473649697766e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (4.642978893765106e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (4.642978893765106e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (5.275783339901089e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (5.275783339901089e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (5.974176776677234e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (5.974176776677234e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (6.5242044079809325e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (6.5242044079809325e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (6.5242044079809325e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (6.5242044079809325e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (7.735870605659221e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (7.735870605659221e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (7.735870605659221e-05) [X0 X1 X7 Z8 Z9 X10]
+ (7.735870605659221e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (0.0004957972885868317) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004957972885868317) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650303448963279) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650303448963279) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650303448963279) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650303448963279) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533831051004429) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533831051004429) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.0009298407044426821) [Y4 Z5 Y6 X10 Z11 X12]
+ (0.0009298407044426821) [X4 Z5 X6 Y10 Z11 Y12]
+ (0.0009298407044426821) [Y5 Z6 Y7 X11 Z12 X13]
+ (0.0009298407044426821) [X5 Z6 X7 Y11 Z12 Y13]
+ (0.0016095335162985376) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.0016095335162985376) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.0016095335162985376) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.0016095335162985376) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0016676137499723903) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.0016676137499723903) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.0016676137499723903) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.0016676137499723903) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.0017278745823321438) [Y1 Z2 Z3 X4 X11 Y12]
+ (0.0017278745823321438) [X1 Z2 Z3 Y4 Y11 X12]
+ (0.0017991930083859425) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.0017991930083859425) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.0022939556230661133) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.0022939556230661133) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.0024629166213989808) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.0024629166213989808) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.0024629166213989808) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.0024629166213989808) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.0039615693730385045) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.0039615693730385045) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.0039615693730385045) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.0039615693730385045) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.004424843668499424) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424843668499424) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004575015188895431) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.004575015188895431) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668615266021896) [Y1 X2 X7 Z8 Z9 Y10]
+ (0.004668615266021896) [X1 Y2 Y7 Z8 Z9 X10]
+ (0.004684920226872946) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.004684920226872946) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.0073067639695988265) [Y4 X5 X7 Z8 Z9 Y10]
+ (0.0073067639695988265) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (0.0073067639695988265) [X4 X5 X7 Z8 Z9 X10]
+ (0.0073067639695988265) [X4 Y5 Y7 Z8 Z9 X10]
+ (0.00812524841012268) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.00812524841012268) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.008764858219385094) [Y2 Z3 Z4 X5 X11 Y12]
+ (0.008764858219385094) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (0.008764858219385094) [X2 Z3 Z4 X5 X11 X12]
+ (0.008764858219385094) [X2 Z3 Z4 Y5 Y11 X12]
+ (0.008764858219385094) [Y3 X4 X10 Z11 Z12 Y13]
+ (0.008764858219385094) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (0.008764858219385094) [X3 X4 X10 Z11 Z12 X13]
+ (0.008764858219385094) [X3 Y4 Y10 Z11 Z12 X13]
+ (0.012214985322756337) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (0.012214985322756337) [Y4 Z5 Y6 X11 Z12 X13]
+ (0.012214985322756337) [X4 Z5 X6 Y11 Z12 Y13]
+ (0.012214985322756337) [X4 Z5 X6 X11 Z12 X13]
+ (0.012214985322756337) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (0.012214985322756337) [Y5 Z6 Y7 X10 Z11 X12]
+ (0.012214985322756337) [X5 Z6 X7 Y10 Z11 Y12]
+ (0.012214985322756337) [X5 Z6 X7 X10 Z11 X12]
+ (0.0144111897700634) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.0144111897700634) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.0144111897700634) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.0144111897700634) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.014564473640804721) [Y7 Z8 Z9 X10 X12 Y13]
+ (0.014564473640804721) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (0.014564473640804721) [X7 Z8 Z9 X10 X12 X13]
+ (0.014564473640804721) [X7 Z8 Z9 Y10 Y12 X13]
+ (0.015225659057125116) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.015225659057125116) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.015225659057125116) [X3 Z4 Z5 X6 X10 X11]
+ (0.015225659057125116) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.04587942403067647) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.04587942403067647) [X0 Z2 Z3 Z4 Z5 X6]
+ (-6.631261805345916e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631261805345916e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.631261805345916e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631261805345916e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.59508130350394e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.59508130350394e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.59508130350394e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.59508130350394e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.04274326006291184) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.04274326006291184) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (0.042743260062911875) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.042743260062911875) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.047642613600139674) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.047642613600139674) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.047642613600139674) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.047642613600139674) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.04171881404453212) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.04171881404453212) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.04171881404453212) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.04171881404453212) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.03956454804163188) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.03956454804163188) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.03956454804163188) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03956454804163188) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0393181072310521) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0393181072310521) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0393181072310521) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0393181072310521) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.025637212809967107) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.025637212809967107) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.025637212809967107) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.025637212809967107) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.023145221653271045) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.023145221653271045) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.02252835424087271) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.02252835424087271) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.019257453001866466) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.019257453001866466) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019028318718288723) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.019028318718288723) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.016024666095520133) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.016024666095520133) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.01522565905712512) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.01522565905712512) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.01460374241072829) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (-0.01460374241072829) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (-0.014564473640804721) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.014564473640804721) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.011755995240384577) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.011755995240384577) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.011285144618313647) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-0.011285144618313647) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-0.009841802923811014) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.009841802923811014) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009612546714446979) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612546714446979) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612546714446979) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612546714446979) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.00846983334136155) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.00846983334136155) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.007306763969598825) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.007306763969598825) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.00592379955560755) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.00592379955560755) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.005708479853867187) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (-0.005708479853867187) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (-0.005708479853867187) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (-0.005708479853867187) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (-0.005652607315223221) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.005652607315223221) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368616111432313) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.005368616111432313) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005262631033415901) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (-0.005262631033415901) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (-0.005262631033415901) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (-0.005262631033415901) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (-0.005114464086473182) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (-0.005114464086473182) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (-0.005114464086473182) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (-0.005114464086473182) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (-0.005114464086473182) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005114464086473182) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005114464086473182) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.005114464086473182) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.004158830716415749) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.004158830716415749) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.003989845257551139) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (-0.003989845257551139) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (-0.003989845257551139) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (-0.003989845257551139) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (-0.0033566679213672505) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.0033566679213672505) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.0033566679213672505) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.0033566679213672505) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.003267514897153897) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.003267514897153897) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.003267514897153897) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.003267514897153897) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.0027790407628801314) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.0027790407628801314) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.0022939556230661133) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939556230661133) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.0022939556230661133) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939556230661133) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.002261970675218995) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (-0.002261970675218995) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (-0.002261970675218995) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (-0.002261970675218995) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (-0.002261970675218995) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (-0.002261970675218995) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (-0.002261970675218995) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (-0.002261970675218995) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (-0.0013038029824605723) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (-0.0013038029824605723) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (-0.0013038029824605723) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (-0.0013038029824605723) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (-0.0005940157673940051) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (-0.0005940157673940051) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (-0.0005940157673940051) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (-0.0005940157673940051) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (-0.0005940157673940051) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (-0.0005940157673940051) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (-0.0005940157673940051) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (-0.0005940157673940051) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (-0.00044584882045128557) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (-0.00044584882045128557) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (-0.00044584882045128557) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (-0.00044584882045128557) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (-0.00024644081057978765) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00024644081057978765) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013838603701095707) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.00013838603701095707) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.00013838603701095707) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.00013838603701095707) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.735870605659221e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (-7.735870605659221e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.610337466116128e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.610337466116128e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.610337466116128e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.610337466116128e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.531661381373781e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.531661381373781e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.531661381373781e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.531661381373781e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.80598179923381e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.80598179923381e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.80598179923381e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.80598179923381e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.089728490634352e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.089728490634352e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.089728490634352e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.089728490634352e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-5.071403536886787e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.071403536886787e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.071403536886787e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.071403536886787e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.7345782623452876e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.7345782623452876e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.7345782623452876e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.7345782623452876e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.728781323614219e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.728781323614219e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.728781323614219e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.728781323614219e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.253118613963339e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.253118613963339e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.183808698706576e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-4.183808698706576e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-3.5443573459476106e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.5443573459476106e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.5443573459476106e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.5443573459476106e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.5443573459476106e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.5443573459476106e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.5443573459476106e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.5443573459476106e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.3130170220248625e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-3.3130170220248625e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-3.3130170220248625e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-3.3130170220248625e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-2.3609471670163387e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609471670163387e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609471670163387e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.3609471670163387e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-1.5224581710915645e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224581710915645e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224581710915645e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224581710915645e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224581710915645e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224581710915645e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224581710915645e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224581710915645e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.2282691138991702e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2282691138991702e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.2282691138991702e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2282691138991702e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.988467768477387e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988467768477387e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988467768477387e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988467768477387e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.867608474234771e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867608474234771e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867608474234771e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.867608474234771e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.189870302940458e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.189870302940458e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (-6.175164773312576e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175164773312576e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.471605954255281e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.471605954255281e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.5611169618151644e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.5611169618151644e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.5611169618151644e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.5611169618151644e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.5233392649655647e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.5233392649655647e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.42735080667198e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.42735080667198e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.42735080667198e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.42735080667198e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.086770850409861e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (-3.086770850409861e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (-2.3712704389242055e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (-2.3712704389242055e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (-1.8393943389216143e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (-1.8393943389216143e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (-1.703543415608743e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.703543415608743e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.703543415608743e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (-1.703543415608743e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (-9.208946699391105e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-9.208946699391105e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-9.208946699391105e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-9.208946699391105e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-8.057465113507991e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (-8.057465113507991e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (-6.772951765071658e-08) [Y1 Z2 Z3 X4 X7 Z8 Z9 Y10]
+ (-6.772951765071658e-08) [X1 Z2 Z3 Y4 Y7 Z8 Z9 X10]
+ (-3.226104922950881e-09) [Y0 Z1 Z2 Y3 X4 Z5 Z6 X7]
+ (-3.226104922950881e-09) [X0 Z1 Z2 X3 Y4 Z5 Z6 Y7]
+ (3.226104922950881e-09) [Y0 Z1 Z2 X3 X4 Z5 Z6 Y7]
+ (3.226104922950881e-09) [X0 Z1 Z2 Y3 Y4 Z5 Z6 X7]
+ (6.046790582158769e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (6.046790582158769e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (6.046790582158769e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (6.046790582158769e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (6.772951765071658e-08) [Y1 Z2 Z3 Y4 X7 Z8 Z9 X10]
+ (6.772951765071658e-08) [X1 Z2 Z3 X4 Y7 Z8 Z9 Y10]
+ (8.057465113507991e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (8.057465113507991e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (1.8393943389216143e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (1.8393943389216143e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (2.3712704389242055e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (2.3712704389242055e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (2.888565083175934e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.888565083175934e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.888565083175934e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (2.888565083175934e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (3.086770850409861e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (3.086770850409861e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (3.328039610517807e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (3.328039610517807e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (3.328039610517807e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (3.328039610517807e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (4.5233392649655647e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.5233392649655647e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.471605954255281e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.471605954255281e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175164773312576e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175164773312576e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189870302940458e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (7.189870302940458e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.3304568326256722e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.3304568326256722e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.3304568326256722e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.3304568326256722e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288377380601748e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288377380601748e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288377380601748e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288377380601748e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.6540900372478615e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6540900372478615e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6540900372478615e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6540900372478615e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.6893056438800277e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.6893056438800277e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.6893056438800277e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.6893056438800277e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (1.942946545564154e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.942946545564154e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.942946545564154e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.942946545564154e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.0110739761190722e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.0110739761190722e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.0110739761190722e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.0110739761190722e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.1031634431129833e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.1031634431129833e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.1031634431129833e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.1031634431129833e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.745510569100956e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.745510569100956e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.745510569100956e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.745510569100956e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.745510569100956e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.745510569100956e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.745510569100956e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.745510569100956e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.211763814974086e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.211763814974086e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.211763814974086e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.211763814974086e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.211763814974086e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.211763814974086e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.211763814974086e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.211763814974086e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.3342618386174305e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.3342618386174305e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (3.769583539967253e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.769583539967253e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.481751878229467e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (6.481751878229467e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.481751878229467e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (6.481751878229467e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (6.652106219790413e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (6.652106219790413e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (6.652106219790413e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (6.652106219790413e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.735870605659221e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (7.735870605659221e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.00024644081057978765) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00024644081057978765) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0008533831051004429) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533831051004429) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533831051004429) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533831051004429) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0009581676927584213) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (0.0009581676927584213) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (0.0009581676927584213) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (0.0009581676927584213) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (0.0009581676927584213) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (0.0009581676927584213) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (0.0009581676927584213) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (0.0009581676927584213) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (0.0010435237104125033) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435237104125033) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435237104125033) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435237104125033) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.0012803055951291884) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.0012803055951291884) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.0012803055951291884) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.0012803055951291884) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.002686042275090566) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (0.002686042275090566) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (0.002686042275090566) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (0.002686042275090566) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (0.004158830716415749) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.004158830716415749) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.0043110386075664005) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.0043110386075664005) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.0043110386075664005) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.0043110386075664005) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.004636973516496439) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.004636973516496439) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.004636973516496439) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.004636973516496439) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.005241543597048339) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.005241543597048339) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.005241543597048339) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.005241543597048339) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.005368616111432313) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005368616111432313) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005379929634059296) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.005379929634059296) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.005379929634059296) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.005379929634059296) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.005652607315223221) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.005652607315223221) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.00592379955560755) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.00592379955560755) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.007306763969598825) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (0.007306763969598825) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.00846983334136155) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.00846983334136155) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.009841802923811014) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.009841802923811014) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.011285144618313647) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (0.011285144618313647) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (0.011755995240384577) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.011755995240384577) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.014564473640804721) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.014564473640804721) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.01460374241072829) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (0.01460374241072829) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (0.01522565905712512) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.01522565905712512) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.016024666095520133) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.016024666095520133) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.018888995077448012) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (0.018888995077448012) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (0.018888995077448012) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.018888995077448012) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.019028318718288723) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.019028318718288723) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.019257453001866466) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.019257453001866466) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.021433980116931893) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.021433980116931893) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.021433980116931893) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.021433980116931893) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.024282031623400126) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (0.024282031623400126) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (0.024755507980792846) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.024755507980792846) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.024755507980792846) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.024755507980792846) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.028730798001259026) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.028730798001259026) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.028730798001259026) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.028730798001259026) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.029903813458293446) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.029903813458293446) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.029903813458293446) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.029903813458293446) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.03560840035232359) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.03560840035232359) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.03935925039152113) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.03935925039152113) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.03935925039152113) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.03935925039152113) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.04587942403067647) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04587942403067647) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.36937137554456756) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.36937137554456756) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.3693713755445675) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.3693713755445675) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.28164335753427716) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.28164335753427716) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.28164335753427716) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.28164335753427716) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.09065142344955007) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.09065142344955007) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.09065142344955007) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.09065142344955007) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684736029518075) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684736029518075) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684736029518075) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684736029518075) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.05859215179547844) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.05859215179547844) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.03490330427072441) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03490330427072441) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03490330427072441) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03490330427072441) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591832088294522) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.024591832088294522) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591832088294522) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.024591832088294522) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.023145221653271045) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.023145221653271045) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.02252835424087271) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.02252835424087271) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.01602466609552013) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.01602466609552013) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.01602466609552013) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.01602466609552013) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.01460374241072829) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (-0.01460374241072829) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (-0.01460374241072829) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (-0.01460374241072829) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (-0.010311472182429891) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311472182429891) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311472182429891) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311472182429891) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541975656796208) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541975656796208) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541975656796208) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541975656796208) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541975656796208) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541975656796208) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541975656796208) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008541975656796208) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0054089707583866425) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.0054089707583866425) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.0054089707583866425) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.0054089707583866425) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.005286569056783093) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.005286569056783093) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.005286569056783093) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.005286569056783093) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.004767276644730846) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.004767276644730846) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.004767276644730846) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.004767276644730846) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.004668615266021896) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.004668615266021896) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0038764821957233313) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (-0.0038764821957233313) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (-0.00380406315436931) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00380406315436931) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00380406315436931) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.00380406315436931) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484154579346787) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484154579346787) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0033566679213672505) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.0033566679213672505) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.003267514897153897) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.003267514897153897) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.002446463422674714) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.002446463422674714) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.002446463422674714) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.002446463422674714) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.0017278745823321438) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.0017278745823321438) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.001640759116706777) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.001640759116706777) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0015324885626633118) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.0015324885626633118) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.0015324885626633118) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.0015324885626633118) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.000787089370558169) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.000787089370558169) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007156737069675678) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0007156737069675678) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007156737069675678) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0007156737069675678) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0005192924120522454) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.0005192924120522454) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.00019401030606567594) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00019401030606567594) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00018787486138339363) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.00018787486138339363) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.00018787486138339363) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.00018787486138339363) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.00013838603701095707) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.00013838603701095707) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.97971083900742e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-7.97971083900742e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-7.97971083900742e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-7.97971083900742e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-7.253185032655708e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-7.253185032655708e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.935743901801585e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-5.935743901801585e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.935743901801585e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-5.935743901801585e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-5.427926539569583e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-5.427926539569583e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.427926539569583e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-5.427926539569583e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-5.15929439406362e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-5.15929439406362e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-5.15929439406362e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-5.15929439406362e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-5.146386671556896e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-5.146386671556896e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-5.146386671556896e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-5.146386671556896e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-5.105462334344541e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-5.105462334344541e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-5.105462334344541e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-5.105462334344541e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-5.071403536886787e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.071403536886787e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.846190819526037e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-3.846190819526037e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-3.846190819526037e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-3.846190819526037e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-3.151295909154233e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.151295909154233e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0882456497679373e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.0882456497679373e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.988412497087695e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.988412497087695e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.947331404712806e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.947331404712806e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.947331404712806e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.947331404712806e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.883653134161737e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-2.883653134161737e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-2.874248504662879e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-2.874248504662879e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-2.3609471670163387e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.3609471670163387e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.3001958520312933e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (-1.3001958520312933e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (-1.1468225986991526e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.1468225986991526e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.1468225986991526e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.1468225986991526e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091539677291558e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091539677291558e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091539677291558e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091539677291558e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091539677291558e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091539677291558e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091539677291558e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091539677291558e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900025574661975e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900025574661975e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900025574661975e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900025574661975e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867608474234771e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.867608474234771e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.99695169729991e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.99695169729991e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.99695169729991e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.99695169729991e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.99695169729991e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.99695169729991e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.99695169729991e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.99695169729991e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.5682004123233e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.5682004123233e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.5682004123233e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.5682004123233e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3766863096990533e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3766863096990533e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3766863096990533e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3766863096990533e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3766863096990533e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3766863096990533e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.3766863096990533e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3766863096990533e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.888565083175934e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.888565083175934e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.686321455063964e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.686321455063964e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (-1.703543415608743e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.703543415608743e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.208946699391105e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.208946699391105e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.204004058720501e-08) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (-3.204004058720501e-08) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (-3.204004058720501e-08) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (-3.204004058720501e-08) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (3.56894770635319e-08) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (3.56894770635319e-08) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (3.56894770635319e-08) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (3.56894770635319e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (3.56894770635319e-08) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.56894770635319e-08) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (3.56894770635319e-08) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (3.56894770635319e-08) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.7068346771340583e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.7068346771340583e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.7068346771340583e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.7068346771340583e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.379737345418462e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (8.379737345418462e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.379737345418462e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (8.379737345418462e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.379737345418462e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (8.379737345418462e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (8.379737345418462e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (8.379737345418462e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (9.208946699391105e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.208946699391105e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.0716844862376386e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0716844862376386e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0716844862376386e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0716844862376386e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.1782130760070079e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.1782130760070079e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.1782130760070079e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.1782130760070079e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.703543415608743e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.703543415608743e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.249897562244477e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.249897562244477e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.249897562244477e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.249897562244477e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.686321455063964e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (2.686321455063964e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.888565083175934e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.888565083175934e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.092161853021214e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.092161853021214e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.092161853021214e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.092161853021214e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.092161853021214e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.092161853021214e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.092161853021214e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.092161853021214e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.449056623655991e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.449056623655991e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.449056623655991e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.449056623655991e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.769457029526076e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.769457029526076e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.769457029526076e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.769457029526076e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246849259548656e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246849259548656e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246849259548656e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246849259548656e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246849259548656e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246849259548656e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246849259548656e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246849259548656e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.560553770656235e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.560553770656235e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (7.560553770656235e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (7.560553770656235e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (7.560553770656235e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.560553770656235e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.560553770656235e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.560553770656235e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.867608474234771e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.867608474234771e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (8.027844037484269e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (8.027844037484269e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (8.027844037484269e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.027844037484269e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (8.39852750519366e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (8.39852750519366e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (8.39852750519366e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (8.39852750519366e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.3001958520312933e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (1.3001958520312933e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (2.3609471670163387e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.3609471670163387e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.874248504662879e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (2.874248504662879e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (2.988412497087695e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.988412497087695e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.0882456497679373e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.0882456497679373e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.151295909154233e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.151295909154233e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (5.071403536886787e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.071403536886787e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (4.204685614132798e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.204685614132798e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.204685614132798e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.204685614132798e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.141566359060194e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (7.141566359060194e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (7.141566359060194e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.141566359060194e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00013838603701095707) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.00013838603701095707) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.00019401030606567594) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00019401030606567594) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00024644081057978765) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024644081057978765) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00024644081057978765) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024644081057978765) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192924120522454) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.0005192924120522454) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.000787089370558169) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.000787089370558169) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.0014528842553233814) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0014528842553233814) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0014528842553233814) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0014528842553233814) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.001640759116706777) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.001640759116706777) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0017278745823321438) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.0017278745823321438) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.0021413489647389167) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (0.0021413489647389167) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (0.003267514897153897) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.003267514897153897) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.0033566679213672505) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.0033566679213672505) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.003484154579346787) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.003484154579346787) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0038764821957233313) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (0.0038764821957233313) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (0.004668615266021896) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.004668615266021896) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00592379955560755) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.00592379955560755) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.00592379955560755) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.00592379955560755) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.00846983334136155) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.00846983334136155) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.00846983334136155) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.00846983334136155) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.00882638756779696) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.00882638756779696) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.00882638756779696) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.00882638756779696) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.009841802923811016) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.009841802923811016) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.009841802923811016) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.009841802923811016) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.010715477345070262) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010715477345070262) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010715477345070262) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010715477345070262) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01075752420121159) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01075752420121159) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.01075752420121159) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01075752420121159) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01709162192226924) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.01709162192226924) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.01709162192226924) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.01709162192226924) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.019299499858007794) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299499858007794) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299499858007794) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299499858007794) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299499858007794) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299499858007794) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.019299499858007794) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299499858007794) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.019538085344943956) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.019538085344943956) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.019538085344943956) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.019538085344943956) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.024282031623400126) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (0.024282031623400126) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (0.03560840035232359) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.03560840035232359) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.06752398179968269) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.06752398179968269) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.06752398179968269) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.06752398179968269) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.07635036936747966) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.07635036936747966) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.07635036936747966) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07635036936747966) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.07165056250045074) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07165056250045074) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.07165056250045072) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07165056250045072) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.775871994916944e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-5.775871994916944e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.775871994916944e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-5.775871994916944e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.05859215179547845) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.05859215179547845) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.019257453001866466) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019257453001866466) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010311472182429891) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311472182429891) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00882638756779696) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.00882638756779696) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.007597461779087032) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.007597461779087032) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.007597461779087032) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.007597461779087032) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733568640018853) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733568640018853) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733568640018853) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733568640018853) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733568640018853) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733568640018853) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733568640018853) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733568640018853) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348047718414965) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005348047718414965) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005348047718414965) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348047718414965) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.004220835998342174) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.004220835998342174) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.004220835998342174) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.004220835998342174) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.003876482195723331) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (-0.003876482195723331) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (-0.003876482195723331) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (-0.003876482195723331) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (-0.00380406315436931) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00380406315436931) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002446463422674714) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.002446463422674714) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.0022494140606720654) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0022494140606720654) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022494140606720654) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0022494140606720654) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.001863893139068179) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.001863893139068179) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.001863893139068179) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.001863893139068179) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.001863893139068179) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.001863893139068179) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.001863893139068179) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.001863893139068179) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012366559235537952) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.0012366559235537952) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.0012366559235537952) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.0012366559235537952) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.0012223373698216065) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223373698216065) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223373698216065) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223373698216065) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223373698216065) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223373698216065) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223373698216065) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0012223373698216065) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0010283270637559305) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0010283270637559305) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0010283270637559305) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0010283270637559305) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0007870893705581691) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0007870893705581691) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870893705581691) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0007870893705581691) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0005192924120522454) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.0005192924120522454) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.0005192924120522454) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.0005192924120522454) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-1.580937997794301e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.580937997794301e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.580937997794301e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.580937997794301e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.401691618239271e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.401691618239271e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.401691618239271e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.401691618239271e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.253185032655708e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.253185032655708e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.988412497087695e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.988412497087695e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.988412497087695e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.988412497087695e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.883653134161737e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-2.883653134161737e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-1.7924637955501742e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7924637955501742e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7924637955501742e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.7924637955501742e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.949307515898229e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-8.949307515898229e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-7.540204100412463e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-7.540204100412463e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-7.189870302940458e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.189870302940458e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.175164773312576e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175164773312576e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.670408071298936e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-4.670408071298936e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-4.670408071298936e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-4.670408071298936e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.5233392649655647e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.5233392649655647e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.013398778029827e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.013398778029827e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.686321455063964e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.686321455063964e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.686321455063964e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.686321455063964e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.850562430984819e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-1.850562430984819e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.6570092932727346e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.6570092932727346e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6570092932727346e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.6570092932727346e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.846699168060408e-08) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (-7.846699168060408e-08) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (-6.772951765071658e-08) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (-6.772951765071658e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (-4.0998292247850583e-08) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-4.0998292247850583e-08) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.0998292247850583e-08) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.0998292247850583e-08) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (6.772951765071658e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (6.772951765071658e-08) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (7.846699168060408e-08) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (7.846699168060408e-08) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (1.850562430984819e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.850562430984819e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.6666797632007025e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.6666797632007025e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.6666797632007025e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6666797632007025e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.904537319991694e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (2.904537319991694e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (2.904537319991694e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (2.904537319991694e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (3.013398778029827e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.013398778029827e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.076662685674804e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.076662685674804e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.076662685674804e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.076662685674804e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.5233392649655647e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.5233392649655647e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (6.175164773312576e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175164773312576e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (6.876530269986348e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (6.876530269986348e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (6.876530269986348e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (6.876530269986348e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (7.189870302940458e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.189870302940458e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.540204100412463e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (7.540204100412463e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (7.661200186796725e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (7.661200186796725e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (7.661200186796725e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (7.661200186796725e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (8.10534094620795e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (8.10534094620795e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (8.10534094620795e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (8.10534094620795e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.955903377189042e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (9.955903377189042e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (9.955903377189042e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.955903377189042e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.0444741420404157e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (1.0444741420404157e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (1.0444741420404157e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (1.0444741420404157e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (1.3001958520312933e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (1.3001958520312933e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (1.3001958520312933e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (1.3001958520312933e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (2.874248504662879e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (2.874248504662879e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (2.874248504662879e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (2.874248504662879e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (1.1462850757151626e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.1462850757151626e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0011726297842395816) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0011726297842395816) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0011726297842395816) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0011726297842395816) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.001640759116706777) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.001640759116706777) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.001640759116706777) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.001640759116706777) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0021413489647389167) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (0.0021413489647389167) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (0.002200956847995512) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002200956847995512) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002200956847995512) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002200956847995512) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394967154061188) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394967154061188) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394967154061188) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394967154061188) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394967154061188) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394967154061188) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002394967154061188) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394967154061188) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002446463422674714) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.002446463422674714) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.002984180074788379) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.002984180074788379) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.002984180074788379) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.002984180074788379) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.00380406315436931) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00380406315436931) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00882638756779696) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00882638756779696) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.010311472182429891) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311472182429891) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257453001866466) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019257453001866466) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.3986653283179593e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.3986653283179593e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.3986653283179595e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.3986653283179595e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484154579346787) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003484154579346787) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984180074788379) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.002984180074788379) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.00019401030606567594) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00019401030606567594) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7924637955501742e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7924637955501742e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.949307515898229e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-8.949307515898229e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-3.013398778029827e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.013398778029827e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.013398778029827e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.013398778029827e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.0998292247850583e-08) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.0998292247850583e-08) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.0998292247850583e-08) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.0998292247850583e-08) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.846699168060408e-08) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (7.846699168060408e-08) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (7.846699168060408e-08) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (7.846699168060408e-08) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (1.850562430984819e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.850562430984819e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.850562430984819e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.850562430984819e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540204100412463e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (7.540204100412463e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (7.540204100412463e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (7.540204100412463e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (1.7924637955501742e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.7924637955501742e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.1462850757151628e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.1462850757151628e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00019401030606567594) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019401030606567594) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002984180074788379) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.002984180074788379) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.003484154579346787) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.003484154579346787) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
  (-73.13873149598145) [I0]
+ (-0.18066757507074516) [Z6]
+ (-0.18066757507074505) [Z7]
+ (-0.1596144358379445) [Z4]
+ (-0.1596144358379445) [Z5]
+ (0.1741998661203756) [Z2]
+ (0.1741998661203756) [Z3]
+ (0.2275732881437355) [Z1]
+ (0.22757328814373556) [Z0]
+ (-7.954224568649387e-06) [Y5 Y7]
+ (-7.954224568649387e-06) [X5 X7]
+ (8.194104959190207e-06) [Y4 Y6]
+ (8.194104959190207e-06) [X4 X6]
+ (0.1127038185912334) [Z4 Z6]
+ (0.1127038185912334) [Z5 Z7]
+ (0.1195244101690037) [Z0 Z4]
+ (0.1195244101690037) [Z1 Z5]
+ (0.13401737372649103) [Z0 Z6]
+ (0.13401737372649103) [Z1 Z7]
+ (0.13734942210165352) [Z0 Z5]
+ (0.13734942210165352) [Z1 Z4]
+ (0.13766859133632645) [Z2 Z4]
+ (0.13766859133632645) [Z3 Z5]
+ (0.14138903590147578) [Z4 Z7]
+ (0.14138903590147578) [Z5 Z6]
+ (0.1472293078327882) [Z2 Z5]
+ (0.1472293078327882) [Z3 Z4]
+ (0.14926347060053669) [Z4 Z5]
+ (0.14973497005437833) [Z2 Z6]
+ (0.14973497005437833) [Z3 Z7]
+ (0.15138342699112373) [Z0 Z7]
+ (0.15138342699112373) [Z1 Z6]
+ (0.1543576006529187) [Z6 Z7]
+ (0.1558228068586172) [Z2 Z7]
+ (0.1558228068586172) [Z3 Z6]
+ (0.16756669356178774) [Z0 Z2]
+ (0.16756669356178774) [Z1 Z3]
+ (0.1814400936294206) [Z0 Z3]
+ (0.1814400936294206) [Z1 Z2]
+ (0.19392574334973628) [Z0 Z1]
+ (0.22003977240299535) [Z2 Z3]
+ (7.038023739376453e-06) [Y4 Z5 Y6]
+ (7.038023739376453e-06) [X4 Z5 X6]
+ (7.038023739376453e-06) [Y5 Z6 Y7]
+ (7.038023739376453e-06) [X5 Z6 X7]
+ (-0.02868521731024235) [Y4 Y5 X6 X7]
+ (-0.02868521731024235) [X4 X5 Y6 Y7]
+ (-0.017825011932649817) [Y0 Y1 X4 X5]
+ (-0.017825011932649817) [X0 X1 Y4 Y5]
+ (-0.017366053264632704) [Y0 Y1 X6 X7]
+ (-0.017366053264632704) [X0 X1 Y6 Y7]
+ (-0.013873400067632893) [Y0 Y1 X2 X3]
+ (-0.013873400067632893) [X0 X1 Y2 Y3]
+ (-0.009560716496461778) [Y2 Y3 X4 X5]
+ (-0.009560716496461778) [X2 X3 Y4 Y5]
+ (-0.006087836804238903) [Y2 Y3 X6 X7]
+ (-0.006087836804238903) [X2 X3 Y6 Y7]
+ (-0.0002922256724510236) [Y1 Y2 X3 X4]
+ (-0.0002922256724510236) [X1 X2 Y3 Y4]
+ (-7.954224568649387e-06) [Y4 Z5 Y6 Z7]
+ (-7.954224568649387e-06) [X4 Z5 X6 Z7]
+ (-6.62842721522788e-07) [Y2 X3 X5 Y6]
+ (-6.62842721522788e-07) [Y2 Y3 Y5 Y6]
+ (-6.62842721522788e-07) [X2 X3 X5 X6]
+ (-6.62842721522788e-07) [X2 Y3 Y5 X6]
+ (9.344970169460065e-07) [Z2 Y5 Z6 Y7]
+ (9.344970169460065e-07) [Z2 X5 Z6 X7]
+ (9.344970169460065e-07) [Z3 Y4 Z5 Y6]
+ (9.344970169460065e-07) [Z3 X4 Z5 X6]
+ (1.0357924710574506e-06) [Y0 X1 X5 Y6]
+ (1.0357924710574506e-06) [Y0 Y1 Y5 Y6]
+ (1.0357924710574506e-06) [X0 X1 X5 X6]
+ (1.0357924710574506e-06) [X0 Y1 Y5 X6]
+ (1.5973397384687945e-06) [Z2 Y4 Z5 Y6]
+ (1.5973397384687945e-06) [Z2 X4 Z5 X6]
+ (1.5973397384687945e-06) [Z3 Y5 Z6 Y7]
+ (1.5973397384687945e-06) [Z3 X5 Z6 X7]
+ (1.8551374825841703e-06) [Z0 Y4 Z5 Y6]
+ (1.8551374825841703e-06) [Z0 X4 Z5 X6]
+ (1.8551374825841703e-06) [Z1 Y5 Z6 Y7]
+ (1.8551374825841703e-06) [Z1 X5 Z6 X7]
+ (2.8909299536398862e-06) [Z0 Y5 Z6 Y7]
+ (2.8909299536398862e-06) [Z0 X5 Z6 X7]
+ (2.8909299536398862e-06) [Z1 Y4 Z5 Y6]
+ (2.8909299536398862e-06) [Z1 X4 Z5 X6]
+ (8.194104959190207e-06) [Z4 Y5 Z6 Y7]
+ (8.194104959190207e-06) [Z4 X5 Z6 X7]
+ (0.0002922256724510236) [Y1 X2 X3 Y4]
+ (0.0002922256724510236) [X1 Y2 Y3 X4]
+ (0.006087836804238903) [Y2 X3 X6 Y7]
+ (0.006087836804238903) [X2 Y3 Y6 X7]
+ (0.009560716496461778) [Y2 X3 X4 Y5]
+ (0.009560716496461778) [X2 Y3 Y4 X5]
+ (0.011307208030038181) [Y1 Z2 Z3 Y5]
+ (0.011307208030038181) [X1 Z2 Z3 X5]
+ (0.013873400067632893) [Y0 X1 X2 Y3]
+ (0.013873400067632893) [X0 Y1 Y2 X3]
+ (0.017366053264632704) [Y0 X1 X6 Y7]
+ (0.017366053264632704) [X0 Y1 Y6 X7]
+ (0.017825011932649817) [Y0 X1 X4 Y5]
+ (0.017825011932649817) [X0 Y1 Y4 X5]
+ (0.02868521731024235) [Y4 X5 X6 Y7]
+ (0.02868521731024235) [X4 Y5 Y6 X7]
+ (0.02981229960113875) [Y0 Z1 Z2 Y4]
+ (0.02981229960113875) [X0 Z1 Z2 X4]
+ (0.02981229960113875) [Y1 Z3 Z4 Y5]
+ (0.02981229960113875) [X1 Z3 Z4 X5]
+ (0.030104525273589772) [Y0 Z1 Z3 Y4]
+ (0.030104525273589772) [X0 Z1 Z3 X4]
+ (0.030104525273589772) [Y1 Z2 Z4 Y5]
+ (0.030104525273589772) [X1 Z2 Z4 X5]
+ (0.030787440718573995) [Y0 Z2 Z3 Y4]
+ (0.030787440718573995) [X0 Z2 Z3 X4]
+ (0.04375171612136726) [Y0 Z1 Z2 Z3 Y4]
+ (0.04375171612136726) [X0 Z1 Z2 Z3 X4]
+ (0.04375171612136727) [Y1 Z2 Z3 Z4 Y5]
+ (0.04375171612136727) [X1 Z2 Z3 Z4 X5]
+ (-0.014564473640804656) [Y1 Z2 Z3 X4 X6 Y7]
+ (-0.014564473640804656) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-0.014564473640804656) [X1 Z2 Z3 X4 X6 X7]
+ (-0.014564473640804656) [X1 Z2 Z3 Y4 Y6 X7]
+ (-4.183808786587667e-06) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-4.183808786587667e-06) [X1 Z2 Z3 Z4 Z5 X7]
+ (-3.313017056041055e-06) [Y1 Z2 Z3 X4 X5 Y6]
+ (-3.313017056041055e-06) [X1 Z2 Z3 Y4 Y5 X6]
+ (-1.0357924710574506e-06) [Y0 X1 X4 Z5 Z6 Y7]
+ (-1.0357924710574506e-06) [X0 Y1 Y4 Z5 Z6 X7]
+ (-6.62842721522788e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-6.62842721522788e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-3.3280396430346564e-07) [Y1 Y2 X3 Z4 Z5 X6]
+ (-3.3280396430346564e-07) [X1 X2 Y3 Z4 Z5 Y6]
+ (3.3280396430346564e-07) [Y1 X2 X3 Z4 Z5 Y6]
+ (3.3280396430346564e-07) [X1 Y2 Y3 Z4 Z5 X6]
+ (6.62842721522788e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (6.62842721522788e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (1.0357924710574506e-06) [Y0 Y1 X4 Z5 Z6 X7]
+ (1.0357924710574506e-06) [X0 X1 Y4 Z5 Z6 Y7]
+ (3.211187420438903e-06) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (3.211187420438903e-06) [X0 Z1 Z2 Z3 Z5 X6]
+ (3.211187420438903e-06) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (3.211187420438903e-06) [X1 Z2 Z3 Z4 Z6 X7]
+ (3.277438261227103e-06) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (3.277438261227103e-06) [X0 Z1 Z2 Z4 Z5 X6]
+ (3.277438261227103e-06) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (3.277438261227103e-06) [X1 Z3 Z4 Z5 Z6 X7]
+ (3.313017056041055e-06) [Y1 Z2 Z3 Y4 X5 X6]
+ (3.313017056041055e-06) [X1 Z2 Z3 X4 Y5 Y6]
+ (3.6102422255305687e-06) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (3.6102422255305687e-06) [X0 Z1 Z3 Z4 Z5 X6]
+ (3.6102422255305687e-06) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (3.6102422255305687e-06) [X1 Z2 Z4 Z5 Z6 X7]
+ (3.76958358288583e-06) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (3.76958358288583e-06) [X0 Z2 Z3 Z4 Z5 X6]
+ (6.524204476487765e-06) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (6.524204476487765e-06) [X0 Z1 Z2 Z3 Z4 X6]
+ (6.524204476487765e-06) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (6.524204476487765e-06) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0002922256724510236) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (0.0002922256724510236) [Y0 Z1 Y2 X3 Z4 X5]
+ (0.0002922256724510236) [X0 Z1 X2 Y3 Z4 Y5]
+ (0.0002922256724510236) [X0 Z1 X2 X3 Z4 X5]
+ (0.010540434329235314) [Y0 Z1 Z2 Z3 Y4 Z7]
+ (0.010540434329235314) [X0 Z1 Z2 Z3 X4 Z7]
+ (0.010540434329235314) [Y1 Z2 Z3 Z4 Y5 Z6]
+ (0.010540434329235314) [X1 Z2 Z3 Z4 X5 Z6]
+ (0.011307208030038181) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (0.011307208030038181) [X0 Z1 Z2 Z3 X4 Z5]
+ (0.02510490797003997) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (0.02510490797003997) [X0 Z1 Z2 Z3 X4 Z6]
+ (0.02510490797003997) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (0.02510490797003997) [X1 Z2 Z3 Z4 X5 Z7]
+ (0.030787440718573995) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (0.030787440718573995) [Z0 X1 Z2 Z3 Z4 X5]
+ (5.105681053778314e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (5.105681053778314e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (5.105681053778314e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (5.105681053778314e-06) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.014564473640804656) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-0.014564473640804656) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-4.183808786587667e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-4.183808786587667e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-3.313017056041055e-06) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (-3.313017056041055e-06) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (-3.313017056041055e-06) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (-3.313017056041055e-06) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (3.3280396430346564e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (3.3280396430346564e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (3.3280396430346564e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (3.3280396430346564e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (3.76958358288583e-06) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (3.76958358288583e-06) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (0.014564473640804656) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (0.014564473640804656) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
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
+ (0.15676396176430993) [Z4 Z9]
+ (0.15676396176430993) [Z5 Z8]
+ (0.16079764534838553) [Z2 Z5]
+ (0.16079764534838553) [Z3 Z4]
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
  (-46.46390691340409) [I0]
+ (0.7829652070489532) [Z10]
+ (0.7829652070489532) [Z11]
+ (0.8084591005181179) [Z12]
+ (0.8084591005181179) [Z13]
+ (1.2034393391307763) [Z5]
+ (1.2034393391307767) [Z4]
+ (1.3096876618631679) [Z7]
+ (1.3096876618631685) [Z6]
+ (1.3693525711706633) [Z8]
+ (1.3693525711706633) [Z9]
+ (1.6538938305469262) [Z3]
+ (1.6538938305469264) [Z2]
+ (12.412630714435085) [Z0]
+ (12.412630714435087) [Z1]
+ (-7.954224685108313e-06) [Y11 Y13]
+ (-7.954224685108313e-06) [X11 X13]
+ (-7.76508229748292e-07) [Y3 Y5]
+ (-7.76508229748292e-07) [X3 X5]
+ (5.929280393046892e-07) [Y4 Y6]
+ (5.929280393046892e-07) [X4 X6]
+ (1.6021751019053765e-06) [Y2 Y4]
+ (1.6021751019053765e-06) [X2 X4]
+ (1.854056513945243e-06) [Y5 Y7]
+ (1.854056513945243e-06) [X5 X7]
+ (8.194105000858265e-06) [Y10 Y12]
+ (8.194105000858265e-06) [X10 X12]
+ (0.0032769650657404935) [Y1 Y3]
+ (0.0032769650657404935) [X1 X3]
+ (0.10433061485304782) [Y0 Y2]
+ (0.10433061485304782) [X0 X2]
+ (0.11270381859120827) [Z10 Z12]
+ (0.11270381859120827) [Z11 Z13]
+ (0.11383573685296709) [Z4 Z12]
+ (0.11383573685296709) [Z5 Z13]
+ (0.11952441016900887) [Z6 Z10]
+ (0.11952441016900887) [Z7 Z11]
+ (0.12489977362992505) [Z4 Z10]
+ (0.12489977362992505) [Z5 Z11]
+ (0.12495799328193033) [Z2 Z4]
+ (0.12495799328193033) [Z3 Z5]
+ (0.1279949280140107) [Z2 Z10]
+ (0.1279949280140107) [Z3 Z11]
+ (0.1340173737265287) [Z6 Z12]
+ (0.1340173737265287) [Z7 Z13]
+ (0.13701191913060043) [Z4 Z6]
+ (0.13701191913060043) [Z5 Z7]
+ (0.13734942210164758) [Z6 Z11]
+ (0.13734942210164758) [Z7 Z10]
+ (0.13739112375017112) [Z2 Z6]
+ (0.13739112375017112) [Z3 Z7]
+ (0.13766859133627865) [Z8 Z10]
+ (0.13766859133627865) [Z9 Z11]
+ (0.14011294749709122) [Z2 Z12]
+ (0.14011294749709122) [Z3 Z13]
+ (0.1413890359014931) [Z10 Z13]
+ (0.1413890359014931) [Z11 Z12]
+ (0.1425799112870569) [Z4 Z11]
+ (0.1425799112870569) [Z5 Z10]
+ (0.14722930783273167) [Z8 Z11]
+ (0.14722930783273167) [Z9 Z10]
+ (0.14899426171385527) [Z4 Z7]
+ (0.14899426171385527) [Z5 Z6]
+ (0.1492634706005175) [Z10 Z11]
+ (0.14960692557251462) [Z4 Z8]
+ (0.14960692557251462) [Z5 Z9]
+ (0.1497349700543461) [Z8 Z12]
+ (0.1497349700543461) [Z9 Z13]
+ (0.1507140548229863) [Z2 Z8]
+ (0.1507140548229863) [Z3 Z9]
+ (0.15138342699114685) [Z6 Z13]
+ (0.15138342699114685) [Z7 Z12]
+ (0.15215040622639142) [Z4 Z13]
+ (0.15215040622639142) [Z5 Z12]
+ (0.15337959171067994) [Z2 Z11]
+ (0.15337959171067994) [Z3 Z10]
+ (0.15435760065293985) [Z12 Z13]
+ (0.15569017457254353) [Z2 Z13]
+ (0.15569017457254353) [Z3 Z12]
+ (0.15582280685857897) [Z8 Z13]
+ (0.15582280685857897) [Z9 Z12]
+ (0.1567638461017026) [Z4 Z9]
+ (0.1567638461017026) [Z5 Z8]
+ (0.1575530380434727) [Z4 Z5]
+ (0.16079755046695676) [Z2 Z5]
+ (0.16079755046695676) [Z3 Z4]
+ (0.16756669356182857) [Z6 Z8]
+ (0.16756669356182857) [Z7 Z9]
+ (0.16853492794653857) [Z2 Z7]
+ (0.16853492794653857) [Z3 Z6]
+ (0.1814400936294669) [Z6 Z9]
+ (0.1814400936294669) [Z7 Z8]
+ (0.1818908124374215) [Z2 Z3]
+ (0.18690814831050503) [Z2 Z9]
+ (0.18690814831050503) [Z3 Z8]
+ (0.19299700269863226) [Z0 Z10]
+ (0.19299700269863226) [Z1 Z11]
+ (0.19392574334989413) [Z6 Z7]
+ (0.1966174995971574) [Z0 Z4]
+ (0.1966174995971574) [Z1 Z5]
+ (0.19936332691257913) [Z0 Z5]
+ (0.19936332691257913) [Z1 Z4]
+ (0.20072843554599068) [Z0 Z11]
+ (0.20072843554599068) [Z1 Z10]
+ (0.21102681234270626) [Z0 Z12]
+ (0.21102681234270626) [Z1 Z13]
+ (0.2163105980965315) [Z0 Z13]
+ (0.2163105980965315) [Z1 Z12]
+ (0.2200397724029339) [Z8 Z9]
+ (0.2367107174041492) [Z0 Z2]
+ (0.2367107174041492) [Z1 Z3]
+ (0.24164696831745325) [Z0 Z6]
+ (0.24164696831745325) [Z1 Z7]
+ (0.2485351728597721) [Z0 Z7]
+ (0.2485351728597721) [Z1 Z6]
+ (0.25129435573115955) [Z0 Z3]
+ (0.25129435573115955) [Z1 Z2]
+ (0.2723251845037709) [Z0 Z8]
+ (0.2723251845037709) [Z1 Z9]
+ (0.27883454573785127) [Z0 Z9]
+ (0.27883454573785127) [Z1 Z8]
+ (1.1861764484123527) [Z0 Z1]
+ (-3.886639537186466e-06) [Y3 Z4 Y5]
+ (-3.886639537186466e-06) [X3 Z4 X5]
+ (-3.886639537186465e-06) [Y2 Z3 Y4]
+ (-3.886639537186465e-06) [X2 Z3 X4]
+ (1.0722748431072061e-05) [Y10 Z11 Y12]
+ (1.0722748431072061e-05) [X10 Z11 X12]
+ (1.0722748431072061e-05) [Y11 Z12 Y13]
+ (1.0722748431072061e-05) [X11 Z12 X13]
+ (1.226027673890479e-05) [Y4 Z5 Y6]
+ (1.226027673890479e-05) [X4 Z5 X6]
+ (1.226027673890479e-05) [Y5 Z6 Y7]
+ (1.226027673890479e-05) [X5 Z6 X7]
+ (0.12507036883967684) [Y0 Z1 Y2]
+ (0.12507036883967684) [X0 Z1 X2]
+ (0.12507036883967684) [Y1 Z2 Y3]
+ (0.12507036883967684) [X1 Z2 X3]
+ (-0.038314669373424336) [Y4 Y5 X12 X13]
+ (-0.038314669373424336) [X4 X5 Y12 Y13]
+ (-0.03619409348751874) [Y2 Y3 X8 X9]
+ (-0.03619409348751874) [X2 X3 Y8 Y9]
+ (-0.035839557185026426) [Y2 Y3 X4 X5]
+ (-0.035839557185026426) [X2 X3 Y4 Y5]
+ (-0.03114380419636743) [Y2 Y3 X6 X7]
+ (-0.03114380419636743) [X2 X3 Y6 Y7]
+ (-0.028685217310284807) [Y10 Y11 X12 X13]
+ (-0.028685217310284807) [X10 X11 Y12 Y13]
+ (-0.02538466369666924) [Y2 Y3 X10 X11]
+ (-0.02538466369666924) [X2 X3 Y10 Y11]
+ (-0.01902831871828428) [Y3 X4 X11 Y12]
+ (-0.01902831871828428) [X3 Y4 Y11 X12]
+ (-0.017825011932638694) [Y6 Y7 X10 X11]
+ (-0.017825011932638694) [X6 X7 Y10 Y11]
+ (-0.017680137657131844) [Y4 Y5 X10 X11]
+ (-0.017680137657131844) [X4 X5 Y10 Y11]
+ (-0.01736605326461815) [Y6 Y7 X12 X13]
+ (-0.01736605326461815) [X6 X7 Y12 Y13]
+ (-0.015577227075452324) [Y2 Y3 X12 X13]
+ (-0.015577227075452324) [X2 X3 Y12 Y13]
+ (-0.014583638327010377) [Y0 Y1 X2 X3]
+ (-0.014583638327010377) [X0 X1 Y2 Y3]
+ (-0.013873400067638332) [Y6 Y7 X8 X9]
+ (-0.013873400067638332) [X6 X7 Y8 Y9]
+ (-0.011982342583254854) [Y4 Y5 X6 X7]
+ (-0.011982342583254854) [X4 X5 Y6 Y7]
+ (-0.011285144618318692) [Y5 X6 X11 Y12]
+ (-0.011285144618318692) [X5 Y6 Y11 X12]
+ (-0.009560716496453035) [Y8 Y9 X10 X11]
+ (-0.009560716496453035) [X8 X9 Y10 Y11]
+ (-0.008125248410129604) [Y1 X2 X8 Y9]
+ (-0.008125248410129604) [Y1 Y2 Y8 Y9]
+ (-0.008125248410129604) [X1 X2 X8 X9]
+ (-0.008125248410129604) [X1 Y2 Y8 X9]
+ (-0.007731432847358378) [Y0 Y1 X10 X11]
+ (-0.007731432847358378) [X0 X1 Y10 Y11]
+ (-0.007156920529187965) [Y4 Y5 X8 X9]
+ (-0.007156920529187965) [X4 X5 Y8 Y9]
+ (-0.006888204542318858) [Y0 Y1 X6 X7]
+ (-0.006888204542318858) [X0 X1 Y6 Y7]
+ (-0.006509361234080356) [Y0 Y1 X8 X9]
+ (-0.006509361234080356) [X0 X1 Y8 Y9]
+ (-0.006087836804232851) [Y8 Y9 X12 X13]
+ (-0.006087836804232851) [X8 X9 Y12 Y13]
+ (-0.0052837857538252445) [Y0 Y1 X12 X13]
+ (-0.0052837857538252445) [X0 X1 Y12 Y13]
+ (-0.005143382387697353) [Y3 Y4 X5 X6]
+ (-0.005143382387697353) [X3 X4 Y5 Y6]
+ (-0.0046849202268751345) [Y1 X2 X6 Y7]
+ (-0.0046849202268751345) [Y1 Y2 Y6 Y7]
+ (-0.0046849202268751345) [X1 X2 X6 X7]
+ (-0.0046849202268751345) [X1 Y2 Y6 X7]
+ (-0.004575015188892522) [Y1 X2 X12 Y13]
+ (-0.004575015188892522) [Y1 Y2 Y12 Y13]
+ (-0.004575015188892522) [X1 X2 X12 X13]
+ (-0.004575015188892522) [X1 Y2 Y12 X13]
+ (-0.004424843668496739) [Y1 X2 X4 Y5]
+ (-0.004424843668496739) [Y1 Y2 Y4 Y5]
+ (-0.004424843668496739) [X1 X2 X4 X5]
+ (-0.004424843668496739) [X1 Y2 Y4 X5]
+ (-0.002745827315421737) [Y0 Y1 X4 X5]
+ (-0.002745827315421737) [X0 X1 Y4 Y5]
+ (-0.0017991930083876967) [Y1 X2 X10 Y11]
+ (-0.0017991930083876967) [Y1 Y2 Y10 Y11]
+ (-0.0017991930083876967) [X1 X2 X10 X11]
+ (-0.0017991930083876967) [X1 Y2 Y10 X11]
+ (-0.0016639606584190588) [Y2 Z3 Z4 Y6]
+ (-0.0016639606584190588) [X2 Z3 Z4 X6]
+ (-0.0016639606584190588) [Y3 Z5 Z6 Y7]
+ (-0.0016639606584190588) [X3 Z5 Z6 X7]
+ (-0.0004957972886229613) [Y2 Z4 Z5 Y6]
+ (-0.0004957972886229613) [X2 Z4 Z5 X6]
+ (-0.0002922256724506923) [Y7 Y8 X9 X10]
+ (-0.0002922256724506923) [X7 X8 Y9 Y10]
+ (-7.954224685108313e-06) [Y10 Z11 Y12 Z13]
+ (-7.954224685108313e-06) [X10 Z11 X12 Z13]
+ (-7.801538391185355e-06) [Y2 Z3 Y4 Z11]
+ (-7.801538391185355e-06) [X2 Z3 X4 Z11]
+ (-7.801538391185355e-06) [Y3 Z4 Y5 Z10]
+ (-7.801538391185355e-06) [X3 Z4 X5 Z10]
+ (-5.9741768991773345e-06) [Y5 X6 X10 Y11]
+ (-5.9741768991773345e-06) [Y5 Y6 Y10 Y11]
+ (-5.9741768991773345e-06) [X5 X6 X10 X11]
+ (-5.9741768991773345e-06) [X5 Y6 Y10 X11]
+ (-4.642979005748879e-06) [Y3 X4 X10 Y11]
+ (-4.642979005748879e-06) [Y3 Y4 Y10 Y11]
+ (-4.642979005748879e-06) [X3 X4 X10 X11]
+ (-4.642979005748879e-06) [X3 Y4 Y10 X11]
+ (-4.281812064196346e-06) [Y4 Z5 Y6 Z11]
+ (-4.281812064196346e-06) [X4 Z5 X6 Z11]
+ (-4.281812064196346e-06) [Y5 Z6 Y7 Z10]
+ (-4.281812064196346e-06) [X5 Z6 X7 Z10]
+ (-3.1585593854343078e-06) [Y2 Z3 Y4 Z10]
+ (-3.1585593854343078e-06) [X2 Z3 X4 Z10]
+ (-3.1585593854343078e-06) [Y3 Z4 Y5 Z11]
+ (-3.1585593854343078e-06) [X3 Z4 X5 Z11]
+ (-2.172638021765241e-06) [Y2 X3 X11 Y12]
+ (-2.172638021765241e-06) [Y2 Y3 Y11 Y12]
+ (-2.172638021765241e-06) [X2 X3 X11 X12]
+ (-2.172638021765241e-06) [X2 Y3 Y11 X12]
+ (-1.878149549626451e-06) [Z4 Y10 Z11 Y12]
+ (-1.878149549626451e-06) [Z4 X10 Z11 X12]
+ (-1.878149549626451e-06) [Z5 Y11 Z12 Y13]
+ (-1.878149549626451e-06) [Z5 X11 Z12 X13]
+ (-1.4548066691723403e-06) [Y3 X4 X6 Y7]
+ (-1.4548066691723403e-06) [Y3 Y4 Y6 Y7]
+ (-1.4548066691723403e-06) [X3 X4 X6 X7]
+ (-1.4548066691723403e-06) [X3 Y4 Y6 X7]
+ (-1.195392055879682e-06) [Y2 Z3 Y4 Z7]
+ (-1.195392055879682e-06) [X2 Z3 X4 Z7]
+ (-1.195392055879682e-06) [Y3 Z4 Y5 Z6]
+ (-1.195392055879682e-06) [X3 Z4 X5 Z6]
+ (-1.1907330995546511e-06) [Z0 Y3 Z4 Y5]
+ (-1.1907330995546511e-06) [Z0 X3 Z4 X5]
+ (-1.1907330995546511e-06) [Z1 Y2 Z3 Y4]
+ (-1.1907330995546511e-06) [Z1 X2 Z3 X4]
+ (-1.1094125009978835e-06) [Z2 Y11 Z12 Y13]
+ (-1.1094125009978835e-06) [Z2 X11 Z12 X13]
+ (-1.1094125009978835e-06) [Z3 Y10 Z11 Y12]
+ (-1.1094125009978835e-06) [Z3 X10 Z11 X12]
+ (-8.336695514197036e-07) [Z0 Y2 Z3 Y4]
+ (-8.336695514197036e-07) [Z0 X2 Z3 X4]
+ (-8.336695514197036e-07) [Z1 Y3 Z4 Y5]
+ (-8.336695514197036e-07) [Z1 X3 Z4 X5]
+ (-7.956666925456609e-07) [Y3 X4 X8 Y9]
+ (-7.956666925456609e-07) [Y3 Y4 Y8 Y9]
+ (-7.956666925456609e-07) [X3 X4 X8 X9]
+ (-7.956666925456609e-07) [X3 Y4 Y8 X9]
+ (-7.76508229748292e-07) [Y2 Z3 Y4 Z5]
+ (-7.76508229748292e-07) [X2 Z3 X4 Z5]
+ (-6.628427287409724e-07) [Y8 X9 X11 Y12]
+ (-6.628427287409724e-07) [Y8 Y9 Y11 Y12]
+ (-6.628427287409724e-07) [X8 X9 X11 X12]
+ (-6.628427287409724e-07) [X8 Y9 Y11 X12]
+ (-5.769436528645783e-07) [Y2 Z3 Y4 Z9]
+ (-5.769436528645783e-07) [X2 Z3 X4 Z9]
+ (-5.769436528645783e-07) [Y3 Z4 Y5 Z8]
+ (-5.769436528645783e-07) [X3 Z4 X5 Z8]
+ (-5.627722041763966e-07) [Y0 X1 X11 Y12]
+ (-5.627722041763966e-07) [Y0 Y1 Y11 Y12]
+ (-5.627722041763966e-07) [X0 X1 X11 X12]
+ (-5.627722041763966e-07) [X0 Y1 Y11 X12]
+ (-5.471606066462955e-07) [Y1 X2 X11 Y12]
+ (-5.471606066462955e-07) [X1 Y2 Y11 X12]
+ (-3.570635481350178e-07) [Y0 X1 X3 Y4]
+ (-3.570635481350178e-07) [Y0 Y1 Y3 Y4]
+ (-3.570635481350178e-07) [X0 X1 X3 X4]
+ (-3.570635481350178e-07) [X0 Y1 Y3 X4]
+ (-1.9332121228344622e-07) [Y1 X2 X3 Y4]
+ (-1.9332121228344622e-07) [X1 Y2 Y3 X4]
+ (-1.2919458250413665e-07) [Y1 Z2 Z3 Y5]
+ (-1.2919458250413665e-07) [X1 Z2 Z3 X5]
+ (-3.226106239649198e-09) [Y1 Y2 X5 X6]
+ (-3.226106239649198e-09) [X1 X2 Y5 Y6]
+ (3.226106239649198e-09) [Y1 X2 X5 Y6]
+ (3.226106239649198e-09) [X1 Y2 Y5 X6]
+ (3.2284022075459995e-08) [Y4 Z5 Y6 Z12]
+ (3.2284022075459995e-08) [X4 Z5 X6 Z12]
+ (3.2284022075459995e-08) [Y5 Z6 Y7 Z13]
+ (3.2284022075459995e-08) [X5 Z6 X7 Z13]
+ (1.6728571377919855e-07) [Y0 Z1 Z3 Y4]
+ (1.6728571377919855e-07) [X0 Z1 Z3 X4]
+ (1.6728571377919855e-07) [Y1 Z2 Z4 Y5]
+ (1.6728571377919855e-07) [X1 Z2 Z4 X5]
+ (1.9332121228344622e-07) [Y1 Y2 X3 X4]
+ (1.9332121228344622e-07) [X1 X2 Y3 Y4]
+ (2.1872303968108262e-07) [Y2 Z3 Y4 Z8]
+ (2.1872303968108262e-07) [X2 Z3 X4 Z8]
+ (2.1872303968108262e-07) [Y3 Z4 Y5 Z9]
+ (2.1872303968108262e-07) [X3 Z4 X5 Z9]
+ (2.1989637723559977e-07) [Y2 X3 X5 Y6]
+ (2.1989637723559977e-07) [Y2 Y3 Y5 Y6]
+ (2.1989637723559977e-07) [X2 X3 X5 X6]
+ (2.1989637723559977e-07) [X2 Y3 Y5 X6]
+ (2.447264788508917e-07) [Y0 X1 X5 Y6]
+ (2.447264788508917e-07) [Y0 Y1 Y5 Y6]
+ (2.447264788508917e-07) [X0 X1 X5 X6]
+ (2.447264788508917e-07) [X0 Y1 Y5 X6]
+ (2.594146132920078e-07) [Y2 Z3 Y4 Z6]
+ (2.594146132920078e-07) [X2 Z3 X4 Z6]
+ (2.594146132920078e-07) [Y3 Z4 Y5 Z7]
+ (2.594146132920078e-07) [X3 Z4 X5 Z7]
+ (3.6060692606305135e-07) [Y0 Z1 Z2 Y4]
+ (3.6060692606305135e-07) [X0 Z1 Z2 X4]
+ (3.6060692606305135e-07) [Y1 Z3 Z4 Y5]
+ (3.6060692606305135e-07) [X1 Z3 Z4 X5]
+ (4.837953298618804e-07) [Y5 X6 X8 Y9]
+ (4.837953298618804e-07) [Y5 Y6 Y8 Y9]
+ (4.837953298618804e-07) [X5 X6 X8 X9]
+ (4.837953298618804e-07) [X5 Y6 Y8 X9]
+ (5.471606066462955e-07) [Y1 Y2 X11 X12]
+ (5.471606066462955e-07) [X1 X2 Y11 Y12]
+ (5.929280393046892e-07) [Z4 Y5 Z6 Y7]
+ (5.929280393046892e-07) [Z4 X5 Z6 X7]
+ (9.344970014028842e-07) [Z8 Y11 Z12 Y13]
+ (9.344970014028842e-07) [Z8 X11 Z12 X13]
+ (9.344970014028842e-07) [Z9 Y10 Z11 Y12]
+ (9.344970014028842e-07) [Z9 X10 Z11 X12]
+ (9.509134339710356e-07) [Z2 Y4 Z5 Y6]
+ (9.509134339710356e-07) [Z2 X4 Z5 X6]
+ (9.509134339710356e-07) [Z3 Y5 Z6 Y7]
+ (9.509134339710356e-07) [Z3 X5 Z6 X7]
+ (1.035792466932712e-06) [Y6 X7 X11 Y12]
+ (1.035792466932712e-06) [Y6 Y7 Y11 Y12]
+ (1.035792466932712e-06) [X6 X7 X11 X12]
+ (1.035792466932712e-06) [X6 Y7 Y11 X12]
+ (1.0632255207684415e-06) [Z2 Y10 Z11 Y12]
+ (1.0632255207684415e-06) [Z2 X10 Z11 X12]
+ (1.0632255207684415e-06) [Z3 Y11 Z12 Y13]
+ (1.0632255207684415e-06) [Z3 X11 Z12 X13]
+ (1.1708098112054427e-06) [Z2 Y5 Z6 Y7]
+ (1.1708098112054427e-06) [Z2 X5 Z6 X7]
+ (1.1708098112054427e-06) [Z3 Y4 Z5 Y6]
+ (1.1708098112054427e-06) [Z3 X4 Z5 X6]
+ (1.398024271413007e-06) [Y4 Z5 Y6 Z8]
+ (1.398024271413007e-06) [X4 Z5 X6 Z8]
+ (1.398024271413007e-06) [Y5 Z6 Y7 Z9]
+ (1.398024271413007e-06) [X5 Z6 X7 Z9]
+ (1.5973397301438565e-06) [Z8 Y10 Z11 Y12]
+ (1.5973397301438565e-06) [Z8 X10 Z11 X12]
+ (1.5973397301438565e-06) [Z9 Y11 Z12 Y13]
+ (1.5973397301438565e-06) [Z9 X11 Z12 X13]
+ (1.6021751019053765e-06) [Z2 Y3 Z4 Y5]
+ (1.6021751019053765e-06) [Z2 X3 Z4 X5]
+ (1.6149608000547968e-06) [Z0 Y11 Z12 Y13]
+ (1.6149608000547968e-06) [Z0 X11 Z12 X13]
+ (1.6149608000547968e-06) [Z1 Y10 Z11 Y12]
+ (1.6149608000547968e-06) [Z1 X10 Z11 X12]
+ (1.6923648349801215e-06) [Y4 Z5 Y6 Z10]
+ (1.6923648349801215e-06) [X4 Z5 X6 Z10]
+ (1.6923648349801215e-06) [Y5 Z6 Y7 Z11]
+ (1.6923648349801215e-06) [X5 Z6 X7 Z11]
+ (1.8163673805686453e-06) [Z4 Y11 Z12 Y13]
+ (1.8163673805686453e-06) [Z4 X11 Z12 X13]
+ (1.8163673805686453e-06) [Z5 Y10 Z11 Y12]
+ (1.8163673805686453e-06) [Z5 X10 Z11 X12]
+ (1.854056513945243e-06) [Y4 Z5 Y6 Z7]
+ (1.854056513945243e-06) [X4 Z5 X6 Z7]
+ (1.855137478105548e-06) [Z6 Y10 Z11 Y12]
+ (1.855137478105548e-06) [Z6 X10 Z11 X12]
+ (1.855137478105548e-06) [Z7 Y11 Z12 Y13]
+ (1.855137478105548e-06) [Z7 X11 Z12 X13]
+ (1.8818196012748874e-06) [Y4 Z5 Y6 Z9]
+ (1.8818196012748874e-06) [X4 Z5 X6 Z9]
+ (1.8818196012748874e-06) [Y5 Z6 Y7 Z8]
+ (1.8818196012748874e-06) [X5 Z6 X7 Z8]
+ (2.1777330042310606e-06) [Z0 Y10 Z11 Y12]
+ (2.1777330042310606e-06) [Z0 X10 Z11 X12]
+ (2.1777330042310606e-06) [Z1 Y11 Z12 Y13]
+ (2.1777330042310606e-06) [Z1 X11 Z12 X13]
+ (2.890929945036959e-06) [Z6 Y11 Z12 Y13]
+ (2.890929945036959e-06) [Z6 X11 Z12 X13]
+ (2.890929945036959e-06) [Z7 Y10 Z11 Y12]
+ (2.890929945036959e-06) [Z7 X10 Z11 X12]
+ (3.099296599366516e-06) [Z0 Y4 Z5 Y6]
+ (3.099296599366516e-06) [Z0 X4 Z5 X6]
+ (3.099296599366516e-06) [Z1 Y5 Z6 Y7]
+ (3.099296599366516e-06) [Z1 X5 Z6 X7]
+ (3.1173663804227842e-06) [Y0 Z2 Z3 Y4]
+ (3.1173663804227842e-06) [X0 Z2 Z3 X4]
+ (3.344023078217333e-06) [Z0 Y5 Z6 Y7]
+ (3.344023078217333e-06) [Z0 X5 Z6 X7]
+ (3.344023078217333e-06) [Z1 Y4 Z5 Y6]
+ (3.344023078217333e-06) [Z1 X4 Z5 X6]
+ (3.539010000267427e-06) [Y2 Z3 Y4 Z12]
+ (3.539010000267427e-06) [X2 Z3 X4 Z12]
+ (3.539010000267427e-06) [Y3 Z4 Y5 Z13]
+ (3.539010000267427e-06) [X3 Z4 X5 Z13]
+ (3.6945169301963975e-06) [Y4 X5 X11 Y12]
+ (3.6945169301963975e-06) [Y4 Y5 Y11 Y12]
+ (3.6945169301963975e-06) [X4 X5 X11 X12]
+ (3.6945169301963975e-06) [X4 Y5 Y11 X12]
+ (4.556473786485483e-06) [Y5 X6 X12 Y13]
+ (4.556473786485483e-06) [Y5 Y6 Y12 Y13]
+ (4.556473786485483e-06) [X5 X6 X12 X13]
+ (4.556473786485483e-06) [X5 Y6 Y12 X13]
+ (4.588757808557907e-06) [Y4 Z5 Y6 Z13]
+ (4.588757808557907e-06) [X4 Z5 X6 Z13]
+ (4.588757808557907e-06) [Y5 Z6 Y7 Z12]
+ (4.588757808557907e-06) [X5 Z6 X7 Z12]
+ (5.275783469220821e-06) [Y3 X4 X12 Y13]
+ (5.275783469220821e-06) [Y3 Y4 Y12 Y13]
+ (5.275783469220821e-06) [X3 X4 X12 X13]
+ (5.275783469220821e-06) [X3 Y4 Y12 X13]
+ (8.194105000858265e-06) [Z10 Y11 Z12 Y13]
+ (8.194105000858265e-06) [Z10 X11 Z12 X13]
+ (8.81479346949085e-06) [Y2 Z3 Y4 Z13]
+ (8.81479346949085e-06) [X2 Z3 X4 Z13]
+ (8.81479346949085e-06) [Y3 Z4 Y5 Z12]
+ (8.81479346949085e-06) [X3 Z4 X5 Z12]
+ (0.0002922256724506923) [Y7 X8 X9 Y10]
+ (0.0002922256724506923) [X7 Y8 Y9 X10]
+ (0.001105898480890369) [Y0 Z1 Y2 Z5]
+ (0.001105898480890369) [X0 Z1 X2 Z5]
+ (0.001105898480890369) [Y1 Z2 Y3 Z4]
+ (0.001105898480890369) [X1 Z2 X3 Z4]
+ (0.0017560659628705292) [Y0 Z1 Y2 Z11]
+ (0.0017560659628705292) [X0 Z1 X2 Z11]
+ (0.0017560659628705292) [Y1 Z2 Y3 Z10]
+ (0.0017560659628705292) [X1 Z2 X3 Z10]
+ (0.002326234847602346) [Y0 Z1 Y2 Z13]
+ (0.002326234847602346) [X0 Z1 X2 Z13]
+ (0.002326234847602346) [Y1 Z2 Y3 Z12]
+ (0.002326234847602346) [X1 Z2 X3 Z12]
+ (0.002745827315421737) [Y0 X1 X4 Y5]
+ (0.002745827315421737) [X0 Y1 Y4 X5]
+ (0.002929768278576192) [Y0 Z1 Y2 Z9]
+ (0.002929768278576192) [X0 Z1 X2 Z9]
+ (0.002929768278576192) [Y1 Z2 Y3 Z8]
+ (0.002929768278576192) [X1 Z2 X3 Z8]
+ (0.0032769650657404935) [Y0 Z1 Y2 Z3]
+ (0.0032769650657404935) [X0 Z1 X2 Z3]
+ (0.0033476264706845194) [Y0 Z1 Y2 Z7]
+ (0.0033476264706845194) [X0 Z1 X2 Z7]
+ (0.0033476264706845194) [Y1 Z2 Y3 Z6]
+ (0.0033476264706845194) [X1 Z2 X3 Z6]
+ (0.0034794217292782948) [Y2 Z3 Z5 Y6]
+ (0.0034794217292782948) [X2 Z3 Z5 X6]
+ (0.0034794217292782948) [Y3 Z4 Z6 Y7]
+ (0.0034794217292782948) [X3 Z4 Z6 X7]
+ (0.003555258971258226) [Y0 Z1 Y2 Z10]
+ (0.003555258971258226) [X0 Z1 X2 Z10]
+ (0.003555258971258226) [Y1 Z2 Y3 Z11]
+ (0.003555258971258226) [X1 Z2 X3 Z11]
+ (0.005143382387697353) [Y3 X4 X5 Y6]
+ (0.005143382387697353) [X3 Y4 Y5 X6]
+ (0.0052837857538252445) [Y0 X1 X12 Y13]
+ (0.0052837857538252445) [X0 Y1 Y12 X13]
+ (0.005530742149387108) [Y0 Z1 Y2 Z4]
+ (0.005530742149387108) [X0 Z1 X2 Z4]
+ (0.005530742149387108) [Y1 Z2 Y3 Z5]
+ (0.005530742149387108) [X1 Z2 X3 Z5]
+ (0.006087836804232851) [Y8 X9 X12 Y13]
+ (0.006087836804232851) [X8 Y9 Y12 X13]
+ (0.006509361234080356) [Y0 X1 X8 Y9]
+ (0.006509361234080356) [X0 Y1 Y8 X9]
+ (0.006888204542318858) [Y0 X1 X6 Y7]
+ (0.006888204542318858) [X0 Y1 Y6 X7]
+ (0.006901250036494868) [Y0 Z1 Y2 Z12]
+ (0.006901250036494868) [X0 Z1 X2 Z12]
+ (0.006901250036494868) [Y1 Z2 Y3 Z13]
+ (0.006901250036494868) [X1 Z2 X3 Z13]
+ (0.007156920529187965) [Y4 X5 X8 Y9]
+ (0.007156920529187965) [X4 Y5 Y8 X9]
+ (0.007731432847358378) [Y0 X1 X10 Y11]
+ (0.007731432847358378) [X0 Y1 Y10 X11]
+ (0.008032546697559654) [Y0 Z1 Y2 Z6]
+ (0.008032546697559654) [X0 Z1 X2 Z6]
+ (0.008032546697559654) [Y1 Z2 Y3 Z7]
+ (0.008032546697559654) [X1 Z2 X3 Z7]
+ (0.009560716496453035) [Y8 X9 X10 Y11]
+ (0.009560716496453035) [X8 Y9 Y10 X11]
+ (0.011055016688705795) [Y0 Z1 Y2 Z8]
+ (0.011055016688705795) [X0 Z1 X2 Z8]
+ (0.011055016688705795) [Y1 Z2 Y3 Z9]
+ (0.011055016688705795) [X1 Z2 X3 Z9]
+ (0.011285144618318692) [Y5 Y6 X11 X12]
+ (0.011285144618318692) [X5 X6 Y11 Y12]
+ (0.011307208030054866) [Y7 Z8 Z9 Y11]
+ (0.011307208030054866) [X7 Z8 Z9 X11]
+ (0.011982342583254854) [Y4 X5 X6 Y7]
+ (0.011982342583254854) [X4 Y5 Y6 X7]
+ (0.013873400067638332) [Y6 X7 X8 Y9]
+ (0.013873400067638332) [X6 Y7 Y8 X9]
+ (0.014583638327010377) [Y0 X1 X2 Y3]
+ (0.014583638327010377) [X0 Y1 Y2 X3]
+ (0.015577227075452324) [Y2 X3 X12 Y13]
+ (0.015577227075452324) [X2 Y3 Y12 X13]
+ (0.01736605326461815) [Y6 X7 X12 Y13]
+ (0.01736605326461815) [X6 Y7 Y12 X13]
+ (0.017680137657131844) [Y4 X5 X10 Y11]
+ (0.017680137657131844) [X4 Y5 Y10 X11]
+ (0.017825011932638694) [Y6 X7 X10 Y11]
+ (0.017825011932638694) [X6 Y7 Y10 X11]
+ (0.01902831871828428) [Y3 Y4 X11 X12]
+ (0.01902831871828428) [X3 X4 Y11 Y12]
+ (0.02538466369666924) [Y2 X3 X10 Y11]
+ (0.02538466369666924) [X2 Y3 Y10 X11]
+ (0.025996206267182822) [Y3 Z4 Z5 Y7]
+ (0.025996206267182822) [X3 Z4 Z5 X7]
+ (0.028685217310284807) [Y10 X11 X12 Y13]
+ (0.028685217310284807) [X10 Y11 Y12 X13]
+ (0.029812299601160802) [Y6 Z7 Z8 Y10]
+ (0.029812299601160802) [X6 Z7 Z8 X10]
+ (0.029812299601160802) [Y7 Z9 Z10 Y11]
+ (0.029812299601160802) [X7 Z9 Z10 X11]
+ (0.030104525273611495) [Y6 Z7 Z9 Y10]
+ (0.030104525273611495) [X6 Z7 Z9 X10]
+ (0.030104525273611495) [Y7 Z8 Z10 Y11]
+ (0.030104525273611495) [X7 Z8 Z10 X11]
+ (0.03078744071862932) [Y6 Z8 Z9 Y10]
+ (0.03078744071862932) [X6 Z8 Z9 X10]
+ (0.03114380419636743) [Y2 X3 X6 Y7]
+ (0.03114380419636743) [X2 Y3 Y6 X7]
+ (0.035839557185026426) [Y2 X3 X4 Y5]
+ (0.035839557185026426) [X2 Y3 Y4 X5]
+ (0.03619409348751874) [Y2 X3 X8 Y9]
+ (0.03619409348751874) [X2 Y3 Y8 X9]
+ (0.038314669373424336) [Y4 X5 X12 Y13]
+ (0.038314669373424336) [X4 Y5 Y12 X13]
+ (0.10433061485304782) [Z0 Y1 Z2 Y3]
+ (0.10433061485304782) [Z0 X1 Z2 X3]
+ (3.204142222412856e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.204142222412856e-06) [X0 Z1 Z2 Z3 X4]
+ (3.204142222412856e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.204142222412856e-06) [X1 Z2 Z3 Z4 X5]
+ (0.12133242248354913) [Y2 Z3 Z4 Z5 Y6]
+ (0.12133242248354913) [X2 Z3 Z4 Z5 X6]
+ (0.12133242248354921) [Y3 Z4 Z5 Z6 Y7]
+ (0.12133242248354921) [X3 Z4 Z5 Z6 X7]
+ (0.22847946311063302) [Y6 Z7 Z8 Z9 Y10]
+ (0.22847946311063302) [X6 Z7 Z8 Z9 X10]
+ (0.22847946311063302) [Y7 Z8 Z9 Z10 Y11]
+ (0.22847946311063302) [X7 Z8 Z9 Z10 X11]
+ (-0.045879424030655) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (-0.045879424030655) [X0 Z2 Z3 Z4 Z5 X6]
+ (-0.02435313608450475) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (-0.02435313608450475) [Y2 Z3 Y4 X11 Z12 X13]
+ (-0.02435313608450475) [X2 Z3 X4 Y11 Z12 Y13]
+ (-0.02435313608450475) [X2 Z3 X4 X11 Z12 X13]
+ (-0.02435313608450475) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (-0.02435313608450475) [Y3 Z4 Y5 X10 Z11 X12]
+ (-0.02435313608450475) [X3 Z4 X5 Y10 Z11 Y12]
+ (-0.02435313608450475) [X3 Z4 X5 X10 Z11 X12]
+ (-0.01558827786511179) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (-0.01558827786511179) [X2 Z3 X4 X10 Z11 X12]
+ (-0.01558827786511179) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (-0.01558827786511179) [X3 Z4 X5 X11 Z12 X13]
+ (-0.015225659057132896) [Y3 Z4 Z5 X6 X10 Y11]
+ (-0.015225659057132896) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (-0.015225659057132896) [X3 Z4 Z5 X6 X10 X11]
+ (-0.015225659057132896) [X3 Z4 Z5 Y6 Y10 X11]
+ (-0.014564473640806812) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564473640806812) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564473640806812) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564473640806812) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.014411189770087644) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (-0.014411189770087644) [X2 Z3 Z4 Z5 X6 Z11]
+ (-0.014411189770087644) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (-0.014411189770087644) [X3 Z4 Z5 Z6 X7 Z10]
+ (-0.012214985322759674) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.012214985322759674) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.012214985322759674) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.012214985322759674) [X4 Z5 X6 X11 Z12 X13]
+ (-0.012214985322759674) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.012214985322759674) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.012214985322759674) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.012214985322759674) [X5 Z6 X7 X10 Z11 X12]
+ (-0.010263460498891326) [Y2 Z3 X4 X10 Z11 Y12]
+ (-0.010263460498891326) [X2 Z3 Y4 Y10 Z11 X12]
+ (-0.010263460498891326) [Y3 Z4 X5 X11 Z12 Y13]
+ (-0.010263460498891326) [X3 Z4 Y5 Y11 Z12 X13]
+ (-0.008125248410129604) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125248410129604) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.007306763969604764) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306763969604764) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306763969604764) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306763969604764) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005324817366220477) [Y2 Z3 Y4 X10 Z11 X12]
+ (-0.005324817366220477) [X2 Z3 X4 Y10 Z11 Y12]
+ (-0.005324817366220477) [Y3 Z4 Y5 X11 Z12 X13]
+ (-0.005324817366220477) [X3 Z4 X5 Y11 Z12 Y13]
+ (-0.0046849202268751345) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.0046849202268751345) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668615266021862) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668615266021862) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.004575015188892521) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.004575015188892521) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424843668496738) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424843668496738) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.003961569373035198) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (-0.003961569373035198) [X0 Z1 Z2 Z4 Z5 X6]
+ (-0.003961569373035198) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (-0.003961569373035198) [X1 Z3 Z4 Z5 Z6 X7]
+ (-0.002462916621394748) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (-0.002462916621394748) [X0 Z1 Z2 Z3 Z5 X6]
+ (-0.002462916621394748) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (-0.002462916621394748) [X1 Z2 Z3 Z4 Z6 X7]
+ (-0.0022939556230648817) [Y1 Y2 X3 Z4 Z5 X6]
+ (-0.0022939556230648817) [X1 X2 Y3 Z4 Z5 Y6]
+ (-0.0017991930083876967) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.0017991930083876967) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278745823304962) [Y1 Z2 Z3 Y4 X11 X12]
+ (-0.0017278745823304962) [X1 Z2 Z3 X4 Y11 Y12]
+ (-0.001667613749970318) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (-0.001667613749970318) [X0 Z1 Z3 Z4 Z5 X6]
+ (-0.001667613749970318) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (-0.001667613749970318) [X1 Z2 Z4 Z5 Z6 X7]
+ (-0.0016095335162950115) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (-0.0016095335162950115) [X0 Z1 Z2 Z3 Z4 X6]
+ (-0.0016095335162950115) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (-0.0016095335162950115) [X1 Z2 Z3 Z5 Z6 X7]
+ (-0.0009298407044409814) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298407044409814) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298407044409814) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298407044409814) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533831050997363) [Y1 Z2 Z3 X4 X5 Y6]
+ (-0.0008533831050997363) [X1 Z2 Z3 Y4 Y5 X6]
+ (-0.0006650303449144551) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (-0.0006650303449144551) [X2 Z3 Z4 Z5 X6 Z12]
+ (-0.0006650303449144551) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (-0.0006650303449144551) [X3 Z4 Z5 Z6 X7 Z13]
+ (-0.0004957972886229613) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0004957972886229613) [Z2 X3 Z4 Z5 Z6 X7]
+ (-7.73587060554835e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.73587060554835e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.73587060554835e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.73587060554835e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-5.9741768991773345e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (-5.9741768991773345e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (-5.275783469220821e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.275783469220821e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.642979005748879e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.642979005748879e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.556473786485483e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (-4.556473786485483e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (-4.1838088461008255e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (-4.1838088461008255e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (-3.6945169301963975e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (-3.6945169301963975e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (-3.3342618576456123e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (-3.3342618576456123e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (-3.313017076081448e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (-3.313017076081448e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (-3.151295976920013e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.151295976920013e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.088245732762963e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.088245732762963e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.172638021765241e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (-2.172638021765241e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (-1.4548066691723403e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.4548066691723403e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.3304568589956375e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (-1.3304568589956375e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (-1.2282691033180075e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (-1.2282691033180075e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (-1.035792466932712e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (-1.035792466932712e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (-7.956666925456609e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956666925456609e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733096811957652e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733096811957652e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733096811957652e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733096811957652e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.628427287409724e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (-6.628427287409724e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (-5.927350278120907e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927350278120907e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927350278120907e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927350278120907e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627722041763966e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (-5.627722041763966e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (-4.837953298618804e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (-4.837953298618804e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (-3.570635481350178e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.570635481350178e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.328039660927245e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (-3.328039660927245e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (-3.086770893099915e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.086770893099915e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.086770893099915e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.086770893099915e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.447264788508917e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (-2.447264788508917e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (-2.3712704491457544e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.3712704491457544e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.3712704491457544e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.3712704491457544e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.1989637723559977e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (-2.1989637723559977e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (-1.9332121228344622e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.9332121228344622e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.9332121228344622e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.9332121228344622e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.839394338978518e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.839394338978518e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.839394338978518e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.839394338978518e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.2919458250413665e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.2919458250413665e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.8395657822106807e-08) [Y0 Z1 X2 X10 Z11 Y12]
+ (-1.8395657822106807e-08) [X0 Z1 Y2 Y10 Z11 X12]
+ (-1.8395657822106807e-08) [Y1 Z2 X3 X11 Z12 Y13]
+ (-1.8395657822106807e-08) [X1 Z2 Y3 Y11 Z12 X13]
+ (-1.0351487511196908e-09) [Y0 Z1 Z2 Z3 Y4 Z7]
+ (-1.0351487511196908e-09) [X0 Z1 Z2 Z3 X4 Z7]
+ (-1.0351487511196908e-09) [Y1 Z2 Z3 Z4 Y5 Z6]
+ (-1.0351487511196908e-09) [X1 Z2 Z3 Z4 X5 Z6]
+ (2.27020820534888e-08) [Y0 Z1 Z2 X3 X5 Y6]
+ (2.27020820534888e-08) [Y0 Z1 Z2 Y3 Y5 Y6]
+ (2.27020820534888e-08) [X0 Z1 Z2 X3 X5 X6]
+ (2.27020820534888e-08) [X0 Z1 Z2 Y3 Y5 X6]
+ (2.27020820534888e-08) [Y1 X2 X4 Z5 Z6 Y7]
+ (2.27020820534888e-08) [Y1 Y2 Y4 Z5 Z6 Y7]
+ (2.27020820534888e-08) [X1 X2 X4 Z5 Z6 X7]
+ (2.27020820534888e-08) [X1 Y2 Y4 Z5 Z6 X7]
+ (2.59281882930948e-08) [Y0 Z1 X2 X4 Z5 Y6]
+ (2.59281882930948e-08) [X0 Z1 Y2 Y4 Z5 X6]
+ (2.59281882930948e-08) [Y1 Z2 X3 X5 Z6 Y7]
+ (2.59281882930948e-08) [X1 Z2 Y3 Y5 Z6 X7]
+ (8.057465338370844e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057465338370844e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057465338370844e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057465338370844e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.64912954803848e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.64912954803848e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.64912954803848e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.64912954803848e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.1076529169380664e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (1.1076529169380664e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (1.1076529169380664e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (1.1076529169380664e-07) [X0 Z1 X2 X11 Z12 X13]
+ (1.1076529169380664e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (1.1076529169380664e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (1.1076529169380664e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (1.1076529169380664e-07) [X1 Z2 X3 X10 Z11 X12]
+ (1.3484968644599164e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (1.3484968644599164e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (1.3484968644599164e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (1.3484968644599164e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (1.380757926852207e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (1.380757926852207e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (1.380757926852207e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (1.380757926852207e-07) [X0 Z1 X2 X5 Z6 X7]
+ (1.380757926852207e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (1.380757926852207e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (1.380757926852207e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (1.380757926852207e-07) [X1 Z2 X3 X4 Z5 X6]
+ (1.6077787473870273e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (1.6077787473870273e-07) [X0 Z1 X2 X4 Z5 X6]
+ (1.6077787473870273e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (1.6077787473870273e-07) [X1 Z2 X3 X5 Z6 X7]
+ (1.8290428514668976e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.8290428514668976e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.8290428514668976e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.8290428514668976e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.1989637723559977e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (2.1989637723559977e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (2.447264788508917e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (2.447264788508917e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (3.2361834039496024e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.2361834039496024e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.2361834039496024e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.2361834039496024e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.328039660927245e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (3.328039660927245e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (3.570635481350178e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.570635481350178e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.837953298618804e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (4.837953298618804e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (5.287649488228877e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (5.287649488228877e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (5.287649488228877e-07) [X0 Z1 Z2 X3 X11 X12]
+ (5.287649488228877e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (5.287649488228877e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (5.287649488228877e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (5.287649488228877e-07) [X1 X2 X10 Z11 Z12 X13]
+ (5.287649488228877e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (5.627722041763966e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (5.627722041763966e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (6.395302405168502e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (6.395302405168502e-07) [X0 Z1 X2 X10 Z11 X12]
+ (6.395302405168502e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (6.395302405168502e-07) [X1 Z2 X3 X11 Z12 X13]
+ (6.57925898340136e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (6.57925898340136e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (6.57925898340136e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (6.57925898340136e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (6.628427287409724e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (6.628427287409724e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (7.956666925456609e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956666925456609e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.306342990563464e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.306342990563464e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.306342990563464e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.306342990563464e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.035792466932712e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (1.035792466932712e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (1.2282691033180075e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (1.2282691033180075e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (1.2393113883667716e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393113883667716e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393113883667716e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393113883667716e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.3304568589956375e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (1.3304568589956375e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (1.4548066691723403e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.4548066691723403e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.172638021765241e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (2.172638021765241e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (3.088245732762963e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.088245732762963e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.117366380422784e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.117366380422784e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.151295976920013e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.151295976920013e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.2111874430119924e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (3.2111874430119924e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (3.2111874430119924e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (3.2111874430119924e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (3.277438290294997e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (3.277438290294997e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (3.277438290294997e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (3.277438290294997e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (3.313017076081448e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (3.313017076081448e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (3.6102422563877214e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (3.6102422563877214e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (3.6102422563877214e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (3.6102422563877214e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (3.6945169301963975e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (3.6945169301963975e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (3.7695836101377935e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (3.7695836101377935e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (4.2531187796146855e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (4.2531187796146855e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (4.556473786485483e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (4.556473786485483e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (4.642979005748879e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.642979005748879e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.275783469220821e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.275783469220821e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.9741768991773345e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (5.9741768991773345e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (6.290019761453916e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (6.290019761453916e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (6.290019761453916e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (6.290019761453916e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (6.524204519092573e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (6.524204519092573e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (6.524204519092573e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (6.524204519092573e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (7.444267475658706e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (7.444267475658706e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (7.444267475658706e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (7.444267475658706e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (7.518288864773008e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (7.518288864773008e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (7.518288864773008e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (7.518288864773008e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (8.774724334654344e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (8.774724334654344e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (8.774724334654344e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (8.774724334654344e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (0.0002922256724506923) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.0002922256724506923) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.0002922256724506923) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.0002922256724506923) [X6 Z7 X8 X9 Z10 X11]
+ (0.0008144692870452469) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (0.0008144692870452469) [X2 Z3 Z4 Z5 X6 Z10]
+ (0.0008144692870452469) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (0.0008144692870452469) [X3 Z4 Z5 Z6 X7 Z11]
+ (0.0008533831050997363) [Y1 Z2 Z3 Y4 X5 X6]
+ (0.0008533831050997363) [X1 Z2 Z3 X4 Y5 Y6]
+ (0.0017278745823304962) [Y1 Z2 Z3 X4 X11 Y12]
+ (0.0017278745823304962) [X1 Z2 Z3 Y4 Y11 X12]
+ (0.0017991930083876967) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.0017991930083876967) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.0022939556230648817) [Y1 X2 X3 Z4 Z5 Y6]
+ (0.0022939556230648817) [X1 Y2 Y3 Z4 Z5 X6]
+ (0.002779040762887437) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (0.002779040762887437) [X1 Z2 Z3 Z4 Z5 X7]
+ (0.0034938003714923204) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (0.0034938003714923204) [X2 Z3 Z4 Z5 X6 Z13]
+ (0.0034938003714923204) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (0.0034938003714923204) [X3 Z4 Z5 Z6 X7 Z12]
+ (0.004158830716406776) [Y3 Z4 Z5 X6 X12 Y13]
+ (0.004158830716406776) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (0.004158830716406776) [X3 Z4 Z5 X6 X12 X13]
+ (0.004158830716406776) [X3 Z4 Z5 Y6 Y12 X13]
+ (0.004424843668496738) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424843668496738) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004575015188892521) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.004575015188892521) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668615266021862) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668615266021862) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.0046849202268751345) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.0046849202268751345) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005143382387697353) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (0.005143382387697353) [Y2 Z3 Y4 X5 Z6 X7]
+ (0.005143382387697353) [X2 Z3 X4 Y5 Z6 Y7]
+ (0.005143382387697353) [X2 Z3 X4 X5 Z6 X7]
+ (0.005368616111430509) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368616111430509) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368616111430509) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368616111430509) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.005652607315219922) [Y0 X1 X3 Z4 Z5 Y6]
+ (0.005652607315219922) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (0.005652607315219922) [X0 X1 X3 Z4 Z5 X6]
+ (0.005652607315219922) [X0 Y1 Y3 Z4 Z5 X6]
+ (0.005805121212342192) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (0.005805121212342192) [X2 Z3 Z4 Z5 X6 Z8]
+ (0.005805121212342192) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (0.005805121212342192) [X3 Z4 Z5 Z6 X7 Z9]
+ (0.007960839634230022) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.007960839634230022) [X4 Z5 X6 X10 Z11 X12]
+ (0.007960839634230022) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.007960839634230022) [X5 Z6 X7 X11 Z12 X13]
+ (0.008125248410129604) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125248410129604) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.008764858219392957) [Y2 Z3 Z4 X5 X11 Y12]
+ (0.008764858219392957) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (0.008764858219392957) [X2 Z3 Z4 X5 X11 X12]
+ (0.008764858219392957) [X2 Z3 Z4 Y5 Y11 X12]
+ (0.008764858219392957) [Y3 X4 X10 Z11 Z12 Y13]
+ (0.008764858219392957) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (0.008764858219392957) [X3 X4 X10 Z11 Z12 X13]
+ (0.008764858219392957) [X3 Y4 Y10 Z11 Z12 X13]
+ (0.008890680338671002) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.008890680338671002) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.008890680338671002) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.008890680338671002) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.010540434329250902) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.010540434329250902) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.010540434329250902) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.010540434329250902) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010959994608951158) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010959994608951158) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010959994608951158) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010959994608951158) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307208030054867) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307208030054867) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.011755995240392947) [Y3 Z4 Z5 X6 X8 Y9]
+ (0.011755995240392947) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (0.011755995240392947) [X3 Z4 Z5 X6 X8 X9]
+ (0.011755995240392947) [X3 Z4 Z5 Y6 Y8 X9]
+ (0.01756111645273514) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (0.01756111645273514) [X2 Z3 Z4 Z5 X6 Z9]
+ (0.01756111645273514) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (0.01756111645273514) [X3 Z4 Z5 Z6 X7 Z8]
+ (0.018266758578555923) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.018266758578555923) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.018266758578555923) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.018266758578555923) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.019020373875168896) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.019020373875168896) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.019020373875168896) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.019020373875168896) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.020175824956989698) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.020175824956989698) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.020175824956989698) [X4 Z5 Z6 X7 X11 X12]
+ (0.020175824956989698) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.020175824956989698) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.020175824956989698) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.020175824956989698) [X5 X6 X10 Z11 Z12 X13]
+ (0.020175824956989698) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.024388989986599406) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.024388989986599406) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.024388989986599406) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.024388989986599406) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.025104907970057714) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.025104907970057714) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.025104907970057714) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.025104907970057714) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.02599620626718282) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (0.02599620626718282) [X2 Z3 Z4 Z5 X6 Z7]
+ (0.027114878580413268) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (0.027114878580413268) [Z0 X2 Z3 Z4 Z5 X6]
+ (0.027114878580413268) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (0.027114878580413268) [Z1 X3 Z4 Z5 Z6 X7]
+ (0.03078744071862932) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.03078744071862932) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.03276748589563319) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (0.03276748589563319) [Z0 X3 Z4 Z5 Z6 X7]
+ (0.03276748589563319) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (0.03276748589563319) [Z1 X2 Z3 Z4 Z5 X6]
+ (0.05600713561694211) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.05600713561694211) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.05600713561694211) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.05600713561694211) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.05608449432299759) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.05608449432299759) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.05608449432299759) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.05608449432299759) [Z1 X7 Z8 Z9 Z10 X11]
+ (-0.04274326006287478) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-0.04274326006287478) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.04274326006287476) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (-0.04274326006287476) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (2.595081367984747e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.595081367984747e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.5950813679847476e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.5950813679847476e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (6.631261987598633e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (6.631261987598633e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (6.631261987598633e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (6.631261987598633e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.04764261360010173) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.04764261360010173) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.04764261360010173) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.04764261360010173) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.045879424030655) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-0.045879424030655) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.04171881404449267) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.04171881404449267) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.04171881404449267) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.04171881404449267) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.03956454804157256) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.03956454804157256) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.03956454804157256) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03956454804157256) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03931810723099392) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03931810723099392) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.03931810723099392) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03931810723099392) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.02563721280991771) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.02563721280991771) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.02563721280991771) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.02563721280991771) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.023145221653182155) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.023145221653182155) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.02252835424077098) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.02252835424077098) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.019257453001858965) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019257453001858965) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.01902831871828428) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.01902831871828428) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.01602466609553815) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.01602466609553815) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.015225659057132898) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.015225659057132898) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-0.014603742410740304) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (-0.014603742410740304) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (-0.01456447364080681) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.01456447364080681) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.011755995240392947) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (-0.011755995240392947) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (-0.011285144618318692) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.011285144618318692) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.009841802923827789) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.009841802923827789) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009612546714379559) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612546714379559) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612546714379559) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612546714379559) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.008469833341357768) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.008469833341357768) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.007306763969604764) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.007306763969604764) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005923799555609058) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.005923799555609058) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.005652607315219922) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005652607315219922) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (-0.005379929634057482) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (-0.005379929634057482) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (-0.005379929634057482) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (-0.005379929634057482) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (-0.005368616111430509) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368616111430509) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.005241543597047592) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (-0.005241543597047592) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (-0.005241543597047592) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (-0.005241543597047592) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (-0.00463697351649648) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (-0.00463697351649648) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (-0.00463697351649648) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (-0.00463697351649648) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (-0.0043110386075626275) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (-0.0043110386075626275) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (-0.0043110386075626275) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (-0.0043110386075626275) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (-0.004158830716406776) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (-0.004158830716406776) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (-0.003989845257549015) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (-0.003989845257549015) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (-0.003989845257549015) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (-0.003989845257549015) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (-0.002261970675218519) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (-0.002261970675218519) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (-0.002261970675218519) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (-0.002261970675218519) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (-0.002261970675218519) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (-0.002261970675218519) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (-0.002261970675218519) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (-0.002261970675218519) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (-0.0013038029824615732) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (-0.0013038029824615732) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (-0.0013038029824615732) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (-0.0013038029824615732) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (-0.0012803055951252018) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (-0.0012803055951252018) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (-0.0012803055951252018) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (-0.0012803055951252018) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (-0.0010435237104085013) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (-0.0010435237104085013) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (-0.0010435237104085013) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (-0.0010435237104085013) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (-0.0008533831050997363) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (-0.0008533831050997363) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (-0.0008533831050997363) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (-0.0008533831050997363) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (-0.00024644081057863796) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00024644081057863796) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.73587060554835e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.73587060554835e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-4.1838088461008255e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-4.1838088461008255e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-3.5443574329199236e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.5443574329199236e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.5443574329199236e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.5443574329199236e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.5443574329199236e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.5443574329199236e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.5443574329199236e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.5443574329199236e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.3342618576456123e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-3.3342618576456123e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-3.313017076081448e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-3.313017076081448e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-3.313017076081448e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-3.313017076081448e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.5224581964854225e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224581964854225e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224581964854225e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224581964854225e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224581964854225e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224581964854225e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224581964854225e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224581964854225e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.3304568589956375e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (-1.3304568589956375e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (-1.3304568589956375e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (-1.3304568589956375e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (-7.988467781666706e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988467781666706e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988467781666706e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988467781666706e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.189870494980767e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.189870494980767e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (-6.175164909318217e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175164909318217e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.471606066462955e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (-5.471606066462955e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (-4.561117001582616e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.561117001582616e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.561117001582616e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.561117001582616e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.523339412533612e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (-4.523339412533612e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (-3.427350780092764e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.427350780092764e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.427350780092764e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.427350780092764e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.086770893099915e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.086770893099915e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.3712704491457544e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.3712704491457544e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.839394338978518e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.839394338978518e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-1.7035434418440555e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035434418440555e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035434418440555e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (-1.7035434418440555e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (-9.208945092652274e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-9.208945092652274e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-9.208945092652274e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-9.208945092652274e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-8.057465338370844e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057465338370844e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (-6.772951846110688e-08) [Y1 Z2 Z3 X4 X7 Z8 Z9 Y10]
+ (-6.772951846110688e-08) [X1 Z2 Z3 Y4 Y7 Z8 Z9 X10]
+ (-3.226106239649198e-09) [Y0 Z1 Z2 Y3 X4 Z5 Z6 X7]
+ (-3.226106239649198e-09) [X0 Z1 Z2 X3 Y4 Z5 Z6 Y7]
+ (3.226106239649198e-09) [Y0 Z1 Z2 X3 X4 Z5 Z6 Y7]
+ (3.226106239649198e-09) [X0 Z1 Z2 Y3 Y4 Z5 Z6 X7]
+ (6.046790266970355e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (6.046790266970355e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (6.046790266970355e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (6.046790266970355e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (6.772951846110688e-08) [Y1 Z2 Z3 Y4 X7 Z8 Z9 X10]
+ (6.772951846110688e-08) [X1 Z2 Z3 X4 Y7 Z8 Z9 Y10]
+ (8.057465338370844e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057465338370844e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (1.839394338978518e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.839394338978518e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.3712704491457544e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.3712704491457544e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (2.88856458934136e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.88856458934136e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.88856458934136e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (2.88856458934136e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (3.086770893099915e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.086770893099915e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (3.328039660927245e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (3.328039660927245e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (3.328039660927245e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (3.328039660927245e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (4.523339412533612e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (4.523339412533612e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (5.471606066462955e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (5.471606066462955e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (6.175164909318217e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175164909318217e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189870494980767e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (7.189870494980767e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (7.867608645287783e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (7.867608645287783e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (7.867608645287783e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (7.867608645287783e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (1.2282691033180075e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (1.2282691033180075e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (1.2282691033180075e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (1.2282691033180075e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (1.628837780435024e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.628837780435024e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.628837780435024e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.628837780435024e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.6540900723898896e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6540900723898896e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6540900723898896e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6540900723898896e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.6893056831031013e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.6893056831031013e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.6893056831031013e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.6893056831031013e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (1.9429465313242425e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.9429465313242425e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.9429465313242425e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.9429465313242425e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.0110740275541654e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.0110740275541654e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.0110740275541654e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.0110740275541654e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.1031634784813387e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.1031634784813387e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.1031634784813387e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.1031634784813387e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.360947223921015e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (2.360947223921015e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (2.360947223921015e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (2.360947223921015e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (2.7455106547526024e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.7455106547526024e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.7455106547526024e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.7455106547526024e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.7455106547526024e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.7455106547526024e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.7455106547526024e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.7455106547526024e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.2117638795902586e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.2117638795902586e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.2117638795902586e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.2117638795902586e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.2117638795902586e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.2117638795902586e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.2117638795902586e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.2117638795902586e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.7695836101377935e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.7695836101377935e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.2531187796146855e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.2531187796146855e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.7287814717740235e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (4.7287814717740235e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (4.7287814717740235e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.7287814717740235e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.73457842342824e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (4.73457842342824e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (4.73457842342824e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (4.73457842342824e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (5.07140368043342e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (5.07140368043342e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (5.07140368043342e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (5.07140368043342e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (6.481752045653005e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (6.481752045653005e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.481752045653005e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (6.481752045653005e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (6.652106389837294e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (6.652106389837294e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (6.652106389837294e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (6.652106389837294e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.089728695694605e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.089728695694605e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (7.089728695694605e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (7.089728695694605e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (9.80598210385819e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (9.80598210385819e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (9.80598210385819e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (9.80598210385819e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (1.5316614231865406e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (1.5316614231865406e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (1.5316614231865406e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (1.5316614231865406e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.610337509639423e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (1.610337509639423e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.610337509639423e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (1.610337509639423e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (7.73587060554835e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.73587060554835e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00013838603700988843) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (0.00013838603700988843) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (0.00013838603700988843) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (0.00013838603700988843) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (0.00024644081057863796) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00024644081057863796) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0004458488204514853) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.0004458488204514853) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.0004458488204514853) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.0004458488204514853) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940157673918117) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940157673918117) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940157673918117) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940157673918117) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940157673918117) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940157673918117) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940157673918117) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940157673918117) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0009581676927569458) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (0.0009581676927569458) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (0.0009581676927569458) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (0.0009581676927569458) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (0.0009581676927569458) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (0.0009581676927569458) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (0.0009581676927569458) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (0.0009581676927569458) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (0.0022939556230648817) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (0.0022939556230648817) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (0.0022939556230648817) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (0.0022939556230648817) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (0.0026860422750874424) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (0.0026860422750874424) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (0.0026860422750874424) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (0.0026860422750874424) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (0.0027790407628874364) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (0.0027790407628874364) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (0.0032675148971541266) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (0.0032675148971541266) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (0.0032675148971541266) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (0.0032675148971541266) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (0.0033566679213712786) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (0.0033566679213712786) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (0.0033566679213712786) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (0.0033566679213712786) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (0.004158830716406776) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (0.004158830716406776) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (0.005114464086473346) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.005114464086473346) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.005114464086473346) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.005114464086473346) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.005114464086473346) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114464086473346) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114464086473346) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005114464086473346) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005262631033413675) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262631033413675) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262631033413675) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262631033413675) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368616111430509) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368616111430509) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005652607315219922) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (0.005652607315219922) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (0.005708479853865158) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708479853865158) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708479853865158) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708479853865158) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.005923799555609058) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.005923799555609058) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.007306763969604764) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.007306763969604764) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469833341357768) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.008469833341357768) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.009841802923827789) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.009841802923827789) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.011285144618318692) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.011285144618318692) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.011755995240392947) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (0.011755995240392947) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (0.01456447364080681) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.01456447364080681) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603742410740304) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (0.014603742410740304) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (0.015225659057132898) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.015225659057132898) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.01602466609553815) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.01602466609553815) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.01888899507741361) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (0.01888899507741361) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (0.01888899507741361) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.01888899507741361) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.01902831871828428) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.01902831871828428) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.019257453001858965) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (0.019257453001858965) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.021433980116924038) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.021433980116924038) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.021433980116924038) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.021433980116924038) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.02428203162338485) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (0.02428203162338485) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (0.024755507980799375) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.024755507980799375) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.024755507980799375) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.024755507980799375) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.028730798001241398) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.028730798001241398) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.028730798001241398) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.028730798001241398) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.029903813458281806) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.029903813458281806) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.029903813458281806) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.029903813458281806) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.035608400352296385) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.035608400352296385) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.03935925039153968) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.03935925039153968) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.03935925039153968) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.03935925039153968) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.36937137554404426) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.36937137554404426) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.36937137554404426) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.36937137554404426) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.28164335753419334) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.28164335753419334) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.28164335753419334) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.28164335753419334) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.09065142344941732) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.09065142344941732) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.09065142344941732) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.09065142344941732) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684736029505066) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684736029505066) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684736029505066) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684736029505066) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.058592151795449784) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.058592151795449784) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.034903304270627294) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.034903304270627294) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.034903304270627294) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.034903304270627294) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591832088221688) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.024591832088221688) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591832088221688) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.024591832088221688) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.02314522165318215) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.02314522165318215) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.02252835424077098) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.02252835424077098) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019299499858005314) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.019299499858005314) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019299499858005314) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.019299499858005314) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019299499858005314) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019299499858005314) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.019299499858005314) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019299499858005314) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.016024666095538154) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.016024666095538154) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.016024666095538154) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.016024666095538154) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.014603742410740304) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (-0.014603742410740304) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (-0.014603742410740304) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (-0.014603742410740304) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (-0.010757524201208565) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010757524201208565) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010757524201208565) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010757524201208565) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010715477345062206) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010715477345062206) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010715477345062206) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010715477345062206) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311472182405612) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311472182405612) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311472182405612) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311472182405612) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005408970758385603) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.005408970758385603) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.005408970758385603) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.005408970758385603) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.005286569056779067) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.005286569056779067) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.005286569056779067) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.005286569056779067) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.004767276644727216) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.004767276644727216) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.004767276644727216) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.004767276644727216) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.004668615266021862) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668615266021862) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.0038764821957279166) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (-0.0038764821957279166) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (-0.0038040631543666357) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0038040631543666357) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0038040631543666357) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0038040631543666357) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484154579343921) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484154579343921) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0033566679213712786) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (-0.0033566679213712786) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (-0.0032675148971541266) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (-0.0032675148971541266) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (-0.0024464634226879947) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.0024464634226879947) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.0024464634226879947) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.0024464634226879947) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.0017278745823304962) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.0017278745823304962) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.0016407591167029436) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0016407591167029436) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0015324885626576876) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.0015324885626576876) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.0015324885626576876) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.0015324885626576876) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.0007870893705571727) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0007870893705571727) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007156737069596011) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0007156737069596011) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007156737069596011) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0007156737069596011) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0005192924120518527) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.0005192924120518527) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.00019401030606793558) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00019401030606793558) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.00018787486137456777) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.00018787486137456777) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.00018787486137456777) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.00018787486137456777) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.00013838603700988843) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.00013838603700988843) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-4.204685614635261e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-4.204685614635261e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-4.204685614635261e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.204685614635261e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-5.07140368043342e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-5.07140368043342e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-3.151295976920013e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.151295976920013e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.088245732762963e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.088245732762963e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.988412563025185e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.988412563025185e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.87424857188298e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.87424857188298e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.360947223921015e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-2.360947223921015e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3001958903322529e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3001958903322529e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-7.867608645287783e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867608645287783e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.996951815028816e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.996951815028816e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.996951815028816e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.996951815028816e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.996951815028816e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.996951815028816e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.996951815028816e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.996951815028816e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.88856458934136e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.88856458934136e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.686321793518272e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.686321793518272e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035434418440555e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7035434418440555e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.208945092652274e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.208945092652274e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.204004648471256e-08) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (-3.204004648471256e-08) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (-3.204004648471256e-08) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (-3.204004648471256e-08) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (3.568947197624524e-08) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (3.568947197624524e-08) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (3.568947197624524e-08) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (3.568947197624524e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (3.568947197624524e-08) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.568947197624524e-08) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (3.568947197624524e-08) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (3.568947197624524e-08) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.706834937775922e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.706834937775922e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.706834937775922e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.706834937775922e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.379737463926672e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (8.379737463926672e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.379737463926672e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (8.379737463926672e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.379737463926672e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (8.379737463926672e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (8.379737463926672e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (8.379737463926672e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (9.208945092652274e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.208945092652274e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.0716845394068115e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0716845394068115e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0716845394068115e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0716845394068115e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.1782130942876391e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.1782130942876391e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.1782130942876391e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.1782130942876391e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.7035434418440555e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.7035434418440555e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.2498976337115268e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.2498976337115268e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.2498976337115268e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.2498976337115268e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.686321793518272e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.686321793518272e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.88856458934136e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.88856458934136e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.3766863874182226e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (3.3766863874182226e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (3.3766863874182226e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (3.3766863874182226e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (3.3766863874182226e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (3.3766863874182226e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (3.3766863874182226e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (3.3766863874182226e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (3.5682004928723724e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (3.5682004928723724e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (3.5682004928723724e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (3.5682004928723724e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.092161945611368e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.092161945611368e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.092161945611368e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.092161945611368e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.092161945611368e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.092161945611368e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.092161945611368e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.092161945611368e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.4490566653755144e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.4490566653755144e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.4490566653755144e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.4490566653755144e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.769457130219862e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.769457130219862e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.769457130219862e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.769457130219862e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246849448732753e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246849448732753e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246849448732753e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246849448732753e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246849448732753e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246849448732753e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246849448732753e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246849448732753e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.560553988758105e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.560553988758105e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (7.560553988758105e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (7.560553988758105e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (7.560553988758105e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.560553988758105e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.560553988758105e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.560553988758105e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.867608645287783e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (7.867608645287783e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.900025799956375e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (7.900025799956375e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (7.900025799956375e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (7.900025799956375e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (8.027844241373502e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (8.027844241373502e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (8.027844241373502e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.027844241373502e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (8.09153990539953e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (8.09153990539953e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (8.09153990539953e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (8.09153990539953e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (8.09153990539953e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (8.09153990539953e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (8.09153990539953e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (8.09153990539953e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (8.398527735152212e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (8.398527735152212e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (8.398527735152212e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (8.398527735152212e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.1468226292817583e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (1.1468226292817583e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (1.1468226292817583e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (1.1468226292817583e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3001958903322529e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3001958903322529e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.360947223921015e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.360947223921015e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.87424857188298e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.87424857188298e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.8836531667268336e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.8836531667268336e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.9473314601272463e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9473314601272463e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.9473314601272463e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9473314601272463e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.988412563025185e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.988412563025185e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.088245732762963e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.088245732762963e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.151295976920013e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.151295976920013e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.846190882290067e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846190882290067e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846190882290067e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846190882290067e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.07140368043342e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (5.07140368043342e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (5.105462424440441e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.105462424440441e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.105462424440441e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.105462424440441e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146386772621886e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146386772621886e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146386772621886e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146386772621886e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159294474401158e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159294474401158e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159294474401158e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159294474401158e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.4279266537514675e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.4279266537514675e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.4279266537514675e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.4279266537514675e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935744023153949e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935744023153949e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935744023153949e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935744023153949e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253185165209452e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.253185165209452e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.97971099632342e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.97971099632342e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.97971099632342e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.97971099632342e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (7.141566359757172e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (7.141566359757172e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (7.141566359757172e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.141566359757172e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00013838603700988843) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.00013838603700988843) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.00019401030606793558) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (0.00019401030606793558) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.00024644081057863796) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024644081057863796) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00024644081057863796) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024644081057863796) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192924120518527) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.0005192924120518527) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.0007870893705571727) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.0007870893705571727) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.0014528842553283748) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0014528842553283748) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0014528842553283748) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0014528842553283748) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0016407591167029436) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0016407591167029436) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0017278745823304962) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.0017278745823304962) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.002141348964746001) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (0.002141348964746001) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (0.0032675148971541266) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (0.0032675148971541266) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (0.0033566679213712786) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (0.0033566679213712786) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (0.003484154579343921) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.003484154579343921) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0038764821957279166) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (0.0038764821957279166) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (0.004668615266021862) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668615266021862) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005923799555609058) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923799555609058) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.005923799555609058) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923799555609058) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.008469833341357768) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.008469833341357768) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.008469833341357768) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.008469833341357768) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.008541975656796756) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.008541975656796756) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.008541975656796756) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (0.008541975656796756) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.008541975656796756) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.008541975656796756) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.008541975656796756) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.008541975656796756) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.008826387567793888) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.008826387567793888) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.008826387567793888) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.008826387567793888) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.00984180292382779) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00984180292382779) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00984180292382779) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00984180292382779) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.017091621922241385) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.017091621922241385) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.017091621922241385) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.017091621922241385) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.019538085344929377) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.019538085344929377) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.019538085344929377) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.019538085344929377) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.02428203162338485) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (0.02428203162338485) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (0.035608400352296385) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.035608400352296385) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.06752398179965079) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.06752398179965079) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.06752398179965079) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.06752398179965079) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.07635036936744467) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.07635036936744467) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.07635036936744467) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07635036936744467) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.07165056250039936) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07165056250039936) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07165056250039935) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07165056250039935) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.775872110146414e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.775872110146414e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.775872110146414e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.775872110146414e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.058592151795449784) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.058592151795449784) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.019257453001858965) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.019257453001858965) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311472182405612) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311472182405612) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008826387567793888) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.008826387567793888) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.007597461779080108) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.007597461779080108) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.007597461779080108) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.007597461779080108) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733568640016672) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733568640016672) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733568640016672) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733568640016672) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733568640016672) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733568640016672) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733568640016672) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733568640016672) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348047718407357) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005348047718407357) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005348047718407357) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348047718407357) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.004220835998336399) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.004220835998336399) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.004220835998336399) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.004220835998336399) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.0038764821957279166) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (-0.0038764821957279166) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (-0.0038764821957279166) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (-0.0038764821957279166) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (-0.0038040631543666357) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0038040631543666357) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0024464634226879943) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.0024464634226879943) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.0023949671540605677) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0023949671540605677) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0023949671540605677) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0023949671540605677) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0023949671540605677) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0023949671540605677) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0023949671540605677) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0023949671540605677) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022494140606727515) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0022494140606727515) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022494140606727515) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0022494140606727515) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.002200956847992632) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.002200956847992632) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.002200956847992632) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.002200956847992632) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638931390634365) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638931390634365) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638931390634365) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638931390634365) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638931390634365) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638931390634365) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0018638931390634365) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638931390634365) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012366559235474643) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.0012366559235474643) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.0012366559235474643) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.0012366559235474643) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.0011726297842397115) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0011726297842397115) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0011726297842397115) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0011726297842397115) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0007870893705571727) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0007870893705571727) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870893705571727) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0007870893705571727) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0005192924120518528) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.0005192924120518528) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.0005192924120518528) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.0005192924120518528) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-1.146285103010531e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.146285103010531e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.87424857188298e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.87424857188298e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.87424857188298e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.87424857188298e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3001958903322529e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3001958903322529e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3001958903322529e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3001958903322529e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.0444741668265267e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.0444741668265267e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.0444741668265267e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.0444741668265267e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.955903604495551e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.955903604495551e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.955903604495551e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.955903604495551e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105341140040869e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105341140040869e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105341140040869e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105341140040869e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661200360698417e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661200360698417e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661200360698417e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661200360698417e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540204281845319e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540204281845319e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189870494980767e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.189870494980767e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.876530414497216e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876530414497216e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876530414497216e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876530414497216e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175164909318217e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175164909318217e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.523339412533612e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.523339412533612e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.076662715415079e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.076662715415079e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.076662715415079e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.076662715415079e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.0133988589987735e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0133988589987735e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.9045373864199477e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.9045373864199477e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.9045373864199477e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.9045373864199477e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.666679758813343e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.666679758813343e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.666679758813343e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.666679758813343e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505624644550887e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.8505624644550887e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.846699462044538e-08) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-7.846699462044538e-08) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-6.772951846110688e-08) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (-6.772951846110688e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (-4.099829565998051e-08) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-4.099829565998051e-08) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.099829565998051e-08) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.099829565998051e-08) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (6.772951846110688e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (6.772951846110688e-08) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (7.846699462044538e-08) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (7.846699462044538e-08) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (1.6570092716143732e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6570092716143732e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6570092716143732e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6570092716143732e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.8505624644550887e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.8505624644550887e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.686321793518272e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.686321793518272e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.686321793518272e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.686321793518272e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0133988589987735e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.0133988589987735e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.523339412533612e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.523339412533612e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.6704081306207446e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.6704081306207446e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.6704081306207446e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.6704081306207446e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175164909318217e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175164909318217e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189870494980767e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.189870494980767e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.540204281845319e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540204281845319e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949307746245383e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949307746245383e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.792463835852362e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.792463835852362e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.792463835852362e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.792463835852362e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.8836531667268336e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.8836531667268336e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.988412563025185e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.988412563025185e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.988412563025185e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.988412563025185e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253185165209452e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253185165209452e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4016916461539249e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4016916461539249e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4016916461539249e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4016916461539249e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809380297391614e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.5809380297391614e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809380297391614e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.5809380297391614e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0010283270637529214) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0010283270637529214) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0010283270637529214) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0010283270637529214) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.001222337369820857) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.001222337369820857) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.001222337369820857) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (0.001222337369820857) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.001222337369820857) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.001222337369820857) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.001222337369820857) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.001222337369820857) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0016407591167029436) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0016407591167029436) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0016407591167029436) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0016407591167029436) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.002141348964746001) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (0.002141348964746001) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (0.0024464634226879943) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.0024464634226879943) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.002984180074788933) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.002984180074788933) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.002984180074788933) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.002984180074788933) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.0038040631543666357) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0038040631543666357) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.008826387567793888) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.008826387567793888) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.010311472182405612) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311472182405612) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257453001858965) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257453001858965) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.39866536227402e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.39866536227402e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.3986653622740196e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3986653622740196e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003484154579343921) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003484154579343921) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984180074788933) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.002984180074788933) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.00019401030606793558) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00019401030606793558) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.146285103010531e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.146285103010531e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.792463835852362e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.792463835852362e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540204281845319e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540204281845319e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540204281845319e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540204281845319e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.8505624644550887e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505624644550887e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505624644550887e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505624644550887e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.846699462044538e-08) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-7.846699462044538e-08) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-7.846699462044538e-08) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-7.846699462044538e-08) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-4.099829565998051e-08) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-4.099829565998051e-08) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.099829565998051e-08) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-4.099829565998051e-08) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0133988589987735e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0133988589987735e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0133988589987735e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0133988589987735e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949307746245383e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949307746245383e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.792463835852362e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.792463835852362e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019401030606793558) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00019401030606793558) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002984180074788933) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.002984180074788933) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.003484154579343921) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.003484154579343921) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
  (-73.13873149597377) [I0]
+ (-0.18066757507002823) [Z6]
+ (-0.18066757507002806) [Z7]
+ (-0.15961443583725843) [Z4]
+ (-0.15961443583725832) [Z5]
+ (0.17419986612126878) [Z3]
+ (0.1741998661212688) [Z2]
+ (0.22757328814464498) [Z1]
+ (0.2275732881446453) [Z0]
+ (-7.95422451894609e-06) [Y5 Y7]
+ (-7.95422451894609e-06) [X5 X7]
+ (8.194104942911562e-06) [Y4 Y6]
+ (8.194104942911562e-06) [X4 X6]
+ (0.11270381859112809) [Z4 Z6]
+ (0.11270381859112809) [Z5 Z7]
+ (0.11952441016888041) [Z0 Z4]
+ (0.11952441016888041) [Z1 Z5]
+ (0.13401737372646577) [Z0 Z6]
+ (0.13401737372646577) [Z1 Z7]
+ (0.1373494221014993) [Z0 Z5]
+ (0.1373494221014993) [Z1 Z4]
+ (0.13766859133617518) [Z2 Z4]
+ (0.13766859133617518) [Z3 Z5]
+ (0.14138903590137478) [Z4 Z7]
+ (0.14138903590137478) [Z5 Z6]
+ (0.14722930783262234) [Z2 Z5]
+ (0.14722930783262234) [Z3 Z4]
+ (0.1492634706003103) [Z4 Z5]
+ (0.14973497005432304) [Z2 Z6]
+ (0.14973497005432304) [Z3 Z7]
+ (0.15138342699108018) [Z0 Z7]
+ (0.15138342699108018) [Z1 Z6]
+ (0.1543576006529091) [Z6 Z7]
+ (0.15582280685855676) [Z2 Z7]
+ (0.15582280685855676) [Z3 Z6]
+ (0.16756669356174117) [Z0 Z2]
+ (0.16756669356174117) [Z1 Z3]
+ (0.18144009362937702) [Z0 Z3]
+ (0.18144009362937702) [Z1 Z2]
+ (0.1939257433497494) [Z0 Z1]
+ (0.2200397724028862) [Z2 Z3]
+ (7.038023723793356e-06) [Y4 Z5 Y6]
+ (7.038023723793356e-06) [X4 Z5 X6]
+ (7.038023723793356e-06) [Y5 Z6 Y7]
+ (7.038023723793356e-06) [X5 Z6 X7]
+ (-0.02868521731024671) [Y4 Y5 X6 X7]
+ (-0.02868521731024671) [X4 X5 Y6 Y7]
+ (-0.01782501193261893) [Y0 Y1 X4 X5]
+ (-0.01782501193261893) [X0 X1 Y4 Y5]
+ (-0.01736605326461441) [Y0 Y1 X6 X7]
+ (-0.01736605326461441) [X0 X1 Y6 Y7]
+ (-0.013873400067635853) [Y0 Y1 X2 X3]
+ (-0.013873400067635853) [X0 X1 Y2 Y3]
+ (-0.009560716496447148) [Y2 Y3 X4 X5]
+ (-0.009560716496447148) [X2 X3 Y4 Y5]
+ (-0.00608783680423372) [Y2 Y3 X6 X7]
+ (-0.00608783680423372) [X2 X3 Y6 Y7]
+ (-0.00029222567245307897) [Y1 Y2 X3 X4]
+ (-0.00029222567245307897) [X1 X2 Y3 Y4]
+ (-7.95422451894609e-06) [Y4 Z5 Y6 Z7]
+ (-7.95422451894609e-06) [X4 Z5 X6 Z7]
+ (-6.628427184628442e-07) [Y2 X3 X5 Y6]
+ (-6.628427184628442e-07) [Y2 Y3 Y5 Y6]
+ (-6.628427184628442e-07) [X2 X3 X5 X6]
+ (-6.628427184628442e-07) [X2 Y3 Y5 X6]
+ (9.344970245128703e-07) [Z2 Y5 Z6 Y7]
+ (9.344970245128703e-07) [Z2 X5 Z6 X7]
+ (9.344970245128703e-07) [Z3 Y4 Z5 Y6]
+ (9.344970245128703e-07) [Z3 X4 Z5 X6]
+ (1.0357924726247733e-06) [Y0 X1 X5 Y6]
+ (1.0357924726247733e-06) [Y0 Y1 Y5 Y6]
+ (1.0357924726247733e-06) [X0 X1 X5 X6]
+ (1.0357924726247733e-06) [X0 Y1 Y5 X6]
+ (1.5973397429757145e-06) [Z2 Y4 Z5 Y6]
+ (1.5973397429757145e-06) [Z2 X4 Z5 X6]
+ (1.5973397429757145e-06) [Z3 Y5 Z6 Y7]
+ (1.5973397429757145e-06) [Z3 X5 Z6 X7]
+ (1.8551374852838337e-06) [Z0 Y4 Z5 Y6]
+ (1.8551374852838337e-06) [Z0 X4 Z5 X6]
+ (1.8551374852838337e-06) [Z1 Y5 Z6 Y7]
+ (1.8551374852838337e-06) [Z1 X5 Z6 X7]
+ (2.8909299579081733e-06) [Z0 Y5 Z6 Y7]
+ (2.8909299579081733e-06) [Z0 X5 Z6 X7]
+ (2.8909299579081733e-06) [Z1 Y4 Z5 Y6]
+ (2.8909299579081733e-06) [Z1 X4 Z5 X6]
+ (8.194104942911562e-06) [Z4 Y5 Z6 Y7]
+ (8.194104942911562e-06) [Z4 X5 Z6 X7]
+ (0.00029222567245307897) [Y1 X2 X3 Y4]
+ (0.00029222567245307897) [X1 Y2 Y3 X4]
+ (0.00608783680423372) [Y2 X3 X6 Y7]
+ (0.00608783680423372) [X2 Y3 Y6 X7]
+ (0.009560716496447148) [Y2 X3 X4 Y5]
+ (0.009560716496447148) [X2 Y3 Y4 X5]
+ (0.011307208030047157) [Y1 Z2 Z3 Y5]
+ (0.011307208030047157) [X1 Z2 Z3 X5]
+ (0.013873400067635853) [Y0 X1 X2 Y3]
+ (0.013873400067635853) [X0 Y1 Y2 X3]
+ (0.01736605326461441) [Y0 X1 X6 Y7]
+ (0.01736605326461441) [X0 Y1 Y6 X7]
+ (0.01782501193261893) [Y0 X1 X4 Y5]
+ (0.01782501193261893) [X0 Y1 Y4 X5]
+ (0.02868521731024671) [Y4 X5 X6 Y7]
+ (0.02868521731024671) [X4 Y5 Y6 X7]
+ (0.029812299601141762) [Y0 Z1 Z2 Y4]
+ (0.029812299601141762) [X0 Z1 Z2 X4]
+ (0.029812299601141762) [Y1 Z3 Z4 Y5]
+ (0.029812299601141762) [X1 Z3 Z4 X5]
+ (0.030104525273594845) [Y0 Z1 Z3 Y4]
+ (0.030104525273594845) [X0 Z1 Z3 X4]
+ (0.030104525273594845) [Y1 Z2 Z4 Y5]
+ (0.030104525273594845) [X1 Z2 Z4 X5]
+ (0.030787440718605365) [Y0 Z2 Z3 Y4]
+ (0.030787440718605365) [X0 Z2 Z3 X4]
+ (0.04375171612137386) [Y1 Z2 Z3 Z4 Y5]
+ (0.04375171612137386) [X1 Z2 Z3 Z4 X5]
+ (0.04375171612137387) [Y0 Z1 Z2 Z3 Y4]
+ (0.04375171612137387) [X0 Z1 Z2 Z3 X4]
+ (-0.014564473640792273) [Y1 Z2 Z3 X4 X6 Y7]
+ (-0.014564473640792273) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-0.014564473640792273) [X1 Z2 Z3 X4 X6 X7]
+ (-0.014564473640792273) [X1 Z2 Z3 Y4 Y6 X7]
+ (-4.1838087621193926e-06) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-4.1838087621193926e-06) [X1 Z2 Z3 Z4 Z5 X7]
+ (-3.3130170473804482e-06) [Y1 Z2 Z3 X4 X5 Y6]
+ (-3.3130170473804482e-06) [X1 Z2 Z3 Y4 Y5 X6]
+ (-1.0357924726247733e-06) [Y0 X1 X4 Z5 Z6 Y7]
+ (-1.0357924726247733e-06) [X0 Y1 Y4 Z5 Z6 X7]
+ (-6.628427184628442e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-6.628427184628442e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-3.3280396354457833e-07) [Y1 Y2 X3 Z4 Z5 X6]
+ (-3.3280396354457833e-07) [X1 X2 Y3 Z4 Z5 Y6]
+ (3.3280396354457833e-07) [Y1 X2 X3 Z4 Z5 Y6]
+ (3.3280396354457833e-07) [X1 Y2 Y3 Z4 Z5 X6]
+ (6.628427184628442e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (6.628427184628442e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (1.0357924726247733e-06) [Y0 Y1 X4 Z5 Z6 X7]
+ (1.0357924726247733e-06) [X0 X1 Y4 Z5 Z6 Y7]
+ (3.211187410890985e-06) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (3.211187410890985e-06) [X0 Z1 Z2 Z3 Z5 X6]
+ (3.211187410890985e-06) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (3.211187410890985e-06) [X1 Z2 Z3 Z4 Z6 X7]
+ (3.277438248835106e-06) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (3.277438248835106e-06) [X0 Z1 Z2 Z4 Z5 X6]
+ (3.277438248835106e-06) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (3.277438248835106e-06) [X1 Z3 Z4 Z5 Z6 X7]
+ (3.3130170473804482e-06) [Y1 Z2 Z3 Y4 X5 X6]
+ (3.3130170473804482e-06) [X1 Z2 Z3 X4 Y5 Y6]
+ (3.610242212379684e-06) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (3.610242212379684e-06) [X0 Z1 Z3 Z4 Z5 X6]
+ (3.610242212379684e-06) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (3.610242212379684e-06) [X1 Z2 Z4 Z5 Z6 X7]
+ (3.769583571113888e-06) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (3.769583571113888e-06) [X0 Z2 Z3 Z4 Z5 X6]
+ (6.524204458272301e-06) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (6.524204458272301e-06) [X0 Z1 Z2 Z3 Z4 X6]
+ (6.524204458272301e-06) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (6.524204458272301e-06) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.00029222567245307897) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (0.00029222567245307897) [Y0 Z1 Y2 X3 Z4 X5]
+ (0.00029222567245307897) [X0 Z1 X2 Y3 Z4 Y5]
+ (0.00029222567245307897) [X0 Z1 X2 X3 Z4 X5]
+ (0.010540434329253778) [Y0 Z1 Z2 Z3 Y4 Z7]
+ (0.010540434329253778) [X0 Z1 Z2 Z3 X4 Z7]
+ (0.010540434329253778) [Y1 Z2 Z3 Z4 Y5 Z6]
+ (0.010540434329253778) [X1 Z2 Z3 Z4 X5 Z6]
+ (0.011307208030047157) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (0.011307208030047157) [X0 Z1 Z2 Z3 X4 Z5]
+ (0.025104907970046053) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (0.025104907970046053) [X0 Z1 Z2 Z3 X4 Z6]
+ (0.025104907970046053) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (0.025104907970046053) [X1 Z2 Z3 Z4 X5 Z7]
+ (0.030787440718605365) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (0.030787440718605365) [Z0 X1 Z2 Z3 Z4 X5]
+ (5.1056810192636525e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (5.1056810192636525e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (5.1056810192636525e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (5.1056810192636525e-06) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.014564473640792273) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-0.014564473640792273) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-4.1838087621193926e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-4.1838087621193926e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-3.3130170473804482e-06) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (-3.3130170473804482e-06) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (-3.3130170473804482e-06) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (-3.3130170473804482e-06) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (3.3280396354457833e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (3.3280396354457833e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (3.3280396354457833e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (3.3280396354457833e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (3.769583571113888e-06) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (3.769583571113888e-06) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (0.014564473640792273) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (0.014564473640792273) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
  (-46.46390678868897) [I0]
+ (0.7829661725950188) [Z10]
+ (0.782966172595019) [Z11]
+ (0.808458196172055) [Z13]
+ (0.8084581961720552) [Z12]
+ (1.203440228914567) [Z4]
+ (1.2034402289145671) [Z5]
+ (1.309686298861554) [Z7]
+ (1.3096862988615543) [Z6]
+ (1.3693525634718235) [Z8]
+ (1.3693525634718235) [Z9]
+ (1.653894222683163) [Z2]
+ (1.653894222683163) [Z3]
+ (12.412630742111784) [Z0]
+ (12.412630742111784) [Z1]
+ (-8.194261373271547e-06) [Y10 Y12]
+ (-8.194261373271547e-06) [X10 X12]
+ (-1.8540608579184492e-06) [Y5 Y7]
+ (-1.8540608579184492e-06) [X5 X7]
+ (-7.764994120768726e-07) [Y3 Y5]
+ (-7.764994120768726e-07) [X3 X5]
+ (-5.929765814699853e-07) [Y4 Y6]
+ (-5.929765814699853e-07) [X4 X6]
+ (1.6021167407057144e-06) [Y2 Y4]
+ (1.6021167407057144e-06) [X2 X4]
+ (7.954413177246877e-06) [Y11 Y13]
+ (7.954413177246877e-06) [X11 X13]
+ (0.0032769719312315372) [Y1 Y3]
+ (0.0032769719312315372) [X1 X3]
+ (0.10433064780651312) [Y0 Y2]
+ (0.10433064780651312) [X0 X2]
+ (0.11270386920332179) [Z10 Z12]
+ (0.11270386920332179) [Z11 Z13]
+ (0.11383573679388637) [Z4 Z12]
+ (0.11383573679388637) [Z5 Z13]
+ (0.11952438964682723) [Z6 Z10]
+ (0.11952438964682723) [Z7 Z11]
+ (0.12489990917237585) [Z4 Z10]
+ (0.12489990917237585) [Z5 Z11]
+ (0.12495807739503174) [Z2 Z4]
+ (0.12495807739503174) [Z3 Z5]
+ (0.12799502492468348) [Z2 Z10]
+ (0.12799502492468348) [Z3 Z11]
+ (0.13401715261963745) [Z6 Z12]
+ (0.13401715261963745) [Z7 Z13]
+ (0.1370119167404079) [Z4 Z6]
+ (0.1370119167404079) [Z5 Z7]
+ (0.1373495306426128) [Z6 Z11]
+ (0.1373495306426128) [Z7 Z10]
+ (0.13739104762683235) [Z2 Z6]
+ (0.13739104762683235) [Z3 Z7]
+ (0.13766872645852554) [Z8 Z10]
+ (0.13766872645852554) [Z9 Z11]
+ (0.1401128986535478) [Z2 Z12]
+ (0.1401128986535478) [Z3 Z13]
+ (0.14138905291942813) [Z10 Z13]
+ (0.14138905291942813) [Z11 Z12]
+ (0.14257997712485745) [Z4 Z11]
+ (0.14257997712485745) [Z5 Z10]
+ (0.14722943218766174) [Z8 Z11]
+ (0.14722943218766174) [Z9 Z10]
+ (0.14899430575065564) [Z4 Z7]
+ (0.14899430575065564) [Z5 Z6]
+ (0.14926355147388937) [Z10 Z11]
+ (0.14960702684445298) [Z4 Z8]
+ (0.14960702684445298) [Z5 Z9]
+ (0.15071408121008262) [Z2 Z8]
+ (0.15071408121008262) [Z3 Z9]
+ (0.1513832716142886) [Z6 Z13]
+ (0.1513832716142886) [Z7 Z12]
+ (0.15215040708869032) [Z4 Z13]
+ (0.15215040708869032) [Z5 Z12]
+ (0.1533796824331412) [Z2 Z11]
+ (0.1533796824331412) [Z3 Z10]
+ (0.15435748657223627) [Z12 Z13]
+ (0.15569010671752426) [Z2 Z13]
+ (0.15569010671752426) [Z3 Z12]
+ (0.15676396176430996) [Z4 Z9]
+ (0.15676396176430996) [Z5 Z8]
+ (0.16079764534838534) [Z2 Z5]
+ (0.16079764534838534) [Z3 Z4]
+ (0.16853486561579917) [Z2 Z7]
+ (0.16853486561579917) [Z3 Z6]
+ (0.18143991440303986) [Z6 Z9]
+ (0.18143991440303986) [Z7 Z8]
+ (0.18189085790751258) [Z2 Z3]
+ (0.18690820476912495) [Z2 Z9]
+ (0.18690820476912495) [Z3 Z8]
+ (0.19299723935364177) [Z0 Z10]
+ (0.19299723935364177) [Z1 Z11]
+ (0.19392534613270435) [Z6 Z7]
+ (0.19661770890342148) [Z0 Z4]
+ (0.19661770890342148) [Z1 Z5]
+ (0.19936354537360831) [Z0 Z5]
+ (0.19936354537360831) [Z1 Z4]
+ (0.20072866460441718) [Z0 Z11]
+ (0.20072866460441718) [Z1 Z10]
+ (0.21102659849791516) [Z0 Z12]
+ (0.21102659849791516) [Z1 Z13]
+ (0.2163103749863181) [Z0 Z13]
+ (0.2163103749863181) [Z1 Z12]
+ (0.22003977334376124) [Z8 Z9]
+ (0.23671080783830278) [Z0 Z2]
+ (0.23671080783830278) [Z1 Z3]
+ (0.24164663936017375) [Z0 Z6]
+ (0.24164663936017375) [Z1 Z7]
+ (0.24853483371314444) [Z0 Z7]
+ (0.24853483371314444) [Z1 Z6]
+ (0.2512944567459152) [Z0 Z3]
+ (0.2512944567459152) [Z1 Z2]
+ (0.2723251830660572) [Z0 Z8]
+ (0.2723251830660572) [Z1 Z9]
+ (0.2788345442672344) [Z0 Z9]
+ (0.2788345442672344) [Z1 Z8]
+ (1.1861763734860484) [Z0 Z1]
+ (-1.2260484988598762e-05) [Y4 Z5 Y6]
+ (-1.2260484988598762e-05) [X4 Z5 X6]
+ (-1.2260484988598762e-05) [Y5 Z6 Y7]
+ (-1.2260484988598762e-05) [X5 Z6 X7]
+ (-1.0722312159042325e-05) [Y10 Z11 Y12]
+ (-1.0722312159042325e-05) [X10 Z11 X12]
+ (-1.0722312159042322e-05) [Y11 Z12 Y13]
+ (-1.0722312159042322e-05) [X11 Z12 X13]
+ (-3.887051672605772e-06) [Y2 Z3 Y4]
+ (-3.887051672605772e-06) [X2 Z3 X4]
+ (-3.887051672605772e-06) [Y3 Z4 Y5]
+ (-3.887051672605772e-06) [X3 Z4 X5]
+ (0.12507032579771896) [Y0 Z1 Y2]
+ (0.12507032579771896) [X0 Z1 X2]
+ (0.12507032579771896) [Y1 Z2 Y3]
+ (0.12507032579771896) [X1 Z2 X3]
+ (-0.03831467029480395) [Y4 Y5 X12 X13]
+ (-0.03831467029480395) [X4 X5 Y12 Y13]
+ (-0.03619412355904233) [Y2 Y3 X8 X9]
+ (-0.03619412355904233) [X2 X3 Y8 Y9]
+ (-0.03583956795335361) [Y2 Y3 X4 X5]
+ (-0.03583956795335361) [X2 X3 Y4 Y5]
+ (-0.028685183716106347) [Y10 Y11 X12 X13]
+ (-0.028685183716106347) [X10 X11 Y12 Y13]
+ (-0.025996177598021683) [Y3 Z4 Z5 Y7]
+ (-0.025996177598021683) [X3 Z4 Z5 X7]
+ (-0.025384657508457722) [Y2 Y3 X10 X11]
+ (-0.025384657508457722) [X2 X3 Y10 Y11]
+ (-0.01902824244384796) [Y3 Y4 X11 X12]
+ (-0.01902824244384796) [X3 X4 Y11 Y12]
+ (-0.017825140995785558) [Y6 Y7 X10 X11]
+ (-0.017825140995785558) [X6 X7 Y10 Y11]
+ (-0.017680067952481598) [Y4 Y5 X10 X11]
+ (-0.017680067952481598) [X4 X5 Y10 Y11]
+ (-0.017366118994651163) [Y6 Y7 X12 X13]
+ (-0.017366118994651163) [X6 X7 Y12 Y13]
+ (-0.015577208063976434) [Y2 Y3 X12 X13]
+ (-0.015577208063976434) [X2 X3 Y12 Y13]
+ (-0.014583648907612446) [Y0 Y1 X2 X3]
+ (-0.014583648907612446) [X0 X1 Y2 Y3]
+ (-0.013873381748426368) [Y6 Y7 X8 X9]
+ (-0.013873381748426368) [X6 X7 Y8 Y9]
+ (-0.011982389010247712) [Y4 Y5 X6 X7]
+ (-0.011982389010247712) [X4 X5 Y6 Y7]
+ (-0.011285190200840525) [Y5 X6 X11 Y12]
+ (-0.011285190200840525) [X5 Y6 Y11 X12]
+ (-0.009560705729136214) [Y8 Y9 X10 X11]
+ (-0.009560705729136214) [X8 X9 Y10 Y11]
+ (-0.008125251921380992) [Y1 X2 X8 Y9]
+ (-0.008125251921380992) [Y1 Y2 Y8 Y9]
+ (-0.008125251921380992) [X1 X2 X8 X9]
+ (-0.008125251921380992) [X1 Y2 Y8 X9]
+ (-0.007731425250775401) [Y0 Y1 X10 X11]
+ (-0.007731425250775401) [X0 X1 Y10 Y11]
+ (-0.006888194352970665) [Y0 Y1 X6 X7]
+ (-0.006888194352970665) [X0 X1 Y6 Y7]
+ (-0.0065093612011772484) [Y0 Y1 X8 X9]
+ (-0.0065093612011772484) [X0 X1 Y8 Y9]
+ (-0.006087822480561942) [Y8 Y9 X12 X13]
+ (-0.006087822480561942) [X8 X9 Y12 Y13]
+ (-0.005283776488402961) [Y0 Y1 X12 X13]
+ (-0.005283776488402961) [X0 X1 Y12 Y13]
+ (-0.00514339176882479) [Y3 X4 X5 Y6]
+ (-0.00514339176882479) [X3 Y4 Y5 X6]
+ (-0.0046849033881552395) [Y1 X2 X6 Y7]
+ (-0.0046849033881552395) [Y1 Y2 Y6 Y7]
+ (-0.0046849033881552395) [X1 X2 X6 X7]
+ (-0.0046849033881552395) [X1 Y2 Y6 X7]
+ (-0.004575007626639164) [Y1 X2 X12 Y13]
+ (-0.004575007626639164) [Y1 Y2 Y12 Y13]
+ (-0.004575007626639164) [X1 X2 X12 X13]
+ (-0.004575007626639164) [X1 Y2 Y12 X13]
+ (-0.004424855449441846) [Y1 X2 X4 Y5]
+ (-0.004424855449441846) [Y1 Y2 Y4 Y5]
+ (-0.004424855449441846) [X1 X2 X4 X5]
+ (-0.004424855449441846) [X1 Y2 Y4 X5]
+ (-0.0034795118903340415) [Y2 Z3 Z5 Y6]
+ (-0.0034795118903340415) [X2 Z3 Z5 X6]
+ (-0.0034795118903340415) [Y3 Z4 Z6 Y7]
+ (-0.0034795118903340415) [X3 Z4 Z6 X7]
+ (-0.0027458364701868176) [Y0 Y1 X4 X5]
+ (-0.0027458364701868176) [X0 X1 Y4 Y5]
+ (-0.0017992194936628748) [Y1 X2 X10 Y11]
+ (-0.0017992194936628748) [Y1 Y2 Y10 Y11]
+ (-0.0017992194936628748) [X1 X2 X10 X11]
+ (-0.0017992194936628748) [X1 Y2 Y10 X11]
+ (-0.00029219862611127646) [Y7 Y8 X9 X10]
+ (-0.00029219862611127646) [X7 X8 Y9 Y10]
+ (-8.194261373271547e-06) [Z10 Y11 Z12 Y13]
+ (-8.194261373271547e-06) [Z10 X11 Z12 X13]
+ (-7.801707501647534e-06) [Y2 Z3 Y4 Z11]
+ (-7.801707501647534e-06) [X2 Z3 X4 Z11]
+ (-7.801707501647534e-06) [Y3 Z4 Y5 Z10]
+ (-7.801707501647534e-06) [X3 Z4 X5 Z10]
+ (-4.643051069161842e-06) [Y3 X4 X10 Y11]
+ (-4.643051069161842e-06) [Y3 Y4 Y10 Y11]
+ (-4.643051069161842e-06) [X3 X4 X10 X11]
+ (-4.643051069161842e-06) [X3 Y4 Y10 X11]
+ (-4.588855156239038e-06) [Y4 Z5 Y6 Z13]
+ (-4.588855156239038e-06) [X4 Z5 X6 Z13]
+ (-4.588855156239038e-06) [Y5 Z6 Y7 Z12]
+ (-4.588855156239038e-06) [X5 Z6 X7 Z12]
+ (-4.556569218961857e-06) [Y5 X6 X12 Y13]
+ (-4.556569218961857e-06) [Y5 Y6 Y12 Y13]
+ (-4.556569218961857e-06) [X5 X6 X12 X13]
+ (-4.556569218961857e-06) [X5 Y6 Y12 X13]
+ (-3.694513295192641e-06) [Y4 X5 X11 Y12]
+ (-3.694513295192641e-06) [Y4 Y5 Y11 Y12]
+ (-3.694513295192641e-06) [X4 X5 X11 X12]
+ (-3.694513295192641e-06) [X4 Y5 Y11 X12]
+ (-3.344081556312352e-06) [Z0 Y5 Z6 Y7]
+ (-3.344081556312352e-06) [Z0 X5 Z6 X7]
+ (-3.344081556312352e-06) [Z1 Y4 Z5 Y6]
+ (-3.344081556312352e-06) [Z1 X4 Z5 X6]
+ (-3.1586564324856924e-06) [Y2 Z3 Y4 Z10]
+ (-3.1586564324856924e-06) [X2 Z3 X4 Z10]
+ (-3.1586564324856924e-06) [Y3 Z4 Y5 Z11]
+ (-3.1586564324856924e-06) [X3 Z4 X5 Z11]
+ (-3.0993492434480336e-06) [Z0 Y4 Z5 Y6]
+ (-3.0993492434480336e-06) [Z0 X4 Z5 X6]
+ (-3.0993492434480336e-06) [Z1 Y5 Z6 Y7]
+ (-3.0993492434480336e-06) [Z1 X5 Z6 X7]
+ (-2.8909678818343255e-06) [Z6 Y11 Z12 Y13]
+ (-2.8909678818343255e-06) [Z6 X11 Z12 X13]
+ (-2.8909678818343255e-06) [Z7 Y10 Z11 Y12]
+ (-2.8909678818343255e-06) [Z7 X10 Z11 X12]
+ (-2.177664605455534e-06) [Z0 Y10 Z11 Y12]
+ (-2.177664605455534e-06) [Z0 X10 Z11 X12]
+ (-2.177664605455534e-06) [Z1 Y11 Z12 Y13]
+ (-2.177664605455534e-06) [Z1 X11 Z12 X13]
+ (-1.8818501831274759e-06) [Y4 Z5 Y6 Z9]
+ (-1.8818501831274759e-06) [X4 Z5 X6 Z9]
+ (-1.8818501831274759e-06) [Y5 Z6 Y7 Z8]
+ (-1.8818501831274759e-06) [X5 Z6 X7 Z8]
+ (-1.8551201218888237e-06) [Z6 Y10 Z11 Y12]
+ (-1.8551201218888237e-06) [Z6 X10 Z11 X12]
+ (-1.8551201218888237e-06) [Z7 Y11 Z12 Y13]
+ (-1.8551201218888237e-06) [Z7 X11 Z12 X13]
+ (-1.8540608579184492e-06) [Y4 Z5 Y6 Z7]
+ (-1.8540608579184492e-06) [X4 Z5 X6 Z7]
+ (-1.8163031702007199e-06) [Z4 Y11 Z12 Y13]
+ (-1.8163031702007199e-06) [Z4 X11 Z12 X13]
+ (-1.8163031702007199e-06) [Z5 Y10 Z11 Y12]
+ (-1.8163031702007199e-06) [Z5 X10 Z11 X12]
+ (-1.6923978287878942e-06) [Y4 Z5 Y6 Z10]
+ (-1.6923978287878942e-06) [X4 Z5 X6 Z10]
+ (-1.6923978287878942e-06) [Y5 Z6 Y7 Z11]
+ (-1.6923978287878942e-06) [X5 Z6 X7 Z11]
+ (-1.6148794142472414e-06) [Z0 Y11 Z12 Y13]
+ (-1.6148794142472414e-06) [Z0 X11 Z12 X13]
+ (-1.6148794142472414e-06) [Z1 Y10 Z11 Y12]
+ (-1.6148794142472414e-06) [Z1 X10 Z11 X12]
+ (-1.5973171980996465e-06) [Z8 Y10 Z11 Y12]
+ (-1.5973171980996465e-06) [Z8 X10 Z11 X12]
+ (-1.5973171980996465e-06) [Z9 Y11 Z12 Y13]
+ (-1.5973171980996465e-06) [Z9 X11 Z12 X13]
+ (-1.454842448903081e-06) [Y3 X4 X6 Y7]
+ (-1.454842448903081e-06) [Y3 Y4 Y6 Y7]
+ (-1.454842448903081e-06) [X3 X4 X6 X7]
+ (-1.454842448903081e-06) [X3 Y4 Y6 X7]
+ (-1.3980449080622432e-06) [Y4 Z5 Y6 Z8]
+ (-1.3980449080622432e-06) [X4 Z5 X6 Z8]
+ (-1.3980449080622432e-06) [Y5 Z6 Y7 Z9]
+ (-1.3980449080622432e-06) [X5 Z6 X7 Z9]
+ (-1.1954890097622128e-06) [Y2 Z3 Y4 Z7]
+ (-1.1954890097622128e-06) [X2 Z3 X4 Z7]
+ (-1.1954890097622128e-06) [Y3 Z4 Y5 Z6]
+ (-1.1954890097622128e-06) [X3 Z4 X5 Z6]
+ (-1.1908508081908382e-06) [Z0 Y3 Z4 Y5]
+ (-1.1908508081908382e-06) [Z0 X3 Z4 X5]
+ (-1.1908508081908382e-06) [Z1 Y2 Z3 Y4]
+ (-1.1908508081908382e-06) [Z1 X2 Z3 X4]
+ (-1.1708301370153798e-06) [Z2 Y5 Z6 Y7]
+ (-1.1708301370153798e-06) [Z2 X5 Z6 X7]
+ (-1.1708301370153798e-06) [Z3 Y4 Z5 Y6]
+ (-1.1708301370153798e-06) [Z3 X4 Z5 X6]
+ (-1.0632283425898855e-06) [Z2 Y10 Z11 Y12]
+ (-1.0632283425898855e-06) [Z2 X10 Z11 X12]
+ (-1.0632283425898855e-06) [Z3 Y11 Z12 Y13]
+ (-1.0632283425898855e-06) [Z3 X11 Z12 X13]
+ (-1.0358477599455018e-06) [Y6 X7 X11 Y12]
+ (-1.0358477599455018e-06) [Y6 Y7 Y11 Y12]
+ (-1.0358477599455018e-06) [X6 X7 X11 X12]
+ (-1.0358477599455018e-06) [X6 Y7 Y11 X12]
+ (-9.509249751185763e-07) [Z2 Y4 Z5 Y6]
+ (-9.509249751185763e-07) [Z2 X4 Z5 X6]
+ (-9.509249751185763e-07) [Z3 Y5 Z6 Y7]
+ (-9.509249751185763e-07) [Z3 X5 Z6 X7]
+ (-9.344557778626778e-07) [Z8 Y11 Z12 Y13]
+ (-9.344557778626778e-07) [Z8 X11 Z12 X13]
+ (-9.344557778626778e-07) [Z9 Y10 Z11 Y12]
+ (-9.344557778626778e-07) [Z9 X10 Z11 X12]
+ (-8.337746753045616e-07) [Z0 Y2 Z3 Y4]
+ (-8.337746753045616e-07) [Z0 X2 Z3 X4]
+ (-8.337746753045616e-07) [Z1 Y3 Z4 Y5]
+ (-8.337746753045616e-07) [Z1 X3 Z4 X5]
+ (-7.956895372096334e-07) [Y3 X4 X8 Y9]
+ (-7.956895372096334e-07) [Y3 Y4 Y8 Y9]
+ (-7.956895372096334e-07) [X3 X4 X8 X9]
+ (-7.956895372096334e-07) [X3 Y4 Y8 X9]
+ (-7.764994120768726e-07) [Y2 Z3 Y4 Z5]
+ (-7.764994120768726e-07) [X2 Z3 X4 Z5]
+ (-5.929765814699853e-07) [Z4 Y5 Z6 Y7]
+ (-5.929765814699853e-07) [Z4 X5 Z6 X7]
+ (-5.770052994265654e-07) [Y2 Z3 Y4 Z9]
+ (-5.770052994265654e-07) [X2 Z3 X4 Z9]
+ (-5.770052994265654e-07) [Y3 Z4 Y5 Z8]
+ (-5.770052994265654e-07) [X3 Z4 X5 Z8]
+ (-5.47164774532699e-07) [Y1 Y2 X11 X12]
+ (-5.47164774532699e-07) [X1 X2 Y11 Y12]
+ (-4.838052750652327e-07) [Y5 X6 X8 Y9]
+ (-4.838052750652327e-07) [Y5 Y6 Y8 Y9]
+ (-4.838052750652327e-07) [X5 X6 X8 X9]
+ (-4.838052750652327e-07) [X5 Y6 Y8 X9]
+ (-3.570761328862766e-07) [Y0 X1 X3 Y4]
+ (-3.570761328862766e-07) [Y0 Y1 Y3 Y4]
+ (-3.570761328862766e-07) [X0 X1 X3 X4]
+ (-3.570761328862766e-07) [X0 Y1 Y3 X4]
+ (-2.447323128643185e-07) [Y0 X1 X5 Y6]
+ (-2.447323128643185e-07) [Y0 Y1 Y5 Y6]
+ (-2.447323128643185e-07) [X0 X1 X5 X6]
+ (-2.447323128643185e-07) [X0 Y1 Y5 X6]
+ (-2.1990516189680353e-07) [Y2 X3 X5 Y6]
+ (-2.1990516189680353e-07) [Y2 Y3 Y5 Y6]
+ (-2.1990516189680353e-07) [X2 X3 X5 X6]
+ (-2.1990516189680353e-07) [X2 Y3 Y5 X6]
+ (-1.9332412769948645e-07) [Y1 X2 X3 Y4]
+ (-1.9332412769948645e-07) [X1 Y2 Y3 X4]
+ (-1.291969486380334e-07) [Y1 Z2 Z3 Y5]
+ (-1.291969486380334e-07) [X1 Z2 Z3 X5]
+ (1.7379332623480798e-07) [Y0 Z1 Z3 Y4]
+ (1.7379332623480798e-07) [X0 Z1 Z3 X4]
+ (1.7379332623480798e-07) [Y1 Z2 Z4 Y5]
+ (1.7379332623480798e-07) [X1 Z2 Z4 X5]
+ (1.9332412769948645e-07) [Y1 Y2 X3 X4]
+ (1.9332412769948645e-07) [X1 X2 Y3 Y4]
+ (2.18684237783068e-07) [Y2 Z3 Y4 Z8]
+ (2.18684237783068e-07) [X2 Z3 X4 Z8]
+ (2.18684237783068e-07) [Y3 Z4 Y5 Z9]
+ (2.18684237783068e-07) [X3 Z4 X5 Z9]
+ (2.5935343914086813e-07) [Y2 Z3 Y4 Z6]
+ (2.5935343914086813e-07) [X2 Z3 X4 Z6]
+ (2.5935343914086813e-07) [Y3 Z4 Y5 Z7]
+ (2.5935343914086813e-07) [X3 Z4 X5 Z7]
+ (3.6060718677953713e-07) [Y0 Z1 Z2 Y4]
+ (3.6060718677953713e-07) [X0 Z1 Z2 X4]
+ (3.6060718677953713e-07) [Y1 Z3 Z4 Y5]
+ (3.6060718677953713e-07) [X1 Z3 Z4 X5]
+ (5.47164774532699e-07) [Y1 X2 X11 Y12]
+ (5.47164774532699e-07) [X1 Y2 Y11 X12]
+ (5.627851912082927e-07) [Y0 X1 X11 Y12]
+ (5.627851912082927e-07) [Y0 Y1 Y11 Y12]
+ (5.627851912082927e-07) [X0 X1 X11 X12]
+ (5.627851912082927e-07) [X0 Y1 Y11 X12]
+ (6.628614202369688e-07) [Y8 X9 X11 Y12]
+ (6.628614202369688e-07) [Y8 Y9 Y11 Y12]
+ (6.628614202369688e-07) [X8 X9 X11 X12]
+ (6.628614202369688e-07) [X8 Y9 Y11 X12]
+ (1.1094407591498868e-06) [Z2 Y11 Z12 Y13]
+ (1.1094407591498868e-06) [Z2 X11 Z12 X13]
+ (1.1094407591498868e-06) [Z3 Y10 Z11 Y12]
+ (1.1094407591498868e-06) [Z3 X10 Z11 X12]
+ (1.6021167407057144e-06) [Z2 Y3 Z4 Y5]
+ (1.6021167407057144e-06) [Z2 X3 Z4 X5]
+ (1.8782101249919213e-06) [Z4 Y10 Z11 Y12]
+ (1.8782101249919213e-06) [Z4 X10 Z11 X12]
+ (1.8782101249919213e-06) [Z5 Y11 Z12 Y13]
+ (1.8782101249919213e-06) [Z5 X11 Z12 X13]
+ (2.1726691017397723e-06) [Y2 X3 X11 Y12]
+ (2.1726691017397723e-06) [Y2 Y3 Y11 Y12]
+ (2.1726691017397723e-06) [X2 X3 X11 X12]
+ (2.1726691017397723e-06) [X2 Y3 Y11 X12]
+ (3.117447945903401e-06) [Y0 Z2 Z3 Y4]
+ (3.117447945903401e-06) [X0 Z2 Z3 X4]
+ (3.5390541849335527e-06) [Y2 Z3 Y4 Z12]
+ (3.5390541849335527e-06) [X2 Z3 X4 Z12]
+ (3.5390541849335527e-06) [Y3 Z4 Y5 Z13]
+ (3.5390541849335527e-06) [X3 Z4 X5 Z13]
+ (4.281913885343779e-06) [Y4 Z5 Y6 Z11]
+ (4.281913885343779e-06) [X4 Z5 X6 Z11]
+ (4.281913885343779e-06) [Y5 Z6 Y7 Z10]
+ (4.281913885343779e-06) [X5 Z6 X7 Z10]
+ (5.275883122776411e-06) [Y3 X4 X12 Y13]
+ (5.275883122776411e-06) [Y3 Y4 Y12 Y13]
+ (5.275883122776411e-06) [X3 X4 X12 X13]
+ (5.275883122776411e-06) [X3 Y4 Y12 X13]
+ (5.9743117141316734e-06) [Y5 X6 X10 Y11]
+ (5.9743117141316734e-06) [Y5 Y6 Y10 Y11]
+ (5.9743117141316734e-06) [X5 X6 X10 X11]
+ (5.9743117141316734e-06) [X5 Y6 Y10 X11]
+ (7.954413177246877e-06) [Y10 Z11 Y12 Z13]
+ (7.954413177246877e-06) [X10 Z11 X12 Z13]
+ (8.814937307709964e-06) [Y2 Z3 Y4 Z13]
+ (8.814937307709964e-06) [X2 Z3 X4 Z13]
+ (8.814937307709964e-06) [Y3 Z4 Y5 Z12]
+ (8.814937307709964e-06) [X3 Z4 X5 Z12]
+ (0.00029219862611127646) [Y7 X8 X9 Y10]
+ (0.00029219862611127646) [X7 Y8 Y9 X10]
+ (0.0004956762314923096) [Y2 Z4 Z5 Y6]
+ (0.0004956762314923096) [X2 Z4 Z5 X6]
+ (0.001105903769189663) [Y0 Z1 Y2 Z5]
+ (0.001105903769189663) [X0 Z1 X2 Z5]
+ (0.001105903769189663) [Y1 Z2 Y3 Z4]
+ (0.001105903769189663) [X1 Z2 X3 Z4]
+ (0.0016638798784907485) [Y2 Z3 Z4 Y6]
+ (0.0016638798784907485) [X2 Z3 Z4 X6]
+ (0.0016638798784907485) [Y3 Z5 Z6 Y7]
+ (0.0016638798784907485) [X3 Z5 Z6 X7]
+ (0.0017560707018411978) [Y0 Z1 Y2 Z11]
+ (0.0017560707018411978) [X0 Z1 X2 Z11]
+ (0.0017560707018411978) [Y1 Z2 Y3 Z10]
+ (0.0017560707018411978) [X1 Z2 X3 Z10]
+ (0.002326230623158063) [Y0 Z1 Y2 Z13]
+ (0.002326230623158063) [X0 Z1 X2 Z13]
+ (0.002326230623158063) [Y1 Z2 Y3 Z12]
+ (0.002326230623158063) [X1 Z2 X3 Z12]
+ (0.0027458364701868176) [Y0 X1 X4 Y5]
+ (0.0027458364701868176) [X0 Y1 Y4 X5]
+ (0.0029297686747510364) [Y0 Z1 Y2 Z9]
+ (0.0029297686747510364) [X0 Z1 X2 Z9]
+ (0.0029297686747510364) [Y1 Z2 Y3 Z8]
+ (0.0029297686747510364) [X1 Z2 X3 Z8]
+ (0.0032769719312315372) [Y0 Z1 Y2 Z3]
+ (0.0032769719312315372) [X0 Z1 X2 Z3]
+ (0.003347617530666232) [Y0 Z1 Y2 Z7]
+ (0.003347617530666232) [X0 Z1 X2 Z7]
+ (0.003347617530666232) [Y1 Z2 Y3 Z6]
+ (0.003347617530666232) [X1 Z2 X3 Z6]
+ (0.003555290195504073) [Y0 Z1 Y2 Z10]
+ (0.003555290195504073) [X0 Z1 X2 Z10]
+ (0.003555290195504073) [Y1 Z2 Y3 Z11]
+ (0.003555290195504073) [X1 Z2 X3 Z11]
+ (0.00514339176882479) [Y3 Y4 X5 X6]
+ (0.00514339176882479) [X3 X4 Y5 Y6]
+ (0.005283776488402961) [Y0 X1 X12 Y13]
+ (0.005283776488402961) [X0 Y1 Y12 X13]
+ (0.005530759218631509) [Y0 Z1 Y2 Z4]
+ (0.005530759218631509) [X0 Z1 X2 Z4]
+ (0.005530759218631509) [Y1 Z2 Y3 Z5]
+ (0.005530759218631509) [X1 Z2 X3 Z5]
+ (0.006087822480561942) [Y8 X9 X12 Y13]
+ (0.006087822480561942) [X8 Y9 Y12 X13]
+ (0.0065093612011772484) [Y0 X1 X8 Y9]
+ (0.0065093612011772484) [X0 Y1 Y8 X9]
+ (0.006888194352970665) [Y0 X1 X6 Y7]
+ (0.006888194352970665) [X0 Y1 Y6 X7]
+ (0.006901238249797227) [Y0 Z1 Y2 Z12]
+ (0.006901238249797227) [X0 Z1 X2 Z12]
+ (0.006901238249797227) [Y1 Z2 Y3 Z13]
+ (0.006901238249797227) [X1 Z2 X3 Z13]
+ (0.007731425250775401) [Y0 X1 X10 Y11]
+ (0.007731425250775401) [X0 Y1 Y10 X11]
+ (0.008032520918821472) [Y0 Z1 Y2 Z6]
+ (0.008032520918821472) [X0 Z1 X2 Z6]
+ (0.008032520918821472) [Y1 Z2 Y3 Z7]
+ (0.008032520918821472) [X1 Z2 X3 Z7]
+ (0.009560705729136214) [Y8 X9 X10 Y11]
+ (0.009560705729136214) [X8 Y9 Y10 X11]
+ (0.011055020596132028) [Y0 Z1 Y2 Z8]
+ (0.011055020596132028) [X0 Z1 X2 Z8]
+ (0.011055020596132028) [Y1 Z2 Y3 Z9]
+ (0.011055020596132028) [X1 Z2 X3 Z9]
+ (0.011285190200840525) [Y5 Y6 X11 X12]
+ (0.011285190200840525) [X5 X6 Y11 Y12]
+ (0.011307274008848183) [Y7 Z8 Z9 Y11]
+ (0.011307274008848183) [X7 Z8 Z9 X11]
+ (0.011982389010247712) [Y4 X5 X6 Y7]
+ (0.011982389010247712) [X4 Y5 Y6 X7]
+ (0.013873381748426368) [Y6 X7 X8 Y9]
+ (0.013873381748426368) [X6 Y7 Y8 X9]
+ (0.014583648907612446) [Y0 X1 X2 Y3]
+ (0.014583648907612446) [X0 Y1 Y2 X3]
+ (0.015577208063976434) [Y2 X3 X12 Y13]
+ (0.015577208063976434) [X2 Y3 Y12 X13]
+ (0.017366118994651163) [Y6 X7 X12 Y13]
+ (0.017366118994651163) [X6 Y7 Y12 X13]
+ (0.017680067952481598) [Y4 X5 X10 Y11]
+ (0.017680067952481598) [X4 Y5 Y10 X11]
+ (0.017825140995785558) [Y6 X7 X10 Y11]
+ (0.017825140995785558) [X6 Y7 Y10 X11]
+ (0.01902824244384796) [Y3 X4 X11 Y12]
+ (0.01902824244384796) [X3 Y4 Y11 X12]
+ (0.025384657508457722) [Y2 X3 X10 Y11]
+ (0.025384657508457722) [X2 Y3 Y10 X11]
+ (0.028685183716106347) [Y10 X11 X12 Y13]
+ (0.028685183716106347) [X10 Y11 Y12 X13]
+ (0.02981242451734488) [Y6 Z7 Z8 Y10]
+ (0.02981242451734488) [X6 Z7 Z8 X10]
+ (0.02981242451734488) [Y7 Z9 Z10 Y11]
+ (0.02981242451734488) [X7 Z9 Z10 X11]
+ (0.03010462314345616) [Y6 Z7 Z9 Y10]
+ (0.03010462314345616) [X6 Z7 Z9 X10]
+ (0.03010462314345616) [Y7 Z8 Z10 Y11]
+ (0.03010462314345616) [X7 Z8 Z10 X11]
+ (0.030787505389143557) [Y6 Z8 Z9 Y10]
+ (0.030787505389143557) [X6 Z8 Z9 X10]
+ (0.03583956795335361) [Y2 X3 X4 Y5]
+ (0.03583956795335361) [X2 Y3 Y4 X5]
+ (0.03619412355904233) [Y2 X3 X8 Y9]
+ (0.03619412355904233) [X2 Y3 Y8 X9]
+ (0.03831467029480395) [Y4 X5 X12 Y13]
+ (0.03831467029480395) [X4 Y5 Y12 X13]
+ (0.10433064780651312) [Z0 Y1 Z2 Y3]
+ (0.10433064780651312) [Z0 X1 Z2 X3]
+ (-0.12133276911042251) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133276911042251) [X3 Z4 Z5 Z6 X7]
+ (-0.12133276911042241) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042241) [X2 Z3 Z4 Z5 X6]
+ (3.202076879547056e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.202076879547056e-06) [X0 Z1 Z2 Z3 X4]
+ (3.202076879547056e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.202076879547056e-06) [X1 Z2 Z3 Z4 X5]
+ (0.22848106564918383) [Y7 Z8 Z9 Z10 Y11]
+ (0.22848106564918383) [X7 Z8 Z9 Z10 X11]
+ (0.22848106564918386) [Y6 Z7 Z8 Z9 Y10]
+ (0.22848106564918386) [X6 Z7 Z8 Z9 X10]
+ (-0.032767657823289796) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.032767657823289796) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.032767657823289796) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.032767657823289796) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845272496) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.027115036845272496) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845272496) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.027115036845272496) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.025996177598021683) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.025996177598021683) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.017561202409646058) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.017561202409646058) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.017561202409646058) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.017561202409646058) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.01456453123117283) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.01456453123117283) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.01456453123117283) [X7 Z8 Z9 X10 X12 X13]
+ (-0.01456453123117283) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.012215040997613471) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.012215040997613471) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.012215040997613471) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.012215040997613471) [X4 Z5 X6 X11 Z12 X13]
+ (-0.012215040997613471) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.012215040997613471) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.012215040997613471) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.012215040997613471) [X5 Z6 X7 X10 Z11 X12]
+ (-0.01175601341981929) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.01175601341981929) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.01175601341981929) [X3 Z4 Z5 X6 X8 X9]
+ (-0.01175601341981929) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.00876482757568912) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.00876482757568912) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.00876482757568912) [X2 Z3 Z4 X5 X11 X12]
+ (-0.00876482757568912) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.00876482757568912) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.00876482757568912) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.00876482757568912) [X3 X4 X10 Z11 Z12 X13]
+ (-0.00876482757568912) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.008125251921380992) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125251921380992) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.007306759928832763) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306759928832763) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306759928832763) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306759928832763) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.0058051889898267685) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.0058051889898267685) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.0058051889898267685) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.0058051889898267685) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.005652620978017296) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.005652620978017296) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.005652620978017296) [X0 X1 X3 Z4 Z5 X6]
+ (-0.005652620978017296) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.00514339176882479) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.00514339176882479) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.00514339176882479) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.00514339176882479) [X2 Z3 X4 X5 Z6 X7]
+ (-0.00468490338815524) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.00468490338815524) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668620318776284) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668620318776284) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.004575007626639164) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.004575007626639164) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424855449441846) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424855449441846) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.004158797381840011) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.004158797381840011) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.004158797381840011) [X3 Z4 Z5 X6 X12 X13]
+ (-0.004158797381840011) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.0034937903598901885) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.0034937903598901885) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.0034937903598901885) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.0034937903598901885) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.0027790267990256473) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.0027790267990256473) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.002293956611352455) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.002293956611352455) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.0017992194936628748) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.0017992194936628748) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278753941369724) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.0017278753941369724) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298507967729457) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298507967729457) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298507967729457) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298507967729457) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533856254125824) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533856254125824) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008145313270956963) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008145313270956963) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008145313270956963) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008145313270956963) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.735036880599797e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735036880599797e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735036880599797e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735036880599797e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.774817865859694e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.774817865859694e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.774817865859694e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.774817865859694e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.5183622167158005e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.5183622167158005e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.5183622167158005e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.5183622167158005e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.4443446770434225e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.4443446770434225e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.4443446770434225e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.4443446770434225e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.524373849485163e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.524373849485163e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.524373849485163e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.524373849485163e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.290028434379142e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.290028434379142e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.290028434379142e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.290028434379142e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.9743117141316734e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.9743117141316734e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.275883122776411e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.275883122776411e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.643051069161842e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.643051069161842e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.556569218961857e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.556569218961857e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.2532242260236684e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.2532242260236684e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.7696594528766753e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.7696594528766753e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.694513295192641e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.694513295192641e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.610297131479009e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.610297131479009e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.610297131479009e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.610297131479009e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.313145500376518e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.313145500376518e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.2774831964510476e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.2774831964510476e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.2774831964510476e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.2774831964510476e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.2112283491086448e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.2112283491086448e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.2112283491086448e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.2112283491086448e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.1513463115752935e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.1513463115752935e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.0882507118879313e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.0882507118879313e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.1726691017397723e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.1726691017397723e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.454842448903081e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.454842448903081e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.3304731888162712e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.3304731888162712e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2283337823366589e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2283337823366589e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.0358477599455018e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.0358477599455018e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956895372096334e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956895372096334e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733197743795952e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733197743795952e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733197743795952e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733197743795952e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.628614202369688e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.628614202369688e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.556281915418245e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.556281915418245e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.556281915418245e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.556281915418245e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.41829157557192e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.41829157557192e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.41829157557192e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.41829157557192e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.927453084097665e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927453084097665e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927453084097665e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927453084097665e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627851912082927e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627851912082927e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660625537027e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660625537027e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660625537027e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660625537027e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660625537027e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660625537027e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660625537027e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660625537027e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.838052750652327e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.838052750652327e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.5707613288627664e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.5707613288627664e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.3281393502796125e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.3281393502796125e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.0868265654491057e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.0868265654491057e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.0868265654491057e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.0868265654491057e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.447323128643185e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.447323128643185e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.3713289477741438e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.3713289477741438e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.3713289477741438e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.3713289477741438e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.1990516189680353e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.1990516189680353e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.9332412769948645e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.9332412769948645e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.9332412769948645e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.9332412769948645e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.83942091533787e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.83942091533787e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.83942091533787e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.83942091533787e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.5510539175161148e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.5510539175161148e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.5510539175161148e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.5510539175161148e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.3807781479747706e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.3807781479747706e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.3807781479747706e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.3807781479747706e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.3807781479747706e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.3807781479747706e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.3807781479747706e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.3807781479747706e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.3807781479747706e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.3807781479747706e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.3807781479747706e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.3807781479747706e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.291969486380334e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.291969486380334e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.107632560063074e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.107632560063074e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.107632560063074e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.107632560063074e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.107632560063074e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.107632560063074e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.107632560063074e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.107632560063074e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.057446596982873e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057446596982873e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057446596982873e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057446596982873e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649310133927333e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649310133927333e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649310133927333e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649310133927333e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.83942091533787e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.83942091533787e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.83942091533787e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.83942091533787e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.1990516189680353e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.1990516189680353e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.447323128643185e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.447323128643185e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.236259961166877e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.236259961166877e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.236259961166877e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.236259961166877e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.3281393502796125e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.3281393502796125e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.5707613288627664e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.5707613288627664e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.838052750652327e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.838052750652327e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.627851912082927e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627851912082927e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.628614202369688e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.628614202369688e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956895372096334e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956895372096334e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.306536652777008e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.306536652777008e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.306536652777008e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.306536652777008e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.0358477599455018e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.0358477599455018e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2283337823366589e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2283337823366589e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.2393363218226114e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393363218226114e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393363218226114e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393363218226114e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.3304731888162712e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.3304731888162712e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.454842448903081e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.454842448903081e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.1726691017397723e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.1726691017397723e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.0882507118879313e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.0882507118879313e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.117447945903401e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.117447945903401e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.1513463115752935e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.1513463115752935e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.313145500376518e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.313145500376518e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.334331289815695e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.334331289815695e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.694513295192641e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.694513295192641e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.1839325596805706e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.1839325596805706e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.556569218961857e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.556569218961857e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.643051069161842e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.643051069161842e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.275883122776411e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.275883122776411e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.9743117141316734e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.9743117141316734e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.00029219862611127646) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.00029219862611127646) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.00029219862611127646) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.00029219862611127646) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004956762314923096) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004956762314923096) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650070219498217) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650070219498217) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650070219498217) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650070219498217) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533856254125824) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533856254125824) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.0016095313817213828) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.0016095313817213828) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.0016095313817213828) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.0016095313817213828) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.001667604181144127) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.001667604181144127) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.001667604181144127) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.001667604181144127) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.0017278753941369724) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.0017278753941369724) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.0017992194936628748) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.0017992194936628748) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.002293956611352455) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.002293956611352455) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.0024629170071339655) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.0024629170071339655) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.0024629170071339655) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.0024629170071339655) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.003961560792496582) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.003961560792496582) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.003961560792496582) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.003961560792496582) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.004424855449441846) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424855449441846) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004575007626639164) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.004575007626639164) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668620318776284) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668620318776284) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.00468490338815524) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.00468490338815524) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005324835234221453) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.005324835234221453) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.005324835234221453) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.005324835234221453) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.005368659358109196) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368659358109196) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368659358109196) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368659358109196) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.007960880725921354) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.007960880725921354) [X4 Z5 X6 X10 Z11 X12]
+ (0.007960880725921354) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.007960880725921354) [X5 Z6 X7 X11 Z12 X13]
+ (0.008125251921380992) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125251921380992) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.008890731522694302) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.008890731522694302) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.008890731522694302) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.008890731522694302) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.010263414868158842) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.010263414868158842) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.010263414868158842) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.010263414868158842) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.010540425907671166) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.010540425907671166) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.010540425907671166) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.010540425907671166) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010960074940542353) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010960074940542353) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010960074940542353) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010960074940542353) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307274008848183) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307274008848183) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.01441109943013123) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.01441109943013123) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.01441109943013123) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.01441109943013123) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.015588250102380297) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.015588250102380297) [X2 Z3 X4 X10 Z11 X12]
+ (0.015588250102380297) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.015588250102380297) [X3 Z4 X5 X11 Z12 X13]
+ (0.018266834869375113) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.018266834869375113) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.018266834869375113) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.018266834869375113) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.0190204231730397) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.0190204231730397) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.0190204231730397) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.0190204231730397) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.02435307767806941) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.02435307767806941) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.02435307767806941) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.02435307767806941) [X2 Z3 X4 X11 Z12 X13]
+ (0.02435307767806941) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.02435307767806941) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.02435307767806941) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.02435307767806941) [X3 Z4 X5 X10 Z11 X12]
+ (0.024389082531148895) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.024389082531148895) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.024389082531148895) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.024389082531148895) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.025104957138843993) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.025104957138843993) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.025104957138843993) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.025104957138843993) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.030787505389143557) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.030787505389143557) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.04587947078129816) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.04587947078129816) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.05600733087780656) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.05600733087780656) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.05600733087780656) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.05600733087780656) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.056084681246612554) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.056084681246612554) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.056084681246612554) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.056084681246612554) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.631277929414951e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631277929414951e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.631277929414951e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631277929414951e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.5950860076753125e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.5950860076753125e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.595086007675312e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.595086007675312e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.04274327701378329) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.04274327701378329) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (0.04274327701378329) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04274327701378329) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.04764261217638321) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.04764261217638321) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.04764261217638321) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.04764261217638321) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.04171881383982181) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.04171881383982181) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.04171881383982181) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.04171881383982181) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.039564416322894244) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.039564416322894244) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.039564416322894244) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.039564416322894244) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03935916802205406) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.03935916802205406) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.03935916802205406) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.03935916802205406) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.03931805194719788) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03931805194719788) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.03931805194719788) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03931805194719788) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.02990378951262533) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.02990378951262533) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.02990378951262533) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.02990378951262533) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.02873077955190581) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.02873077955190581) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.02873077955190581) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.02873077955190581) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.02563723829602681) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.02563723829602681) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.02563723829602681) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.02563723829602681) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.02475546329289162) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.02475546329289162) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.02475546329289162) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.02475546329289162) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.024282117354693142) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.024282117354693142) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.023145130929529183) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.023145130929529183) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.022528440196013123) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.022528440196013123) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.021433810721602067) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.021433810721602067) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.021433810721602067) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.021433810721602067) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251492) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.019257505095251492) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.01902824244384796) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.01902824244384796) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.01888903030494323) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.01888903030494323) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.01888903030494323) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.01888903030494323) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.016024603689179545) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.016024603689179545) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.014603704729162446) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.014603704729162446) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564531231172828) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564531231172828) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.01175601341981929) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.01175601341981929) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.011285190200840525) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.011285190200840525) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.009841749246962576) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009841749246962576) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.009612634606847265) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612634606847265) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612634606847265) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612634606847265) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.008469978791023267) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.008469978791023267) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.007306759928832763) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.007306759928832763) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005923798336561394) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.005923798336561394) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.005652620978017296) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.005652620978017296) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368659358109196) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368659358109196) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.004158797381840011) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.004158797381840011) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0033566705638329183) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.0033566705638329183) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.0033566705638329183) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.0033566705638329183) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.003267513854423556) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.003267513854423556) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.003267513854423556) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.003267513854423556) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.002779026799025647) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.002779026799025647) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.0026860409778066753) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.0026860409778066753) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.0026860409778066753) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.0026860409778066753) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.002293956611352455) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.002293956611352455) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.002293956611352455) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.002293956611352455) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.0009581655836697025) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.0009581655836697025) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0009581655836697025) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.0009581655836697025) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.0009581655836697025) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.0009581655836697025) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0009581655836697025) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.0009581655836697025) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.0002463643756963586) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0002463643756963586) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0001384017730354104) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.0001384017730354104) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.0001384017730354104) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.0001384017730354104) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.735036880599797e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735036880599797e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585308142128e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585308142128e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.6103585308142128e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.6103585308142128e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808797633827e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.5316808797633827e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808797633827e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5316808797633827e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.806102776482861e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.806102776482861e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.806102776482861e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.806102776482861e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.089799468512604e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.089799468512604e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.089799468512604e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.089799468512604e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.6522096711809665e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.6522096711809665e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.6522096711809665e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.6522096711809665e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851835644192e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.481851835644192e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851835644192e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.481851835644192e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.071480737013287e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.071480737013287e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.071480737013287e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.071480737013287e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.734622039469574e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.734622039469574e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.734622039469574e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.734622039469574e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.728843147883176e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.728843147883176e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.728843147883176e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.728843147883176e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.2532242260236684e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.2532242260236684e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.7696594528766753e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.7696594528766753e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.544395429803341e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.544395429803341e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.544395429803341e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.544395429803341e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.544395429803341e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.544395429803341e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.544395429803341e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.544395429803341e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.3609563206294275e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563206294275e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563206294275e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.3609563206294275e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.1032156052140107e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.1032156052140107e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.1032156052140107e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.1032156052140107e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.011122098916486e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.011122098916486e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.011122098916486e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.011122098916486e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9429468370935413e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.9429468370935413e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9429468370935413e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9429468370935413e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174777454353e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.6541174777454353e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174777454353e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.6541174777454353e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.5224930677900511e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224930677900511e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224930677900511e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224930677900511e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224930677900511e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930677900511e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930677900511e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224930677900511e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.2283337823366589e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337823366589e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.2283337823366589e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337823366589e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.988770288004828e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988770288004828e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988770288004828e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988770288004828e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.867765105083012e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765105083012e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765105083012e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.867765105083012e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.18999097655633e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.18999097655633e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246207637869e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246207637869e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.47164774532699e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.47164774532699e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.5614471791540975e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.5614471791540975e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.5614471791540975e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.5614471791540975e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.523389678863949e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.523389678863949e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.42732310885073e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.42732310885073e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.42732310885073e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.42732310885073e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.3281393502796125e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.3281393502796125e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.3281393502796125e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.3281393502796125e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.0868265654491057e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.0868265654491057e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.8882935934810595e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935934810595e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935934810595e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.8882935934810595e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.3713289477741438e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.3713289477741438e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.83942091533787e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.83942091533787e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.057446596982873e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057446596982873e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.53717809436948e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.53717809436948e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.53717809436948e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.53717809436948e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.057446596982873e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057446596982873e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.209350629752486e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350629752486e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350629752486e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.209350629752486e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.703578355367755e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.703578355367755e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.703578355367755e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.703578355367755e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.83942091533787e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.83942091533787e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.3713289477741438e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.3713289477741438e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.0868265654491057e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.0868265654491057e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.523389678863949e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.523389678863949e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.47164774532699e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.47164774532699e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246207637869e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246207637869e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.18999097655633e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.18999097655633e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.3304731888162712e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.3304731888162712e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.3304731888162712e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.3304731888162712e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288532437852424e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288532437852424e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288532437852424e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288532437852424e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.689348951710372e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.689348951710372e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.689348951710372e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.689348951710372e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.7455184010028583e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.7455184010028583e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.7455184010028583e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.7455184010028583e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.7455184010028583e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.7455184010028583e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.7455184010028583e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.7455184010028583e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.2118420195004232e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420195004232e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.2118420195004232e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420195004232e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.2118420195004232e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420195004232e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.2118420195004232e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420195004232e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.313145500376518e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.313145500376518e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.313145500376518e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.313145500376518e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.334331289815695e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.334331289815695e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.1839325596805706e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.1839325596805706e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.735036880599797e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735036880599797e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.0002463643756963586) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.0002463643756963586) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00044585351288405604) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.00044585351288405604) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.00044585351288405604) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.00044585351288405604) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940221543005028) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005028) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005028) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005028) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005028) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005028) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940221543005028) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005028) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533856254125824) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533856254125824) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533856254125824) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533856254125824) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0010435246534907525) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435246534907525) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435246534907525) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435246534907525) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.0012803060973496734) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.0012803060973496734) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.0012803060973496734) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.0012803060973496734) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.001303800478812695) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.001303800478812695) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.001303800478812695) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.001303800478812695) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.0022619660624823975) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.0022619660624823975) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.0022619660624823975) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.0022619660624823975) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.0022619660624823975) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.0022619660624823975) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.0022619660624823975) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.0022619660624823975) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.00398984145661937) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.00398984145661937) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.00398984145661937) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.00398984145661937) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.004158797381840011) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.004158797381840011) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.004311038507914308) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.004311038507914308) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.004311038507914308) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.004311038507914308) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.004636976661182592) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.004636976661182592) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.004636976661182592) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.004636976661182592) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.00511447383166034) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.00511447383166034) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.00511447383166034) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.00511447383166034) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.00511447383166034) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.00511447383166034) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.00511447383166034) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.00511447383166034) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005241535382803948) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.005241535382803948) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.005241535382803948) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.005241535382803948) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.005262642473076786) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262642473076786) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262642473076786) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262642473076786) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368659358109196) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368659358109196) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005379937155839358) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.005379937155839358) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.005379937155839358) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.005379937155839358) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.005652620978017296) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.005652620978017296) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.0057084959859608425) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.0057084959859608425) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.0057084959859608425) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.0057084959859608425) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.005923798336561394) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.005923798336561394) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.007306759928832763) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.007306759928832763) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469978791023267) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.008469978791023267) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.009841749246962576) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.009841749246962576) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.011285190200840525) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.011285190200840525) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.01175601341981929) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.01175601341981929) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.014564531231172828) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564531231172828) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603704729162446) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.014603704729162446) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.016024603689179545) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.016024603689179545) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.01902824244384796) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.01902824244384796) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.019257505095251492) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.019257505095251492) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.04587947078129816) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04587947078129816) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.3693708936615636) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.3693708936615636) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.3693708936615636) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.3693708936615636) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.2816425776702358) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.2816425776702358) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.28164257767023576) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.28164257767023576) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.09065144207036499) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.09065144207036499) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.09065144207036499) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.09065144207036499) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863643) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684737589863643) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863643) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684737589863643) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635021950635121) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635021950635121) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635021950635121) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635021950635121) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214124) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.06752385099214124) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214124) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.06752385099214124) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.034903343373661404) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.034903343373661404) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.034903343373661404) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.034903343373661404) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02459186088382998) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.02459186088382998) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02459186088382998) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.02459186088382998) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.024282117354693142) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.024282117354693142) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.023145130929529183) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.023145130929529183) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.022528440196013123) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.022528440196013123) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019538050311315287) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.019538050311315287) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.019538050311315287) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.019538050311315287) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.01709155315589935) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.01709155315589935) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.01709155315589935) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.01709155315589935) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.016024603689179545) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.016024603689179545) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.016024603689179545) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.016024603689179545) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.010311482489831424) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831424) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831424) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311482489831424) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.009841749246962576) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962576) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.009841749246962576) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962576) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209955) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209955) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209955) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008826368514209955) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.00854199662545488) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00854199662545488) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00854199662545488) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00854199662545488) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.00854199662545488) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00854199662545488) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00854199662545488) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00854199662545488) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008469978791023267) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023267) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.008469978791023267) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023267) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.004668620318776284) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668620318776284) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.0038764708993370392) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.0038764708993370392) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.003804066171728546) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728546) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728546) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003804066171728546) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178574) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178574) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0033566705638329187) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.0033566705638329187) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.003267513854423556) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.003267513854423556) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0021413612231015347) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.0021413612231015347) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.0017278753941369724) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.0017278753941369724) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.0016407548553124137) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0016407548553124137) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214168653) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0014528843214168653) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214168653) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0014528843214168653) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870896771024866) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007870896771024866) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192743499488541) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192743499488541) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.00019400857029756911) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00019400857029756911) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0001384017730354104) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.0001384017730354104) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.141625221149875e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.141625221149875e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.141625221149875e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.141625221149875e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.071480737013287e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.071480737013287e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.1513463115752935e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.1513463115752935e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0882507118879313e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.0882507118879313e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.9885117063151845e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.9885117063151845e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8742990717034916e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.8742990717034916e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.3609563206294275e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.3609563206294275e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.30029465637946e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.30029465637946e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.146837650951188e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.146837650951188e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.146837650951188e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.146837650951188e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.352332104805574e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.352332104805574e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.352332104805574e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.352332104805574e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091637200571497e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637200571497e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637200571497e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637200571497e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637200571497e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637200571497e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637200571497e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091637200571497e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.074305987438299e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.074305987438299e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.074305987438299e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.074305987438299e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128987804333e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128987804333e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128987804333e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128987804333e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867765105083015e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.867765105083015e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560692466385009e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692466385009e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692466385009e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692466385009e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692466385009e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692466385009e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692466385009e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560692466385009e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.997018422737375e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.997018422737375e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.997018422737375e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.997018422737375e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.997018422737375e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.997018422737375e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.997018422737375e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.997018422737375e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.5682475217075477e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.5682475217075477e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.5682475217075477e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.5682475217075477e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393089403845e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393089403845e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393089403845e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393089403845e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393089403845e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393089403845e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.3767393089403845e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393089403845e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.8882935934810595e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.8882935934810595e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6863815495080335e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.6863815495080335e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.703578355367755e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.703578355367755e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.209350629752486e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.209350629752486e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773246512673e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773246512673e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773246512673e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773246512673e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773246512673e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773246512673e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773246512673e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773246512673e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253803404673e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253803404673e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9742253803404673e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9742253803404673e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0474716557868944e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.0474716557868944e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.0474716557868944e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.0474716557868944e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.209350629752486e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.209350629752486e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.071728218817147e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.071728218817147e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.071728218817147e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.071728218817147e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.2004287495295103e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.2004287495295103e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.2004287495295103e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.2004287495295103e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.703578355367755e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.703578355367755e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.3120943056183369e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.3120943056183369e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.3120943056183369e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.3120943056183369e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.6863815495080335e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815495080335e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.8882935934810595e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.8882935934810595e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.09225061665781e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.09225061665781e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.09225061665781e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.09225061665781e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.09225061665781e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.09225061665781e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.09225061665781e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.09225061665781e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.444597854288772e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.444597854288772e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.444597854288772e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.444597854288772e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.6849150957046933e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.6849150957046933e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.6849150957046933e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.6849150957046933e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246974426455016e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246974426455016e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246974426455016e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246974426455016e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246974426455016e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246974426455016e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246974426455016e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246974426455016e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.867765105083015e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.867765105083015e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.30029465637946e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.30029465637946e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.3609563206294275e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.3609563206294275e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.8742990717034916e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.8742990717034916e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.8836765763222763e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.8836765763222763e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.9473560122110823e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9473560122110823e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.9473560122110823e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9473560122110823e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117063151845e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9885117063151845e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.0882507118879313e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.0882507118879313e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.1513463115752935e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.1513463115752935e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.846201671745117e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846201671745117e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846201671745117e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846201671745117e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.071480737013287e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.071480737013287e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.105526722554976e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.105526722554976e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.105526722554976e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.105526722554976e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.1464963281245765e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.1464963281245765e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.1464963281245765e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.1464963281245765e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159350502306835e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159350502306835e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159350502306835e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159350502306835e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427988657257638e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427988657257638e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427988657257638e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427988657257638e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935867718526267e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935867718526267e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935867718526267e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935867718526267e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.25327334891381e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.25327334891381e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979825794258467e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979825794258467e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979825794258467e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979825794258467e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.205548411211936e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.205548411211936e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.205548411211936e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.205548411211936e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0001384017730354104) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.0001384017730354104) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.00018787053389554805) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.00018787053389554805) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.00018787053389554805) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00018787053389554805) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00019400857029756911) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00019400857029756911) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0002463643756963586) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0002463643756963586) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0002463643756963586) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0002463643756963586) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192743499488541) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192743499488541) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156734248909878) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156734248909878) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156734248909878) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156734248909878) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024866) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007870896771024866) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.0015324835230730984) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.0015324835230730984) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.0015324835230730984) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.0015324835230730984) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.0016407548553124137) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0016407548553124137) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0017278753941369724) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.0017278753941369724) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.002446497155415934) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.002446497155415934) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.002446497155415934) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.002446497155415934) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.003267513854423556) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.003267513854423556) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.0033566705638329187) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.0033566705638329187) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.0034841573002178574) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0034841573002178574) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0038764708993370392) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.0038764708993370392) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.004668620318776284) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668620318776284) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278228) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.004767272188278228) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.004767272188278228) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278228) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.005286546538227082) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.005286546538227082) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.005286546538227082) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.005286546538227082) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.005408954422410138) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.005408954422410138) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.005408954422410138) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.005408954422410138) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.005923798336561394) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561394) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.005923798336561394) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561394) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.010757563953908728) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010757563953908728) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010757563953908728) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010757563953908728) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.014603704729162448) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.014603704729162448) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.014603704729162448) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.014603704729162448) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.019299560579363606) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363606) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363606) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363606) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363606) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363606) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.019299560579363606) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363606) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.05859198873386312) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.05859198873386312) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.775950527903153e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.775950527903153e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.775950527903153e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.775950527903153e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.07165035181002946) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07165035181002946) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.07165035181002953) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.07165035181002953) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251492) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019257505095251492) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010311482489831422) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311482489831422) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008826368514209955) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209955) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.007597464029770596) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.007597464029770596) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.007597464029770596) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.007597464029770596) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0057335697473118626) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0057335697473118626) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0057335697473118626) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0057335697473118626) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0057335697473118626) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0057335697473118626) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0057335697473118626) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0057335697473118626) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676589) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005348051582676589) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005348051582676589) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676589) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0038040661717285464) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0038040661717285464) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984166168122003) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.002984166168122003) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.002984166168122003) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.002984166168122003) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.0024464971554159338) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.0024464971554159338) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.0022494124470940056) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0022494124470940056) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022494124470940056) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0022494124470940056) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0021413612231015347) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.0021413612231015347) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.0018638942824587325) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587325) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587325) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587325) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587325) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587325) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0018638942824587325) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587325) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0016407548553124137) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124137) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0016407548553124137) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124137) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0012223378081538763) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538763) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538763) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538763) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538763) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538763) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538763) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0012223378081538763) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.001028329237856307) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.001028329237856307) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.001028329237856307) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.001028329237856307) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1463061454595246e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1463061454595246e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.8742990717034916e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990717034916e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.8742990717034916e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990717034916e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.30029465637946e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.30029465637946e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.30029465637946e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.30029465637946e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.044494129974238e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.044494129974238e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.044494129974238e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.044494129974238e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.956079231600506e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.956079231600506e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.956079231600506e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.956079231600506e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.10551503878448e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.10551503878448e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.10551503878448e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.10551503878448e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661347214379298e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661347214379298e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661347214379298e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661347214379298e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341414516252e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341414516252e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.18999097655633e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.18999097655633e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876621659415331e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876621659415331e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876621659415331e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876621659415331e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246207637869e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246207637869e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.523389678863949e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.523389678863949e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.076732532571163e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.076732532571163e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.076732532571163e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.076732532571163e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.013471459268586e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.013471459268586e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.904599885226127e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.904599885226127e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.904599885226127e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.904599885226127e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.666731754998546e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.666731754998546e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.666731754998546e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.666731754998546e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641928160259e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.8505641928160259e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309311944625e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6569309311944625e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309311944625e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6569309311944625e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.8505641928160259e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.8505641928160259e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815495080335e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815495080335e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.6863815495080335e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815495080335e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471459268586e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.013471459268586e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.523389678863949e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.523389678863949e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.6704023904630485e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.6704023904630485e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.6704023904630485e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.6704023904630485e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246207637869e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246207637869e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.18999097655633e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.18999097655633e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341414516252e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341414516252e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949476487877785e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949476487877785e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.7924939579223098e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939579223098e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939579223098e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.7924939579223098e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.8836765763222763e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.8836765763222763e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.9885117063151845e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117063151845e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117063151845e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117063151845e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.25327334891381e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.25327334891381e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4017109736784404e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4017109736784404e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4017109736784404e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4017109736784404e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603694706716e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.5809603694706716e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603694706716e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.5809603694706716e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192743499488541) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499488541) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192743499488541) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499488541) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.0007870896771024866) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024866) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024866) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024866) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.001172634831644148) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.001172634831644148) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.001172634831644148) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.001172634831644148) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012366478019245728) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.0012366478019245728) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.0012366478019245728) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.0012366478019245728) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.0022009640695004546) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0022009640695004546) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0022009640695004546) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0022009640695004546) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798025) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798025) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798025) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798025) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798025) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798025) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002394972639798025) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798025) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0024464971554159338) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.0024464971554159338) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.0038040661717285464) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0038040661717285464) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0038764708993370392) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.0038764708993370392) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.0038764708993370392) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.0038764708993370392) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.0042208139700465755) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.0042208139700465755) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.0042208139700465755) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.0042208139700465755) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.008826368514209955) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.008826368514209955) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.010311482489831422) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311482489831422) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257505095251492) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019257505095251492) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.05859198873386312) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.05859198873386312) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.398700901845449e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.398700901845449e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.398700901845449e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.398700901845449e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178574) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178574) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984166168122003) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.002984166168122003) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.00019400857029756911) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00019400857029756911) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061454595246e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061454595246e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7924939579223098e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7924939579223098e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540341414516252e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341414516252e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540341414516252e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341414516252e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.8505641928160259e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641928160259e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641928160259e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641928160259e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471459268586e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471459268586e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471459268586e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471459268586e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949476487877785e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949476487877785e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.7924939579223098e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7924939579223098e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756911) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756911) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002984166168122003) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.002984166168122003) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.0034841573002178574) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0034841573002178574) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
 </code>
 </pre>
 </details>

---

## 14. tutorial_error_mitigation.html <a name="demo13"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_error_mitigation.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
0.9244013324461877
0: ──RY(4.56)─╭●──RY(5.93)──RY(-5.93)──────────────────────────────────────────
1: ──RY(3.60)─╰Z──RY(5.90)─╭●──────────RY(5.18)──RY(-5.18)─────────────────────
2: ──RY(4.05)─╭●──RY(3.32)─╰Z──────────RY(1.07)──RY(-1.07)──RY(1.07)──RY(-1.07)
3: ──RY(3.51)─╰Z──RY(3.66)──RY(-3.66)──────────────────────────────────────────
────────────────╭●──RY(-4.56)─┤
──╭●──RY(-5.90)─╰Z──RY(-3.60)─┤
──╰Z──RY(-3.32)─╭●──RY(-4.05)─┤
────────────────╰Z──RY(-3.51)─┤
0: ──RY(4.56)─╭●──RY(5.93)──RY(-5.93)─────────────────────────────────────────
1: ──RY(3.60)─╰Z──RY(5.90)─╭●──────────RY(5.18)──RY(-5.18)─╭●─╭●─╭●──RY(-5.90)
2: ──RY(4.05)─╭●──RY(3.32)─╰Z──────────RY(1.07)──RY(-1.07)─╰Z─╰Z─╰Z──RY(-3.32)
3: ──RY(3.51)─╰Z──RY(3.66)──RY(-3.66)─────────────────────────────────────────
──╭●──RY(-4.56)─┤
──╰Z──RY(-3.60)─┤
──╭●──RY(-4.05)─┤
──╰Z──RY(-3.51)─┤
0: ──RY(4.56)─╭●──RY(5.93)──RY(-5.93)─────────────────────────────────────────
1: ──RY(3.60)─╰Z──RY(5.90)─╭●──────────RY(5.18)──RY(-5.18)─╭●─╭●─╭●──RY(-5.90)
2: ──RY(4.05)─╭●──RY(3.32)─╰Z──────────RY(1.07)──RY(-1.07)─╰Z─╰Z─╰Z──RY(-3.32)
3: ──RY(3.51)─╰Z──RY(3.66)──RY(-3.66)─────────────────────────────────────────
──╭●──RY(-4.56)─┤
──╰Z──RY(-3.60)─┤
──╭●──RY(-4.05)─┤
──╰Z──RY(-3.51)─┤
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
0.9475501495316461
0: ──RY(4.56)─╭●──RY(5.93)──RY(-5.93)─────────────────────────────────────────
1: ──RY(3.60)─╰Z──RY(5.90)─╭●──────────RY(5.18)──RY(-5.18)─╭●─╭●─╭●──RY(-5.90)
2: ──RY(4.05)─╭●──RY(3.32)─╰Z──────────RY(1.07)──RY(-1.07)─╰Z─╰Z─╰Z──RY(-3.32)
3: ──RY(3.51)─╰Z──RY(3.66)──RY(-3.66)─────────────────────────────────────────
──╭●──RY(-4.56)─┤
──╰Z──RY(-3.60)─┤
──╭●──RY(-4.05)─┤
──╰Z──RY(-3.51)─┤
0: ──RY(4.56)─╭●──RY(5.93)──RY(-5.93)────────────────────────────────────╭●
1: ──RY(3.60)─╰Z──RY(5.90)─╭●──────────RY(5.18)──RY(-5.18)─╭●──RY(-5.90)─╰Z
2: ──RY(4.05)─╭●──RY(3.32)─╰Z──────────RY(1.07)──RY(-1.07)─╰Z──RY(-3.32)─╭●
3: ──RY(3.51)─╰Z──RY(3.66)──RY(-3.66)────────────────────────────────────╰Z
───RY(-4.56)───────────────┤
───RY(-3.60)───────────────┤
──╭●─────────╭●──RY(-4.05)─┤
──╰Z─────────╰Z──RY(-3.51)─┤
0: ──RY(4.56)──RY(-4.56)──RY(4.56)─╭●──────────RY(5.93)──RY(-5.93)──────────
1: ──RY(3.60)──────────────────────╰Z──────────RY(5.90)─╭●──────────RY(5.18)
2: ──RY(4.05)─╭●──────────RY(3.32)──────────────────────╰Z──────────RY(1.07)
3: ──RY(3.51)─╰Z──────────RY(3.66)──RY(-3.66)───────────────────────────────
───────────────────────────╭●──RY(-4.56)─┤
───RY(-5.18)─╭●──RY(-5.90)─╰Z──RY(-3.60)─┤
───RY(-1.07)─╰Z──RY(-3.32)─╭●──RY(-4.05)─┤
───────────────────────────╰Z──RY(-3.51)─┤
 </code>
 </pre>
 </details>

---

## 15. tutorial_vqe_spin_sectors.html <a name="demo14"></a>

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

## 16. tutorial_jax_transformations.html <a name="demo15"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_jax_transformations.html):

```
No jit time: 0.0061 seconds
First run time: 0.0886 seconds
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_jax_transformations.html):

```
No jit time: 0.0057 seconds
First run time: 0.0824 seconds
```

---

## 17. tutorial_measurement_optimize.html <a name="demo16"></a>

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

