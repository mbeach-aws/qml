Last update: 2021-12-18  22:31:06 (All times shown in Eastern time)
# List of differences in demonstration outputs

# Table of contents

1. [tutorial_quantum_transfer_learning.html](#demo0)
2. [tutorial_error_mitigation.html](#demo1)
3. [tutorial_measurement_optimize.html](#demo2)
4. [tutorial_quantum_chemistry.html](#demo3)
5. [tutorial_qnn_module_tf.html](#demo4)
6. [tutorial_jax_transformations.html](#demo5)
7. [tutorial_adaptive_circuits.html](#demo6)
8. [tutorial_quanvolution.html](#demo7)
9. [tutorial_backprop.html](#demo8)


Number of demos different/all demos: 9/55

## 1. tutorial_quantum_transfer_learning.html <a name="demo0"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
  0%|          | 176k/44.7M [00:00<00:26, 1.75MB/s]
  3%|2         | 1.25M/44.7M [00:00<00:06, 7.30MB/s]
 10%|9         | 4.30M/44.7M [00:00<00:02, 18.4MB/s]
 14%|#3        | 6.06M/44.7M [00:00<00:02, 18.0MB/s]
 17%|#7        | 7.79M/44.7M [00:00<00:02, 16.9MB/s]
 21%|##1       | 9.41M/44.7M [00:00<00:02, 15.9MB/s]
 25%|##5       | 11.2M/44.7M [00:00<00:02, 16.3MB/s]
 29%|##8       | 12.7M/44.7M [00:00<00:02, 15.5MB/s]
 32%|###2      | 14.5M/44.7M [00:00<00:01, 16.3MB/s]
 36%|###6      | 16.2M/44.7M [00:01<00:01, 16.6MB/s]
 40%|####      | 18.0M/44.7M [00:01<00:01, 17.0MB/s]
 44%|####3     | 19.6M/44.7M [00:01<00:01, 14.3MB/s]
 48%|####7     | 21.4M/44.7M [00:01<00:01, 15.3MB/s]
 51%|#####1    | 23.0M/44.7M [00:01<00:01, 15.6MB/s]
 55%|#####5    | 24.7M/44.7M [00:01<00:01, 16.1MB/s]
 60%|######    | 26.8M/44.7M [00:01<00:01, 17.7MB/s]
 64%|######3   | 28.5M/44.7M [00:01<00:01, 15.0MB/s]
 68%|######7   | 30.4M/44.7M [00:02<00:00, 16.0MB/s]
 72%|#######1  | 32.1M/44.7M [00:02<00:00, 15.6MB/s]
 76%|#######6  | 34.0M/44.7M [00:02<00:00, 16.5MB/s]
 80%|#######9  | 35.6M/44.7M [00:02<00:00, 16.3MB/s]
 85%|########4 | 37.9M/44.7M [00:02<00:00, 18.4MB/s]
 89%|########8 | 39.7M/44.7M [00:02<00:00, 18.4MB/s]
 94%|#########3| 41.9M/44.7M [00:02<00:00, 19.5MB/s]
 98%|#########7| 43.8M/44.7M [00:02<00:00, 19.2MB/s]
100%|##########| 44.7M/44.7M [00:02<00:00, 16.7MB/s]
Training started:
Phase: train Epoch: 1/3 Iter: 1/62 Batch time: 0.2254
Phase: train Epoch: 1/3 Iter: 2/62 Batch time: 0.2224
Phase: train Epoch: 1/3 Iter: 3/62 Batch time: 0.2236
Phase: train Epoch: 1/3 Iter: 4/62 Batch time: 0.2278
Phase: train Epoch: 1/3 Iter: 5/62 Batch time: 0.2238
Phase: train Epoch: 1/3 Iter: 6/62 Batch time: 0.2232
Phase: train Epoch: 1/3 Iter: 7/62 Batch time: 0.2233
Phase: train Epoch: 1/3 Iter: 8/62 Batch time: 0.2238
Phase: train Epoch: 1/3 Iter: 9/62 Batch time: 0.2251
Phase: train Epoch: 1/3 Iter: 10/62 Batch time: 0.2242
Phase: train Epoch: 1/3 Iter: 11/62 Batch time: 0.2266
Phase: train Epoch: 1/3 Iter: 12/62 Batch time: 0.2239
Phase: train Epoch: 1/3 Iter: 13/62 Batch time: 0.2254
Phase: train Epoch: 1/3 Iter: 14/62 Batch time: 0.2239
Phase: train Epoch: 1/3 Iter: 15/62 Batch time: 0.2247
Phase: train Epoch: 1/3 Iter: 16/62 Batch time: 0.2237
Phase: train Epoch: 1/3 Iter: 17/62 Batch time: 0.2243
Phase: train Epoch: 1/3 Iter: 18/62 Batch time: 0.2237
Phase: train Epoch: 1/3 Iter: 19/62 Batch time: 0.2255
Phase: train Epoch: 1/3 Iter: 20/62 Batch time: 0.2246
Phase: train Epoch: 1/3 Iter: 21/62 Batch time: 0.2250
Phase: train Epoch: 1/3 Iter: 22/62 Batch time: 0.2244
Phase: train Epoch: 1/3 Iter: 23/62 Batch time: 0.2259
Phase: train Epoch: 1/3 Iter: 24/62 Batch time: 0.2237
Phase: train Epoch: 1/3 Iter: 25/62 Batch time: 0.2239
Phase: train Epoch: 1/3 Iter: 26/62 Batch time: 0.2240
Phase: train Epoch: 1/3 Iter: 27/62 Batch time: 0.2257
Phase: train Epoch: 1/3 Iter: 28/62 Batch time: 0.2267
Phase: train Epoch: 1/3 Iter: 29/62 Batch time: 0.2257
Phase: train Epoch: 1/3 Iter: 30/62 Batch time: 0.2249
Phase: train Epoch: 1/3 Iter: 31/62 Batch time: 0.2230
Phase: train Epoch: 1/3 Iter: 32/62 Batch time: 0.2243
Phase: train Epoch: 1/3 Iter: 33/62 Batch time: 0.2244
Phase: train Epoch: 1/3 Iter: 34/62 Batch time: 0.2235
Phase: train Epoch: 1/3 Iter: 35/62 Batch time: 0.2251
Phase: train Epoch: 1/3 Iter: 36/62 Batch time: 0.2257
Phase: train Epoch: 1/3 Iter: 37/62 Batch time: 0.2253
Phase: train Epoch: 1/3 Iter: 38/62 Batch time: 0.2238
Phase: train Epoch: 1/3 Iter: 39/62 Batch time: 0.2245
Phase: train Epoch: 1/3 Iter: 40/62 Batch time: 0.2257
Phase: train Epoch: 1/3 Iter: 41/62 Batch time: 0.2259
Phase: train Epoch: 1/3 Iter: 42/62 Batch time: 0.2261
Phase: train Epoch: 1/3 Iter: 43/62 Batch time: 0.2269
Phase: train Epoch: 1/3 Iter: 44/62 Batch time: 0.2257
Phase: train Epoch: 1/3 Iter: 45/62 Batch time: 0.2238
Phase: train Epoch: 1/3 Iter: 46/62 Batch time: 0.2237
Phase: train Epoch: 1/3 Iter: 47/62 Batch time: 0.2246
Phase: train Epoch: 1/3 Iter: 48/62 Batch time: 0.2230
Phase: train Epoch: 1/3 Iter: 49/62 Batch time: 0.2239
Phase: train Epoch: 1/3 Iter: 50/62 Batch time: 0.2247
Phase: train Epoch: 1/3 Iter: 51/62 Batch time: 0.2245
Phase: train Epoch: 1/3 Iter: 52/62 Batch time: 0.2248
Phase: train Epoch: 1/3 Iter: 53/62 Batch time: 0.2244
Phase: train Epoch: 1/3 Iter: 54/62 Batch time: 0.2252
Phase: train Epoch: 1/3 Iter: 55/62 Batch time: 0.2264
Phase: train Epoch: 1/3 Iter: 56/62 Batch time: 0.2260
Phase: train Epoch: 1/3 Iter: 57/62 Batch time: 0.2245
Phase: train Epoch: 1/3 Iter: 58/62 Batch time: 0.2245
Phase: train Epoch: 1/3 Iter: 59/62 Batch time: 0.2234
Phase: train Epoch: 1/3 Iter: 60/62 Batch time: 0.2258
Phase: train Epoch: 1/3 Iter: 61/62 Batch time: 0.2271
Phase: train Epoch: 1/3 Loss: 0.6993 Acc: 0.5246
Phase: validation Epoch: 1/3 Iter: 1/39 Batch time: 0.1704
Phase: validation Epoch: 1/3 Iter: 2/39 Batch time: 0.1675
Phase: validation Epoch: 1/3 Iter: 3/39 Batch time: 0.1665
Phase: validation Epoch: 1/3 Iter: 4/39 Batch time: 0.1685
Phase: validation Epoch: 1/3 Iter: 5/39 Batch time: 0.1686
Phase: validation Epoch: 1/3 Iter: 6/39 Batch time: 0.1671
Phase: validation Epoch: 1/3 Iter: 7/39 Batch time: 0.1670
Phase: validation Epoch: 1/3 Iter: 8/39 Batch time: 0.1684
Phase: validation Epoch: 1/3 Iter: 9/39 Batch time: 0.1673
Phase: validation Epoch: 1/3 Iter: 10/39 Batch time: 0.1671
Phase: validation Epoch: 1/3 Iter: 11/39 Batch time: 0.1671
Phase: validation Epoch: 1/3 Iter: 12/39 Batch time: 0.1678
Phase: validation Epoch: 1/3 Iter: 13/39 Batch time: 0.1665
Phase: validation Epoch: 1/3 Iter: 14/39 Batch time: 0.1673
Phase: validation Epoch: 1/3 Iter: 15/39 Batch time: 0.1677
Phase: validation Epoch: 1/3 Iter: 16/39 Batch time: 0.1655
Phase: validation Epoch: 1/3 Iter: 17/39 Batch time: 0.1679
Phase: validation Epoch: 1/3 Iter: 18/39 Batch time: 0.1711
Phase: validation Epoch: 1/3 Iter: 19/39 Batch time: 0.1666
Phase: validation Epoch: 1/3 Iter: 20/39 Batch time: 0.1671
Phase: validation Epoch: 1/3 Iter: 21/39 Batch time: 0.1664
Phase: validation Epoch: 1/3 Iter: 22/39 Batch time: 0.1670
Phase: validation Epoch: 1/3 Iter: 23/39 Batch time: 0.1663
Phase: validation Epoch: 1/3 Iter: 24/39 Batch time: 0.1669
Phase: validation Epoch: 1/3 Iter: 25/39 Batch time: 0.1662
Phase: validation Epoch: 1/3 Iter: 26/39 Batch time: 0.1673
Phase: validation Epoch: 1/3 Iter: 27/39 Batch time: 0.1694
Phase: validation Epoch: 1/3 Iter: 28/39 Batch time: 0.1670
Phase: validation Epoch: 1/3 Iter: 29/39 Batch time: 0.1668
Phase: validation Epoch: 1/3 Iter: 30/39 Batch time: 0.1679
Phase: validation Epoch: 1/3 Iter: 31/39 Batch time: 0.1665
Phase: validation Epoch: 1/3 Iter: 32/39 Batch time: 0.1675
Phase: validation Epoch: 1/3 Iter: 33/39 Batch time: 0.1661
Phase: validation Epoch: 1/3 Iter: 34/39 Batch time: 0.1662
Phase: validation Epoch: 1/3 Iter: 35/39 Batch time: 0.1668
Phase: validation Epoch: 1/3 Iter: 36/39 Batch time: 0.1663
Phase: validation Epoch: 1/3 Iter: 37/39 Batch time: 0.1668
Phase: validation Epoch: 1/3 Iter: 38/39 Batch time: 0.1667
Phase: validation Epoch: 1/3 Iter: 39/39 Batch time: 0.0585
Phase: validation   Epoch: 1/3 Loss: 0.6432 Acc: 0.6536
Phase: train Epoch: 2/3 Iter: 1/62 Batch time: 0.2194
Phase: train Epoch: 2/3 Iter: 2/62 Batch time: 0.2221
Phase: train Epoch: 2/3 Iter: 3/62 Batch time: 0.2238
Phase: train Epoch: 2/3 Iter: 4/62 Batch time: 0.2233
Phase: train Epoch: 2/3 Iter: 5/62 Batch time: 0.2252
Phase: train Epoch: 2/3 Iter: 6/62 Batch time: 0.2263
Phase: train Epoch: 2/3 Iter: 7/62 Batch time: 0.2275
Phase: train Epoch: 2/3 Iter: 8/62 Batch time: 0.2296
Phase: train Epoch: 2/3 Iter: 9/62 Batch time: 0.2255
Phase: train Epoch: 2/3 Iter: 10/62 Batch time: 0.2247
Phase: train Epoch: 2/3 Iter: 11/62 Batch time: 0.2255
Phase: train Epoch: 2/3 Iter: 12/62 Batch time: 0.2242
Phase: train Epoch: 2/3 Iter: 13/62 Batch time: 0.2307
Phase: train Epoch: 2/3 Iter: 14/62 Batch time: 0.2264
Phase: train Epoch: 2/3 Iter: 15/62 Batch time: 0.2256
Phase: train Epoch: 2/3 Iter: 16/62 Batch time: 0.2246
Phase: train Epoch: 2/3 Iter: 17/62 Batch time: 0.2255
Phase: train Epoch: 2/3 Iter: 18/62 Batch time: 0.2247
Phase: train Epoch: 2/3 Iter: 19/62 Batch time: 0.2241
Phase: train Epoch: 2/3 Iter: 20/62 Batch time: 0.2248
Phase: train Epoch: 2/3 Iter: 21/62 Batch time: 0.2254
Phase: train Epoch: 2/3 Iter: 22/62 Batch time: 0.2257
Phase: train Epoch: 2/3 Iter: 23/62 Batch time: 0.2260
Phase: train Epoch: 2/3 Iter: 24/62 Batch time: 0.2247
Phase: train Epoch: 2/3 Iter: 25/62 Batch time: 0.2251
Phase: train Epoch: 2/3 Iter: 26/62 Batch time: 0.2248
Phase: train Epoch: 2/3 Iter: 27/62 Batch time: 0.2259
Phase: train Epoch: 2/3 Iter: 28/62 Batch time: 0.2239
Phase: train Epoch: 2/3 Iter: 29/62 Batch time: 0.2357
Phase: train Epoch: 2/3 Iter: 30/62 Batch time: 0.2248
Phase: train Epoch: 2/3 Iter: 31/62 Batch time: 0.2263
Phase: train Epoch: 2/3 Iter: 32/62 Batch time: 0.2294
Phase: train Epoch: 2/3 Iter: 33/62 Batch time: 0.2277
Phase: train Epoch: 2/3 Iter: 34/62 Batch time: 0.2252
Phase: train Epoch: 2/3 Iter: 35/62 Batch time: 0.2285
Phase: train Epoch: 2/3 Iter: 36/62 Batch time: 0.2250
Phase: train Epoch: 2/3 Iter: 37/62 Batch time: 0.2258
Phase: train Epoch: 2/3 Iter: 38/62 Batch time: 0.2246
Phase: train Epoch: 2/3 Iter: 39/62 Batch time: 0.2253
Phase: train Epoch: 2/3 Iter: 40/62 Batch time: 0.2248
Phase: train Epoch: 2/3 Iter: 41/62 Batch time: 0.2259
Phase: train Epoch: 2/3 Iter: 42/62 Batch time: 0.2257
Phase: train Epoch: 2/3 Iter: 43/62 Batch time: 0.2245
Phase: train Epoch: 2/3 Iter: 44/62 Batch time: 0.2243
Phase: train Epoch: 2/3 Iter: 45/62 Batch time: 0.2245
Phase: train Epoch: 2/3 Iter: 46/62 Batch time: 0.2233
Phase: train Epoch: 2/3 Iter: 47/62 Batch time: 0.2256
Phase: train Epoch: 2/3 Iter: 48/62 Batch time: 0.2248
Phase: train Epoch: 2/3 Iter: 49/62 Batch time: 0.2247
Phase: train Epoch: 2/3 Iter: 50/62 Batch time: 0.2260
Phase: train Epoch: 2/3 Iter: 51/62 Batch time: 0.2263
Phase: train Epoch: 2/3 Iter: 52/62 Batch time: 0.2268
Phase: train Epoch: 2/3 Iter: 53/62 Batch time: 0.2260
Phase: train Epoch: 2/3 Iter: 54/62 Batch time: 0.2241
Phase: train Epoch: 2/3 Iter: 55/62 Batch time: 0.2254
Phase: train Epoch: 2/3 Iter: 56/62 Batch time: 0.2245
Phase: train Epoch: 2/3 Iter: 57/62 Batch time: 0.2240
Phase: train Epoch: 2/3 Iter: 58/62 Batch time: 0.2243
Phase: train Epoch: 2/3 Iter: 59/62 Batch time: 0.2254
Phase: train Epoch: 2/3 Iter: 60/62 Batch time: 0.2271
Phase: train Epoch: 2/3 Iter: 61/62 Batch time: 0.2246
Phase: train Epoch: 2/3 Loss: 0.6141 Acc: 0.7049
Phase: validation Epoch: 2/3 Iter: 1/39 Batch time: 0.1706
Phase: validation Epoch: 2/3 Iter: 2/39 Batch time: 0.1671
Phase: validation Epoch: 2/3 Iter: 3/39 Batch time: 0.1670
Phase: validation Epoch: 2/3 Iter: 4/39 Batch time: 0.1678
Phase: validation Epoch: 2/3 Iter: 5/39 Batch time: 0.1680
Phase: validation Epoch: 2/3 Iter: 6/39 Batch time: 0.1671
Phase: validation Epoch: 2/3 Iter: 7/39 Batch time: 0.1674
Phase: validation Epoch: 2/3 Iter: 8/39 Batch time: 0.1673
Phase: validation Epoch: 2/3 Iter: 9/39 Batch time: 0.1674
Phase: validation Epoch: 2/3 Iter: 10/39 Batch time: 0.1660
Phase: validation Epoch: 2/3 Iter: 11/39 Batch time: 0.1685
Phase: validation Epoch: 2/3 Iter: 12/39 Batch time: 0.1699
Phase: validation Epoch: 2/3 Iter: 13/39 Batch time: 0.1664
Phase: validation Epoch: 2/3 Iter: 14/39 Batch time: 0.1671
Phase: validation Epoch: 2/3 Iter: 15/39 Batch time: 0.1671
Phase: validation Epoch: 2/3 Iter: 16/39 Batch time: 0.1672
Phase: validation Epoch: 2/3 Iter: 17/39 Batch time: 0.1675
Phase: validation Epoch: 2/3 Iter: 18/39 Batch time: 0.1668
Phase: validation Epoch: 2/3 Iter: 19/39 Batch time: 0.1664
Phase: validation Epoch: 2/3 Iter: 20/39 Batch time: 0.1677
Phase: validation Epoch: 2/3 Iter: 21/39 Batch time: 0.1663
Phase: validation Epoch: 2/3 Iter: 22/39 Batch time: 0.1684
Phase: validation Epoch: 2/3 Iter: 23/39 Batch time: 0.1681
Phase: validation Epoch: 2/3 Iter: 24/39 Batch time: 0.1682
Phase: validation Epoch: 2/3 Iter: 25/39 Batch time: 0.1665
Phase: validation Epoch: 2/3 Iter: 26/39 Batch time: 0.1674
Phase: validation Epoch: 2/3 Iter: 27/39 Batch time: 0.1673
Phase: validation Epoch: 2/3 Iter: 28/39 Batch time: 0.1669
Phase: validation Epoch: 2/3 Iter: 29/39 Batch time: 0.1670
Phase: validation Epoch: 2/3 Iter: 30/39 Batch time: 0.1667
Phase: validation Epoch: 2/3 Iter: 31/39 Batch time: 0.1678
Phase: validation Epoch: 2/3 Iter: 32/39 Batch time: 0.1678
Phase: validation Epoch: 2/3 Iter: 33/39 Batch time: 0.1674
Phase: validation Epoch: 2/3 Iter: 34/39 Batch time: 0.1678
Phase: validation Epoch: 2/3 Iter: 35/39 Batch time: 0.1670
Phase: validation Epoch: 2/3 Iter: 36/39 Batch time: 0.1672
Phase: validation Epoch: 2/3 Iter: 37/39 Batch time: 0.1680
Phase: validation Epoch: 2/3 Iter: 38/39 Batch time: 0.1680
Phase: validation Epoch: 2/3 Iter: 39/39 Batch time: 0.0517
Phase: validation   Epoch: 2/3 Loss: 0.5392 Acc: 0.8235
Phase: train Epoch: 3/3 Iter: 1/62 Batch time: 0.2172
Phase: train Epoch: 3/3 Iter: 2/62 Batch time: 0.2209
Phase: train Epoch: 3/3 Iter: 3/62 Batch time: 0.2252
Phase: train Epoch: 3/3 Iter: 4/62 Batch time: 0.2258
Phase: train Epoch: 3/3 Iter: 5/62 Batch time: 0.2278
Phase: train Epoch: 3/3 Iter: 6/62 Batch time: 0.2257
Phase: train Epoch: 3/3 Iter: 7/62 Batch time: 0.2270
Phase: train Epoch: 3/3 Iter: 8/62 Batch time: 0.2249
Phase: train Epoch: 3/3 Iter: 9/62 Batch time: 0.2260
Phase: train Epoch: 3/3 Iter: 10/62 Batch time: 0.2255
Phase: train Epoch: 3/3 Iter: 11/62 Batch time: 0.2248
Phase: train Epoch: 3/3 Iter: 12/62 Batch time: 0.2277
Phase: train Epoch: 3/3 Iter: 13/62 Batch time: 0.2242
Phase: train Epoch: 3/3 Iter: 14/62 Batch time: 0.2254
Phase: train Epoch: 3/3 Iter: 15/62 Batch time: 0.2254
Phase: train Epoch: 3/3 Iter: 16/62 Batch time: 0.2248
Phase: train Epoch: 3/3 Iter: 17/62 Batch time: 0.2249
Phase: train Epoch: 3/3 Iter: 18/62 Batch time: 0.2243
Phase: train Epoch: 3/3 Iter: 19/62 Batch time: 0.2253
Phase: train Epoch: 3/3 Iter: 20/62 Batch time: 0.2230
Phase: train Epoch: 3/3 Iter: 21/62 Batch time: 0.2247
Phase: train Epoch: 3/3 Iter: 22/62 Batch time: 0.2250
Phase: train Epoch: 3/3 Iter: 23/62 Batch time: 0.2258
Phase: train Epoch: 3/3 Iter: 24/62 Batch time: 0.2260
Phase: train Epoch: 3/3 Iter: 25/62 Batch time: 0.2253
Phase: train Epoch: 3/3 Iter: 26/62 Batch time: 0.2258
Phase: train Epoch: 3/3 Iter: 27/62 Batch time: 0.2247
Phase: train Epoch: 3/3 Iter: 28/62 Batch time: 0.2246
Phase: train Epoch: 3/3 Iter: 29/62 Batch time: 0.2266
Phase: train Epoch: 3/3 Iter: 30/62 Batch time: 0.2274
Phase: train Epoch: 3/3 Iter: 31/62 Batch time: 0.2264
Phase: train Epoch: 3/3 Iter: 32/62 Batch time: 0.2251
Phase: train Epoch: 3/3 Iter: 33/62 Batch time: 0.2266
Phase: train Epoch: 3/3 Iter: 34/62 Batch time: 0.2247
Phase: train Epoch: 3/3 Iter: 35/62 Batch time: 0.2275
Phase: train Epoch: 3/3 Iter: 36/62 Batch time: 0.2294
Phase: train Epoch: 3/3 Iter: 37/62 Batch time: 0.2270
Phase: train Epoch: 3/3 Iter: 38/62 Batch time: 0.2245
Phase: train Epoch: 3/3 Iter: 39/62 Batch time: 0.2250
Phase: train Epoch: 3/3 Iter: 40/62 Batch time: 0.2292
Phase: train Epoch: 3/3 Iter: 41/62 Batch time: 0.2249
Phase: train Epoch: 3/3 Iter: 42/62 Batch time: 0.2267
Phase: train Epoch: 3/3 Iter: 43/62 Batch time: 0.2257
Phase: train Epoch: 3/3 Iter: 44/62 Batch time: 0.2268
Phase: train Epoch: 3/3 Iter: 45/62 Batch time: 0.2250
Phase: train Epoch: 3/3 Iter: 46/62 Batch time: 0.2244
Phase: train Epoch: 3/3 Iter: 47/62 Batch time: 0.2248
Phase: train Epoch: 3/3 Iter: 48/62 Batch time: 0.2248
Phase: train Epoch: 3/3 Iter: 49/62 Batch time: 0.2265
Phase: train Epoch: 3/3 Iter: 50/62 Batch time: 0.2261
Phase: train Epoch: 3/3 Iter: 51/62 Batch time: 0.2285
Phase: train Epoch: 3/3 Iter: 52/62 Batch time: 0.2261
Phase: train Epoch: 3/3 Iter: 53/62 Batch time: 0.2246
Phase: train Epoch: 3/3 Iter: 54/62 Batch time: 0.2249
Phase: train Epoch: 3/3 Iter: 55/62 Batch time: 0.2241
Phase: train Epoch: 3/3 Iter: 56/62 Batch time: 0.2246
Phase: train Epoch: 3/3 Iter: 57/62 Batch time: 0.2252
Phase: train Epoch: 3/3 Iter: 58/62 Batch time: 0.2252
Phase: train Epoch: 3/3 Iter: 59/62 Batch time: 0.2273
Phase: train Epoch: 3/3 Iter: 60/62 Batch time: 0.2247
Phase: train Epoch: 3/3 Iter: 61/62 Batch time: 0.2240
Phase: train Epoch: 3/3 Loss: 0.5652 Acc: 0.7336
Phase: validation Epoch: 3/3 Iter: 1/39 Batch time: 0.1712
Phase: validation Epoch: 3/3 Iter: 2/39 Batch time: 0.1679
Phase: validation Epoch: 3/3 Iter: 3/39 Batch time: 0.1660
Phase: validation Epoch: 3/3 Iter: 4/39 Batch time: 0.1667
Phase: validation Epoch: 3/3 Iter: 5/39 Batch time: 0.1665
Phase: validation Epoch: 3/3 Iter: 6/39 Batch time: 0.1666
Phase: validation Epoch: 3/3 Iter: 7/39 Batch time: 0.1676
Phase: validation Epoch: 3/3 Iter: 8/39 Batch time: 0.1666
Phase: validation Epoch: 3/3 Iter: 9/39 Batch time: 0.1668
Phase: validation Epoch: 3/3 Iter: 10/39 Batch time: 0.1669
Phase: validation Epoch: 3/3 Iter: 11/39 Batch time: 0.1668
Phase: validation Epoch: 3/3 Iter: 12/39 Batch time: 0.1673
Phase: validation Epoch: 3/3 Iter: 13/39 Batch time: 0.1670
Phase: validation Epoch: 3/3 Iter: 14/39 Batch time: 0.1682
Phase: validation Epoch: 3/3 Iter: 15/39 Batch time: 0.1667
Phase: validation Epoch: 3/3 Iter: 16/39 Batch time: 0.1665
Phase: validation Epoch: 3/3 Iter: 17/39 Batch time: 0.1675
Phase: validation Epoch: 3/3 Iter: 18/39 Batch time: 0.1675
Phase: validation Epoch: 3/3 Iter: 19/39 Batch time: 0.1664
Phase: validation Epoch: 3/3 Iter: 20/39 Batch time: 0.1674
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
 21%|##1       | 9.49M/44.7M [00:00<00:00, 99.5MB/s]
 47%|####6     | 20.9M/44.7M [00:00<00:00, 111MB/s]
 74%|#######4  | 33.2M/44.7M [00:00<00:00, 119MB/s]
100%|##########| 44.7M/44.7M [00:00<00:00, 119MB/s]
Training started:
Phase: train Epoch: 1/3 Iter: 1/62 Batch time: 0.2599
Phase: train Epoch: 1/3 Iter: 2/62 Batch time: 0.2692
Phase: train Epoch: 1/3 Iter: 3/62 Batch time: 0.2545
Phase: train Epoch: 1/3 Iter: 4/62 Batch time: 0.2535
Phase: train Epoch: 1/3 Iter: 5/62 Batch time: 0.2519
Phase: train Epoch: 1/3 Iter: 6/62 Batch time: 0.2477
Phase: train Epoch: 1/3 Iter: 7/62 Batch time: 0.2512
Phase: train Epoch: 1/3 Iter: 8/62 Batch time: 0.2455
Phase: train Epoch: 1/3 Iter: 9/62 Batch time: 0.2476
Phase: train Epoch: 1/3 Iter: 10/62 Batch time: 0.2485
Phase: train Epoch: 1/3 Iter: 11/62 Batch time: 0.2724
Phase: train Epoch: 1/3 Iter: 12/62 Batch time: 0.2654
Phase: train Epoch: 1/3 Iter: 13/62 Batch time: 0.2524
Phase: train Epoch: 1/3 Iter: 14/62 Batch time: 0.2780
Phase: train Epoch: 1/3 Iter: 15/62 Batch time: 0.2532
Phase: train Epoch: 1/3 Iter: 16/62 Batch time: 0.2613
Phase: train Epoch: 1/3 Iter: 17/62 Batch time: 0.2548
Phase: train Epoch: 1/3 Iter: 18/62 Batch time: 0.2702
Phase: train Epoch: 1/3 Iter: 19/62 Batch time: 0.2570
Phase: train Epoch: 1/3 Iter: 20/62 Batch time: 0.2526
Phase: train Epoch: 1/3 Iter: 21/62 Batch time: 0.2635
Phase: train Epoch: 1/3 Iter: 22/62 Batch time: 0.2549
Phase: train Epoch: 1/3 Iter: 23/62 Batch time: 0.2477
Phase: train Epoch: 1/3 Iter: 24/62 Batch time: 0.2502
Phase: train Epoch: 1/3 Iter: 25/62 Batch time: 0.2500
Phase: train Epoch: 1/3 Iter: 26/62 Batch time: 0.2518
Phase: train Epoch: 1/3 Iter: 27/62 Batch time: 0.2522
Phase: train Epoch: 1/3 Iter: 28/62 Batch time: 0.2496
Phase: train Epoch: 1/3 Iter: 29/62 Batch time: 0.2554
Phase: train Epoch: 1/3 Iter: 30/62 Batch time: 0.2510
Phase: train Epoch: 1/3 Iter: 31/62 Batch time: 0.2502
Phase: train Epoch: 1/3 Iter: 32/62 Batch time: 0.2567
Phase: train Epoch: 1/3 Iter: 33/62 Batch time: 0.2580
Phase: train Epoch: 1/3 Iter: 34/62 Batch time: 0.2500
Phase: train Epoch: 1/3 Iter: 35/62 Batch time: 0.2530
Phase: train Epoch: 1/3 Iter: 36/62 Batch time: 0.2458
Phase: train Epoch: 1/3 Iter: 37/62 Batch time: 0.2453
Phase: train Epoch: 1/3 Iter: 38/62 Batch time: 0.2475
Phase: train Epoch: 1/3 Iter: 39/62 Batch time: 0.2497
Phase: train Epoch: 1/3 Iter: 40/62 Batch time: 0.2561
Phase: train Epoch: 1/3 Iter: 41/62 Batch time: 0.2546
Phase: train Epoch: 1/3 Iter: 42/62 Batch time: 0.2544
Phase: train Epoch: 1/3 Iter: 43/62 Batch time: 0.2493
Phase: train Epoch: 1/3 Iter: 44/62 Batch time: 0.2468
Phase: train Epoch: 1/3 Iter: 45/62 Batch time: 0.2440
Phase: train Epoch: 1/3 Iter: 46/62 Batch time: 0.2563
Phase: train Epoch: 1/3 Iter: 47/62 Batch time: 0.2467
Phase: train Epoch: 1/3 Iter: 48/62 Batch time: 0.2561
Phase: train Epoch: 1/3 Iter: 49/62 Batch time: 0.2479
Phase: train Epoch: 1/3 Iter: 50/62 Batch time: 0.2500
Phase: train Epoch: 1/3 Iter: 51/62 Batch time: 0.2507
Phase: train Epoch: 1/3 Iter: 52/62 Batch time: 0.2506
Phase: train Epoch: 1/3 Iter: 53/62 Batch time: 0.2768
Phase: train Epoch: 1/3 Iter: 54/62 Batch time: 0.2603
Phase: train Epoch: 1/3 Iter: 55/62 Batch time: 0.2483
Phase: train Epoch: 1/3 Iter: 56/62 Batch time: 0.2512
Phase: train Epoch: 1/3 Iter: 57/62 Batch time: 0.2471
Phase: train Epoch: 1/3 Iter: 58/62 Batch time: 0.2417
Phase: train Epoch: 1/3 Iter: 59/62 Batch time: 0.2441
Phase: train Epoch: 1/3 Iter: 60/62 Batch time: 0.2495
Phase: train Epoch: 1/3 Iter: 61/62 Batch time: 0.2470
Phase: train Epoch: 1/3 Loss: 0.6993 Acc: 0.5246
Phase: validation Epoch: 1/3 Iter: 1/39 Batch time: 0.1861
Phase: validation Epoch: 1/3 Iter: 2/39 Batch time: 0.1901
Phase: validation Epoch: 1/3 Iter: 3/39 Batch time: 0.1932
Phase: validation Epoch: 1/3 Iter: 4/39 Batch time: 0.1928
Phase: validation Epoch: 1/3 Iter: 5/39 Batch time: 0.1935
Phase: validation Epoch: 1/3 Iter: 6/39 Batch time: 0.1904
Phase: validation Epoch: 1/3 Iter: 7/39 Batch time: 0.1924
Phase: validation Epoch: 1/3 Iter: 8/39 Batch time: 0.1926
Phase: validation Epoch: 1/3 Iter: 9/39 Batch time: 0.1864
Phase: validation Epoch: 1/3 Iter: 10/39 Batch time: 0.1883
Phase: validation Epoch: 1/3 Iter: 11/39 Batch time: 0.1900
Phase: validation Epoch: 1/3 Iter: 12/39 Batch time: 0.1895
Phase: validation Epoch: 1/3 Iter: 13/39 Batch time: 0.1884
Phase: validation Epoch: 1/3 Iter: 14/39 Batch time: 0.1845
Phase: validation Epoch: 1/3 Iter: 15/39 Batch time: 0.1877
Phase: validation Epoch: 1/3 Iter: 16/39 Batch time: 0.1851
Phase: validation Epoch: 1/3 Iter: 17/39 Batch time: 0.1861
Phase: validation Epoch: 1/3 Iter: 18/39 Batch time: 0.1856
Phase: validation Epoch: 1/3 Iter: 19/39 Batch time: 0.1888
Phase: validation Epoch: 1/3 Iter: 20/39 Batch time: 0.1887
Phase: validation Epoch: 1/3 Iter: 21/39 Batch time: 0.1915
Phase: validation Epoch: 1/3 Iter: 22/39 Batch time: 0.1951
Phase: validation Epoch: 1/3 Iter: 23/39 Batch time: 0.1861
Phase: validation Epoch: 1/3 Iter: 24/39 Batch time: 0.1861
Phase: validation Epoch: 1/3 Iter: 25/39 Batch time: 0.1919
Phase: validation Epoch: 1/3 Iter: 26/39 Batch time: 0.1903
Phase: validation Epoch: 1/3 Iter: 27/39 Batch time: 0.1884
Phase: validation Epoch: 1/3 Iter: 28/39 Batch time: 0.1894
Phase: validation Epoch: 1/3 Iter: 29/39 Batch time: 0.1885
Phase: validation Epoch: 1/3 Iter: 30/39 Batch time: 0.1902
Phase: validation Epoch: 1/3 Iter: 31/39 Batch time: 0.1772
Phase: validation Epoch: 1/3 Iter: 32/39 Batch time: 0.1774
Phase: validation Epoch: 1/3 Iter: 33/39 Batch time: 0.1826
Phase: validation Epoch: 1/3 Iter: 34/39 Batch time: 0.1915
Phase: validation Epoch: 1/3 Iter: 35/39 Batch time: 0.1844
Phase: validation Epoch: 1/3 Iter: 36/39 Batch time: 0.1854
Phase: validation Epoch: 1/3 Iter: 37/39 Batch time: 0.1816
Phase: validation Epoch: 1/3 Iter: 38/39 Batch time: 0.1882
Phase: validation Epoch: 1/3 Iter: 39/39 Batch time: 0.0640
Phase: validation   Epoch: 1/3 Loss: 0.6432 Acc: 0.6536
Phase: train Epoch: 2/3 Iter: 1/62 Batch time: 0.2472
Phase: train Epoch: 2/3 Iter: 2/62 Batch time: 0.2495
Phase: train Epoch: 2/3 Iter: 3/62 Batch time: 0.2619
Phase: train Epoch: 2/3 Iter: 4/62 Batch time: 0.2525
Phase: train Epoch: 2/3 Iter: 5/62 Batch time: 0.2541
Phase: train Epoch: 2/3 Iter: 6/62 Batch time: 0.2557
Phase: train Epoch: 2/3 Iter: 7/62 Batch time: 0.2545
Phase: train Epoch: 2/3 Iter: 8/62 Batch time: 0.2485
Phase: train Epoch: 2/3 Iter: 9/62 Batch time: 0.2476
Phase: train Epoch: 2/3 Iter: 10/62 Batch time: 0.2512
Phase: train Epoch: 2/3 Iter: 11/62 Batch time: 0.2452
Phase: train Epoch: 2/3 Iter: 12/62 Batch time: 0.2537
Phase: train Epoch: 2/3 Iter: 13/62 Batch time: 0.2479
Phase: train Epoch: 2/3 Iter: 14/62 Batch time: 0.2574
Phase: train Epoch: 2/3 Iter: 15/62 Batch time: 0.2578
Phase: train Epoch: 2/3 Iter: 16/62 Batch time: 0.2729
Phase: train Epoch: 2/3 Iter: 17/62 Batch time: 0.2779
Phase: train Epoch: 2/3 Iter: 18/62 Batch time: 0.2634
Phase: train Epoch: 2/3 Iter: 19/62 Batch time: 0.2569
Phase: train Epoch: 2/3 Iter: 20/62 Batch time: 0.2486
Phase: train Epoch: 2/3 Iter: 21/62 Batch time: 0.2580
Phase: train Epoch: 2/3 Iter: 22/62 Batch time: 0.2544
Phase: train Epoch: 2/3 Iter: 23/62 Batch time: 0.2489
Phase: train Epoch: 2/3 Iter: 24/62 Batch time: 0.2647
Phase: train Epoch: 2/3 Iter: 25/62 Batch time: 0.2628
Phase: train Epoch: 2/3 Iter: 26/62 Batch time: 0.2612
Phase: train Epoch: 2/3 Iter: 27/62 Batch time: 0.2647
Phase: train Epoch: 2/3 Iter: 28/62 Batch time: 0.2667
Phase: train Epoch: 2/3 Iter: 29/62 Batch time: 0.2497
Phase: train Epoch: 2/3 Iter: 30/62 Batch time: 0.2520
Phase: train Epoch: 2/3 Iter: 31/62 Batch time: 0.2509
Phase: train Epoch: 2/3 Iter: 32/62 Batch time: 0.2463
Phase: train Epoch: 2/3 Iter: 33/62 Batch time: 0.2495
Phase: train Epoch: 2/3 Iter: 34/62 Batch time: 0.2523
Phase: train Epoch: 2/3 Iter: 35/62 Batch time: 0.2479
Phase: train Epoch: 2/3 Iter: 36/62 Batch time: 0.2455
Phase: train Epoch: 2/3 Iter: 37/62 Batch time: 0.2444
Phase: train Epoch: 2/3 Iter: 38/62 Batch time: 0.2480
Phase: train Epoch: 2/3 Iter: 39/62 Batch time: 0.2543
Phase: train Epoch: 2/3 Iter: 40/62 Batch time: 0.2544
Phase: train Epoch: 2/3 Iter: 41/62 Batch time: 0.2504
Phase: train Epoch: 2/3 Iter: 42/62 Batch time: 0.2541
Phase: train Epoch: 2/3 Iter: 43/62 Batch time: 0.2530
Phase: train Epoch: 2/3 Iter: 44/62 Batch time: 0.2791
Phase: train Epoch: 2/3 Iter: 45/62 Batch time: 0.2562
Phase: train Epoch: 2/3 Iter: 46/62 Batch time: 0.2596
Phase: train Epoch: 2/3 Iter: 47/62 Batch time: 0.2586
Phase: train Epoch: 2/3 Iter: 48/62 Batch time: 0.2626
Phase: train Epoch: 2/3 Iter: 49/62 Batch time: 0.2567
Phase: train Epoch: 2/3 Iter: 50/62 Batch time: 0.2576
Phase: train Epoch: 2/3 Iter: 51/62 Batch time: 0.2570
Phase: train Epoch: 2/3 Iter: 52/62 Batch time: 0.2565
Phase: train Epoch: 2/3 Iter: 53/62 Batch time: 0.2552
Phase: train Epoch: 2/3 Iter: 54/62 Batch time: 0.2584
Phase: train Epoch: 2/3 Iter: 55/62 Batch time: 0.2556
Phase: train Epoch: 2/3 Iter: 56/62 Batch time: 0.2591
Phase: train Epoch: 2/3 Iter: 57/62 Batch time: 0.2582
Phase: train Epoch: 2/3 Iter: 58/62 Batch time: 0.2543
Phase: train Epoch: 2/3 Iter: 59/62 Batch time: 0.2558
Phase: train Epoch: 2/3 Iter: 60/62 Batch time: 0.2541
Phase: train Epoch: 2/3 Iter: 61/62 Batch time: 0.2533
Phase: train Epoch: 2/3 Loss: 0.6141 Acc: 0.7049
Phase: validation Epoch: 2/3 Iter: 1/39 Batch time: 0.1911
Phase: validation Epoch: 2/3 Iter: 2/39 Batch time: 0.1898
Phase: validation Epoch: 2/3 Iter: 3/39 Batch time: 0.1881
Phase: validation Epoch: 2/3 Iter: 4/39 Batch time: 0.1924
Phase: validation Epoch: 2/3 Iter: 5/39 Batch time: 0.1861
Phase: validation Epoch: 2/3 Iter: 6/39 Batch time: 0.1900
Phase: validation Epoch: 2/3 Iter: 7/39 Batch time: 0.1891
Phase: validation Epoch: 2/3 Iter: 8/39 Batch time: 0.1909
Phase: validation Epoch: 2/3 Iter: 9/39 Batch time: 0.1889
Phase: validation Epoch: 2/3 Iter: 10/39 Batch time: 0.1953
Phase: validation Epoch: 2/3 Iter: 11/39 Batch time: 0.1936
Phase: validation Epoch: 2/3 Iter: 12/39 Batch time: 0.1921
Phase: validation Epoch: 2/3 Iter: 13/39 Batch time: 0.1973
Phase: validation Epoch: 2/3 Iter: 14/39 Batch time: 0.1878
Phase: validation Epoch: 2/3 Iter: 15/39 Batch time: 0.1928
Phase: validation Epoch: 2/3 Iter: 16/39 Batch time: 0.1979
Phase: validation Epoch: 2/3 Iter: 17/39 Batch time: 0.1896
Phase: validation Epoch: 2/3 Iter: 18/39 Batch time: 0.2036
Phase: validation Epoch: 2/3 Iter: 19/39 Batch time: 0.1873
Phase: validation Epoch: 2/3 Iter: 20/39 Batch time: 0.1896
Phase: validation Epoch: 2/3 Iter: 21/39 Batch time: 0.1892
Phase: validation Epoch: 2/3 Iter: 22/39 Batch time: 0.1891
Phase: validation Epoch: 2/3 Iter: 23/39 Batch time: 0.1861
Phase: validation Epoch: 2/3 Iter: 24/39 Batch time: 0.1919
Phase: validation Epoch: 2/3 Iter: 25/39 Batch time: 0.1907
Phase: validation Epoch: 2/3 Iter: 26/39 Batch time: 0.1884
Phase: validation Epoch: 2/3 Iter: 27/39 Batch time: 0.1885
Phase: validation Epoch: 2/3 Iter: 28/39 Batch time: 0.1969
Phase: validation Epoch: 2/3 Iter: 29/39 Batch time: 0.1936
Phase: validation Epoch: 2/3 Iter: 30/39 Batch time: 0.1870
Phase: validation Epoch: 2/3 Iter: 31/39 Batch time: 0.1968
Phase: validation Epoch: 2/3 Iter: 32/39 Batch time: 0.1899
Phase: validation Epoch: 2/3 Iter: 33/39 Batch time: 0.1914
Phase: validation Epoch: 2/3 Iter: 34/39 Batch time: 0.1909
Phase: validation Epoch: 2/3 Iter: 35/39 Batch time: 0.2010
Phase: validation Epoch: 2/3 Iter: 36/39 Batch time: 0.1956
Phase: validation Epoch: 2/3 Iter: 37/39 Batch time: 0.1938
Phase: validation Epoch: 2/3 Iter: 38/39 Batch time: 0.1935
Phase: validation Epoch: 2/3 Iter: 39/39 Batch time: 0.0618
Phase: validation   Epoch: 2/3 Loss: 0.5392 Acc: 0.8235
Phase: train Epoch: 3/3 Iter: 1/62 Batch time: 0.2506
Phase: train Epoch: 3/3 Iter: 2/62 Batch time: 0.2495
Phase: train Epoch: 3/3 Iter: 3/62 Batch time: 0.2592
Phase: train Epoch: 3/3 Iter: 4/62 Batch time: 0.2586
Phase: train Epoch: 3/3 Iter: 5/62 Batch time: 0.2622
Phase: train Epoch: 3/3 Iter: 6/62 Batch time: 0.2626
Phase: train Epoch: 3/3 Iter: 7/62 Batch time: 0.2702
Phase: train Epoch: 3/3 Iter: 8/62 Batch time: 0.2742
Phase: train Epoch: 3/3 Iter: 9/62 Batch time: 0.2641
Phase: train Epoch: 3/3 Iter: 10/62 Batch time: 0.2578
Phase: train Epoch: 3/3 Iter: 11/62 Batch time: 0.2718
Phase: train Epoch: 3/3 Iter: 12/62 Batch time: 0.2555
Phase: train Epoch: 3/3 Iter: 13/62 Batch time: 0.2617
Phase: train Epoch: 3/3 Iter: 14/62 Batch time: 0.2656
Phase: train Epoch: 3/3 Iter: 15/62 Batch time: 0.2592
Phase: train Epoch: 3/3 Iter: 16/62 Batch time: 0.2546
Phase: train Epoch: 3/3 Iter: 17/62 Batch time: 0.2570
Phase: train Epoch: 3/3 Iter: 18/62 Batch time: 0.2644
Phase: train Epoch: 3/3 Iter: 19/62 Batch time: 0.2631
Phase: train Epoch: 3/3 Iter: 20/62 Batch time: 0.2593
Phase: train Epoch: 3/3 Iter: 21/62 Batch time: 0.2598
Phase: train Epoch: 3/3 Iter: 22/62 Batch time: 0.2620
Phase: train Epoch: 3/3 Iter: 23/62 Batch time: 0.2574
Phase: train Epoch: 3/3 Iter: 24/62 Batch time: 0.2628
Phase: train Epoch: 3/3 Iter: 25/62 Batch time: 0.2594
Phase: train Epoch: 3/3 Iter: 26/62 Batch time: 0.2549
Phase: train Epoch: 3/3 Iter: 27/62 Batch time: 0.2607
Phase: train Epoch: 3/3 Iter: 28/62 Batch time: 0.2519
Phase: train Epoch: 3/3 Iter: 29/62 Batch time: 0.2541
Phase: train Epoch: 3/3 Iter: 30/62 Batch time: 0.2520
Phase: train Epoch: 3/3 Iter: 31/62 Batch time: 0.2586
Phase: train Epoch: 3/3 Iter: 32/62 Batch time: 0.2470
Phase: train Epoch: 3/3 Iter: 33/62 Batch time: 0.2488
Phase: train Epoch: 3/3 Iter: 34/62 Batch time: 0.2534
Phase: train Epoch: 3/3 Iter: 35/62 Batch time: 0.2585
Phase: train Epoch: 3/3 Iter: 36/62 Batch time: 0.2563
Phase: train Epoch: 3/3 Iter: 37/62 Batch time: 0.2512
Phase: train Epoch: 3/3 Iter: 38/62 Batch time: 0.2550
Phase: train Epoch: 3/3 Iter: 39/62 Batch time: 0.2602
Phase: train Epoch: 3/3 Iter: 40/62 Batch time: 0.2877
Phase: train Epoch: 3/3 Iter: 41/62 Batch time: 0.2711
Phase: train Epoch: 3/3 Iter: 42/62 Batch time: 0.2802
Phase: train Epoch: 3/3 Iter: 43/62 Batch time: 0.2560
Phase: train Epoch: 3/3 Iter: 44/62 Batch time: 0.2561
Phase: train Epoch: 3/3 Iter: 45/62 Batch time: 0.2533
Phase: train Epoch: 3/3 Iter: 46/62 Batch time: 0.2507
Phase: train Epoch: 3/3 Iter: 47/62 Batch time: 0.2595
Phase: train Epoch: 3/3 Iter: 48/62 Batch time: 0.2581
Phase: train Epoch: 3/3 Iter: 49/62 Batch time: 0.2530
Phase: train Epoch: 3/3 Iter: 50/62 Batch time: 0.2465
Phase: train Epoch: 3/3 Iter: 51/62 Batch time: 0.2553
Phase: train Epoch: 3/3 Iter: 52/62 Batch time: 0.2528
Phase: train Epoch: 3/3 Iter: 53/62 Batch time: 0.2533
Phase: train Epoch: 3/3 Iter: 54/62 Batch time: 0.2519
Phase: train Epoch: 3/3 Iter: 55/62 Batch time: 0.2508
Phase: train Epoch: 3/3 Iter: 56/62 Batch time: 0.2475
Phase: train Epoch: 3/3 Iter: 57/62 Batch time: 0.2489
Phase: train Epoch: 3/3 Iter: 58/62 Batch time: 0.2516
Phase: train Epoch: 3/3 Iter: 59/62 Batch time: 0.2603
Phase: train Epoch: 3/3 Iter: 60/62 Batch time: 0.2592
Phase: train Epoch: 3/3 Iter: 61/62 Batch time: 0.2481
Phase: train Epoch: 3/3 Loss: 0.5652 Acc: 0.7336
Phase: validation Epoch: 3/3 Iter: 1/39 Batch time: 0.1892
Phase: validation Epoch: 3/3 Iter: 2/39 Batch time: 0.1898
Phase: validation Epoch: 3/3 Iter: 3/39 Batch time: 0.1897
Phase: validation Epoch: 3/3 Iter: 4/39 Batch time: 0.1850
Phase: validation Epoch: 3/3 Iter: 5/39 Batch time: 0.1916
Phase: validation Epoch: 3/3 Iter: 6/39 Batch time: 0.1921
Phase: validation Epoch: 3/3 Iter: 7/39 Batch time: 0.1939
Phase: validation Epoch: 3/3 Iter: 8/39 Batch time: 0.1896
Phase: validation Epoch: 3/3 Iter: 9/39 Batch time: 0.1866
Phase: validation Epoch: 3/3 Iter: 10/39 Batch time: 0.1864
Phase: validation Epoch: 3/3 Iter: 11/39 Batch time: 0.1815
Phase: validation Epoch: 3/3 Iter: 12/39 Batch time: 0.1844
Phase: validation Epoch: 3/3 Iter: 13/39 Batch time: 0.1838
Phase: validation Epoch: 3/3 Iter: 14/39 Batch time: 0.1833
Phase: validation Epoch: 3/3 Iter: 15/39 Batch time: 0.1893
Phase: validation Epoch: 3/3 Iter: 16/39 Batch time: 0.1857
Phase: validation Epoch: 3/3 Iter: 17/39 Batch time: 0.1941
Phase: validation Epoch: 3/3 Iter: 18/39 Batch time: 0.1924
Phase: validation Epoch: 3/3 Iter: 19/39 Batch time: 0.1898
Phase: validation Epoch: 3/3 Iter: 20/39 Batch time: 0.1930
Phase: validation Epoch: 3/3 Iter: 21/39 Batch time: 0.1872
Phase: validation Epoch: 3/3 Iter: 22/39 Batch time: 0.1902
Phase: validation Epoch: 3/3 Iter: 23/39 Batch time: 0.1864
Phase: validation Epoch: 3/3 Iter: 24/39 Batch time: 0.1958
Phase: validation Epoch: 3/3 Iter: 25/39 Batch time: 0.1917
Phase: validation Epoch: 3/3 Iter: 26/39 Batch time: 0.1874
Phase: validation Epoch: 3/3 Iter: 27/39 Batch time: 0.1923
Phase: validation Epoch: 3/3 Iter: 28/39 Batch time: 0.1911
Phase: validation Epoch: 3/3 Iter: 29/39 Batch time: 0.1872
Phase: validation Epoch: 3/3 Iter: 30/39 Batch time: 0.1903
Phase: validation Epoch: 3/3 Iter: 31/39 Batch time: 0.1850
Phase: validation Epoch: 3/3 Iter: 32/39 Batch time: 0.1933
Phase: validation Epoch: 3/3 Iter: 33/39 Batch time: 0.1867
Phase: validation Epoch: 3/3 Iter: 34/39 Batch time: 0.1874
Phase: validation Epoch: 3/3 Iter: 35/39 Batch time: 0.1898
Phase: validation Epoch: 3/3 Iter: 36/39 Batch time: 0.1836
Phase: validation Epoch: 3/3 Iter: 37/39 Batch time: 0.1916
Phase: validation Epoch: 3/3 Iter: 38/39 Batch time: 0.1899
Phase: validation Epoch: 3/3 Iter: 39/39 Batch time: 0.0592
Phase: validation   Epoch: 3/3 Loss: 0.4484 Acc: 0.8497
Training completed in 1m 18s
Best test loss: 0.4484 | Best test accuracy: 0.8497
 </code>
 </pre>
 </details>

---

## 2. tutorial_error_mitigation.html <a name="demo1"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_error_mitigation.html):

```
2: ──RY(4.05)──╭C──RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)──╰Z──RY(-3.32)──╭C───RY(-4.05)─────────────────┤
3: ──RY(3.51)──╰Z──RY(3.66)───RY(-3.66)──────────────────────────────────────╰Z───RY(-3.51)─────────────────┤
0: ──RY(4.56)──╭C──RY(5.93)───RY(-5.93)───────────────────────────────────────────────────────────╭C──RY(-4.56)──┤
1: ──RY(3.6)───╰Z──RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)──RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z──RY(-3.6)───┤
2: ──RY(4.05)──╭C──RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)───────────────────────╰Z──RY(-3.32)──╭C──RY(-4.05)──┤
3: ──RY(3.51)──╰Z──RY(3.66)───RY(-3.66)───────────────────────────────────────────────────────────╰Z──RY(-3.51)──┤
0.9667333558741233
0: ──RY(4.56)──╭C──RY(5.93)───RY(-5.93)──────────────────────────────────────────────╭C──RY(-4.56)──┤
1: ──RY(3.6)───╰Z──RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)──╭C──╭C──╭C──RY(-5.9)───╰Z──RY(-3.6)───┤
2: ──RY(4.05)──╭C──RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)──╰Z──╰Z──╰Z──RY(-3.32)──╭C──RY(-4.05)──┤
3: ──RY(3.51)──╰Z──RY(3.66)───RY(-3.66)──────────────────────────────────────────────╰Z──RY(-3.51)──┤
0: ──RY(4.56)──╭C──RY(5.93)───RY(-5.93)──────────────────────────────────────╭C──╭C──────────╭C──RY(-4.56)──┤
1: ──RY(3.6)───╰Z──RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z──╰Z──────────╰Z──RY(-3.6)───┤
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_error_mitigation.html):

```
2: ──RY(4.05)──╭C──RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)──╰Z──RY(-3.32)──╭C──RY(-4.05)───────────────────────┤
3: ──RY(3.51)──╰Z──RY(3.66)───RY(-3.66)──────────────────────────────────────╰Z──RY(-3.51)──RY(3.51)──RY(-3.51)──┤
0: ──RY(4.56)──╭C──RY(5.93)───RY(-5.93)──────────────────────────────────────╭C───RY(-4.56)─────────────────┤
1: ──RY(3.6)───╰Z──RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z───RY(-3.6)──────────────────┤
2: ──RY(4.05)──╭C──RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)──╰Z──RY(-3.32)──╭C──╭C──────────╭C──RY(-4.05)──┤
3: ──RY(3.51)──╰Z──RY(3.66)───RY(-3.66)──────────────────────────────────────╰Z──╰Z──────────╰Z──RY(-3.51)──┤
0.9713169438097291
0: ──RY(4.56)──╭C──RY(5.93)───RY(-5.93)───────────────────────────────────────────────────────────╭C──RY(-4.56)──┤
1: ──RY(3.6)───╰Z──RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)──╭C──RY(-5.9)────RY(5.9)──RY(-5.9)───╰Z──RY(-3.6)───┤
2: ──RY(4.05)──╭C──RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)──╰Z──RY(-3.32)──╭C────────RY(-4.05)─────────────────┤
3: ──RY(3.51)──╰Z──RY(3.66)───RY(-3.66)──────────────────────────────────────╰Z────────RY(-3.51)─────────────────┤
0: ──RY(4.56)──╭C──RY(5.93)───RY(-5.93)──────────────────────────────────────╭C──RY(-4.56)───────────────────────┤
1: ──RY(3.6)───╰Z──RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z──RY(-3.6)────────────────────────┤
```

---

## 3. tutorial_measurement_optimize.html <a name="demo2"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_measurement_optimize.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
+ (0.7829661725950183) [Z10]
+ (0.7829661725950183) [Z11]
+ (0.8084581961720481) [Z12]
+ (0.8084581961720482) [Z13]
+ (1.2034402289145627) [Z4]
+ (1.2034402289145631) [Z5]
+ (1.3096862988615443) [Z7]
+ (1.3096862988615445) [Z6]
+ (1.3693525634718182) [Z9]
+ (1.6538942226831719) [Z2]
+ (1.6538942226831719) [Z3]
+ (12.412630742111759) [Z0]
+ (12.412630742111759) [Z1]
+ (-8.194261373008289e-06) [Y10 Y12]
+ (-8.194261373008289e-06) [X10 X12]
+ (-1.8540608577606868e-06) [Y5 Y7]
+ (-1.8540608577606868e-06) [X5 X7]
+ (-7.764994116405362e-07) [Y3 Y5]
+ (-7.764994116405362e-07) [X3 X5]
+ (-5.92976581585045e-07) [Y4 Y6]
+ (-5.92976581585045e-07) [X4 X6]
+ (1.6021167402774385e-06) [Y2 Y4]
+ (1.6021167402774385e-06) [X2 X4]
+ (7.954413176965578e-06) [Y11 Y13]
+ (7.954413176965578e-06) [X11 X13]
+ (0.003276971931231682) [Y1 Y3]
+ (0.003276971931231682) [X1 X3]
+ (0.10433064780651399) [Y0 Y2]
+ (0.10433064780651399) [X0 X2]
+ (0.1127038692033221) [Z10 Z12]
+ (0.1127038692033221) [Z11 Z13]
+ (0.11383573679388657) [Z4 Z12]
+ (0.11383573679388657) [Z5 Z13]
+ (0.11952438964682675) [Z6 Z10]
+ (0.11952438964682675) [Z7 Z11]
+ (0.12489990917237596) [Z4 Z10]
+ (0.12489990917237596) [Z5 Z11]
+ (0.1249580773950322) [Z2 Z4]
+ (0.1249580773950322) [Z3 Z5]
+ (0.12799502492468404) [Z2 Z10]
+ (0.12799502492468404) [Z3 Z11]
+ (0.13401715261963718) [Z6 Z12]
+ (0.13401715261963718) [Z7 Z13]
+ (0.13701191674040755) [Z4 Z6]
+ (0.13701191674040755) [Z5 Z7]
+ (0.13734953064261313) [Z6 Z11]
+ (0.13734953064261313) [Z7 Z10]
+ (0.13739104762683232) [Z2 Z6]
+ (0.13739104762683232) [Z3 Z7]
+ (0.13766872645852565) [Z8 Z10]
+ (0.13766872645852565) [Z9 Z11]
+ (0.14011289865354817) [Z2 Z12]
+ (0.14011289865354817) [Z3 Z13]
+ (0.1413890529194281) [Z10 Z13]
+ (0.1413890529194281) [Z11 Z12]
+ (0.14257997712485748) [Z4 Z11]
+ (0.14257997712485748) [Z5 Z10]
+ (0.14722943218766166) [Z8 Z11]
+ (0.14722943218766166) [Z9 Z10]
+ (0.14926355147388898) [Z10 Z11]
+ (0.14960702684445293) [Z4 Z8]
+ (0.14960702684445293) [Z5 Z9]
+ (0.15071408121008295) [Z2 Z8]
+ (0.15071408121008295) [Z3 Z9]
+ (0.1513832716142885) [Z6 Z13]
+ (0.1513832716142885) [Z7 Z12]
+ (0.15215040708869046) [Z4 Z13]
+ (0.15215040708869046) [Z5 Z12]
+ (0.15337968243314143) [Z2 Z11]
+ (0.15337968243314143) [Z3 Z10]
+ (0.15435748657223639) [Z12 Z13]
+ (0.15569010671752465) [Z2 Z13]
+ (0.15569010671752465) [Z3 Z12]
+ (0.15582269051553113) [Z8 Z13]
+ (0.15582269051553113) [Z9 Z12]
+ (0.1567639617643099) [Z4 Z9]
+ (0.1567639617643099) [Z5 Z8]
+ (0.15755314797985656) [Z4 Z5]
+ (0.16079764534838567) [Z2 Z5]
+ (0.16079764534838567) [Z3 Z4]
+ (0.1675665326546128) [Z6 Z8]
+ (0.1675665326546128) [Z7 Z9]
+ (0.16853486561579956) [Z2 Z7]
+ (0.16853486561579956) [Z3 Z6]
+ (0.18143991440303886) [Z6 Z9]
+ (0.18143991440303886) [Z7 Z8]
+ (0.1869082047691256) [Z2 Z9]
+ (0.1869082047691256) [Z3 Z8]
+ (0.19299723935364205) [Z0 Z10]
+ (0.19299723935364205) [Z1 Z11]
+ (0.19392534613270213) [Z6 Z7]
+ (0.19661770890342123) [Z0 Z4]
+ (0.19661770890342123) [Z1 Z5]
+ (0.199363545373608) [Z0 Z5]
+ (0.199363545373608) [Z1 Z4]
+ (0.20072866460441735) [Z0 Z11]
+ (0.20072866460441735) [Z1 Z10]
+ (0.21102659849791497) [Z0 Z12]
+ (0.21102659849791497) [Z1 Z13]
+ (0.21631037498631792) [Z0 Z13]
+ (0.21631037498631792) [Z1 Z12]
+ (0.23671080783830406) [Z0 Z2]
+ (0.23671080783830406) [Z1 Z3]
+ (0.24164663936017197) [Z0 Z6]
+ (0.24164663936017197) [Z1 Z7]
+ (0.2485348337131425) [Z0 Z7]
+ (0.2485348337131425) [Z1 Z6]
+ (0.2512944567459167) [Z0 Z3]
+ (0.2512944567459167) [Z1 Z2]
+ (0.27232518306605663) [Z0 Z8]
+ (0.27232518306605663) [Z1 Z9]
+ (0.27883454426723386) [Z0 Z9]
+ (0.27883454426723386) [Z1 Z8]
+ (1.1861763734860473) [Z0 Z1]
+ (-1.2260484987950837e-05) [Y5 Z6 Y7]
+ (-1.2260484987950837e-05) [X5 Z6 X7]
+ (-1.2260484987950836e-05) [Y4 Z5 Y6]
+ (-1.2260484987950836e-05) [X4 Z5 X6]
+ (-1.0722312157392391e-05) [Y10 Z11 Y12]
+ (-1.0722312157392391e-05) [X10 Z11 X12]
+ (-1.072231215739239e-05) [Y11 Z12 Y13]
+ (-1.072231215739239e-05) [X11 Z12 X13]
+ (-3.887051671834741e-06) [Y2 Z3 Y4]
+ (-3.887051671834741e-06) [X2 Z3 X4]
+ (-3.887051671834741e-06) [Y3 Z4 Y5]
+ (-3.887051671834741e-06) [X3 Z4 X5]
+ (0.12507032579772107) [Y1 Z2 Y3]
+ (0.12507032579772107) [X1 Z2 X3]
+ (0.1250703257977211) [Y0 Z1 Y2]
+ (0.1250703257977211) [X0 Z1 X2]
+ (-0.03831467029480387) [Y4 Y5 X12 X13]
+ (-0.03831467029480387) [X4 X5 Y12 Y13]
+ (-0.03619412355904263) [Y2 Y3 X8 X9]
+ (-0.03619412355904263) [X2 X3 Y8 Y9]
+ (-0.035839567953353475) [Y2 Y3 X4 X5]
+ (-0.035839567953353475) [X2 X3 Y4 Y5]
+ (-0.031143817988967207) [Y2 Y3 X6 X7]
+ (-0.031143817988967207) [X2 X3 Y6 Y7]
+ (-0.028685183716106004) [Y10 Y11 X12 X13]
+ (-0.028685183716106004) [X10 X11 Y12 Y13]
+ (-0.02599617759802123) [Y3 Z4 Z5 Y7]
+ (-0.02599617759802123) [X3 Z4 Z5 X7]
+ (-0.025384657508457396) [Y2 Y3 X10 X11]
+ (-0.025384657508457396) [X2 X3 Y10 Y11]
+ (-0.01902824244384734) [Y3 Y4 X11 X12]
+ (-0.01902824244384734) [X3 X4 Y11 Y12]
+ (-0.017825140995786384) [Y6 Y7 X10 X11]
+ (-0.017825140995786384) [X6 X7 Y10 Y11]
+ (-0.017680067952481532) [Y4 Y5 X10 X11]
+ (-0.017680067952481532) [X4 X5 Y10 Y11]
+ (-0.017366118994651316) [Y6 Y7 X12 X13]
+ (-0.017366118994651316) [X6 X7 Y12 Y13]
+ (-0.015577208063976474) [Y2 Y3 X12 X13]
+ (-0.015577208063976474) [X2 X3 Y12 Y13]
+ (-0.014583648907612668) [Y0 Y1 X2 X3]
+ (-0.014583648907612668) [X0 X1 Y2 Y3]
+ (-0.013873381748426087) [Y6 Y7 X8 X9]
+ (-0.013873381748426087) [X6 X7 Y8 Y9]
+ (-0.011982389010247911) [Y4 Y5 X6 X7]
+ (-0.011982389010247911) [X4 X5 Y6 Y7]
+ (-0.011285190200840851) [Y5 X6 X11 Y12]
+ (-0.011285190200840851) [X5 Y6 Y11 X12]
+ (-0.009560705729135968) [Y8 Y9 X10 X11]
+ (-0.009560705729135968) [X8 X9 Y10 Y11]
+ (-0.008125251921381025) [Y1 X2 X8 Y9]
+ (-0.008125251921381025) [Y1 Y2 Y8 Y9]
+ (-0.008125251921381025) [X1 X2 X8 X9]
+ (-0.008125251921381025) [X1 Y2 Y8 X9]
+ (-0.007731425250775297) [Y0 Y1 X10 X11]
+ (-0.007731425250775297) [X0 X1 Y10 Y11]
+ (-0.007156934919856944) [Y4 Y5 X8 X9]
+ (-0.007156934919856944) [X4 X5 Y8 Y9]
+ (-0.006888194352970551) [Y0 Y1 X6 X7]
+ (-0.006888194352970551) [X0 X1 Y6 Y7]
+ (-0.006509361201177231) [Y0 Y1 X8 X9]
+ (-0.006509361201177231) [X0 X1 Y8 Y9]
+ (-0.006087822480561859) [Y8 Y9 X12 X13]
+ (-0.006087822480561859) [X8 X9 Y12 Y13]
+ (-0.005283776488402952) [Y0 Y1 X12 X13]
+ (-0.005283776488402952) [X0 X1 Y12 Y13]
+ (-0.005143391768825095) [Y3 X4 X5 Y6]
+ (-0.005143391768825095) [X3 Y4 Y5 X6]
+ (-0.004684903388155233) [Y1 X2 X6 Y7]
+ (-0.004684903388155233) [Y1 Y2 Y6 Y7]
+ (-0.004684903388155233) [X1 X2 X6 X7]
+ (-0.004684903388155233) [X1 Y2 Y6 X7]
+ (-0.004575007626639204) [Y1 X2 X12 Y13]
+ (-0.004575007626639204) [Y1 Y2 Y12 Y13]
+ (-0.004575007626639204) [X1 X2 X12 X13]
+ (-0.004575007626639204) [X1 Y2 Y12 X13]
+ (-0.004424855449441854) [Y1 X2 X4 Y5]
+ (-0.004424855449441854) [Y1 Y2 Y4 Y5]
+ (-0.004424855449441854) [X1 X2 X4 X5]
+ (-0.004424855449441854) [X1 Y2 Y4 X5]
+ (-0.003479511890334311) [Y2 Z3 Z5 Y6]
+ (-0.003479511890334311) [X2 Z3 Z5 X6]
+ (-0.003479511890334311) [Y3 Z4 Z6 Y7]
+ (-0.003479511890334311) [X3 Z4 Z6 X7]
+ (-0.0027458364701868077) [Y0 Y1 X4 X5]
+ (-0.0027458364701868077) [X0 X1 Y4 Y5]
+ (-0.0017992194936629795) [Y1 X2 X10 Y11]
+ (-0.0017992194936629795) [Y1 Y2 Y10 Y11]
+ (-0.0017992194936629795) [X1 X2 X10 X11]
+ (-0.0017992194936629795) [X1 Y2 Y10 X11]
+ (-0.00029219862611107165) [Y7 Y8 X9 X10]
+ (-0.00029219862611107165) [X7 X8 Y9 Y10]
+ (-8.194261373008287e-06) [Z10 Y11 Z12 Y13]
+ (-8.194261373008287e-06) [Z10 X11 Z12 X13]
+ (-7.801707501383158e-06) [Y2 Z3 Y4 Z11]
+ (-7.801707501383158e-06) [X2 Z3 X4 Z11]
+ (-7.801707501383158e-06) [Y3 Z4 Y5 Z10]
+ (-7.801707501383158e-06) [X3 Z4 X5 Z10]
+ (-4.64305106893822e-06) [Y3 X4 X10 Y11]
+ (-4.64305106893822e-06) [Y3 Y4 Y10 Y11]
+ (-4.64305106893822e-06) [X3 X4 X10 X11]
+ (-4.64305106893822e-06) [X3 Y4 Y10 X11]
+ (-4.5888551560282864e-06) [Y4 Z5 Y6 Z13]
+ (-4.5888551560282864e-06) [X4 Z5 X6 Z13]
+ (-4.5888551560282864e-06) [Y5 Z6 Y7 Z12]
+ (-4.5888551560282864e-06) [X5 Z6 X7 Z12]
+ (-4.556569218702232e-06) [Y5 X6 X12 Y13]
+ (-4.556569218702232e-06) [Y5 Y6 Y12 Y13]
+ (-4.556569218702232e-06) [X5 X6 X12 X13]
+ (-4.556569218702232e-06) [X5 Y6 Y12 X13]
+ (-3.6945132947170326e-06) [Y4 X5 X11 Y12]
+ (-3.6945132947170326e-06) [Y4 Y5 Y11 Y12]
+ (-3.6945132947170326e-06) [X4 X5 X11 X12]
+ (-3.6945132947170326e-06) [X4 Y5 Y11 X12]
+ (-3.344081556159941e-06) [Z0 Y5 Z6 Y7]
+ (-3.344081556159941e-06) [Z0 X5 Z6 X7]
+ (-3.344081556159941e-06) [Z1 Y4 Z5 Y6]
+ (-3.344081556159941e-06) [Z1 X4 Z5 X6]
+ (-3.1586564324449382e-06) [Y2 Z3 Y4 Z10]
+ (-3.1586564324449382e-06) [X2 Z3 X4 Z10]
+ (-3.1586564324449382e-06) [Y3 Z4 Y5 Z11]
+ (-3.1586564324449382e-06) [X3 Z4 X5 Z11]
+ (-3.0993492433109997e-06) [Z0 Y4 Z5 Y6]
+ (-3.0993492433109997e-06) [Z0 X4 Z5 X6]
+ (-3.0993492433109997e-06) [Z1 Y5 Z6 Y7]
+ (-3.0993492433109997e-06) [Z1 X5 Z6 X7]
+ (-2.8909678817957004e-06) [Z6 Y11 Z12 Y13]
+ (-2.8909678817957004e-06) [Z6 X11 Z12 X13]
+ (-2.8909678817957004e-06) [Z7 Y10 Z11 Y12]
+ (-2.8909678817957004e-06) [Z7 X10 Z11 X12]
+ (-2.1776646051847334e-06) [Z0 Y10 Z11 Y12]
+ (-2.1776646051847334e-06) [Z0 X10 Z11 X12]
+ (-2.1776646051847334e-06) [Z1 Y11 Z12 Y13]
+ (-2.1776646051847334e-06) [Z1 X11 Z12 X13]
+ (-1.8818501830359057e-06) [Y4 Z5 Y6 Z9]
+ (-1.8818501830359057e-06) [X4 Z5 X6 Z9]
+ (-1.8818501830359057e-06) [Y5 Z6 Y7 Z8]
+ (-1.8818501830359057e-06) [X5 Z6 X7 Z8]
+ (-1.855120121733482e-06) [Z6 Y10 Z11 Y12]
+ (-1.855120121733482e-06) [Z6 X10 Z11 X12]
+ (-1.855120121733482e-06) [Z7 Y11 Z12 Y13]
+ (-1.855120121733482e-06) [Z7 X11 Z12 X13]
+ (-1.8540608577606868e-06) [Y4 Z5 Y6 Z7]
+ (-1.8540608577606868e-06) [X4 Z5 X6 Z7]
+ (-1.8163031696846047e-06) [Z4 Y11 Z12 Y13]
+ (-1.8163031696846047e-06) [Z4 X11 Z12 X13]
+ (-1.8163031696846047e-06) [Z5 Y10 Z11 Y12]
+ (-1.8163031696846047e-06) [Z5 X10 Z11 X12]
+ (-1.692397828655042e-06) [Y4 Z5 Y6 Z10]
+ (-1.692397828655042e-06) [X4 Z5 X6 Z10]
+ (-1.692397828655042e-06) [Y5 Z6 Y7 Z11]
+ (-1.692397828655042e-06) [X5 Z6 X7 Z11]
+ (-1.6148794140156238e-06) [Z0 Y11 Z12 Y13]
+ (-1.6148794140156238e-06) [Z0 X11 Z12 X13]
+ (-1.6148794140156238e-06) [Z1 Y10 Z11 Y12]
+ (-1.6148794140156238e-06) [Z1 X10 Z11 X12]
+ (-1.597317197928938e-06) [Z8 Y10 Z11 Y12]
+ (-1.597317197928938e-06) [Z8 X10 Z11 X12]
+ (-1.597317197928938e-06) [Z9 Y11 Z12 Y13]
+ (-1.597317197928938e-06) [Z9 X11 Z12 X13]
+ (-1.4548424489056125e-06) [Y3 X4 X6 Y7]
+ (-1.4548424489056125e-06) [Y3 Y4 Y6 Y7]
+ (-1.4548424489056125e-06) [X3 X4 X6 X7]
+ (-1.4548424489056125e-06) [X3 Y4 Y6 X7]
+ (-1.398044908011661e-06) [Y4 Z5 Y6 Z8]
+ (-1.398044908011661e-06) [X4 Z5 X6 Z8]
+ (-1.398044908011661e-06) [Y5 Z6 Y7 Z9]
+ (-1.398044908011661e-06) [X5 Z6 X7 Z9]
+ (-1.195489009879766e-06) [Y2 Z3 Y4 Z7]
+ (-1.195489009879766e-06) [X2 Z3 X4 Z7]
+ (-1.195489009879766e-06) [Y3 Z4 Y5 Z6]
+ (-1.195489009879766e-06) [X3 Z4 X5 Z6]
+ (-1.1908508082388385e-06) [Z0 Y3 Z4 Y5]
+ (-1.1908508082388385e-06) [Z0 X3 Z4 X5]
+ (-1.1908508082388385e-06) [Z1 Y2 Z3 Y4]
+ (-1.1908508082388385e-06) [Z1 X2 Z3 X4]
+ (-1.1708301369089434e-06) [Z2 Y5 Z6 Y7]
+ (-1.1708301369089434e-06) [Z2 X5 Z6 X7]
+ (-1.1708301369089434e-06) [Z3 Y4 Z5 Y6]
+ (-1.1708301369089434e-06) [Z3 X4 Z5 X6]
+ (-1.06322834262159e-06) [Z2 Y10 Z11 Y12]
+ (-1.06322834262159e-06) [Z2 X10 Z11 X12]
+ (-1.06322834262159e-06) [Z3 Y11 Z12 Y13]
+ (-1.06322834262159e-06) [Z3 X11 Z12 X13]
+ (-1.035847760062219e-06) [Y6 X7 X11 Y12]
+ (-1.035847760062219e-06) [Y6 Y7 Y11 Y12]
+ (-1.035847760062219e-06) [X6 X7 X11 X12]
+ (-1.035847760062219e-06) [X6 Y7 Y11 X12]
+ (-9.509249750980548e-07) [Z2 Y4 Z5 Y6]
+ (-9.509249750980548e-07) [Z2 X4 Z5 X6]
+ (-9.509249750980548e-07) [Z3 Y5 Z6 Y7]
+ (-9.509249750980548e-07) [Z3 X5 Z6 X7]
+ (-9.34455777701197e-07) [Z8 Y11 Z12 Y13]
+ (-9.34455777701197e-07) [Z8 X11 Z12 X13]
+ (-9.34455777701197e-07) [Z9 Y10 Z11 Y12]
+ (-9.34455777701197e-07) [Z9 X10 Z11 X12]
+ (-8.337746753772892e-07) [Z0 Y2 Z3 Y4]
+ (-8.337746753772892e-07) [Z0 X2 Z3 X4]
+ (-8.337746753772892e-07) [Z1 Y3 Z4 Y5]
+ (-8.337746753772892e-07) [Z1 X3 Z4 X5]
+ (-7.956895371248119e-07) [Y3 X4 X8 Y9]
+ (-7.956895371248119e-07) [Y3 Y4 Y8 Y9]
+ (-7.956895371248119e-07) [X3 X4 X8 X9]
+ (-7.956895371248119e-07) [X3 Y4 Y8 X9]
+ (-7.764994116405362e-07) [Y2 Z3 Y4 Z5]
+ (-7.764994116405362e-07) [X2 Z3 X4 Z5]
+ (-5.92976581585045e-07) [Z4 Y5 Z6 Y7]
+ (-5.92976581585045e-07) [Z4 X5 Z6 X7]
+ (-5.77005299452339e-07) [Y2 Z3 Y4 Z9]
+ (-5.77005299452339e-07) [X2 Z3 X4 Z9]
+ (-5.77005299452339e-07) [Y3 Z4 Y5 Z8]
+ (-5.77005299452339e-07) [X3 Z4 X5 Z8]
+ (-5.47164774490617e-07) [Y1 Y2 X11 X12]
+ (-5.47164774490617e-07) [X1 X2 Y11 Y12]
+ (-4.838052750242447e-07) [Y5 X6 X8 Y9]
+ (-4.838052750242447e-07) [Y5 Y6 Y8 Y9]
+ (-4.838052750242447e-07) [X5 X6 X8 X9]
+ (-4.838052750242447e-07) [X5 Y6 Y8 X9]
+ (-3.570761328615493e-07) [Y0 X1 X3 Y4]
+ (-3.570761328615493e-07) [Y0 Y1 Y3 Y4]
+ (-3.570761328615493e-07) [X0 X1 X3 X4]
+ (-3.570761328615493e-07) [X0 Y1 Y3 X4]
+ (-2.447323128489411e-07) [Y0 X1 X5 Y6]
+ (-2.447323128489411e-07) [Y0 Y1 Y5 Y6]
+ (-2.447323128489411e-07) [X0 X1 X5 X6]
+ (-2.447323128489411e-07) [X0 Y1 Y5 X6]
+ (-2.1990516181088855e-07) [Y2 X3 X5 Y6]
+ (-2.1990516181088855e-07) [Y2 Y3 Y5 Y6]
+ (-2.1990516181088855e-07) [X2 X3 X5 X6]
+ (-2.1990516181088855e-07) [X2 Y3 Y5 X6]
+ (-1.9332412767656069e-07) [Y1 X2 X3 Y4]
+ (-1.9332412767656069e-07) [X1 Y2 Y3 X4]
+ (-1.2919694858902884e-07) [Y1 Z2 Z3 Y5]
+ (-1.2919694858902884e-07) [X1 Z2 Z3 X5]
+ (1.7379332621190776e-07) [Y0 Z1 Z3 Y4]
+ (1.7379332621190776e-07) [X0 Z1 Z3 X4]
+ (1.7379332621190776e-07) [Y1 Z2 Z4 Y5]
+ (1.7379332621190776e-07) [X1 Z2 Z4 X5]
+ (1.9332412767656069e-07) [Y1 Y2 X3 X4]
+ (1.9332412767656069e-07) [X1 X2 Y3 Y4]
+ (2.1868423767247297e-07) [Y2 Z3 Y4 Z8]
+ (2.1868423767247297e-07) [X2 Z3 X4 Z8]
+ (2.1868423767247297e-07) [Y3 Z4 Y5 Z9]
+ (2.1868423767247297e-07) [X3 Z4 X5 Z9]
+ (2.5935343902584645e-07) [Y2 Z3 Y4 Z6]
+ (2.5935343902584645e-07) [X2 Z3 X4 Z6]
+ (2.5935343902584645e-07) [Y3 Z4 Y5 Z7]
+ (2.5935343902584645e-07) [X3 Z4 X5 Z7]
+ (3.6060718673343366e-07) [Y0 Z1 Z2 Y4]
+ (3.6060718673343366e-07) [X0 Z1 Z2 X4]
+ (3.6060718673343366e-07) [Y1 Z3 Z4 Y5]
+ (3.6060718673343366e-07) [X1 Z3 Z4 X5]
+ (5.47164774490617e-07) [Y1 X2 X11 Y12]
+ (5.47164774490617e-07) [X1 Y2 Y11 X12]
+ (5.627851911691098e-07) [Y0 X1 X11 Y12]
+ (5.627851911691098e-07) [Y0 Y1 Y11 Y12]
+ (5.627851911691098e-07) [X0 X1 X11 X12]
+ (5.627851911691098e-07) [X0 Y1 Y11 X12]
+ (6.628614202277411e-07) [Y8 X9 X11 Y12]
+ (6.628614202277411e-07) [Y8 Y9 Y11 Y12]
+ (6.628614202277411e-07) [X8 X9 X11 X12]
+ (6.628614202277411e-07) [X8 Y9 Y11 X12]
+ (1.1094407589181227e-06) [Z2 Y11 Z12 Y13]
+ (1.1094407589181227e-06) [Z2 X11 Z12 X13]
+ (1.1094407589181227e-06) [Z3 Y10 Z11 Y12]
+ (1.1094407589181227e-06) [Z3 X10 Z11 X12]
+ (1.6021167402774383e-06) [Z2 Y3 Z4 Y5]
+ (1.6021167402774383e-06) [Z2 X3 Z4 X5]
+ (1.8782101250324277e-06) [Z4 Y10 Z11 Y12]
+ (1.8782101250324277e-06) [Z4 X10 Z11 X12]
+ (1.8782101250324277e-06) [Z5 Y11 Z12 Y13]
+ (1.8782101250324277e-06) [Z5 X11 Z12 X13]
+ (2.1726691015397125e-06) [Y2 X3 X11 Y12]
+ (2.1726691015397125e-06) [Y2 Y3 Y11 Y12]
+ (2.1726691015397125e-06) [X2 X3 X11 X12]
+ (2.1726691015397125e-06) [X2 Y3 Y11 X12]
+ (3.1174479456279366e-06) [Y0 Z2 Z3 Y4]
+ (3.1174479456279366e-06) [X0 Z2 Z3 X4]
+ (3.539054184885913e-06) [Y2 Z3 Y4 Z12]
+ (3.539054184885913e-06) [X2 Z3 X4 Z12]
+ (3.539054184885913e-06) [Y3 Z4 Y5 Z13]
+ (3.539054184885913e-06) [X3 Z4 X5 Z13]
+ (4.28191388521047e-06) [Y4 Z5 Y6 Z11]
+ (4.28191388521047e-06) [X4 Z5 X6 Z11]
+ (4.28191388521047e-06) [Y5 Z6 Y7 Z10]
+ (4.28191388521047e-06) [X5 Z6 X7 Z10]
+ (5.275883122444495e-06) [Y3 X4 X12 Y13]
+ (5.275883122444495e-06) [Y3 Y4 Y12 Y13]
+ (5.275883122444495e-06) [X3 X4 X12 X13]
+ (5.275883122444495e-06) [X3 Y4 Y12 X13]
+ (5.974311713865512e-06) [Y5 X6 X10 Y11]
+ (5.974311713865512e-06) [Y5 Y6 Y10 Y11]
+ (5.974311713865512e-06) [X5 X6 X10 X11]
+ (5.974311713865512e-06) [X5 Y6 Y10 X11]
+ (7.954413176965578e-06) [Y10 Z11 Y12 Z13]
+ (7.954413176965578e-06) [X10 Z11 X12 Z13]
+ (8.814937307330408e-06) [Y2 Z3 Y4 Z13]
+ (8.814937307330408e-06) [X2 Z3 X4 Z13]
+ (8.814937307330408e-06) [Y3 Z4 Y5 Z12]
+ (8.814937307330408e-06) [X3 Z4 X5 Z12]
+ (0.00029219862611107165) [Y7 X8 X9 Y10]
+ (0.00029219862611107165) [X7 Y8 Y9 X10]
+ (0.0004956762314916185) [Y2 Z4 Z5 Y6]
+ (0.0004956762314916185) [X2 Z4 Z5 X6]
+ (0.0011059037691896955) [Y0 Z1 Y2 Z5]
+ (0.0011059037691896955) [X0 Z1 X2 Z5]
+ (0.0011059037691896955) [Y1 Z2 Y3 Z4]
+ (0.0011059037691896955) [X1 Z2 X3 Z4]
+ (0.001663879878490785) [Y2 Z3 Z4 Y6]
+ (0.001663879878490785) [X2 Z3 Z4 X6]
+ (0.001663879878490785) [Y3 Z5 Z6 Y7]
+ (0.001663879878490785) [X3 Z5 Z6 X7]
+ (0.001756070701841255) [Y0 Z1 Y2 Z11]
+ (0.001756070701841255) [X0 Z1 X2 Z11]
+ (0.001756070701841255) [Y1 Z2 Y3 Z10]
+ (0.001756070701841255) [X1 Z2 X3 Z10]
+ (0.002326230623158088) [Y0 Z1 Y2 Z13]
+ (0.002326230623158088) [X0 Z1 X2 Z13]
+ (0.002326230623158088) [Y1 Z2 Y3 Z12]
+ (0.002326230623158088) [X1 Z2 X3 Z12]
+ (0.0027458364701868077) [Y0 X1 X4 Y5]
+ (0.0027458364701868077) [X0 Y1 Y4 X5]
+ (0.0029297686747510716) [Y0 Z1 Y2 Z9]
+ (0.0029297686747510716) [X0 Z1 X2 Z9]
+ (0.0029297686747510716) [Y1 Z2 Y3 Z8]
+ (0.0029297686747510716) [X1 Z2 X3 Z8]
+ (0.003276971931231682) [Y0 Z1 Y2 Z3]
+ (0.003276971931231682) [X0 Z1 X2 Z3]
+ (0.0033476175306661883) [Y0 Z1 Y2 Z7]
+ (0.0033476175306661883) [X0 Z1 X2 Z7]
+ (0.0033476175306661883) [Y1 Z2 Y3 Z6]
+ (0.0033476175306661883) [X1 Z2 X3 Z6]
+ (0.0035552901955042348) [Y0 Z1 Y2 Z10]
+ (0.0035552901955042348) [X0 Z1 X2 Z10]
+ (0.0035552901955042348) [Y1 Z2 Y3 Z11]
+ (0.0035552901955042348) [X1 Z2 X3 Z11]
+ (0.005143391768825095) [Y3 Y4 X5 X6]
+ (0.005143391768825095) [X3 X4 Y5 Y6]
+ (0.005283776488402952) [Y0 X1 X12 Y13]
+ (0.005283776488402952) [X0 Y1 Y12 X13]
+ (0.005530759218631549) [Y0 Z1 Y2 Z4]
+ (0.005530759218631549) [X0 Z1 X2 Z4]
+ (0.005530759218631549) [Y1 Z2 Y3 Z5]
+ (0.005530759218631549) [X1 Z2 X3 Z5]
+ (0.006087822480561859) [Y8 X9 X12 Y13]
+ (0.006087822480561859) [X8 Y9 Y12 X13]
+ (0.006509361201177231) [Y0 X1 X8 Y9]
+ (0.006509361201177231) [X0 Y1 Y8 X9]
+ (0.006888194352970551) [Y0 X1 X6 Y7]
+ (0.006888194352970551) [X0 Y1 Y6 X7]
+ (0.006901238249797289) [Y0 Z1 Y2 Z12]
+ (0.006901238249797289) [X0 Z1 X2 Z12]
+ (0.006901238249797289) [Y1 Z2 Y3 Z13]
+ (0.006901238249797289) [X1 Z2 X3 Z13]
+ (0.007156934919856944) [Y4 X5 X8 Y9]
+ (0.007156934919856944) [X4 Y5 Y8 X9]
+ (0.007731425250775297) [Y0 X1 X10 Y11]
+ (0.007731425250775297) [X0 Y1 Y10 X11]
+ (0.008032520918821423) [Y0 Z1 Y2 Z6]
+ (0.008032520918821423) [X0 Z1 X2 Z6]
+ (0.008032520918821423) [Y1 Z2 Y3 Z7]
+ (0.008032520918821423) [X1 Z2 X3 Z7]
+ (0.009560705729135968) [Y8 X9 X10 Y11]
+ (0.009560705729135968) [X8 Y9 Y10 X11]
+ (0.011055020596132097) [Y0 Z1 Y2 Z8]
+ (0.011055020596132097) [X0 Z1 X2 Z8]
+ (0.011055020596132097) [Y1 Z2 Y3 Z9]
+ (0.011055020596132097) [X1 Z2 X3 Z9]
+ (0.011285190200840851) [Y5 Y6 X11 X12]
+ (0.011285190200840851) [X5 X6 Y11 Y12]
+ (0.011307274008848175) [Y7 Z8 Z9 Y11]
+ (0.011307274008848175) [X7 Z8 Z9 X11]
+ (0.011982389010247911) [Y4 X5 X6 Y7]
+ (0.011982389010247911) [X4 Y5 Y6 X7]
+ (0.013873381748426087) [Y6 X7 X8 Y9]
+ (0.013873381748426087) [X6 Y7 Y8 X9]
+ (0.014583648907612668) [Y0 X1 X2 Y3]
+ (0.014583648907612668) [X0 Y1 Y2 X3]
+ (0.015577208063976474) [Y2 X3 X12 Y13]
+ (0.015577208063976474) [X2 Y3 Y12 X13]
+ (0.017366118994651316) [Y6 X7 X12 Y13]
+ (0.017366118994651316) [X6 Y7 Y12 X13]
+ (0.017680067952481532) [Y4 X5 X10 Y11]
+ (0.017680067952481532) [X4 Y5 Y10 X11]
+ (0.017825140995786384) [Y6 X7 X10 Y11]
+ (0.017825140995786384) [X6 Y7 Y10 X11]
+ (0.01902824244384734) [Y3 X4 X11 Y12]
+ (0.01902824244384734) [X3 Y4 Y11 X12]
+ (0.025384657508457396) [Y2 X3 X10 Y11]
+ (0.025384657508457396) [X2 Y3 Y10 X11]
+ (0.028685183716106004) [Y10 X11 X12 Y13]
+ (0.028685183716106004) [X10 Y11 Y12 X13]
+ (0.02981242451734569) [Y6 Z7 Z8 Y10]
+ (0.02981242451734569) [X6 Z7 Z8 X10]
+ (0.02981242451734569) [Y7 Z9 Z10 Y11]
+ (0.02981242451734569) [X7 Z9 Z10 X11]
+ (0.030104623143456764) [Y6 Z7 Z9 Y10]
+ (0.030104623143456764) [X6 Z7 Z9 X10]
+ (0.030104623143456764) [Y7 Z8 Z10 Y11]
+ (0.030104623143456764) [X7 Z8 Z10 X11]
+ (0.030787505389143852) [Y6 Z8 Z9 Y10]
+ (0.030787505389143852) [X6 Z8 Z9 X10]
+ (0.031143817988967207) [Y2 X3 X6 Y7]
+ (0.031143817988967207) [X2 Y3 Y6 X7]
+ (0.035839567953353475) [Y2 X3 X4 Y5]
+ (0.035839567953353475) [X2 Y3 Y4 X5]
+ (0.03619412355904263) [Y2 X3 X8 Y9]
+ (0.03619412355904263) [X2 Y3 Y8 X9]
+ (0.03831467029480387) [Y4 X5 X12 Y13]
+ (0.03831467029480387) [X4 Y5 Y12 X13]
+ (0.10433064780651399) [Z0 Y1 Z2 Y3]
+ (0.10433064780651399) [Z0 X1 Z2 X3]
+ (-0.12133276911042352) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042352) [X2 Z3 Z4 Z5 X6]
+ (-0.12133276911042351) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133276911042351) [X3 Z4 Z5 Z6 X7]
+ (3.202076879899267e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.202076879899267e-06) [X0 Z1 Z2 Z3 X4]
+ (3.202076879899267e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.202076879899267e-06) [X1 Z2 Z3 Z4 X5]
+ (0.22848106564918716) [Y7 Z8 Z9 Z10 Y11]
+ (0.22848106564918716) [X7 Z8 Z9 Z10 X11]
+ (0.2284810656491872) [Y6 Z7 Z8 Z9 Y10]
+ (0.2284810656491872) [X6 Z7 Z8 Z9 X10]
+ (-0.032767657823290365) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.032767657823290365) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.032767657823290365) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.032767657823290365) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273045) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.027115036845273045) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273045) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.027115036845273045) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.025996177598021225) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.025996177598021225) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.017561202409646127) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.017561202409646127) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.017561202409646127) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.017561202409646127) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.01456453123117298) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.01456453123117298) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.01456453123117298) [X7 Z8 Z9 X10 X12 X13]
+ (-0.01456453123117298) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.012215040997613908) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.012215040997613908) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.012215040997613908) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.012215040997613908) [X4 Z5 X6 X11 Z12 X13]
+ (-0.012215040997613908) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.012215040997613908) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.012215040997613908) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.012215040997613908) [X5 Z6 X7 X10 Z11 X12]
+ (-0.011756013419819206) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.011756013419819206) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.011756013419819206) [X3 Z4 Z5 X6 X8 X9]
+ (-0.011756013419819206) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.008764827575688802) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.008764827575688802) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.008764827575688802) [X2 Z3 Z4 X5 X11 X12]
+ (-0.008764827575688802) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.008764827575688802) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.008764827575688802) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.008764827575688802) [X3 X4 X10 Z11 Z12 X13]
+ (-0.008764827575688802) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.008125251921381025) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125251921381025) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.007306759928832944) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306759928832944) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306759928832944) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306759928832944) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005805188989826919) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.005805188989826919) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.005805188989826919) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.005805188989826919) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.005652620978017324) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.005652620978017324) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.005652620978017324) [X0 X1 X3 Z4 Z5 X6]
+ (-0.005652620978017324) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.005143391768825095) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.005143391768825095) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.005143391768825095) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.005143391768825095) [X2 Z3 X4 X5 Z6 X7]
+ (-0.0046849033881552335) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.0046849033881552335) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668620318776287) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668620318776287) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.004575007626639204) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.004575007626639204) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424855449441854) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424855449441854) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.004158797381840053) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.004158797381840053) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.004158797381840053) [X3 Z4 Z5 X6 X12 X13]
+ (-0.004158797381840053) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.0034937903598901373) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.0034937903598901373) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.0034937903598901373) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.0034937903598901373) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.0027790267990255575) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.0027790267990255575) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.0022939566113524576) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.0022939566113524576) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.0017992194936629795) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.0017992194936629795) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.00172787539413697) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.00172787539413697) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298507967730586) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298507967730586) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298507967730586) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298507967730586) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533856254125441) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533856254125441) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008145313270956364) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008145313270956364) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008145313270956364) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008145313270956364) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.735036880590737e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735036880590737e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735036880590737e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735036880590737e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.774817865210006e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.774817865210006e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.774817865210006e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.774817865210006e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.518362216193156e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.518362216193156e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.518362216193156e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.518362216193156e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.4443446765064765e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.4443446765064765e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.4443446765064765e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.4443446765064765e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.524373849240547e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.524373849240547e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.524373849240547e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.524373849240547e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.290028433812558e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.290028433812558e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.290028433812558e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.290028433812558e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.974311713865512e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.974311713865512e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.275883122444495e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.275883122444495e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.64305106893822e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.64305106893822e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.5565692187022315e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.5565692187022315e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.253224225948371e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.253224225948371e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.76965945255441e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.76965945255441e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.694513294717032e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.694513294717032e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.610297131140192e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.610297131140192e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.610297131140192e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.610297131140192e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.3131455004022796e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.3131455004022796e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.2774831960918074e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.2774831960918074e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.2774831960918074e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.2774831960918074e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.2112283488382676e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.2112283488382676e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.2112283488382676e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.2112283488382676e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.1513463115058428e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.1513463115058428e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.0882507115680205e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.0882507115680205e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.1726691015397125e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.1726691015397125e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.4548424489056125e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.4548424489056125e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.3304731887035298e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.3304731887035298e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2283337823805978e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2283337823805978e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.035847760062219e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.035847760062219e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956895371248119e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956895371248119e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733197743390365e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733197743390365e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733197743390365e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733197743390365e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.628614202277411e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.628614202277411e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.556281914810466e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.556281914810466e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.556281914810466e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.556281914810466e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.418291574810995e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.418291574810995e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.418291574810995e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.418291574810995e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.92745308362649e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.92745308362649e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.92745308362649e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.92745308362649e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627851911691099e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627851911691099e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660624913814e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660624913814e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660624913814e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660624913814e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660624913814e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660624913814e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660624913814e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660624913814e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.838052750242447e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.838052750242447e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.570761328615493e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.570761328615493e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.328139350483846e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.328139350483846e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.086826564864162e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.086826564864162e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.086826564864162e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.086826564864162e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.447323128489411e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.447323128489411e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.3713289475408984e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.3713289475408984e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.3713289475408984e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.3713289475408984e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.1990516181088855e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.1990516181088855e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.9332412767656066e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.9332412767656066e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.9332412767656066e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.9332412767656066e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.8394209151992418e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.8394209151992418e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.8394209151992418e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.8394209151992418e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.5510539174708965e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.5510539174708965e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.5510539174708965e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.5510539174708965e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.3807781478766498e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.3807781478766498e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.3807781478766498e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.3807781478766498e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.3807781478766498e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.3807781478766498e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.3807781478766498e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.3807781478766498e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.3807781478766498e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.3807781478766498e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.3807781478766498e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.3807781478766498e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.2919694858902884e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.2919694858902884e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.1076325599007398e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.1076325599007398e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.1076325599007398e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.1076325599007398e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.1076325599007398e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.1076325599007398e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.1076325599007398e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.1076325599007398e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.057446597638762e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057446597638762e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057446597638762e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057446597638762e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649310133499915e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649310133499915e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649310133499915e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649310133499915e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.8394209151992418e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.8394209151992418e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.8394209151992418e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.8394209151992418e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.1990516181088855e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.1990516181088855e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.447323128489411e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.447323128489411e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.23625996089089e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.23625996089089e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.23625996089089e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.23625996089089e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.328139350483846e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.328139350483846e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.570761328615493e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.570761328615493e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.838052750242447e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.838052750242447e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.627851911691099e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627851911691099e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.628614202277411e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.628614202277411e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956895371248119e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956895371248119e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.306536652539577e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.306536652539577e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.306536652539577e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.306536652539577e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.035847760062219e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.035847760062219e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2283337823805978e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2283337823805978e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.239336321740374e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.239336321740374e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.239336321740374e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.239336321740374e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.3304731887035298e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.3304731887035298e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.4548424489056125e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.4548424489056125e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.1726691015397125e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.1726691015397125e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.0882507115680205e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.0882507115680205e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.1174479456279366e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.1174479456279366e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.1513463115058428e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.1513463115058428e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.3131455004022796e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.3131455004022796e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.3343312899025664e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.3343312899025664e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.694513294717032e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.694513294717032e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.183932559622907e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.183932559622907e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.5565692187022315e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.5565692187022315e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.64305106893822e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.64305106893822e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.275883122444495e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.275883122444495e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.974311713865512e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.974311713865512e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.00029219862611107165) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.00029219862611107165) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.00029219862611107165) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.00029219862611107165) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004956762314916185) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004956762314916185) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650070219499161) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650070219499161) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650070219499161) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650070219499161) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533856254125441) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533856254125441) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.0016095313817213672) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.0016095313817213672) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.0016095313817213672) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.0016095313817213672) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0016676041811440443) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.0016676041811440443) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.0016676041811440443) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.0016676041811440443) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.00172787539413697) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.00172787539413697) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.0017992194936629795) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.0017992194936629795) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.0022939566113524576) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.0022939566113524576) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.0024629170071339113) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.0024629170071339113) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.0024629170071339113) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.0024629170071339113) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.003961560792496502) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.003961560792496502) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.003961560792496502) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.003961560792496502) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.004424855449441854) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424855449441854) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004575007626639204) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.004575007626639204) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668620318776287) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668620318776287) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.0046849033881552335) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.0046849033881552335) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005324835234221681) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.005324835234221681) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.005324835234221681) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.005324835234221681) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.005368659358109516) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368659358109516) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368659358109516) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368659358109516) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.007960880725921519) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.007960880725921519) [X4 Z5 X6 X10 Z11 X12]
+ (0.007960880725921519) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.007960880725921519) [X5 Z6 X7 X11 Z12 X13]
+ (0.008125251921381025) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125251921381025) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.008890731522694576) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.008890731522694576) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.008890731522694576) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.008890731522694576) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.01026341486815854) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.01026341486815854) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.01026341486815854) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.01026341486815854) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.010540425907671479) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.010540425907671479) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.010540425907671479) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.010540425907671479) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010960074940542588) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010960074940542588) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010960074940542588) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010960074940542588) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307274008848175) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307274008848175) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.01441109943013099) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.01441109943013099) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.01441109943013099) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.01441109943013099) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.015225630757226626) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.015225630757226626) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.015225630757226626) [X3 Z4 Z5 X6 X10 X11]
+ (0.015225630757226626) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.01558825010238022) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.01558825010238022) [X2 Z3 X4 X10 Z11 X12]
+ (0.01558825010238022) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.01558825010238022) [X3 Z4 X5 X11 Z12 X13]
+ (0.018266834869375533) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.018266834869375533) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.018266834869375533) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.018266834869375533) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.019020423173039973) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.019020423173039973) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.019020423173039973) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.019020423173039973) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.020175921723535425) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.020175921723535425) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.020175921723535425) [X4 Z5 Z6 X7 X11 X12]
+ (0.020175921723535425) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.020175921723535425) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.020175921723535425) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.020175921723535425) [X5 X6 X10 Z11 Z12 X13]
+ (0.020175921723535425) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.024353077678069025) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.024353077678069025) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.024353077678069025) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.024353077678069025) [X2 Z3 X4 X11 Z12 X13]
+ (0.024353077678069025) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.024353077678069025) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.024353077678069025) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.024353077678069025) [X3 Z4 X5 X10 Z11 X12]
+ (0.02438908253114949) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.02438908253114949) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.02438908253114949) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.02438908253114949) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.025104957138844457) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.025104957138844457) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.025104957138844457) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.025104957138844457) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.030787505389143852) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.030787505389143852) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.04587947078129771) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.04587947078129771) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.05600733087780749) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.05600733087780749) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.05600733087780749) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.05600733087780749) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.056084681246613394) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.056084681246613394) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.056084681246613394) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.056084681246613394) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.631277928896302e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631277928896302e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.631277928896302e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631277928896302e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.5950860074516666e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.5950860074516666e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.5950860074516636e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.5950860074516636e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.04274327701378273) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04274327701378273) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (0.04274327701378274) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.04274327701378274) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (-0.04764261217638309) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.04764261217638309) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.04764261217638309) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.04764261217638309) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.04171881383982175) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.04171881383982175) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.04171881383982175) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.04171881383982175) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.039564416322893425) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.039564416322893425) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.039564416322893425) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.039564416322893425) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03935916802205318) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.03935916802205318) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.03935916802205318) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.03935916802205318) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.039318051947197584) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.039318051947197584) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.039318051947197584) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.039318051947197584) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03560837898831259) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.03560837898831259) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.02990378951262486) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.02990378951262486) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.02990378951262486) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.02990378951262486) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.028730779551905523) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.028730779551905523) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.028730779551905523) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.028730779551905523) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.02563723829602678) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.02563723829602678) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.02563723829602678) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.02563723829602678) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.02475546329289101) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.02475546329289101) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.02475546329289101) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.02475546329289101) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.02428211735469294) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.02428211735469294) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.023145130929529027) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.023145130929529027) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.022528440196013) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.022528440196013) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.021433810721601085) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.021433810721601085) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.021433810721601085) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.021433810721601085) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251568) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.019257505095251568) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.01902824244384734) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.01902824244384734) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.018889030304942933) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.018889030304942933) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.018889030304942933) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.018889030304942933) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.01602460368917958) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.01602460368917958) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.015225630757226626) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225630757226626) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.01460370472916217) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.01460370472916217) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.01456453123117298) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.01456453123117298) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.011756013419819206) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.011756013419819206) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.011285190200840851) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.011285190200840851) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.009841749246962593) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009841749246962593) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.009612634606847201) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612634606847201) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612634606847201) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612634606847201) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.008469978791023776) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.008469978791023776) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.007306759928832944) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.007306759928832944) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005923798336561344) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.005923798336561344) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.005652620978017324) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.005652620978017324) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368659358109516) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368659358109516) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.004158797381840053) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.004158797381840053) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.003356670563832872) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.003356670563832872) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.003356670563832872) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.003356670563832872) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.0032675138544235377) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.0032675138544235377) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.0032675138544235377) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.0032675138544235377) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.0027790267990255575) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.0027790267990255575) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.002686040977806631) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.002686040977806631) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.002686040977806631) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.002686040977806631) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.0022939566113524576) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524576) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.0022939566113524576) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524576) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.0009581655836696603) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.0009581655836696603) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0009581655836696603) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.0009581655836696603) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.0009581655836696603) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.0009581655836696603) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0009581655836696603) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.0009581655836696603) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.00024636437569584336) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00024636437569584336) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303548788) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.00013840177303548788) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.00013840177303548788) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.00013840177303548788) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.735036880590737e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735036880590737e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585306929434e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585306929434e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.6103585306929434e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.6103585306929434e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808796482367e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.5316808796482367e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808796482367e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5316808796482367e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.806102775975604e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.806102775975604e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.806102775975604e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.806102775975604e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.0897994678620595e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.0897994678620595e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.0897994678620595e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.0897994678620595e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.652209670522857e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.652209670522857e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.652209670522857e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.652209670522857e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851834996993e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.481851834996993e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851834996993e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.481851834996993e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.071480736866439e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.071480736866439e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.071480736866439e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.071480736866439e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.734622039109165e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.734622039109165e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.734622039109165e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.734622039109165e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.72884314750138e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.72884314750138e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.72884314750138e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.72884314750138e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.253224225948371e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.253224225948371e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.76965945255441e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.76965945255441e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.5443954294967363e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.5443954294967363e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.5443954294967363e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.5443954294967363e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.5443954294967363e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.5443954294967363e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.5443954294967363e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.5443954294967363e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.3609563203606796e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563203606796e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563203606796e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.3609563203606796e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.103215605088806e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.103215605088806e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.103215605088806e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.103215605088806e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.011122098883924e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.011122098883924e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.011122098883924e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.011122098883924e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.942946836685357e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.942946836685357e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.942946836685357e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.942946836685357e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174774613229e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.6541174774613229e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174774613229e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.6541174774613229e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.5224930678037013e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224930678037013e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224930678037013e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224930678037013e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224930678037013e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930678037013e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930678037013e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224930678037013e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.2283337823805978e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337823805978e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.2283337823805978e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337823805978e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.988770288253298e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988770288253298e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988770288253298e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988770288253298e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.867765104470655e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104470655e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104470655e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.867765104470655e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.189990976104226e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.189990976104226e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246207476842e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246207476842e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.47164774490617e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.47164774490617e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.5614471792871495e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.5614471792871495e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.5614471792871495e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.5614471792871495e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.52338967855793e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.52338967855793e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.427323108966149e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.427323108966149e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.427323108966149e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.427323108966149e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.328139350483846e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350483846e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.328139350483846e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350483846e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.086826564864162e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.086826564864162e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.888293592240339e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.888293592240339e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.888293592240339e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.888293592240339e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.3713289475408982e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.3713289475408982e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.8394209151992418e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.8394209151992418e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.057446597638763e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057446597638763e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.537178095656623e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.537178095656623e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.537178095656623e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.537178095656623e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.057446597638763e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057446597638763e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.209350620488144e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350620488144e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350620488144e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.209350620488144e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783552586466e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783552586466e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783552586466e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783552586466e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.8394209151992418e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.8394209151992418e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.3713289475408982e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.3713289475408982e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.086826564864162e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.086826564864162e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.52338967855793e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.52338967855793e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.47164774490617e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.47164774490617e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246207476842e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246207476842e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189990976104226e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.189990976104226e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.3304731887035298e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.3304731887035298e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.3304731887035298e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.3304731887035298e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288532437021413e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288532437021413e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288532437021413e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288532437021413e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.6893489516453344e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.6893489516453344e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.6893489516453344e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.6893489516453344e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.745518400671406e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.745518400671406e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.745518400671406e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.745518400671406e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.745518400671406e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.745518400671406e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.745518400671406e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.745518400671406e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.2118420194490367e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420194490367e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.2118420194490367e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420194490367e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.2118420194490367e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420194490367e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.2118420194490367e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420194490367e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.3131455004022796e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.3131455004022796e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.3131455004022796e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.3131455004022796e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.3343312899025664e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.3343312899025664e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.183932559622907e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.183932559622907e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.735036880590737e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735036880590737e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00024636437569584336) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00024636437569584336) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0004458535128840954) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.0004458535128840954) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.0004458535128840954) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.0004458535128840954) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940221543005412) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005412) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005412) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005412) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005412) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005412) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940221543005412) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005412) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533856254125441) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533856254125441) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533856254125441) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533856254125441) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.001043524653490743) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.001043524653490743) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.001043524653490743) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.001043524653490743) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.0012803060973496515) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.0012803060973496515) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.0012803060973496515) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.0012803060973496515) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.0013038004788126954) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.0013038004788126954) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.0013038004788126954) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.0013038004788126954) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.0022619660624823563) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.0022619660624823563) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.0022619660624823563) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.0022619660624823563) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.0022619660624823563) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.0022619660624823563) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.0022619660624823563) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.0022619660624823563) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.0039898414566193275) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.0039898414566193275) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.0039898414566193275) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.0039898414566193275) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.004158797381840053) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.004158797381840053) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.0043110385079142815) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.0043110385079142815) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.0043110385079142815) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.0043110385079142815) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.004636976661182525) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.004636976661182525) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.004636976661182525) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.004636976661182525) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.0051144738316603825) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.0051144738316603825) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.0051144738316603825) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.0051144738316603825) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.0051144738316603825) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.0051144738316603825) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.0051144738316603825) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.0051144738316603825) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005241535382803865) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.005241535382803865) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.005241535382803865) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.005241535382803865) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.005262642473076829) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262642473076829) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262642473076829) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262642473076829) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368659358109516) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368659358109516) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005379937155839353) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.005379937155839353) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.005379937155839353) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.005379937155839353) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.005652620978017324) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.005652620978017324) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.005708495985960923) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708495985960923) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708495985960923) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708495985960923) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.005923798336561344) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.005923798336561344) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.007306759928832944) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.007306759928832944) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469978791023776) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.008469978791023776) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.009841749246962593) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.009841749246962593) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.011285190200840851) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.011285190200840851) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.011756013419819206) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.011756013419819206) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.01456453123117298) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.01456453123117298) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.01460370472916217) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.01460370472916217) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.015225630757226626) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225630757226626) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.01602460368917958) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.01602460368917958) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.01902824244384734) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.01902824244384734) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.019257505095251568) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.019257505095251568) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.04587947078129771) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04587947078129771) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.3693708936615616) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.3693708936615616) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.3693708936615616) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.3693708936615616) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.28164257767023076) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.28164257767023076) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.2816425776702307) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.2816425776702307) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.09065144207036467) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.09065144207036467) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.09065144207036467) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.09065144207036467) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863614) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684737589863614) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863614) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684737589863614) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635021950635018) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635021950635018) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635021950635018) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635021950635018) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214031) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.06752385099214031) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214031) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.06752385099214031) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03560837898831259) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.03560837898831259) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.024591860883830016) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.024591860883830016) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883830016) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.024591860883830016) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.02428211735469294) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.02428211735469294) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.023145130929529023) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.023145130929529023) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.022528440196013) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.022528440196013) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019538050311314753) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.019538050311314753) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.019538050311314753) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.019538050311314753) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.017091553155898897) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.017091553155898897) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.017091553155898897) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.017091553155898897) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.01602460368917958) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.01602460368917958) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.01602460368917958) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.01602460368917958) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.010311482489831793) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831793) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831793) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311482489831793) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.009841749246962593) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962593) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.009841749246962593) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962593) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209874) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209874) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209874) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008826368514209874) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.00854199662545482) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00854199662545482) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00854199662545482) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00854199662545482) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.00854199662545482) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00854199662545482) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00854199662545482) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00854199662545482) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008469978791023776) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023776) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.008469978791023776) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023776) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.004668620318776287) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668620318776287) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.0038764708993369573) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.0038764708993369573) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.003804066171728536) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728536) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728536) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003804066171728536) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484157300217883) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484157300217883) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0033566705638328723) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.0033566705638328723) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.0032675138544235385) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.0032675138544235385) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.002141361223101548) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.002141361223101548) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.00172787539413697) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.00172787539413697) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.001640754855312435) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.001640754855312435) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.001452884321416869) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.001452884321416869) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.001452884321416869) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.001452884321416869) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870896771024503) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007870896771024503) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192743499487952) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192743499487952) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.00019400857029757237) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00019400857029757237) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303548788) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.00013840177303548788) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.141625221152357e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.141625221152357e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.141625221152357e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.141625221152357e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.07148073686644e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.07148073686644e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.1513463115058428e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.1513463115058428e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0882507115680205e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.0882507115680205e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.9885117063761883e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.9885117063761883e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8742990715745842e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.8742990715745842e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.3609563203606796e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.3609563203606796e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.3002946565342154e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3002946565342154e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.14683765086842e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.14683765086842e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.14683765086842e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.14683765086842e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.352332103983379e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.352332103983379e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.352332103983379e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.352332103983379e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091637200042973e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637200042973e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637200042973e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637200042973e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637200042973e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637200042973e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637200042973e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091637200042973e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.074305986766067e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.074305986766067e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.074305986766067e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.074305986766067e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128987199159e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128987199159e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128987199159e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128987199159e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867765104470655e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.867765104470655e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560692465731765e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692465731765e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692465731765e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692465731765e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692465731765e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692465731765e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692465731765e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560692465731765e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.997018422752229e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.997018422752229e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.997018422752229e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.997018422752229e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.997018422752229e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.997018422752229e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.997018422752229e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.997018422752229e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.5682475214850427e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.5682475214850427e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.5682475214850427e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.5682475214850427e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393086412305e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393086412305e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393086412305e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393086412305e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393086412305e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393086412305e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.3767393086412305e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393086412305e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.888293592240339e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.888293592240339e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.686381547484834e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.686381547484834e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035783552586466e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7035783552586466e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.209350620488144e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.209350620488144e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244567267e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773244567267e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244567267e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773244567267e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244567267e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773244567267e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773244567267e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773244567267e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.974225379653935e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.974225379653935e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.974225379653935e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.974225379653935e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.047471655576416e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.047471655576416e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.047471655576416e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.047471655576416e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.209350620488144e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.209350620488144e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0717282186474643e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0717282186474643e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0717282186474643e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0717282186474643e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.2004287494361549e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.2004287494361549e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.2004287494361549e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.2004287494361549e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.7035783552586466e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7035783552586466e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.3120943053647826e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.3120943053647826e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.3120943053647826e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.3120943053647826e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.686381547484834e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.686381547484834e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.888293592240339e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.888293592240339e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.092250616421344e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.092250616421344e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.092250616421344e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.092250616421344e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.092250616421344e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.092250616421344e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.092250616421344e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.092250616421344e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.444597854071759e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.444597854071759e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.444597854071759e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.444597854071759e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.684915095348326e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.684915095348326e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.684915095348326e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.684915095348326e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246974426124306e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246974426124306e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246974426124306e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246974426124306e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246974426124306e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246974426124306e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246974426124306e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246974426124306e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.867765104470655e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.867765104470655e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3002946565342154e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3002946565342154e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.3609563203606796e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.3609563203606796e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.8742990715745842e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.8742990715745842e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.883676576107212e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.883676576107212e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.947356011974971e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.947356011974971e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.947356011974971e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.947356011974971e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117063761883e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9885117063761883e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.0882507115680205e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.0882507115680205e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.1513463115058428e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.1513463115058428e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.846201671424787e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846201671424787e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846201671424787e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846201671424787e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.07148073686644e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.07148073686644e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.105526722162774e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.105526722162774e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.105526722162774e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.105526722162774e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146496327959003e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146496327959003e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146496327959003e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146496327959003e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159350502022389e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159350502022389e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159350502022389e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159350502022389e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427988656770872e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427988656770872e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427988656770872e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427988656770872e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.93586771835116e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.93586771835116e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.93586771835116e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.93586771835116e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348371043e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.253273348371043e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.97982579373736e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.97982579373736e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.97982579373736e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.97982579373736e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.205548411214278e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.205548411214278e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.205548411214278e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.205548411214278e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00013840177303548788) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.00013840177303548788) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.00018787053389556605) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.00018787053389556605) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.00018787053389556605) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00018787053389556605) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00019400857029757237) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00019400857029757237) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00024636437569584336) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569584336) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00024636437569584336) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569584336) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192743499487952) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192743499487952) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156734248909267) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156734248909267) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156734248909267) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156734248909267) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024503) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007870896771024503) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.0015324835230730808) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.0015324835230730808) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.0015324835230730808) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.0015324835230730808) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.001640754855312435) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.001640754855312435) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.00172787539413697) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.00172787539413697) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.0024464971554158587) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.0024464971554158587) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.0024464971554158587) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.0024464971554158587) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.0032675138544235385) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.0032675138544235385) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.0033566705638328723) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.0033566705638328723) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.003484157300217883) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.003484157300217883) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0038764708993369573) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.0038764708993369573) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.004668620318776287) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668620318776287) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.00476727218827813) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.00476727218827813) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.00476727218827813) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.00476727218827813) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.0052865465382269244) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.0052865465382269244) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.0052865465382269244) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.0052865465382269244) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.005408954422410038) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.005408954422410038) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.005408954422410038) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.005408954422410038) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.005923798336561344) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561344) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.005923798336561344) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561344) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.010715508469796752) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010715508469796752) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010715508469796752) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010715508469796752) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.010757563953908898) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010757563953908898) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010757563953908898) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010757563953908898) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01460370472916217) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.01460370472916217) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.01460370472916217) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.01460370472916217) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.019299560579363714) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363714) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363714) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363714) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363714) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363714) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.019299560579363714) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363714) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0585919887338621) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0585919887338621) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.7759505274754735e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.7759505274754735e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.775950527475474e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.775950527475474e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.07165035181002781) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07165035181002781) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.07165035181002782) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.07165035181002782) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251568) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019257505095251568) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010311482489831793) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311482489831793) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008826368514209874) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209874) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.007597464029770611) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.007597464029770611) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.007597464029770611) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.007597464029770611) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00573356974731187) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00573356974731187) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00573356974731187) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.00573356974731187) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.00573356974731187) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00573356974731187) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00573356974731187) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00573356974731187) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676623) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005348051582676623) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005348051582676623) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676623) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.003804066171728536) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003804066171728536) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219408) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.0029841661681219408) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.0029841661681219408) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.0029841661681219408) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.0024464971554158587) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.0024464971554158587) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.0022494124470939882) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0022494124470939882) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022494124470939882) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0022494124470939882) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.002141361223101548) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.002141361223101548) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.0018638942824587405) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587405) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587405) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587405) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587405) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587405) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0018638942824587405) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587405) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0016407548553124352) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124352) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0016407548553124352) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124352) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.00122233780815383) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00122233780815383) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00122233780815383) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00122233780815383) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.00122233780815383) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00122233780815383) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00122233780815383) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00122233780815383) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0010283292378562578) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0010283292378562578) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0010283292378562578) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0010283292378562578) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1463061453634972e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1463061453634972e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.8742990715745842e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990715745842e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.8742990715745842e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990715745842e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3002946565342154e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3002946565342154e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3002946565342154e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3002946565342154e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.0444941298260059e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.0444941298260059e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.0444941298260059e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.0444941298260059e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.956079230282669e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.956079230282669e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.956079230282669e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.956079230282669e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105515037401444e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105515037401444e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105515037401444e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105515037401444e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661347213323811e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661347213323811e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661347213323811e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661347213323811e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341414278971e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341414278971e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189990976104226e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.189990976104226e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876621658404002e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876621658404002e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876621658404002e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876621658404002e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246207476842e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246207476842e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.52338967855793e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.52338967855793e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0767325312222293e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.0767325312222293e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0767325312222293e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.0767325312222293e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.013471459202296e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.013471459202296e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.904599883981087e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.904599883981087e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.904599883981087e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.904599883981087e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.6667317537723684e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.6667317537723684e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.6667317537723684e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6667317537723684e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641928812233e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.8505641928812233e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309315175325e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6569309315175325e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309315175325e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6569309315175325e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.8505641928812233e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.8505641928812233e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.686381547484834e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.686381547484834e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.686381547484834e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.686381547484834e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471459202296e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.013471459202296e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.52338967855793e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.52338967855793e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.6704023907198287e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.6704023907198287e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.6704023907198287e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.6704023907198287e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246207476842e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246207476842e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189990976104226e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.189990976104226e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341414278971e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341414278971e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.94947648844945e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.94947648844945e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.7924939577889983e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939577889983e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939577889983e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.7924939577889983e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.8836765761072125e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.8836765761072125e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.9885117063761883e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117063761883e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117063761883e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117063761883e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348371043e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253273348371043e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4017109735840579e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4017109735840579e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4017109735840579e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4017109735840579e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603693629575e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.5809603693629575e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603693629575e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.5809603693629575e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192743499487951) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487951) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192743499487951) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487951) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.0007870896771024503) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024503) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024503) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024503) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0011726348316441829) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0011726348316441829) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0011726348316441829) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0011726348316441829) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.001236647801924566) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.001236647801924566) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.001236647801924566) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.001236647801924566) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.0022009640695004403) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0022009640695004403) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0022009640695004403) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0022009640695004403) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980127) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980127) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0023949726397980127) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980127) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0023949726397980127) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0023949726397980127) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0023949726397980127) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0023949726397980127) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0024464971554158587) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.0024464971554158587) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.003804066171728536) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.003804066171728536) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0038764708993369577) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.0038764708993369577) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.0038764708993369577) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.0038764708993369577) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.004220813970046507) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.004220813970046507) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.004220813970046507) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.004220813970046507) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.008826368514209874) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.008826368514209874) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.010311482489831793) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311482489831793) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257505095251568) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019257505095251568) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0585919887338621) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0585919887338621) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.398700901327259e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.398700901327259e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.398700901327259e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.398700901327259e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003484157300217883) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003484157300217883) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984166168121941) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.002984166168121941) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.00019400857029757237) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00019400857029757237) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453634972e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453634972e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7924939577889983e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7924939577889983e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540341414278971e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341414278971e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540341414278971e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341414278971e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.8505641928812233e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641928812233e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641928812233e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641928812233e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471459202296e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471459202296e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471459202296e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471459202296e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.94947648844945e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.94947648844945e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.7924939577889983e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7924939577889983e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029757237) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029757237) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002984166168121941) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.002984166168121941) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.003484157300217883) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.003484157300217883) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
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
+ (0.7829661725950187) [Z10]
+ (0.7829661725950188) [Z11]
+ (0.8084581961720474) [Z12]
+ (0.8084581961720475) [Z13]
+ (1.2034402289145631) [Z4]
+ (1.2034402289145634) [Z5]
+ (1.3096862988615419) [Z7]
+ (1.309686298861542) [Z6]
+ (1.3693525634718184) [Z9]
+ (1.6538942226831712) [Z2]
+ (1.6538942226831714) [Z3]
+ (12.412630742111762) [Z0]
+ (12.412630742111762) [Z1]
+ (-8.194261372104591e-06) [Y10 Y12]
+ (-8.194261372104591e-06) [X10 X12]
+ (-1.8540608580057544e-06) [Y5 Y7]
+ (-1.8540608580057544e-06) [X5 X7]
+ (-7.764994120631389e-07) [Y3 Y5]
+ (-7.764994120631389e-07) [X3 X5]
+ (-5.929765815349836e-07) [Y4 Y6]
+ (-5.929765815349836e-07) [X4 X6]
+ (1.6021167406707133e-06) [Y2 Y4]
+ (1.6021167406707133e-06) [X2 X4]
+ (7.954413176219533e-06) [Y11 Y13]
+ (7.954413176219533e-06) [X11 X13]
+ (0.0032769719312315633) [Y1 Y3]
+ (0.0032769719312315633) [X1 X3]
+ (0.10433064780651373) [Y0 Y2]
+ (0.10433064780651373) [X0 X2]
+ (0.11270386920332219) [Z10 Z12]
+ (0.11270386920332219) [Z11 Z13]
+ (0.11383573679388662) [Z4 Z12]
+ (0.11383573679388662) [Z5 Z13]
+ (0.11952438964682673) [Z6 Z10]
+ (0.11952438964682673) [Z7 Z11]
+ (0.12489990917237606) [Z4 Z10]
+ (0.12489990917237606) [Z5 Z11]
+ (0.12495807739503219) [Z2 Z4]
+ (0.12495807739503219) [Z3 Z5]
+ (0.12799502492468412) [Z2 Z10]
+ (0.12799502492468412) [Z3 Z11]
+ (0.134017152619637) [Z6 Z12]
+ (0.134017152619637) [Z7 Z13]
+ (0.13701191674040752) [Z4 Z6]
+ (0.13701191674040752) [Z5 Z7]
+ (0.13734953064261318) [Z6 Z11]
+ (0.13734953064261318) [Z7 Z10]
+ (0.1373910476268322) [Z2 Z6]
+ (0.1373910476268322) [Z3 Z7]
+ (0.13766872645852574) [Z8 Z10]
+ (0.13766872645852574) [Z9 Z11]
+ (0.14011289865354812) [Z2 Z12]
+ (0.14011289865354812) [Z3 Z13]
+ (0.14138905291942805) [Z10 Z13]
+ (0.14138905291942805) [Z11 Z12]
+ (0.14257997712485754) [Z4 Z11]
+ (0.14257997712485754) [Z5 Z10]
+ (0.14722943218766169) [Z8 Z11]
+ (0.14722943218766169) [Z9 Z10]
+ (0.149263551473889) [Z10 Z11]
+ (0.149607026844453) [Z4 Z8]
+ (0.149607026844453) [Z5 Z9]
+ (0.15071408121008292) [Z2 Z8]
+ (0.15071408121008292) [Z3 Z9]
+ (0.15138327161428844) [Z6 Z13]
+ (0.15138327161428844) [Z7 Z12]
+ (0.1521504070886905) [Z4 Z13]
+ (0.1521504070886905) [Z5 Z12]
+ (0.15337968243314154) [Z2 Z11]
+ (0.15337968243314154) [Z3 Z10]
+ (0.15435748657223636) [Z12 Z13]
+ (0.15569010671752456) [Z2 Z13]
+ (0.15569010671752456) [Z3 Z12]
+ (0.1558226905155311) [Z8 Z13]
+ (0.1558226905155311) [Z9 Z12]
+ (0.15676396176430998) [Z4 Z9]
+ (0.15676396176430998) [Z5 Z8]
+ (0.15755314797985673) [Z4 Z5]
+ (0.1607976453483857) [Z2 Z5]
+ (0.1607976453483857) [Z3 Z4]
+ (0.16756653265461263) [Z6 Z8]
+ (0.16756653265461263) [Z7 Z9]
+ (0.1685348656157993) [Z2 Z7]
+ (0.1685348656157993) [Z3 Z6]
+ (0.18143991440303875) [Z6 Z9]
+ (0.18143991440303875) [Z7 Z8]
+ (0.18690820476912554) [Z2 Z9]
+ (0.18690820476912554) [Z3 Z8]
+ (0.19299723935364227) [Z0 Z10]
+ (0.19299723935364227) [Z1 Z11]
+ (0.193925346132702) [Z6 Z7]
+ (0.19661770890342142) [Z0 Z4]
+ (0.19661770890342142) [Z1 Z5]
+ (0.19936354537360826) [Z0 Z5]
+ (0.19936354537360826) [Z1 Z4]
+ (0.20072866460441757) [Z0 Z11]
+ (0.20072866460441757) [Z1 Z10]
+ (0.21102659849791505) [Z0 Z12]
+ (0.21102659849791505) [Z1 Z13]
+ (0.216310374986318) [Z0 Z13]
+ (0.216310374986318) [Z1 Z12]
+ (0.2367108078383041) [Z0 Z2]
+ (0.2367108078383041) [Z1 Z3]
+ (0.24164663936017186) [Z0 Z6]
+ (0.24164663936017186) [Z1 Z7]
+ (0.24853483371314244) [Z0 Z7]
+ (0.24853483371314244) [Z1 Z6]
+ (0.25129445674591666) [Z0 Z3]
+ (0.25129445674591666) [Z1 Z2]
+ (0.2723251830660567) [Z0 Z8]
+ (0.2723251830660567) [Z1 Z9]
+ (0.278834544267234) [Z0 Z9]
+ (0.278834544267234) [Z1 Z8]
+ (1.1861763734860487) [Z0 Z1]
+ (-1.226048498920083e-05) [Y4 Z5 Y6]
+ (-1.226048498920083e-05) [X4 Z5 X6]
+ (-1.226048498920083e-05) [Y5 Z6 Y7]
+ (-1.226048498920083e-05) [X5 Z6 X7]
+ (-1.0722312157126663e-05) [Y10 Z11 Y12]
+ (-1.0722312157126663e-05) [X10 Z11 X12]
+ (-1.0722312157126663e-05) [Y11 Z12 Y13]
+ (-1.0722312157126663e-05) [X11 Z12 X13]
+ (-3.887051674548793e-06) [Y2 Z3 Y4]
+ (-3.887051674548793e-06) [X2 Z3 X4]
+ (-3.88705167454879e-06) [Y3 Z4 Y5]
+ (-3.88705167454879e-06) [X3 Z4 X5]
+ (0.12507032579771776) [Y0 Z1 Y2]
+ (0.12507032579771776) [X0 Z1 X2]
+ (0.1250703257977178) [Y1 Z2 Y3]
+ (0.1250703257977178) [X1 Z2 X3]
+ (-0.038314670294803906) [Y4 Y5 X12 X13]
+ (-0.038314670294803906) [X4 X5 Y12 Y13]
+ (-0.036194123559042606) [Y2 Y3 X8 X9]
+ (-0.036194123559042606) [X2 X3 Y8 Y9]
+ (-0.035839567953353496) [Y2 Y3 X4 X5]
+ (-0.035839567953353496) [X2 X3 Y4 Y5]
+ (-0.03114381798896709) [Y2 Y3 X6 X7]
+ (-0.03114381798896709) [X2 X3 Y6 Y7]
+ (-0.02868518371610587) [Y10 Y11 X12 X13]
+ (-0.02868518371610587) [X10 X11 Y12 Y13]
+ (-0.025996177598021218) [Y3 Z4 Z5 Y7]
+ (-0.025996177598021218) [X3 Z4 Z5 X7]
+ (-0.025384657508457423) [Y2 Y3 X10 X11]
+ (-0.025384657508457423) [X2 X3 Y10 Y11]
+ (-0.019028242443847296) [Y3 Y4 X11 X12]
+ (-0.019028242443847296) [X3 X4 Y11 Y12]
+ (-0.017825140995786453) [Y6 Y7 X10 X11]
+ (-0.017825140995786453) [X6 X7 Y10 Y11]
+ (-0.0176800679524815) [Y4 Y5 X10 X11]
+ (-0.0176800679524815) [X4 X5 Y10 Y11]
+ (-0.017366118994651413) [Y6 Y7 X12 X13]
+ (-0.017366118994651413) [X6 X7 Y12 Y13]
+ (-0.015577208063976456) [Y2 Y3 X12 X13]
+ (-0.015577208063976456) [X2 X3 Y12 Y13]
+ (-0.014583648907612616) [Y0 Y1 X2 X3]
+ (-0.014583648907612616) [X0 X1 Y2 Y3]
+ (-0.013873381748426119) [Y6 Y7 X8 X9]
+ (-0.013873381748426119) [X6 X7 Y8 Y9]
+ (-0.011982389010247955) [Y4 Y5 X6 X7]
+ (-0.011982389010247955) [X4 X5 Y6 Y7]
+ (-0.011285190200840905) [Y5 X6 X11 Y12]
+ (-0.011285190200840905) [X5 Y6 Y11 X12]
+ (-0.00956070572913594) [Y8 Y9 X10 X11]
+ (-0.00956070572913594) [X8 X9 Y10 Y11]
+ (-0.008125251921381029) [Y1 X2 X8 Y9]
+ (-0.008125251921381029) [Y1 Y2 Y8 Y9]
+ (-0.008125251921381029) [X1 X2 X8 X9]
+ (-0.008125251921381029) [X1 Y2 Y8 X9]
+ (-0.0077314252507752965) [Y0 Y1 X10 X11]
+ (-0.0077314252507752965) [X0 X1 Y10 Y11]
+ (-0.00715693491985695) [Y4 Y5 X8 X9]
+ (-0.00715693491985695) [X4 X5 Y8 Y9]
+ (-0.0068881943529705714) [Y0 Y1 X6 X7]
+ (-0.0068881943529705714) [X0 X1 Y6 Y7]
+ (-0.0065093612011772346) [Y0 Y1 X8 X9]
+ (-0.0065093612011772346) [X0 X1 Y8 Y9]
+ (-0.006087822480561857) [Y8 Y9 X12 X13]
+ (-0.006087822480561857) [X8 X9 Y12 Y13]
+ (-0.005283776488402956) [Y0 Y1 X12 X13]
+ (-0.005283776488402956) [X0 X1 Y12 Y13]
+ (-0.0051433917688250945) [Y3 X4 X5 Y6]
+ (-0.0051433917688250945) [X3 Y4 Y5 X6]
+ (-0.004684903388155199) [Y1 X2 X6 Y7]
+ (-0.004684903388155199) [Y1 Y2 Y6 Y7]
+ (-0.004684903388155199) [X1 X2 X6 X7]
+ (-0.004684903388155199) [X1 Y2 Y6 X7]
+ (-0.004575007626639205) [Y1 X2 X12 Y13]
+ (-0.004575007626639205) [Y1 Y2 Y12 Y13]
+ (-0.004575007626639205) [X1 X2 X12 X13]
+ (-0.004575007626639205) [X1 Y2 Y12 X13]
+ (-0.0044248554494418614) [Y1 X2 X4 Y5]
+ (-0.0044248554494418614) [Y1 Y2 Y4 Y5]
+ (-0.0044248554494418614) [X1 X2 X4 X5]
+ (-0.0044248554494418614) [X1 Y2 Y4 X5]
+ (-0.003479511890334338) [Y2 Z3 Z5 Y6]
+ (-0.003479511890334338) [X2 Z3 Z5 X6]
+ (-0.003479511890334338) [Y3 Z4 Z6 Y7]
+ (-0.003479511890334338) [X3 Z4 Z6 X7]
+ (-0.002745836470186813) [Y0 Y1 X4 X5]
+ (-0.002745836470186813) [X0 X1 Y4 Y5]
+ (-0.001799219493663018) [Y1 X2 X10 Y11]
+ (-0.001799219493663018) [Y1 Y2 Y10 Y11]
+ (-0.001799219493663018) [X1 X2 X10 X11]
+ (-0.001799219493663018) [X1 Y2 Y10 X11]
+ (-0.000292198626111066) [Y7 Y8 X9 X10]
+ (-0.000292198626111066) [X7 X8 Y9 Y10]
+ (-8.194261372104591e-06) [Z10 Y11 Z12 Y13]
+ (-8.194261372104591e-06) [Z10 X11 Z12 X13]
+ (-7.801707500488298e-06) [Y2 Z3 Y4 Z11]
+ (-7.801707500488298e-06) [X2 Z3 X4 Z11]
+ (-7.801707500488298e-06) [Y3 Z4 Y5 Z10]
+ (-7.801707500488298e-06) [X3 Z4 X5 Z10]
+ (-4.643051068445003e-06) [Y3 X4 X10 Y11]
+ (-4.643051068445003e-06) [Y3 Y4 Y10 Y11]
+ (-4.643051068445003e-06) [X3 X4 X10 X11]
+ (-4.643051068445003e-06) [X3 Y4 Y10 X11]
+ (-4.588855155614931e-06) [Y4 Z5 Y6 Z13]
+ (-4.588855155614931e-06) [X4 Z5 X6 Z13]
+ (-4.588855155614931e-06) [Y5 Z6 Y7 Z12]
+ (-4.588855155614931e-06) [X5 Z6 X7 Z12]
+ (-4.556569218082908e-06) [Y5 X6 X12 Y13]
+ (-4.556569218082908e-06) [Y5 Y6 Y12 Y13]
+ (-4.556569218082908e-06) [X5 X6 X12 X13]
+ (-4.556569218082908e-06) [X5 Y6 Y12 X13]
+ (-3.6945132945013098e-06) [Y4 X5 X11 Y12]
+ (-3.6945132945013098e-06) [Y4 Y5 Y11 Y12]
+ (-3.6945132945013098e-06) [X4 X5 X11 X12]
+ (-3.6945132945013098e-06) [X4 Y5 Y11 X12]
+ (-3.3440815565429857e-06) [Z0 Y5 Z6 Y7]
+ (-3.3440815565429857e-06) [Z0 X5 Z6 X7]
+ (-3.3440815565429857e-06) [Z1 Y4 Z5 Y6]
+ (-3.3440815565429857e-06) [Z1 X4 Z5 X6]
+ (-3.1586564320432955e-06) [Y2 Z3 Y4 Z10]
+ (-3.1586564320432955e-06) [X2 Z3 X4 Z10]
+ (-3.1586564320432955e-06) [Y3 Z4 Y5 Z11]
+ (-3.1586564320432955e-06) [X3 Z4 X5 Z11]
+ (-3.09934924364991e-06) [Z0 Y4 Z5 Y6]
+ (-3.09934924364991e-06) [Z0 X4 Z5 X6]
+ (-3.09934924364991e-06) [Z1 Y5 Z6 Y7]
+ (-3.09934924364991e-06) [Z1 X5 Z6 X7]
+ (-2.89096788164369e-06) [Z6 Y11 Z12 Y13]
+ (-2.89096788164369e-06) [Z6 X11 Z12 X13]
+ (-2.89096788164369e-06) [Z7 Y10 Z11 Y12]
+ (-2.89096788164369e-06) [Z7 X10 Z11 X12]
+ (-2.1776646049556584e-06) [Z0 Y10 Z11 Y12]
+ (-2.1776646049556584e-06) [Z0 X10 Z11 X12]
+ (-2.1776646049556584e-06) [Z1 Y11 Z12 Y13]
+ (-2.1776646049556584e-06) [Z1 X11 Z12 X13]
+ (-1.8818501832396868e-06) [Y4 Z5 Y6 Z9]
+ (-1.8818501832396868e-06) [X4 Z5 X6 Z9]
+ (-1.8818501832396868e-06) [Y5 Z6 Y7 Z8]
+ (-1.8818501832396868e-06) [X5 Z6 X7 Z8]
+ (-1.855120121481681e-06) [Z6 Y10 Z11 Y12]
+ (-1.855120121481681e-06) [Z6 X10 Z11 X12]
+ (-1.855120121481681e-06) [Z7 Y11 Z12 Y13]
+ (-1.855120121481681e-06) [Z7 X11 Z12 X13]
+ (-1.8540608580057544e-06) [Y4 Z5 Y6 Z7]
+ (-1.8540608580057544e-06) [X4 Z5 X6 Z7]
+ (-1.816303169781348e-06) [Z4 Y11 Z12 Y13]
+ (-1.816303169781348e-06) [Z4 X11 Z12 X13]
+ (-1.816303169781348e-06) [Z5 Y10 Z11 Y12]
+ (-1.816303169781348e-06) [Z5 X10 Z11 X12]
+ (-1.6923978285555383e-06) [Y4 Z5 Y6 Z10]
+ (-1.6923978285555383e-06) [X4 Z5 X6 Z10]
+ (-1.6923978285555383e-06) [Y5 Z6 Y7 Z11]
+ (-1.6923978285555383e-06) [X5 Z6 X7 Z11]
+ (-1.6148794138092402e-06) [Z0 Y11 Z12 Y13]
+ (-1.6148794138092402e-06) [Z0 X11 Z12 X13]
+ (-1.6148794138092402e-06) [Z1 Y10 Z11 Y12]
+ (-1.6148794138092402e-06) [Z1 X10 Z11 X12]
+ (-1.5973171977557215e-06) [Z8 Y10 Z11 Y12]
+ (-1.5973171977557215e-06) [Z8 X10 Z11 X12]
+ (-1.5973171977557215e-06) [Z9 Y11 Z12 Y13]
+ (-1.5973171977557215e-06) [Z9 X11 Z12 X13]
+ (-1.454842449115496e-06) [Y3 X4 X6 Y7]
+ (-1.454842449115496e-06) [Y3 Y4 Y6 Y7]
+ (-1.454842449115496e-06) [X3 X4 X6 X7]
+ (-1.454842449115496e-06) [X3 Y4 Y6 X7]
+ (-1.3980449081237831e-06) [Y4 Z5 Y6 Z8]
+ (-1.3980449081237831e-06) [X4 Z5 X6 Z8]
+ (-1.3980449081237831e-06) [Y5 Z6 Y7 Z9]
+ (-1.3980449081237831e-06) [X5 Z6 X7 Z9]
+ (-1.1954890100595117e-06) [Y2 Z3 Y4 Z7]
+ (-1.1954890100595117e-06) [X2 Z3 X4 Z7]
+ (-1.1954890100595117e-06) [Y3 Z4 Y5 Z6]
+ (-1.1954890100595117e-06) [X3 Z4 X5 Z6]
+ (-1.1908508085541575e-06) [Z0 Y3 Z4 Y5]
+ (-1.1908508085541575e-06) [Z0 X3 Z4 X5]
+ (-1.1908508085541575e-06) [Z1 Y2 Z3 Y4]
+ (-1.1908508085541575e-06) [Z1 X2 Z3 X4]
+ (-1.1708301370320723e-06) [Z2 Y5 Z6 Y7]
+ (-1.1708301370320723e-06) [Z2 X5 Z6 X7]
+ (-1.1708301370320723e-06) [Z3 Y4 Z5 Y6]
+ (-1.1708301370320723e-06) [Z3 X4 Z5 X6]
+ (-1.0632283422747001e-06) [Z2 Y10 Z11 Y12]
+ (-1.0632283422747001e-06) [Z2 X10 Z11 X12]
+ (-1.0632283422747001e-06) [Z3 Y11 Z12 Y13]
+ (-1.0632283422747001e-06) [Z3 X11 Z12 X13]
+ (-1.0358477601620094e-06) [Y6 X7 X11 Y12]
+ (-1.0358477601620094e-06) [Y6 Y7 Y11 Y12]
+ (-1.0358477601620094e-06) [X6 X7 X11 X12]
+ (-1.0358477601620094e-06) [X6 Y7 Y11 X12]
+ (-9.509249751500799e-07) [Z2 Y4 Z5 Y6]
+ (-9.509249751500799e-07) [Z2 X4 Z5 X6]
+ (-9.509249751500799e-07) [Z3 Y5 Z6 Y7]
+ (-9.509249751500799e-07) [Z3 X5 Z6 X7]
+ (-9.344557775859593e-07) [Z8 Y11 Z12 Y13]
+ (-9.344557775859593e-07) [Z8 X11 Z12 X13]
+ (-9.344557775859593e-07) [Z9 Y10 Z11 Y12]
+ (-9.344557775859593e-07) [Z9 X10 Z11 X12]
+ (-8.337746756154614e-07) [Z0 Y2 Z3 Y4]
+ (-8.337746756154614e-07) [Z0 X2 Z3 X4]
+ (-8.337746756154614e-07) [Z1 Y3 Z4 Y5]
+ (-8.337746756154614e-07) [Z1 X3 Z4 X5]
+ (-7.956895373214412e-07) [Y3 X4 X8 Y9]
+ (-7.956895373214412e-07) [Y3 Y4 Y8 Y9]
+ (-7.956895373214412e-07) [X3 X4 X8 X9]
+ (-7.956895373214412e-07) [X3 Y4 Y8 X9]
+ (-7.764994120631389e-07) [Y2 Z3 Y4 Z5]
+ (-7.764994120631389e-07) [X2 Z3 X4 Z5]
+ (-5.929765815349836e-07) [Z4 Y5 Z6 Y7]
+ (-5.929765815349836e-07) [Z4 X5 Z6 X7]
+ (-5.770052996231166e-07) [Y2 Z3 Y4 Z9]
+ (-5.770052996231166e-07) [X2 Z3 X4 Z9]
+ (-5.770052996231166e-07) [Y3 Z4 Y5 Z8]
+ (-5.770052996231166e-07) [X3 Z4 X5 Z8]
+ (-5.471647744634499e-07) [Y1 Y2 X11 X12]
+ (-5.471647744634499e-07) [X1 X2 Y11 Y12]
+ (-4.838052751159036e-07) [Y5 X6 X8 Y9]
+ (-4.838052751159036e-07) [Y5 Y6 Y8 Y9]
+ (-4.838052751159036e-07) [X5 X6 X8 X9]
+ (-4.838052751159036e-07) [X5 Y6 Y8 X9]
+ (-3.5707613293869596e-07) [Y0 X1 X3 Y4]
+ (-3.5707613293869596e-07) [Y0 Y1 Y3 Y4]
+ (-3.5707613293869596e-07) [X0 X1 X3 X4]
+ (-3.5707613293869596e-07) [X0 Y1 Y3 X4]
+ (-2.4473231289307566e-07) [Y0 X1 X5 Y6]
+ (-2.4473231289307566e-07) [Y0 Y1 Y5 Y6]
+ (-2.4473231289307566e-07) [X0 X1 X5 X6]
+ (-2.4473231289307566e-07) [X0 Y1 Y5 X6]
+ (-2.199051618819924e-07) [Y2 X3 X5 Y6]
+ (-2.199051618819924e-07) [Y2 Y3 Y5 Y6]
+ (-2.199051618819924e-07) [X2 X3 X5 X6]
+ (-2.199051618819924e-07) [X2 Y3 Y5 X6]
+ (-1.9332412771900524e-07) [Y1 X2 X3 Y4]
+ (-1.9332412771900524e-07) [X1 Y2 Y3 X4]
+ (-1.291969486489774e-07) [Y1 Z2 Z3 Y5]
+ (-1.291969486489774e-07) [X1 Z2 Z3 X5]
+ (1.7379332624390665e-07) [Y0 Z1 Z3 Y4]
+ (1.7379332624390665e-07) [X0 Z1 Z3 X4]
+ (1.7379332624390665e-07) [Y1 Z2 Z4 Y5]
+ (1.7379332624390665e-07) [X1 Z2 Z4 X5]
+ (1.9332412771900524e-07) [Y1 Y2 X3 X4]
+ (1.9332412771900524e-07) [X1 X2 Y3 Y4]
+ (2.1868423769832442e-07) [Y2 Z3 Y4 Z8]
+ (2.1868423769832442e-07) [X2 Z3 X4 Z8]
+ (2.1868423769832442e-07) [Y3 Z4 Y5 Z9]
+ (2.1868423769832442e-07) [X3 Z4 X5 Z9]
+ (2.5935343905598463e-07) [Y2 Z3 Y4 Z6]
+ (2.5935343905598463e-07) [X2 Z3 X4 Z6]
+ (2.5935343905598463e-07) [Y3 Z4 Y5 Z7]
+ (2.5935343905598463e-07) [X3 Z4 X5 Z7]
+ (3.6060718680453e-07) [Y0 Z1 Z2 Y4]
+ (3.6060718680453e-07) [X0 Z1 Z2 X4]
+ (3.6060718680453e-07) [Y1 Z3 Z4 Y5]
+ (3.6060718680453e-07) [X1 Z3 Z4 X5]
+ (5.471647744634499e-07) [Y1 X2 X11 Y12]
+ (5.471647744634499e-07) [X1 Y2 Y11 X12]
+ (5.62785191146418e-07) [Y0 X1 X11 Y12]
+ (5.62785191146418e-07) [Y0 Y1 Y11 Y12]
+ (5.62785191146418e-07) [X0 X1 X11 X12]
+ (5.62785191146418e-07) [X0 Y1 Y11 X12]
+ (6.628614201697622e-07) [Y8 X9 X11 Y12]
+ (6.628614201697622e-07) [Y8 Y9 Y11 Y12]
+ (6.628614201697622e-07) [X8 X9 X11 X12]
+ (6.628614201697622e-07) [X8 Y9 Y11 X12]
+ (1.1094407592795622e-06) [Z2 Y11 Z12 Y13]
+ (1.1094407592795622e-06) [Z2 X11 Z12 X13]
+ (1.1094407592795622e-06) [Z3 Y10 Z11 Y12]
+ (1.1094407592795622e-06) [Z3 X10 Z11 X12]
+ (1.6021167406707135e-06) [Z2 Y3 Z4 Y5]
+ (1.6021167406707135e-06) [Z2 X3 Z4 X5]
+ (1.878210124719962e-06) [Z4 Y10 Z11 Y12]
+ (1.878210124719962e-06) [Z4 X10 Z11 X12]
+ (1.878210124719962e-06) [Z5 Y11 Z12 Y13]
+ (1.878210124719962e-06) [Z5 X11 Z12 X13]
+ (2.1726691015542624e-06) [Y2 X3 X11 Y12]
+ (2.1726691015542624e-06) [Y2 Y3 Y11 Y12]
+ (2.1726691015542624e-06) [X2 X3 X11 X12]
+ (2.1726691015542624e-06) [X2 Y3 Y11 X12]
+ (3.117447946285276e-06) [Y0 Z2 Z3 Y4]
+ (3.117447946285276e-06) [X0 Z2 Z3 X4]
+ (3.5390541843831875e-06) [Y2 Z3 Y4 Z12]
+ (3.5390541843831875e-06) [X2 Z3 X4 Z12]
+ (3.5390541843831875e-06) [Y3 Z4 Y5 Z13]
+ (3.5390541843831875e-06) [X3 Z4 X5 Z13]
+ (4.281913884858942e-06) [Y4 Z5 Y6 Z11]
+ (4.281913884858942e-06) [X4 Z5 X6 Z11]
+ (4.281913884858942e-06) [Y5 Z6 Y7 Z10]
+ (4.281913884858942e-06) [X5 Z6 X7 Z10]
+ (5.275883122146418e-06) [Y3 X4 X12 Y13]
+ (5.275883122146418e-06) [Y3 Y4 Y12 Y13]
+ (5.275883122146418e-06) [X3 X4 X12 X13]
+ (5.275883122146418e-06) [X3 Y4 Y12 X13]
+ (5.974311713414479e-06) [Y5 X6 X10 Y11]
+ (5.974311713414479e-06) [Y5 Y6 Y10 Y11]
+ (5.974311713414479e-06) [X5 X6 X10 X11]
+ (5.974311713414479e-06) [X5 Y6 Y10 X11]
+ (7.954413176219533e-06) [Y10 Z11 Y12 Z13]
+ (7.954413176219533e-06) [X10 Z11 X12 Z13]
+ (8.814937306529606e-06) [Y2 Z3 Y4 Z13]
+ (8.814937306529606e-06) [X2 Z3 X4 Z13]
+ (8.814937306529606e-06) [Y3 Z4 Y5 Z12]
+ (8.814937306529606e-06) [X3 Z4 X5 Z12]
+ (0.000292198626111066) [Y7 X8 X9 Y10]
+ (0.000292198626111066) [X7 Y8 Y9 X10]
+ (0.0004956762314916611) [Y2 Z4 Z5 Y6]
+ (0.0004956762314916611) [X2 Z4 Z5 X6]
+ (0.0011059037691896261) [Y0 Z1 Y2 Z5]
+ (0.0011059037691896261) [X0 Z1 X2 Z5]
+ (0.0011059037691896261) [Y1 Z2 Y3 Z4]
+ (0.0011059037691896261) [X1 Z2 X3 Z4]
+ (0.0016638798784907568) [Y2 Z3 Z4 Y6]
+ (0.0016638798784907568) [X2 Z3 Z4 X6]
+ (0.0016638798784907568) [Y3 Z5 Z6 Y7]
+ (0.0016638798784907568) [X3 Z5 Z6 X7]
+ (0.001756070701841173) [Y0 Z1 Y2 Z11]
+ (0.001756070701841173) [X0 Z1 X2 Z11]
+ (0.001756070701841173) [Y1 Z2 Y3 Z10]
+ (0.001756070701841173) [X1 Z2 X3 Z10]
+ (0.0023262306231580133) [Y0 Z1 Y2 Z13]
+ (0.0023262306231580133) [X0 Z1 X2 Z13]
+ (0.0023262306231580133) [Y1 Z2 Y3 Z12]
+ (0.0023262306231580133) [X1 Z2 X3 Z12]
+ (0.002745836470186813) [Y0 X1 X4 Y5]
+ (0.002745836470186813) [X0 Y1 Y4 X5]
+ (0.00292976867475097) [Y0 Z1 Y2 Z9]
+ (0.00292976867475097) [X0 Z1 X2 Z9]
+ (0.00292976867475097) [Y1 Z2 Y3 Z8]
+ (0.00292976867475097) [X1 Z2 X3 Z8]
+ (0.0032769719312315633) [Y0 Z1 Y2 Z3]
+ (0.0032769719312315633) [X0 Z1 X2 Z3]
+ (0.0033476175306661146) [Y0 Z1 Y2 Z7]
+ (0.0033476175306661146) [X0 Z1 X2 Z7]
+ (0.0033476175306661146) [Y1 Z2 Y3 Z6]
+ (0.0033476175306661146) [X1 Z2 X3 Z6]
+ (0.003555290195504191) [Y0 Z1 Y2 Z10]
+ (0.003555290195504191) [X0 Z1 X2 Z10]
+ (0.003555290195504191) [Y1 Z2 Y3 Z11]
+ (0.003555290195504191) [X1 Z2 X3 Z11]
+ (0.0051433917688250945) [Y3 Y4 X5 X6]
+ (0.0051433917688250945) [X3 X4 Y5 Y6]
+ (0.005283776488402956) [Y0 X1 X12 Y13]
+ (0.005283776488402956) [X0 Y1 Y12 X13]
+ (0.005530759218631487) [Y0 Z1 Y2 Z4]
+ (0.005530759218631487) [X0 Z1 X2 Z4]
+ (0.005530759218631487) [Y1 Z2 Y3 Z5]
+ (0.005530759218631487) [X1 Z2 X3 Z5]
+ (0.006087822480561857) [Y8 X9 X12 Y13]
+ (0.006087822480561857) [X8 Y9 Y12 X13]
+ (0.0065093612011772346) [Y0 X1 X8 Y9]
+ (0.0065093612011772346) [X0 Y1 Y8 X9]
+ (0.0068881943529705714) [Y0 X1 X6 Y7]
+ (0.0068881943529705714) [X0 Y1 Y6 X7]
+ (0.006901238249797218) [Y0 Z1 Y2 Z12]
+ (0.006901238249797218) [X0 Z1 X2 Z12]
+ (0.006901238249797218) [Y1 Z2 Y3 Z13]
+ (0.006901238249797218) [X1 Z2 X3 Z13]
+ (0.00715693491985695) [Y4 X5 X8 Y9]
+ (0.00715693491985695) [X4 Y5 Y8 X9]
+ (0.0077314252507752965) [Y0 X1 X10 Y11]
+ (0.0077314252507752965) [X0 Y1 Y10 X11]
+ (0.008032520918821312) [Y0 Z1 Y2 Z6]
+ (0.008032520918821312) [X0 Z1 X2 Z6]
+ (0.008032520918821312) [Y1 Z2 Y3 Z7]
+ (0.008032520918821312) [X1 Z2 X3 Z7]
+ (0.00956070572913594) [Y8 X9 X10 Y11]
+ (0.00956070572913594) [X8 Y9 Y10 X11]
+ (0.011055020596132) [Y0 Z1 Y2 Z8]
+ (0.011055020596132) [X0 Z1 X2 Z8]
+ (0.011055020596132) [Y1 Z2 Y3 Z9]
+ (0.011055020596132) [X1 Z2 X3 Z9]
+ (0.011285190200840905) [Y5 Y6 X11 X12]
+ (0.011285190200840905) [X5 X6 Y11 Y12]
+ (0.011307274008848154) [Y7 Z8 Z9 Y11]
+ (0.011307274008848154) [X7 Z8 Z9 X11]
+ (0.011982389010247955) [Y4 X5 X6 Y7]
+ (0.011982389010247955) [X4 Y5 Y6 X7]
+ (0.013873381748426119) [Y6 X7 X8 Y9]
+ (0.013873381748426119) [X6 Y7 Y8 X9]
+ (0.014583648907612616) [Y0 X1 X2 Y3]
+ (0.014583648907612616) [X0 Y1 Y2 X3]
+ (0.015577208063976456) [Y2 X3 X12 Y13]
+ (0.015577208063976456) [X2 Y3 Y12 X13]
+ (0.017366118994651413) [Y6 X7 X12 Y13]
+ (0.017366118994651413) [X6 Y7 Y12 X13]
+ (0.0176800679524815) [Y4 X5 X10 Y11]
+ (0.0176800679524815) [X4 Y5 Y10 X11]
+ (0.017825140995786453) [Y6 X7 X10 Y11]
+ (0.017825140995786453) [X6 Y7 Y10 X11]
+ (0.019028242443847296) [Y3 X4 X11 Y12]
+ (0.019028242443847296) [X3 Y4 Y11 X12]
+ (0.025384657508457423) [Y2 X3 X10 Y11]
+ (0.025384657508457423) [X2 Y3 Y10 X11]
+ (0.02868518371610587) [Y10 X11 X12 Y13]
+ (0.02868518371610587) [X10 Y11 Y12 X13]
+ (0.029812424517345757) [Y6 Z7 Z8 Y10]
+ (0.029812424517345757) [X6 Z7 Z8 X10]
+ (0.029812424517345757) [Y7 Z9 Z10 Y11]
+ (0.029812424517345757) [X7 Z9 Z10 X11]
+ (0.03010462314345682) [Y6 Z7 Z9 Y10]
+ (0.03010462314345682) [X6 Z7 Z9 X10]
+ (0.03010462314345682) [Y7 Z8 Z10 Y11]
+ (0.03010462314345682) [X7 Z8 Z10 X11]
+ (0.030787505389143953) [Y6 Z8 Z9 Y10]
+ (0.030787505389143953) [X6 Z8 Z9 X10]
+ (0.03114381798896709) [Y2 X3 X6 Y7]
+ (0.03114381798896709) [X2 Y3 Y6 X7]
+ (0.035839567953353496) [Y2 X3 X4 Y5]
+ (0.035839567953353496) [X2 Y3 Y4 X5]
+ (0.036194123559042606) [Y2 X3 X8 Y9]
+ (0.036194123559042606) [X2 Y3 Y8 X9]
+ (0.038314670294803906) [Y4 X5 X12 Y13]
+ (0.038314670294803906) [X4 Y5 Y12 X13]
+ (0.10433064780651373) [Z0 Y1 Z2 Y3]
+ (0.10433064780651373) [Z0 X1 Z2 X3]
+ (-0.12133276911042383) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133276911042383) [X3 Z4 Z5 Z6 X7]
+ (-0.12133276911042379) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042379) [X2 Z3 Z4 Z5 X6]
+ (3.202076880068996e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.202076880068996e-06) [X0 Z1 Z2 Z3 X4]
+ (3.202076880068996e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.202076880068996e-06) [X1 Z2 Z3 Z4 X5]
+ (0.22848106564918869) [Y6 Z7 Z8 Z9 Y10]
+ (0.22848106564918869) [X6 Z7 Z8 Z9 X10]
+ (0.22848106564918874) [Y7 Z8 Z9 Z10 Y11]
+ (0.22848106564918874) [X7 Z8 Z9 Z10 X11]
+ (-0.03276765782329051) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.03276765782329051) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.03276765782329051) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.03276765782329051) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.02711503684527317) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.02711503684527317) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.02711503684527317) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.02711503684527317) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.025996177598021218) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.025996177598021218) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.017561202409646186) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.017561202409646186) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.017561202409646186) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.017561202409646186) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.014564531231172996) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564531231172996) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564531231172996) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564531231172996) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.012215040997613925) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.012215040997613925) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.012215040997613925) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.012215040997613925) [X4 Z5 X6 X11 Z12 X13]
+ (-0.012215040997613925) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.012215040997613925) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.012215040997613925) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.012215040997613925) [X5 Z6 X7 X10 Z11 X12]
+ (-0.011756013419819281) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.011756013419819281) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.011756013419819281) [X3 Z4 Z5 X6 X8 X9]
+ (-0.011756013419819281) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.008764827575688774) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.008764827575688774) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.008764827575688774) [X2 Z3 Z4 X5 X11 X12]
+ (-0.008764827575688774) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.008764827575688774) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.008764827575688774) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.008764827575688774) [X3 X4 X10 Z11 Z12 X13]
+ (-0.008764827575688774) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.008125251921381029) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125251921381029) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.007306759928832955) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306759928832955) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306759928832955) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306759928832955) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005805188989826905) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.005805188989826905) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.005805188989826905) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.005805188989826905) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.005652620978017348) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.005652620978017348) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.005652620978017348) [X0 X1 X3 Z4 Z5 X6]
+ (-0.005652620978017348) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.0051433917688250945) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.0051433917688250945) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.0051433917688250945) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.0051433917688250945) [X2 Z3 X4 X5 Z6 X7]
+ (-0.004684903388155198) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.004684903388155198) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668620318776299) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668620318776299) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.004575007626639205) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.004575007626639205) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.0044248554494418614) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.0044248554494418614) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.0041587973818400506) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.0041587973818400506) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.0041587973818400506) [X3 Z4 Z5 X6 X12 X13]
+ (-0.0041587973818400506) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.0034937903598901503) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.0034937903598901503) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.0034937903598901503) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.0034937903598901503) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.0027790267990255566) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.0027790267990255566) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.002293956611352464) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.002293956611352464) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.0017992194936630181) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.0017992194936630181) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278753941369718) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.0017278753941369718) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298507967730222) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298507967730222) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298507967730222) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298507967730222) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533856254125535) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533856254125535) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008145313270957037) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008145313270957037) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008145313270957037) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008145313270957037) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.735036880591678e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735036880591678e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735036880591678e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735036880591678e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.77481786459836e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.77481786459836e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.77481786459836e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.77481786459836e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.518362215686205e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.518362215686205e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.518362215686205e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.518362215686205e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.444344675906052e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.444344675906052e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.444344675906052e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.444344675906052e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.524373848492274e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.524373848492274e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.524373848492274e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.524373848492274e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.290028433171593e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.290028433171593e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.290028433171593e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.290028433171593e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.9743117134144805e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.9743117134144805e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.275883122146418e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.275883122146418e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.643051068445004e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.643051068445004e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.556569218082908e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.556569218082908e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.253224225549225e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.253224225549225e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.7696594518934165e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.7696594518934165e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.6945132945013098e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.6945132945013098e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.6102971304888104e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.6102971304888104e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.6102971304888104e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.6102971304888104e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.3131455001521017e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.3131455001521017e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.2774831954202635e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.2774831954202635e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.2774831954202635e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.2774831954202635e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.2112283483401737e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.2112283483401737e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.2112283483401737e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.2112283483401737e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.151346311101363e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.151346311101363e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.0882507112921664e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.0882507112921664e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.1726691015542624e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.1726691015542624e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.4548424491154965e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.4548424491154965e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.3304731886923088e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.3304731886923088e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2283337825146114e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2283337825146114e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.0358477601620094e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.0358477601620094e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956895373214412e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956895373214412e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733197742216139e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733197742216139e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733197742216139e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733197742216139e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.628614201697622e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.628614201697622e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.556281914616713e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.556281914616713e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.556281914616713e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.556281914616713e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.418291574656516e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.418291574656516e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.418291574656516e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.418291574656516e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.927453082733694e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927453082733694e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927453082733694e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927453082733694e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.62785191146418e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.62785191146418e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660624698296e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660624698296e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660624698296e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660624698296e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660624698296e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660624698296e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660624698296e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660624698296e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.838052751159036e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.838052751159036e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.5707613293869585e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.5707613293869585e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.3281393506854705e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.3281393506854705e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.086826565297468e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.086826565297468e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.086826565297468e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.086826565297468e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.4473231289307566e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.4473231289307566e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.3713289480877252e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.3713289480877252e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.3713289480877252e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.3713289480877252e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.199051618819924e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.199051618819924e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.9332412771900524e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.9332412771900524e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.9332412771900524e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.9332412771900524e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.8394209155849203e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.8394209155849203e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.8394209155849203e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.8394209155849203e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.551053917649649e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.551053917649649e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.551053917649649e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.551053917649649e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.3807781481153723e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.3807781481153723e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.3807781481153723e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.3807781481153723e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.3807781481153723e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.3807781481153723e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.3807781481153723e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.3807781481153723e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.3807781481153723e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.3807781481153723e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.3807781481153723e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.3807781481153723e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.291969486489774e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.291969486489774e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.1076325599702166e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.1076325599702166e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.1076325599702166e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.1076325599702166e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.1076325599702166e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.1076325599702166e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.1076325599702166e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.1076325599702166e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.057446594824443e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057446594824443e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057446594824443e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057446594824443e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649310134323865e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649310134323865e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649310134323865e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649310134323865e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.8394209155849205e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.8394209155849205e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.8394209155849205e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.8394209155849205e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.199051618819924e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.199051618819924e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.4473231289307566e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.4473231289307566e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.2362599615201115e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.2362599615201115e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.2362599615201115e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.2362599615201115e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.3281393506854705e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.3281393506854705e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.5707613293869585e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.5707613293869585e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.838052751159036e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.838052751159036e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.62785191146418e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.62785191146418e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.628614201697622e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.628614201697622e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956895373214412e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956895373214412e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.306536651838714e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.306536651838714e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.306536651838714e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.306536651838714e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.0358477601620094e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.0358477601620094e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2283337825146114e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2283337825146114e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.2393363217136181e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393363217136181e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393363217136181e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393363217136181e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.3304731886923088e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.3304731886923088e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.4548424491154965e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.4548424491154965e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.1726691015542624e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.1726691015542624e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.0882507112921664e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.0882507112921664e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.117447946285276e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.117447946285276e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.151346311101363e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.151346311101363e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.3131455001521017e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.3131455001521017e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.334331289336217e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.334331289336217e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.6945132945013098e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.6945132945013098e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.183932559401312e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.183932559401312e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.556569218082908e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.556569218082908e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.643051068445004e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.643051068445004e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.275883122146418e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.275883122146418e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.9743117134144805e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.9743117134144805e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.000292198626111066) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.000292198626111066) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.000292198626111066) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.000292198626111066) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004956762314916611) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004956762314916611) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650070219499007) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650070219499007) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650070219499007) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650070219499007) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533856254125535) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533856254125535) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.001609531381721375) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.001609531381721375) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.001609531381721375) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.001609531381721375) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0016676041811440607) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.0016676041811440607) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.0016676041811440607) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.0016676041811440607) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.0017278753941369718) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.0017278753941369718) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.0017992194936630181) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.0017992194936630181) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.002293956611352464) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.002293956611352464) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.0024629170071339282) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.0024629170071339282) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.0024629170071339282) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.0024629170071339282) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.003961560792496524) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.003961560792496524) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.003961560792496524) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.003961560792496524) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.0044248554494418614) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.0044248554494418614) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004575007626639205) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.004575007626639205) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668620318776299) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668620318776299) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.004684903388155198) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.004684903388155198) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005324835234221664) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.005324835234221664) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.005324835234221664) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.005324835234221664) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.0053686593581095225) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.0053686593581095225) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.0053686593581095225) [X2 X3 X7 Z8 Z9 X10]
+ (0.0053686593581095225) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.007960880725921583) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.007960880725921583) [X4 Z5 X6 X10 Z11 X12]
+ (0.007960880725921583) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.007960880725921583) [X5 Z6 X7 X11 Z12 X13]
+ (0.008125251921381029) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125251921381029) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.008890731522694602) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.008890731522694602) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.008890731522694602) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.008890731522694602) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.010263414868158526) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.010263414868158526) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.010263414868158526) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.010263414868158526) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.010540425907671524) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.010540425907671524) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.010540425907671524) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.010540425907671524) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010960074940542599) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010960074940542599) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010960074940542599) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010960074940542599) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307274008848154) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307274008848154) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.014411099430130869) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411099430130869) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411099430130869) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411099430130869) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.015225630757226574) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.015225630757226574) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.015225630757226574) [X3 Z4 Z5 X6 X10 X11]
+ (0.015225630757226574) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.015588250102380187) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.015588250102380187) [X2 Z3 X4 X10 Z11 X12]
+ (0.015588250102380187) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.015588250102380187) [X3 Z4 X5 X11 Z12 X13]
+ (0.018266834869375557) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.018266834869375557) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.018266834869375557) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.018266834869375557) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.019020423173039952) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.019020423173039952) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.019020423173039952) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.019020423173039952) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.02017592172353551) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.02017592172353551) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.02017592172353551) [X4 Z5 Z6 X7 X11 X12]
+ (0.02017592172353551) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.02017592172353551) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.02017592172353551) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.02017592172353551) [X5 X6 X10 Z11 Z12 X13]
+ (0.02017592172353551) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.024353077678068963) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.024353077678068963) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.024353077678068963) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.024353077678068963) [X2 Z3 X4 X11 Z12 X13]
+ (0.024353077678068963) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.024353077678068963) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.024353077678068963) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.024353077678068963) [X3 Z4 X5 X10 Z11 X12]
+ (0.024389082531149475) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.024389082531149475) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.024389082531149475) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.024389082531149475) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.02510495713884452) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.02510495713884452) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.02510495713884452) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.02510495713884452) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.030787505389143953) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.030787505389143953) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.04587947078129809) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.04587947078129809) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.056007330877807585) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.056007330877807585) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.056007330877807585) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.056007330877807585) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661349) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.05608468124661349) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661349) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.05608468124661349) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.631277928437841e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631277928437841e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-6.631277928437839e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631277928437839e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.5950860069082784e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.5950860069082784e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.5950860069082784e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.5950860069082784e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0427432770137827) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.0427432770137827) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (0.042743277013782714) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.042743277013782714) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (-0.04764261217638315) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.04764261217638315) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.04764261217638315) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.04764261217638315) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.0417188138398218) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.0417188138398218) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.0417188138398218) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.0417188138398218) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.03956441632289341) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.03956441632289341) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.03956441632289341) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03956441632289341) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03935916802205307) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.03935916802205307) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.03935916802205307) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.03935916802205307) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.039318051947197605) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.039318051947197605) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.039318051947197605) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.039318051947197605) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03560837898831257) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.03560837898831257) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.029903789512624814) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.029903789512624814) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.029903789512624814) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.029903789512624814) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.028730779551905505) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.028730779551905505) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.028730779551905505) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.028730779551905505) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.02563723829602687) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.02563723829602687) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.02563723829602687) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.02563723829602687) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.02475546329289097) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.02475546329289097) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.02475546329289097) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.02475546329289097) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.024282117354693003) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.024282117354693003) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.023145130929529086) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.023145130929529086) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.022528440196013074) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.022528440196013074) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.021433810721600853) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.021433810721600853) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.021433810721600853) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.021433810721600853) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251603) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.019257505095251603) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019028242443847296) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.019028242443847296) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.018889030304942895) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.018889030304942895) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.018889030304942895) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.018889030304942895) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.016024603689179517) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.016024603689179517) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.015225630757226574) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225630757226574) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.0146037047291621) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.0146037047291621) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564531231172994) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564531231172994) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.01175601341981928) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.01175601341981928) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.011285190200840905) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.011285190200840905) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.009841749246962607) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009841749246962607) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.009612634606847352) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612634606847352) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612634606847352) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612634606847352) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.00846997879102396) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.00846997879102396) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.007306759928832955) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.007306759928832955) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0059237983365613475) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.0059237983365613475) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.005652620978017348) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.005652620978017348) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.0053686593581095225) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0053686593581095225) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.0041587973818400506) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.0041587973818400506) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.003356670563832898) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.003356670563832898) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.003356670563832898) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.003356670563832898) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.003267513854423557) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.003267513854423557) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.003267513854423557) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.003267513854423557) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.0027790267990255566) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.0027790267990255566) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.002686040977806622) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.002686040977806622) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.002686040977806622) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.002686040977806622) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.002293956611352464) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.002293956611352464) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.002293956611352464) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.002293956611352464) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.0009581655836696515) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.0009581655836696515) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0009581655836696515) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.0009581655836696515) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.0009581655836696515) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.0009581655836696515) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0009581655836696515) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.0009581655836696515) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.00024636437569580525) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00024636437569580525) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0001384017730355091) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.0001384017730355091) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.0001384017730355091) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.0001384017730355091) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.735036880591678e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735036880591678e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585305727173e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585305727173e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.6103585305727173e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.6103585305727173e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.531680879529636e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.531680879529636e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.531680879529636e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.531680879529636e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.806102775124812e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.806102775124812e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.806102775124812e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.806102775124812e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.089799467596695e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.089799467596695e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.089799467596695e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.089799467596695e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.6522096693092575e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.6522096693092575e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.6522096693092575e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.6522096693092575e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.48185183375765e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.48185183375765e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.48185183375765e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.48185183375765e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.071480736371916e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.071480736371916e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.071480736371916e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.071480736371916e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.734622038752895e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.734622038752895e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.734622038752895e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.734622038752895e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.728843147219162e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.728843147219162e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.728843147219162e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.728843147219162e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.253224225549225e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.253224225549225e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.7696594518934165e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.7696594518934165e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.544395429301296e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.544395429301296e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.544395429301296e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.544395429301296e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.544395429301296e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.544395429301296e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.544395429301296e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.544395429301296e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.360956320377532e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.360956320377532e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.360956320377532e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.360956320377532e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.103215604618379e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.103215604618379e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.103215604618379e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.103215604618379e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220980989068e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.0111220980989068e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220980989068e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.0111220980989068e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9429468366904828e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.9429468366904828e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9429468366904828e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9429468366904828e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.654117477086397e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.654117477086397e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.654117477086397e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.654117477086397e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.5224930675859628e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224930675859628e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224930675859628e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224930675859628e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224930675859628e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930675859628e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930675859628e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224930675859628e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.2283337825146114e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337825146114e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.2283337825146114e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337825146114e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.988770288807506e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988770288807506e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988770288807506e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988770288807506e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.867765104308155e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104308155e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104308155e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.867765104308155e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.189990975233443e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.189990975233443e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246207017154e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246207017154e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.471647744634499e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.471647744634499e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.561447180091295e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.561447180091295e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.561447180091295e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.561447180091295e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.5233896777599155e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.5233896777599155e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.427323108716213e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.427323108716213e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.427323108716213e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.427323108716213e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.3281393506854705e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.3281393506854705e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.3281393506854705e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.3281393506854705e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.086826565297468e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.086826565297468e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.888293596040855e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.888293596040855e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.888293596040855e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.888293596040855e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.3713289480877252e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.3713289480877252e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.8394209155849203e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.8394209155849203e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.057446594824443e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057446594824443e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.5371780952884815e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.5371780952884815e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.5371780952884815e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.5371780952884815e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.057446594824443e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057446594824443e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.209350651947177e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350651947177e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350651947177e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.209350651947177e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.703578355516078e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.703578355516078e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.703578355516078e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.703578355516078e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.8394209155849203e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.8394209155849203e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.3713289480877252e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.3713289480877252e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.086826565297468e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.086826565297468e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.5233896777599155e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.5233896777599155e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.471647744634499e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.471647744634499e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246207017154e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246207017154e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189990975233443e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.189990975233443e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.3304731886923088e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.3304731886923088e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.3304731886923088e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.3304731886923088e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288532435154002e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288532435154002e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288532435154002e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288532435154002e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.6893489514523282e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.6893489514523282e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.6893489514523282e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.6893489514523282e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.745518400420546e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.745518400420546e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.745518400420546e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.745518400420546e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.745518400420546e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.745518400420546e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.745518400420546e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.745518400420546e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.2118420190382905e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420190382905e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.2118420190382905e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420190382905e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.2118420190382905e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420190382905e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.2118420190382905e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420190382905e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.313145500152102e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.313145500152102e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.313145500152102e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.313145500152102e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.3343312893362165e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.3343312893362165e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.183932559401312e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.183932559401312e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.735036880591678e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735036880591678e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00024636437569580525) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00024636437569580525) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00044585351288408965) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.00044585351288408965) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.00044585351288408965) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.00044585351288408965) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940221543005268) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005268) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005268) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005268) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005268) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005268) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940221543005268) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005268) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533856254125535) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533856254125535) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533856254125535) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533856254125535) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0010435246534907516) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435246534907516) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435246534907516) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435246534907516) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.0012803060973496658) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.0012803060973496658) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.0012803060973496658) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.0012803060973496658) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.0013038004788126964) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.0013038004788126964) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.0013038004788126964) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.0013038004788126964) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.002261966062482348) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.002261966062482348) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.002261966062482348) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.002261966062482348) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.002261966062482348) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.002261966062482348) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.002261966062482348) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.002261966062482348) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.0039898414566193205) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.0039898414566193205) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.0039898414566193205) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.0039898414566193205) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.0041587973818400506) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.0041587973818400506) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.004311038507914309) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.004311038507914309) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.004311038507914309) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.004311038507914309) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.004636976661182564) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.004636976661182564) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.004636976661182564) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.004636976661182564) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.0051144738316603894) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.0051144738316603894) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.0051144738316603894) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.0051144738316603894) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.0051144738316603894) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.0051144738316603894) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.0051144738316603894) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.0051144738316603894) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00524153538280386) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.00524153538280386) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.00524153538280386) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.00524153538280386) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.005262642473076826) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262642473076826) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262642473076826) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262642473076826) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.0053686593581095225) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.0053686593581095225) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.0053799371558393705) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.0053799371558393705) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.0053799371558393705) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.0053799371558393705) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.005652620978017348) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.005652620978017348) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.005708495985960916) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708495985960916) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708495985960916) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708495985960916) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.0059237983365613475) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.0059237983365613475) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.007306759928832955) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.007306759928832955) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00846997879102396) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.00846997879102396) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.009841749246962607) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.009841749246962607) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.011285190200840905) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.011285190200840905) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.01175601341981928) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.01175601341981928) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.014564531231172994) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564531231172994) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.0146037047291621) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.0146037047291621) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.015225630757226574) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225630757226574) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.016024603689179517) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.016024603689179517) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.019028242443847296) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.019028242443847296) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.019257505095251603) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.019257505095251603) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.04587947078129809) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04587947078129809) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.3693708936615621) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.3693708936615621) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.3693708936615621) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.3693708936615621) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.2816425776702291) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.2816425776702291) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.281642577670229) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.281642577670229) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0906514420703648) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0906514420703648) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0906514420703648) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0906514420703648) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863627) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684737589863627) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863627) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684737589863627) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635021950635007) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635021950635007) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635021950635007) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635021950635007) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214021) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.06752385099214021) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214021) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.06752385099214021) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03560837898831257) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.03560837898831257) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.024591860883830034) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.024591860883830034) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883830034) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.024591860883830034) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.024282117354693003) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.024282117354693003) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.023145130929529086) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.023145130929529086) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.022528440196013074) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.022528440196013074) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.01953805031131471) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.01953805031131471) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.01953805031131471) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.01953805031131471) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.017091553155898824) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.017091553155898824) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.017091553155898824) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.017091553155898824) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.016024603689179517) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.016024603689179517) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.016024603689179517) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.016024603689179517) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.010311482489831781) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831781) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831781) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311482489831781) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.009841749246962607) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962607) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.009841749246962607) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962607) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209842) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209842) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209842) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008826368514209842) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008541996625454852) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454852) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454852) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454852) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454852) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454852) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454852) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008541996625454852) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00846997879102396) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.00846997879102396) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.00846997879102396) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.00846997879102396) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.004668620318776299) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668620318776299) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.0038764708993369434) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.0038764708993369434) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.003804066171728542) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728542) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728542) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003804066171728542) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484157300217885) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484157300217885) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003356670563832898) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.003356670563832898) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.003267513854423557) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.003267513854423557) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.002141361223101576) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.002141361223101576) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.0017278753941369718) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.0017278753941369718) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.0016407548553123874) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0016407548553123874) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214168768) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0014528843214168768) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214168768) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0014528843214168768) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870896771024452) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007870896771024452) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192743499487605) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192743499487605) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.00019400857029756424) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00019400857029756424) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0001384017730355091) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.0001384017730355091) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.141625221153919e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.141625221153919e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.141625221153919e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.141625221153919e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.071480736371916e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.071480736371916e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.151346311101363e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.151346311101363e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0882507112921664e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.0882507112921664e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.9885117063844667e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.9885117063844667e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8742990713834513e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.8742990713834513e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.3609563203775326e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.3609563203775326e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.300294656203075e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.300294656203075e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.1468376507366776e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.1468376507366776e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.1468376507366776e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.1468376507366776e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.352332102992766e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.352332102992766e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.352332102992766e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.352332102992766e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091637198911083e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637198911083e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637198911083e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637198911083e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637198911083e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637198911083e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637198911083e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091637198911083e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.074305985841887e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.074305985841887e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.074305985841887e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.074305985841887e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128986215609e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128986215609e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128986215609e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128986215609e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867765104308155e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.867765104308155e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560692464775057e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692464775057e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692464775057e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692464775057e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692464775057e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692464775057e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692464775057e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560692464775057e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.99701842203579e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.99701842203579e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.99701842203579e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.99701842203579e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.99701842203579e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.99701842203579e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.99701842203579e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.99701842203579e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.5682475211511667e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.5682475211511667e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.5682475211511667e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.5682475211511667e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376739308455691e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.376739308455691e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376739308455691e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.376739308455691e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376739308455691e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.376739308455691e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.376739308455691e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.376739308455691e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.888293596040855e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.888293596040855e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6863815449968896e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.6863815449968896e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035783555160782e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7035783555160782e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.209350651947177e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.209350651947177e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244130775e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773244130775e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244130775e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773244130775e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244130775e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773244130775e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773244130775e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773244130775e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.974225379363584e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.974225379363584e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.974225379363584e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.974225379363584e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.04747165552811e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.04747165552811e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.04747165552811e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.04747165552811e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.209350651947177e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.209350651947177e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0717282183369733e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0717282183369733e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0717282183369733e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0717282183369733e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.2004287494042225e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.2004287494042225e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.2004287494042225e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.2004287494042225e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.7035783555160782e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7035783555160782e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.3120943052298415e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.3120943052298415e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.3120943052298415e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.3120943052298415e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.6863815449968896e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815449968896e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.888293596040855e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.888293596040855e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.092250616044878e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.092250616044878e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.092250616044878e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.092250616044878e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.092250616044878e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.092250616044878e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.092250616044878e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.092250616044878e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.444597854181368e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.444597854181368e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.444597854181368e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.444597854181368e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.684915095222506e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.684915095222506e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.684915095222506e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.684915095222506e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246974425354128e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246974425354128e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246974425354128e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246974425354128e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246974425354128e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246974425354128e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246974425354128e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246974425354128e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.867765104308155e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.867765104308155e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.300294656203075e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.300294656203075e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.3609563203775326e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.3609563203775326e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.8742990713834513e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.8742990713834513e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.8836765760585263e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.8836765760585263e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.947356011630908e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.947356011630908e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.947356011630908e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.947356011630908e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117063844667e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9885117063844667e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.0882507112921664e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.0882507112921664e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.151346311101363e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.151346311101363e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.846201671262716e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846201671262716e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846201671262716e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846201671262716e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.071480736371916e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.071480736371916e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.105526722004969e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.105526722004969e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.105526722004969e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.105526722004969e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.1464963274657916e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.1464963274657916e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.1464963274657916e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.1464963274657916e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.1593505019469574e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.1593505019469574e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.1593505019469574e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.1593505019469574e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427988656446647e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427988656446647e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427988656446647e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427988656446647e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935867718015374e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935867718015374e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935867718015374e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935867718015374e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348099169e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.253273348099169e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.97982579338842e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.97982579338842e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.97982579338842e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.97982579338842e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.2055484112195254e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.2055484112195254e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.2055484112195254e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.2055484112195254e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0001384017730355091) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.0001384017730355091) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.0001878705338955105) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0001878705338955105) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0001878705338955105) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0001878705338955105) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00019400857029756424) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00019400857029756424) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00024636437569580525) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569580525) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00024636437569580525) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569580525) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192743499487605) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192743499487605) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156734248909061) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156734248909061) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156734248909061) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156734248909061) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024452) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007870896771024452) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.0015324835230730654) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.0015324835230730654) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.0015324835230730654) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.0015324835230730654) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.0016407548553123874) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0016407548553123874) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0017278753941369718) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.0017278753941369718) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.002446497155415889) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.002446497155415889) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.002446497155415889) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.002446497155415889) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.003267513854423557) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.003267513854423557) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.003356670563832898) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.003356670563832898) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.003484157300217885) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.003484157300217885) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0038764708993369434) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.0038764708993369434) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.004668620318776299) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668620318776299) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278141) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.004767272188278141) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.004767272188278141) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278141) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.005286546538226901) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.005286546538226901) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.005286546538226901) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.005286546538226901) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.0054089544224100086) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.0054089544224100086) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.0054089544224100086) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.0054089544224100086) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.0059237983365613475) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.0059237983365613475) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.0059237983365613475) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.0059237983365613475) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.010715508469796754) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010715508469796754) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010715508469796754) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010715508469796754) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01075756395390895) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01075756395390895) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.01075756395390895) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01075756395390895) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.014603704729162099) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.014603704729162099) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.014603704729162099) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.014603704729162099) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.0192995605793638) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0192995605793638) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0192995605793638) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0192995605793638) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0192995605793638) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0192995605793638) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0192995605793638) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0192995605793638) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.05859198873386197) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.05859198873386197) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.77595052730298e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.77595052730298e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.7759505273029824e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.7759505273029824e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.07165035181002777) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.07165035181002777) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.07165035181002778) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07165035181002778) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.019257505095251603) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019257505095251603) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010311482489831781) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311482489831781) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008826368514209842) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209842) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.007597464029770591) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.007597464029770591) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.007597464029770591) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.007597464029770591) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311878) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311878) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311878) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311878) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311878) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311878) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311878) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311878) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676598) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005348051582676598) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005348051582676598) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676598) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.003804066171728542) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003804066171728542) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984166168121928) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.002984166168121928) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.002984166168121928) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.002984166168121928) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.0024464971554158887) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.0024464971554158887) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.002249412447093992) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.002249412447093992) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.002249412447093992) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002249412447093992) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.002141361223101576) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.002141361223101576) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.0018638942824587136) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587136) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587136) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587136) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587136) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587136) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0018638942824587136) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587136) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0016407548553123874) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553123874) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0016407548553123874) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553123874) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0012223378081538411) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538411) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538411) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538411) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538411) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538411) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538411) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0012223378081538411) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0010283292378562767) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0010283292378562767) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0010283292378562767) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0010283292378562767) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1463061453052203e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1463061453052203e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.8742990713834513e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990713834513e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.8742990713834513e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990713834513e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.300294656203075e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.300294656203075e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.300294656203075e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.300294656203075e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.0444941298097393e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.0444941298097393e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.0444941298097393e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.0444941298097393e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.956079230061112e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.956079230061112e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.956079230061112e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.956079230061112e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105515037185826e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105515037185826e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105515037185826e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105515037185826e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661347213156329e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661347213156329e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661347213156329e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661347213156329e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341413661166e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341413661166e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189990975233443e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.189990975233443e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876621658263563e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876621658263563e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876621658263563e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876621658263563e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246207017154e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246207017154e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.5233896777599155e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.5233896777599155e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.076732532204164e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.076732532204164e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.076732532204164e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.076732532204164e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.0134714588989965e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0134714588989965e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.9045998844362263e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.9045998844362263e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.9045998844362263e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.9045998844362263e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.6667317550221856e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.6667317550221856e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.6667317550221856e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6667317550221856e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.850564192875286e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.850564192875286e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309315890846e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6569309315890846e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309315890846e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6569309315890846e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.850564192875286e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.850564192875286e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815449968896e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815449968896e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.6863815449968896e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815449968896e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714588989965e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.0134714588989965e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.5233896777599155e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.5233896777599155e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.670402390488081e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.670402390488081e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.670402390488081e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.670402390488081e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246207017154e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246207017154e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189990975233443e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.189990975233443e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341413661166e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341413661166e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949476487165279e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949476487165279e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.7924939577021011e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939577021011e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939577021011e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.7924939577021011e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.8836765760585263e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.8836765760585263e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.9885117063844667e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117063844667e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117063844667e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117063844667e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348099169e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253273348099169e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.401710973524949e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.401710973524949e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.401710973524949e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.401710973524949e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.580960369295159e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.580960369295159e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.580960369295159e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.580960369295159e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192743499487605) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487605) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192743499487605) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487605) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.0007870896771024452) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024452) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024452) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024452) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0011726348316441865) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0011726348316441865) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0011726348316441865) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0011726348316441865) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012366478019245515) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.0012366478019245515) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.0012366478019245515) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.0012366478019245515) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.0022009640695004632) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0022009640695004632) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0022009640695004632) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0022009640695004632) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980275) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980275) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0023949726397980275) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980275) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0023949726397980275) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0023949726397980275) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0023949726397980275) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0023949726397980275) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0024464971554158887) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.0024464971554158887) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.003804066171728542) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.003804066171728542) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.003876470899336943) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.003876470899336943) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.003876470899336943) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.003876470899336943) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.00422081397004648) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.00422081397004648) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.00422081397004648) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.00422081397004648) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.008826368514209842) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.008826368514209842) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.010311482489831781) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311482489831781) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257505095251603) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019257505095251603) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.05859198873386197) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.05859198873386197) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3987009015878976e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3987009015878976e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.3987009015878972e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.3987009015878972e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484157300217885) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003484157300217885) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984166168121928) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.002984166168121928) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.00019400857029756424) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00019400857029756424) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453052203e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453052203e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7924939577021014e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7924939577021014e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540341413661166e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413661166e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540341413661166e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413661166e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.850564192875286e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.850564192875286e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.850564192875286e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.850564192875286e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458898996e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471458898996e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458898996e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471458898996e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949476487165279e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949476487165279e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.7924939577021014e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7924939577021014e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756424) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756424) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002984166168121928) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.002984166168121928) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.003484157300217885) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.003484157300217885) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
 </code>
 </pre>
 </details>

---

## 4. tutorial_quantum_chemistry.html <a name="demo3"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quantum_chemistry.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
(-46.46390678868898+0j) [] +
(-3.570761329486416e-07+0j) [X0 X1 Y2 Z3 Z4 Y5] +
(-0.00565262097801737+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7] +
(-0.00882636851420987+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.7924939576737372e-06+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.570761329486416e-07+0j) [X0 X1 X3 X4] +
(-0.00565262097801737+0j) [X0 X1 X3 Z4 Z5 X6] +
(-0.008826368514209872+0j) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.7924939576737372e-06+0j) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.002745836470186805+0j) [X0 X1 Y4 Y5] +
(-2.447323129047035e-07+0j) [X0 X1 Y4 Z5 Z6 Y7] +
(-7.867765104126099e-07+0j) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.003804066171728539+0j) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.447323129047035e-07+0j) [X0 X1 X5 X6] +
(-7.867765104126099e-07+0j) [X0 X1 X5 Z6 Z7 Z8 Z9 X10] +
(-0.003804066171728539+0j) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.006888194352970575+0j) [X0 X1 Y6 Y7] +
(-7.735036880591616e-05+0j) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11] +
(1.703578355677888e-07+0j) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.735036880591616e-05+0j) [X0 X1 X7 Z8 Z9 X10] +
(1.703578355677888e-07+0j) [X0 X1 X7 Z8 Z9 Z10 Z11 X12] +
(-0.006509361201177239+0j) [X0 X1 Y8 Y9] +
(-0.0077314252507753095+0j) [X0 X1 Y10 Y11] +
(5.627851911379409e-07+0j) [X0 X1 Y10 Z11 Z12 Y13] +
(5.627851911379408e-07+0j) [X0 X1 X11 X12] +
(-0.005283776488402964+0j) [X0 X1 Y12 Y13] +
(3.570761329486416e-07+0j) [X0 Y1 Y2 Z3 Z4 X5] +
(0.00565262097801737+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7] +
(0.00882636851420987+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-1.7924939576737372e-06+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.570761329486416e-07+0j) [X0 Y1 Y3 X4] +
(-0.00565262097801737+0j) [X0 Y1 Y3 Z4 Z5 X6] +
(-0.008826368514209872+0j) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.7924939576737372e-06+0j) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.002745836470186805+0j) [X0 Y1 Y4 X5] +
(2.447323129047035e-07+0j) [X0 Y1 Y4 Z5 Z6 X7] +
(7.867765104126099e-07+0j) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.003804066171728539+0j) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.447323129047035e-07+0j) [X0 Y1 Y5 X6] +
(-7.867765104126099e-07+0j) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.003804066171728539+0j) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.006888194352970575+0j) [X0 Y1 Y6 X7] +
(7.735036880591616e-05+0j) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11] +
(-1.703578355677888e-07+0j) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.735036880591616e-05+0j) [X0 Y1 Y7 Z8 Z9 X10] +
(1.703578355677888e-07+0j) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12] +
(0.006509361201177239+0j) [X0 Y1 Y8 X9] +
(0.0077314252507753095+0j) [X0 Y1 Y10 X11] +
(-5.627851911379409e-07+0j) [X0 Y1 Y10 Z11 Z12 X13] +
(5.627851911379408e-07+0j) [X0 Y1 Y11 X12] +
(0.005283776488402964+0j) [X0 Y1 Y12 X13] +
(0.12507032579772265+0j) [X0 Z1 X2] +
(-1.9332412772834116e-07+0j) [X0 Z1 X2 X3 Z4 X5] +
(-0.002293956611352475+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 X7] +
(-0.001640754855312421+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(3.013471458804342e-07+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.9332412772834116e-07+0j) [X0 Z1 X2 Y3 Z4 Y5] +
(-0.002293956611352475+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7] +
(-0.001640754855312421+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(3.013471458804342e-07+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0032769719312316786+0j) [X0 Z1 X2 Z3] +
(-1.551053917739348e-07+0j) [X0 Z1 X2 X4 Z5 X6] +
(-1.146837650686876e-06+0j) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.007597464029770616+0j) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.380778148193924e-07+0j) [X0 Z1 X2 Y4 Z5 Y6] +
(-7.900128985790538e-07+0j) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.005348051582676623+0j) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005530759218631541+0j) [X0 Z1 X2 Z4] +
(-1.380778148193924e-07+0j) [X0 Z1 X2 X5 Z6 X7] +
(-3.3767393083651703e-07+0j) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0018638942824587411+0j) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.380778148193924e-07+0j) [X0 Z1 X2 Y5 Z6 Y7] +
(-3.3767393083651703e-07+0j) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0018638942824587411+0j) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0011059037691896955+0j) [X0 Z1 X2 Z5] +
(0.0057084959859609406+0j) [X0 Z1 X2 X6 Z7 Z8 Z9 X10] +
(-8.352332102497864e-07+0j) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
1.9742253791047746e-08j [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005262642473076857+0j) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10] +
(-8.074305985408966e-07+0j) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.008032520918821392+0j) [X0 Z1 X2 Z6] +
(0.0005940221543005461+0j) [X0 Z1 X2 X7 Z8 Z9 Z10 X11] +
(-8.37977324328611e-08+0j) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005940221543005461+0j) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11] +
(-8.37977324328611e-08+0j) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0033476175306662017+0j) [X0 Z1 X2 Z7] +
(0.011055020596132097+0j) [X0 Z1 X2 Z8] +
(0.0029297686747510676+0j) [X0 Z1 X2 Z9] +
(-6.418291574545231e-07+0j) [X0 Z1 X2 X10 Z11 X12] +
(-6.556281914498774e-07+0j) [X0 Z1 X2 Y10 Z11 Y12] +
(0.003555290195504281+0j) [X0 Z1 X2 Z10] +
(-1.1076325599347869e-07+0j) [X0 Z1 X2 X11 Z12 X13] +
(-1.1076325599347869e-07+0j) [X0 Z1 X2 Y11 Z12 Y13] +
(0.0017560707018412598+0j) [X0 Z1 X2 Z11] +
(0.006901238249797304+0j) [X0 Z1 X2 Z12] +
(0.002326230623158096+0j) [X0 Z1 X2 Z13] +
(-3.5682475210782223e-07+0j) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.002249412447093991+0j) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.0474716554238044e-08+0j) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.00044585351288408716+0j) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10] +
(-1.9742253790470122e-08+0j) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441847+0j) [X0 Z1 Z2 X3 Y4 Y5] +
(-4.523389677425367e-07+0j) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0034841573002178834+0j) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-8.09163719850359e-07+0j) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10] +
(-0.005733569747311876+0j) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004684903388155191+0j) [X0 Z1 Z2 X3 Y6 Y7] +
(0.00466862031877631+0j) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11] +
(-7.189990974896306e-07+0j) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0051144738316603955+0j) [X0 Z1 Z2 X3 X7 Z8 Z9 X10] +
(-7.560692464353301e-07+0j) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381029+0j) [X0 Z1 Z2 X3 Y8 Y9] +
(-0.0017992194936630212+0j) [X0 Z1 Z2 X3 Y10 Y11] +
(-5.471647744557047e-07+0j) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13] +
(-5.287660624617383e-07+0j) [X0 Z1 Z2 X3 X11 X12] +
(-0.004575007626639207+0j) [X0 Z1 Z2 X3 Y12 Y13] +
(0.004424855449441847+0j) [X0 Z1 Z2 Y3 Y4 X5] +
(4.523389677425367e-07+0j) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0034841573002178834+0j) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.09163719850359e-07+0j) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.005733569747311876+0j) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.004684903388155191+0j) [X0 Z1 Z2 Y3 Y6 X7] +
(-0.00466862031877631+0j) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11] +
(7.189990974896306e-07+0j) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0051144738316603955+0j) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10] +
(-7.560692464353301e-07+0j) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12] +
(0.008125251921381029+0j) [X0 Z1 Z2 Y3 Y8 X9] +
(0.0017992194936630212+0j) [X0 Z1 Z2 Y3 Y10 X11] +
(5.471647744557047e-07+0j) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13] +
(-5.287660624617383e-07+0j) [X0 Z1 Z2 Y3 Y11 X12] +
(0.004575007626639207+0j) [X0 Z1 Z2 Y3 Y12 X13] +
(3.202076880447949e-06+0j) [X0 Z1 Z2 Z3 X4] +
(0.0008533856254125472+0j) [X0 Z1 Z2 Z3 X4 X5 Z6 X7] +
(0.0007870896771024451+0j) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0008533856254125472+0j) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7] +
(0.0007870896771024451+0j) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.2919694863380308e-07+0j) [X0 Z1 Z2 Z3 X4 Z5] +
(4.4445978542416875e-07+0j) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10] +
(0.001172634831644189+0j) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.6849150951351447e-07+0j) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10] +
(0.0022009640695004637+0j) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.8394209156697395e-07+0j) [X0 Z1 Z2 Z3 X4 Z6] +
(4.0922506159816434e-07+0j) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11] +
(0.0023949726397980205+0j) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.0922506159816434e-07+0j) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11] +
(0.0023949726397980205+0j) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.236259961715071e-07+0j) [X0 Z1 Z2 Z3 X4 Z8] +
(8.649310135731738e-08+0j) [X0 Z1 Z2 Z3 X4 Z9] +
(0.0013038004788126934+0j) [X0 Z1 Z2 Z3 X4 X10 Z11 X12] +
(0.003989841456619313+0j) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12] +
(-6.733197741794053e-07+0j) [X0 Z1 Z2 Z3 X4 Z10] +
(0.002261966062482341+0j) [X0 Z1 Z2 Z3 X4 X11 Z12 X13] +
(0.002261966062482341+0j) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13] +
(-5.92745308233278e-07+0j) [X0 Z1 Z2 Z3 X4 Z11] +
(1.239336321719257e-06+0j) [X0 Z1 Z2 Z3 X4 Z12] +
(9.306536651743931e-07+0j) [X0 Z1 Z2 Z3 X4 Z13] +
(-0.001028329237856275+0j) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.0026860409778066193+0j) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12] +
(-1.8394209156697398e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7] +
(-0.00019400857029755713+0j) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0012223378081538318+0j) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-2.3713289481418972e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9] +
(8.057446594612721e-08+0j) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11] +
(0.0017278753941369718+0j) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13] +
(-0.000958165583669648+0j) [X0 Z1 Z2 Z3 Z4 X5 X11 X12] +
(-3.0868265654486404e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13] +
(1.8394209156697398e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7] +
(0.00019400857029755713+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0012223378081538318+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(2.3713289481418972e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9] +
(-8.057446594612721e-08+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11] +
(-0.0017278753941369718+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13] +
(-0.000958165583669648+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12] +
(3.0868265654486404e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13] +
(0.04274327701378341+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6] +
(0.0005192743499487665+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(-1.8505641929368898e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005192743499487665+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(-1.8505641929368898e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0027790267990255306+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7] +
(0.004636976661182572+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8] +
(0.0012803060973496806+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9] +
(2.3120943051453685e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12] +
(1.07172821813145e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12] +
(0.0053799371558393904+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10] +
(7.246974425083315e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13] +
(7.246974425083315e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13] +
(0.005241535382803877+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11] +
(0.004311038507914323+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12] +
(0.001043524653490765+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13] +
(1.2004287493969867e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12] +
(-0.0033566705638328927+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9] +
(-0.00013840177303551388+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11] +
(-6.175246206951864e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(-4.997018421848276e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12] +
(-0.0032675138544235576+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13] +
(0.0033566705638328927+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9] +
(0.00013840177303551388+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11] +
(6.175246206951864e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(-4.997018421848276e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12] +
(0.0032675138544235576+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13] +
(0.003876470899336945+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(-7.540341413482891e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.003876470899336945+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(-7.540341413482891e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(0.07165035181002839+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0021413612231015954+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(0.004220813970046477+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(0.0012366478019245426+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(-0.002984166168121934+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(0.002984166168121934+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-1.3987009016643128e-05+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(8.949476486905753e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-6.876621658182731e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(-7.661347213055165e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(0.0015324835230730574+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10] +
(-2.904599884492375e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(0.005408954422410002+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10] +
(-1.0444941297975267e-06+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(0.00476727218827813+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10] +
(-8.10551503697011e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(0.005286546538226896+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10] +
(-9.956079229907e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0016095313817213858+0j) [X0 Z1 Z2 Z3 Z4 X6] +
(-7.141625221155571e-05+0j) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10] +
(-2.6667317550582486e-07+0j) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.002462917007133933+0j) [X0 Z1 Z2 Z3 Z5 X6] +
(0.0007156734248908893+0j) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10] +
(-3.0767325321484006e-07+0j) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.606071868267355e-07+0j) [X0 Z1 Z2 X4] +
(0.003961560792496537+0j) [X0 Z1 Z2 Z4 Z5 X6] +
(0.00018787053389551813+0j) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.6569309315286678e-07+0j) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.7379332625590598e-07+0j) [X0 Z1 Z3 X4] +
(0.0016676041811440625+0j) [X0 Z1 Z3 Z4 Z5 X6] +
(-0.0014528843214169026+0j) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(4.6704023903330105e-07+0j) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.10433064780651427+0j) [X0 X2] +
(3.1174479464120664e-06+0j) [X0 Z2 Z3 X4] +
(0.04587947078129816+0j) [X0 Z2 Z3 Z4 Z5 X6] +
(0.05859198873386204+0j) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-1.1463061452870514e-05+0j) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.570761329486416e-07+0j) [Y0 X1 X2 Z3 Z4 Y5] +
(0.00565262097801737+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7] +
(0.00882636851420987+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.7924939576737372e-06+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.570761329486416e-07+0j) [Y0 X1 X3 Y4] +
(-0.00565262097801737+0j) [Y0 X1 X3 Z4 Z5 Y6] +
(-0.008826368514209872+0j) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.7924939576737372e-06+0j) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.002745836470186805+0j) [Y0 X1 X4 Y5] +
(2.447323129047035e-07+0j) [Y0 X1 X4 Z5 Z6 Y7] +
(7.867765104126099e-07+0j) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.003804066171728539+0j) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.447323129047035e-07+0j) [Y0 X1 X5 Y6] +
(-7.867765104126099e-07+0j) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.003804066171728539+0j) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.006888194352970575+0j) [Y0 X1 X6 Y7] +
(7.735036880591616e-05+0j) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11] +
(-1.703578355677888e-07+0j) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.735036880591616e-05+0j) [Y0 X1 X7 Z8 Z9 Y10] +
(1.703578355677888e-07+0j) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12] +
(0.006509361201177239+0j) [Y0 X1 X8 Y9] +
(0.0077314252507753095+0j) [Y0 X1 X10 Y11] +
(-5.627851911379409e-07+0j) [Y0 X1 X10 Z11 Z12 Y13] +
(5.627851911379408e-07+0j) [Y0 X1 X11 Y12] +
(0.005283776488402964+0j) [Y0 X1 X12 Y13] +
(-3.570761329486416e-07+0j) [Y0 Y1 X2 Z3 Z4 X5] +
(-0.00565262097801737+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7] +
(-0.00882636851420987+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.7924939576737372e-06+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.570761329486416e-07+0j) [Y0 Y1 Y3 Y4] +
(-0.00565262097801737+0j) [Y0 Y1 Y3 Z4 Z5 Y6] +
(-0.008826368514209872+0j) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.7924939576737372e-06+0j) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.002745836470186805+0j) [Y0 Y1 X4 X5] +
(-2.447323129047035e-07+0j) [Y0 Y1 X4 Z5 Z6 X7] +
(-7.867765104126099e-07+0j) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.003804066171728539+0j) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.447323129047035e-07+0j) [Y0 Y1 Y5 Y6] +
(-7.867765104126099e-07+0j) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.003804066171728539+0j) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.006888194352970575+0j) [Y0 Y1 X6 X7] +
(-7.735036880591616e-05+0j) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11] +
(1.703578355677888e-07+0j) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.735036880591616e-05+0j) [Y0 Y1 Y7 Z8 Z9 Y10] +
(1.703578355677888e-07+0j) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.006509361201177239+0j) [Y0 Y1 X8 X9] +
(-0.0077314252507753095+0j) [Y0 Y1 X10 X11] +
(5.627851911379409e-07+0j) [Y0 Y1 X10 Z11 Z12 X13] +
(5.627851911379408e-07+0j) [Y0 Y1 Y11 Y12] +
(-0.005283776488402964+0j) [Y0 Y1 X12 X13] +
(-3.5682475210782223e-07+0j) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.002249412447093991+0j) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.00044585351288408716+0j) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10] +
(-1.9742253790470122e-08+0j) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.0474716554238044e-08+0j) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.12507032579772265+0j) [Y0 Z1 Y2] +
(-1.9332412772834116e-07+0j) [Y0 Z1 Y2 X3 Z4 X5] +
(-0.002293956611352475+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7] +
(-0.001640754855312421+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(3.013471458804342e-07+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.9332412772834116e-07+0j) [Y0 Z1 Y2 Y3 Z4 Y5] +
(-0.002293956611352475+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7] +
(-0.001640754855312421+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(3.013471458804342e-07+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0032769719312316786+0j) [Y0 Z1 Y2 Z3] +
(-1.380778148193924e-07+0j) [Y0 Z1 Y2 X4 Z5 X6] +
(-7.900128985790538e-07+0j) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.005348051582676623+0j) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.551053917739348e-07+0j) [Y0 Z1 Y2 Y4 Z5 Y6] +
(-1.146837650686876e-06+0j) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.007597464029770616+0j) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005530759218631541+0j) [Y0 Z1 Y2 Z4] +
(-1.380778148193924e-07+0j) [Y0 Z1 Y2 X5 Z6 X7] +
(-3.3767393083651703e-07+0j) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0018638942824587411+0j) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.380778148193924e-07+0j) [Y0 Z1 Y2 Y5 Z6 Y7] +
(-3.3767393083651703e-07+0j) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0018638942824587411+0j) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0011059037691896955+0j) [Y0 Z1 Y2 Z5] +
(0.005262642473076857+0j) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10] +
(-8.074305985408966e-07+0j) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0057084959859609406+0j) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10] +
-1.9742253791047746e-08j [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-8.352332102497864e-07+0j) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.008032520918821392+0j) [Y0 Z1 Y2 Z6] +
(0.0005940221543005461+0j) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11] +
(-8.37977324328611e-08+0j) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005940221543005461+0j) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11] +
(-8.37977324328611e-08+0j) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0033476175306662017+0j) [Y0 Z1 Y2 Z7] +
(0.011055020596132097+0j) [Y0 Z1 Y2 Z8] +
(0.0029297686747510676+0j) [Y0 Z1 Y2 Z9] +
(-6.556281914498774e-07+0j) [Y0 Z1 Y2 X10 Z11 X12] +
(-6.418291574545231e-07+0j) [Y0 Z1 Y2 Y10 Z11 Y12] +
(0.003555290195504281+0j) [Y0 Z1 Y2 Z10] +
(-1.1076325599347869e-07+0j) [Y0 Z1 Y2 X11 Z12 X13] +
(-1.1076325599347869e-07+0j) [Y0 Z1 Y2 Y11 Z12 Y13] +
(0.0017560707018412598+0j) [Y0 Z1 Y2 Z11] +
(0.006901238249797304+0j) [Y0 Z1 Y2 Z12] +
(0.002326230623158096+0j) [Y0 Z1 Y2 Z13] +
(0.004424855449441847+0j) [Y0 Z1 Z2 X3 X4 Y5] +
(4.523389677425367e-07+0j) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.0034841573002178834+0j) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-8.09163719850359e-07+0j) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.005733569747311876+0j) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.004684903388155191+0j) [Y0 Z1 Z2 X3 X6 Y7] +
(-0.00466862031877631+0j) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11] +
(7.189990974896306e-07+0j) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0051144738316603955+0j) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10] +
(-7.560692464353301e-07+0j) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12] +
(0.008125251921381029+0j) [Y0 Z1 Z2 X3 X8 Y9] +
(0.0017992194936630212+0j) [Y0 Z1 Z2 X3 X10 Y11] +
(5.471647744557047e-07+0j) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13] +
(-5.287660624617383e-07+0j) [Y0 Z1 Z2 X3 X11 Y12] +
(0.004575007626639207+0j) [Y0 Z1 Z2 X3 X12 Y13] +
(-0.004424855449441847+0j) [Y0 Z1 Z2 Y3 X4 X5] +
(-4.523389677425367e-07+0j) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0034841573002178834+0j) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.09163719850359e-07+0j) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.005733569747311876+0j) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004684903388155191+0j) [Y0 Z1 Z2 Y3 X6 X7] +
(0.00466862031877631+0j) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11] +
(-7.189990974896306e-07+0j) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0051144738316603955+0j) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10] +
(-7.560692464353301e-07+0j) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381029+0j) [Y0 Z1 Z2 Y3 X8 X9] +
(-0.0017992194936630212+0j) [Y0 Z1 Z2 Y3 X10 X11] +
(-5.471647744557047e-07+0j) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13] +
(-5.287660624617383e-07+0j) [Y0 Z1 Z2 Y3 Y11 Y12] +
(-0.004575007626639207+0j) [Y0 Z1 Z2 Y3 X12 X13] +
(-0.001028329237856275+0j) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.0026860409778066193+0j) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12] +
(3.202076880447949e-06+0j) [Y0 Z1 Z2 Z3 Y4] +
(0.0008533856254125472+0j) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7] +
(0.0007870896771024451+0j) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0008533856254125472+0j) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7] +
(0.0007870896771024451+0j) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.2919694863380308e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z5] +
(4.6849150951351447e-07+0j) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10] +
(0.0022009640695004637+0j) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.4445978542416875e-07+0j) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10] +
(0.001172634831644189+0j) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.8394209156697395e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z6] +
(4.0922506159816434e-07+0j) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11] +
(0.0023949726397980205+0j) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.0922506159816434e-07+0j) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11] +
(0.0023949726397980205+0j) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.236259961715071e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z8] +
(8.649310135731738e-08+0j) [Y0 Z1 Z2 Z3 Y4 Z9] +
(0.003989841456619313+0j) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12] +
(0.0013038004788126934+0j) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12] +
(-6.733197741794053e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z10] +
(0.002261966062482341+0j) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13] +
(0.002261966062482341+0j) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13] +
(-5.92745308233278e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z11] +
(1.239336321719257e-06+0j) [Y0 Z1 Z2 Z3 Y4 Z12] +
(9.306536651743931e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z13] +
(1.8394209156697398e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7] +
(0.00019400857029755713+0j) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0012223378081538318+0j) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(2.3713289481418972e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9] +
(-8.057446594612721e-08+0j) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11] +
(-0.0017278753941369718+0j) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13] +
(-0.000958165583669648+0j) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12] +
(3.0868265654486404e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13] +
(-1.8394209156697398e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7] +
(-0.00019400857029755713+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0012223378081538318+0j) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-2.3713289481418972e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9] +
(8.057446594612721e-08+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11] +
(0.0017278753941369718+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13] +
(-0.000958165583669648+0j) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12] +
(-3.0868265654486404e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13] +
(1.2004287493969867e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12] +
(0.04274327701378341+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6] +
(0.0005192743499487665+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(-1.8505641929368898e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005192743499487665+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(-1.8505641929368898e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0027790267990255306+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7] +
(0.004636976661182572+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8] +
(0.0012803060973496806+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9] +
(1.07172821813145e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12] +
(2.3120943051453685e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12] +
(0.0053799371558393904+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10] +
(7.246974425083315e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13] +
(7.246974425083315e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13] +
(0.005241535382803877+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11] +
(0.004311038507914323+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12] +
(0.001043524653490765+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13] +
(0.0033566705638328927+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9] +
(0.00013840177303551388+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11] +
(6.175246206951864e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(-4.997018421848276e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12] +
(0.0032675138544235576+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13] +
(-0.0033566705638328927+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9] +
(-0.00013840177303551388+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11] +
(-6.175246206951864e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(-4.997018421848276e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12] +
(-0.0032675138544235576+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13] +
(0.003876470899336945+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(-7.540341413482891e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.003876470899336945+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(-7.540341413482891e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(0.07165035181002839+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0021413612231015954+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(0.004220813970046477+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(0.0012366478019245426+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(0.002984166168121934+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-0.002984166168121934+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-1.3987009016643128e-05+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(8.949476486905753e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-6.876621658182731e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(-7.661347213055165e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(0.0015324835230730574+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10] +
(-2.904599884492375e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(0.005408954422410002+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10] +
(-1.0444941297975267e-06+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(0.00476727218827813+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10] +
(-8.10551503697011e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(0.005286546538226896+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10] +
(-9.956079229907e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0016095313817213858+0j) [Y0 Z1 Z2 Z3 Z4 Y6] +
(-7.141625221155571e-05+0j) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10] +
(-2.6667317550582486e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.002462917007133933+0j) [Y0 Z1 Z2 Z3 Z5 Y6] +
(0.0007156734248908893+0j) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10] +
(-3.0767325321484006e-07+0j) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(3.606071868267355e-07+0j) [Y0 Z1 Z2 Y4] +
(0.003961560792496537+0j) [Y0 Z1 Z2 Z4 Z5 Y6] +
(0.00018787053389551813+0j) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.6569309315286678e-07+0j) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.7379332625590598e-07+0j) [Y0 Z1 Z3 Y4] +
(0.0016676041811440625+0j) [Y0 Z1 Z3 Z4 Z5 Y6] +
(-0.0014528843214169026+0j) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(4.6704023903330105e-07+0j) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.10433064780651427+0j) [Y0 Y2] +
(3.1174479464120664e-06+0j) [Y0 Z2 Z3 Y4] +
(0.04587947078129816+0j) [Y0 Z2 Z3 Z4 Z5 Y6] +
(0.05859198873386204+0j) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-1.1463061452870514e-05+0j) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(12.412630742111777+0j) [Z0] +
(0.10433064780651427+0j) [Z0 X1 Z2 X3] +
(3.1174479464120664e-06+0j) [Z0 X1 Z2 Z3 Z4 X5] +
(0.04587947078129815+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7] +
(0.05859198873386204+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-1.1463061452870516e-05+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.10433064780651427+0j) [Z0 Y1 Z2 Y3] +
(3.1174479464120664e-06+0j) [Z0 Y1 Z2 Z3 Z4 Y5] +
(0.04587947078129815+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7] +
(0.05859198873386204+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.1463061452870516e-05+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.1861763734860507+0j) [Z0 Z1] +
(-8.337746755477111e-07+0j) [Z0 X2 Z3 X4] +
(-0.027115036845273204+0j) [Z0 X2 Z3 Z4 Z5 X6] +
(-0.0675238509921403+0j) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.401710973499001e-05+0j) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-8.337746755477111e-07+0j) [Z0 Y2 Z3 Y4] +
(-0.027115036845273204+0j) [Z0 Y2 Z3 Z4 Z5 Y6] +
(-0.0675238509921403+0j) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.401710973499001e-05+0j) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.1908508084963526e-06+0j) [Z0 X3 Z4 X5] +
(-0.03276765782329057+0j) [Z0 X3 Z4 Z5 Z6 X7] +
(-0.07635021950635017+0j) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.5809603692663747e-05+0j) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.1908508084963526e-06+0j) [Z0 Y3 Z4 Y5] +
(-0.03276765782329057+0j) [Z0 Y3 Z4 Z5 Z6 Y7] +
(-0.07635021950635017+0j) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.5809603692663747e-05+0j) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.099349243855689e-06+0j) [Z0 X4 Z5 X6] +
(-1.5316808794754993e-05+0j) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.08684737589863621+0j) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-3.099349243855689e-06+0j) [Z0 Y4 Z5 Y6] +
(-1.5316808794754993e-05+0j) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.08684737589863621+0j) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.19661770890342142+0j) [Z0 Z4] +
(-3.3440815567603924e-06+0j) [Z0 X5 Z6 X7] +
(-1.6103585305167606e-05+0j) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.09065144207036474+0j) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.3440815567603924e-06+0j) [Z0 Y5 Z6 Y7] +
(-1.6103585305167606e-05+0j) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.09065144207036474+0j) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.19936354537360823+0j) [Z0 Z5] +
(0.056084681246613644+0j) [Z0 X6 Z7 Z8 Z9 X10] +
(-6.6522096687127125e-06+0j) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.056084681246613644+0j) [Z0 Y6 Z7 Z8 Z9 Y10] +
(-6.6522096687127125e-06+0j) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.24164663936017194+0j) [Z0 Z6] +
(0.05600733087780772+0j) [Z0 X7 Z8 Z9 Z10 X11] +
(-6.481851833144925e-06+0j) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.05600733087780772+0j) [Z0 Y7 Z8 Z9 Z10 Y11] +
(-6.481851833144925e-06+0j) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.2485348337131425+0j) [Z0 Z7] +
(0.2723251830660568+0j) [Z0 Z8] +
(0.2788345442672341+0j) [Z0 Z9] +
(-2.1776646049405766e-06+0j) [Z0 X10 Z11 X12] +
(-2.1776646049405766e-06+0j) [Z0 Y10 Z11 Y12] +
(0.1929972393536426+0j) [Z0 Z10] +
(-1.6148794138026357e-06+0j) [Z0 X11 Z12 X13] +
(-1.6148794138026357e-06+0j) [Z0 Y11 Z12 Y13] +
(0.2007286646044179+0j) [Z0 Z11] +
(0.21102659849791522+0j) [Z0 Z12] +
(0.21631037498631817+0j) [Z0 Z13] +
(1.9332412772834116e-07+0j) [X1 X2 Y3 Y4] +
(0.002293956611352475+0j) [X1 X2 Y3 Z4 Z5 Y6] +
(0.0016407548553124208+0j) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-3.013471458804342e-07+0j) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004424855449441847+0j) [X1 X2 X4 X5] +
(-8.09163719850359e-07+0j) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.005733569747311876+0j) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-4.523389677425367e-07+0j) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.0034841573002178834+0j) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.00468490338815519+0j) [X1 X2 X6 X7] +
(0.0051144738316603955+0j) [X1 X2 X6 Z7 Z8 Z9 Z10 X11] +
(-7.560692464353301e-07+0j) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.00466862031877631+0j) [X1 X2 Y7 Z8 Z9 Y10] +
(-7.189990974896306e-07+0j) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381029+0j) [X1 X2 X8 X9] +
(-0.0017992194936630212+0j) [X1 X2 X10 X11] +
(-5.287660624617383e-07+0j) [X1 X2 X10 Z11 Z12 X13] +
(-5.471647744557047e-07+0j) [X1 X2 Y11 Y12] +
(-0.004575007626639208+0j) [X1 X2 X12 X13] +
(-1.9332412772834116e-07+0j) [X1 Y2 Y3 X4] +
(-0.002293956611352475+0j) [X1 Y2 Y3 Z4 Z5 X6] +
(-0.0016407548553124208+0j) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(3.013471458804342e-07+0j) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441847+0j) [X1 Y2 Y4 X5] +
(-8.09163719850359e-07+0j) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.005733569747311876+0j) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.523389677425367e-07+0j) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10] +
(0.0034841573002178834+0j) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.00468490338815519+0j) [X1 Y2 Y6 X7] +
(0.0051144738316603955+0j) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11] +
(-7.560692464353301e-07+0j) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.00466862031877631+0j) [X1 Y2 Y7 Z8 Z9 X10] +
(7.189990974896306e-07+0j) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381029+0j) [X1 Y2 Y8 X9] +
(-0.0017992194936630212+0j) [X1 Y2 Y10 X11] +
(-5.287660624617383e-07+0j) [X1 Y2 Y10 Z11 Z12 X13] +
(5.471647744557047e-07+0j) [X1 Y2 Y11 X12] +
(-0.004575007626639208+0j) [X1 Y2 Y12 X13] +
(0.1250703257977226+0j) [X1 Z2 X3] +
(-1.380778148193924e-07+0j) [X1 Z2 X3 X4 Z5 X6] +
(-3.3767393083651703e-07+0j) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0018638942824587411+0j) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.380778148193924e-07+0j) [X1 Z2 X3 Y4 Z5 Y6] +
(-3.3767393083651703e-07+0j) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0018638942824587411+0j) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0011059037691896955+0j) [X1 Z2 X3 Z4] +
(-1.551053917739348e-07+0j) [X1 Z2 X3 X5 Z6 X7] +
(-1.146837650686876e-06+0j) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.007597464029770616+0j) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.380778148193924e-07+0j) [X1 Z2 X3 Y5 Z6 Y7] +
(-7.900128985790538e-07+0j) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.005348051582676623+0j) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005530759218631541+0j) [X1 Z2 X3 Z5] +
(0.0005940221543005461+0j) [X1 Z2 X3 X6 Z7 Z8 Z9 X10] +
(-8.37977324328611e-08+0j) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0005940221543005461+0j) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10] +
(-8.37977324328611e-08+0j) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0033476175306662017+0j) [X1 Z2 X3 Z6] +
(0.0057084959859609406+0j) [X1 Z2 X3 X7 Z8 Z9 Z10 X11] +
(-8.352332102497864e-07+0j) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
1.9742253791047746e-08j [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005262642473076857+0j) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11] +
(-8.074305985408966e-07+0j) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.008032520918821392+0j) [X1 Z2 X3 Z7] +
(0.0029297686747510676+0j) [X1 Z2 X3 Z8] +
(0.011055020596132097+0j) [X1 Z2 X3 Z9] +
(-1.1076325599347869e-07+0j) [X1 Z2 X3 X10 Z11 X12] +
(-1.1076325599347869e-07+0j) [X1 Z2 X3 Y10 Z11 Y12] +
(0.0017560707018412598+0j) [X1 Z2 X3 Z10] +
(-6.418291574545231e-07+0j) [X1 Z2 X3 X11 Z12 X13] +
(-6.556281914498774e-07+0j) [X1 Z2 X3 Y11 Z12 Y13] +
(0.003555290195504281+0j) [X1 Z2 X3 Z11] +
(0.002326230623158096+0j) [X1 Z2 X3 Z12] +
(0.006901238249797304+0j) [X1 Z2 X3 Z13] +
(-3.5682475210782223e-07+0j) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.002249412447093991+0j) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.0474716554238044e-08+0j) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.00044585351288408716+0j) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11] +
(-1.9742253790470122e-08+0j) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0007870896771024451+0j) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10] +
(-1.8394209156697395e-07+0j) [X1 Z2 Z3 X4 X6 X7] +
(-0.0012223378081538318+0j) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.00019400857029755713+0j) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12] +
(-2.3713289481418972e-07+0j) [X1 Z2 Z3 X4 X8 X9] +
(8.057446594612721e-08+0j) [X1 Z2 Z3 X4 X10 X11] +
(-0.000958165583669648+0j) [X1 Z2 Z3 X4 X10 Z11 Z12 X13] +
(0.0017278753941369718+0j) [X1 Z2 Z3 X4 Y11 Y12] +
(-3.0868265654486404e-07+0j) [X1 Z2 Z3 X4 X12 X13] +
(0.0007870896771024451+0j) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10] +
(-1.8394209156697395e-07+0j) [X1 Z2 Z3 Y4 Y6 X7] +
(-0.0012223378081538318+0j) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.00019400857029755713+0j) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12] +
(-2.3713289481418972e-07+0j) [X1 Z2 Z3 Y4 Y8 X9] +
(8.057446594612721e-08+0j) [X1 Z2 Z3 Y4 Y10 X11] +
(-0.000958165583669648+0j) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13] +
(-0.0017278753941369718+0j) [X1 Z2 Z3 Y4 Y11 X12] +
(-3.0868265654486404e-07+0j) [X1 Z2 Z3 Y4 Y12 X13] +
(3.202076880447948e-06+0j) [X1 Z2 Z3 Z4 X5] +
(4.0922506159816434e-07+0j) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10] +
(0.0023949726397980205+0j) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.0922506159816434e-07+0j) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10] +
(0.0023949726397980205+0j) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.4445978542416875e-07+0j) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11] +
(0.001172634831644189+0j) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.6849150951351447e-07+0j) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11] +
(0.0022009640695004637+0j) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.8394209156697395e-07+0j) [X1 Z2 Z3 Z4 X5 Z7] +
(8.649310135731738e-08+0j) [X1 Z2 Z3 Z4 X5 Z8] +
(3.236259961715071e-07+0j) [X1 Z2 Z3 Z4 X5 Z9] +
(0.002261966062482341+0j) [X1 Z2 Z3 Z4 X5 X10 Z11 X12] +
(0.002261966062482341+0j) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12] +
(-5.92745308233278e-07+0j) [X1 Z2 Z3 Z4 X5 Z10] +
(0.0013038004788126934+0j) [X1 Z2 Z3 Z4 X5 X11 Z12 X13] +
(0.003989841456619313+0j) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13] +
(-6.733197741794053e-07+0j) [X1 Z2 Z3 Z4 X5 Z11] +
(9.306536651743931e-07+0j) [X1 Z2 Z3 Z4 X5 Z12] +
(1.239336321719257e-06+0j) [X1 Z2 Z3 Z4 X5 Z13] +
(-0.001028329237856275+0j) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0026860409778066193+0j) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13] +
(-0.0005192743499487665+0j) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10] +
(1.8505641929368898e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.0033566705638328927+0j) [X1 Z2 Z3 Z4 Z5 X6 X8 X9] +
(-0.00013840177303551388+0j) [X1 Z2 Z3 Z4 Z5 X6 X10 X11] +
(-4.997018421848276e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13] +
(-6.175246206951864e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12] +
(-0.0032675138544235576+0j) [X1 Z2 Z3 Z4 Z5 X6 X12 X13] +
(0.0005192743499487665+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10] +
(-1.8505641929368898e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.0033566705638328927+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9] +
(-0.00013840177303551388+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11] +
(-4.997018421848276e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13] +
(6.175246206951864e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12] +
(-0.0032675138544235576+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13] +
(0.04274327701378341+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7] +
(0.0012803060973496806+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8] +
(0.004636976661182572+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9] +
(7.246974425083315e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12] +
(7.246974425083315e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12] +
(0.005241535382803877+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10] +
(2.3120943051453685e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13] +
(1.07172821813145e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13] +
(0.0053799371558393904+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11] +
(0.001043524653490765+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12] +
(0.004311038507914323+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13] +
(1.2004287493969867e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13] +
(-0.003876470899336945+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10] +
(7.540341413482891e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(0.003876470899336945+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10] +
(-7.540341413482891e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(-0.0029841661681219342+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-0.0029841661681219342+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(0.07165035181002839+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0012366478019245426+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(0.004220813970046477+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(-1.3987009016643125e-05+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(8.949476486905753e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(-7.661347213055165e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(-0.0021413612231015954+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11] +
(-6.876621658182731e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(0.005408954422410002+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11] +
(-1.0444941297975267e-06+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(0.0015324835230730574+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11] +
(-2.904599884492375e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(0.005286546538226896+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11] +
(-9.956079229907e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0027790267990255306+0j) [X1 Z2 Z3 Z4 Z5 X7] +
(0.00476727218827813+0j) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11] +
(-8.10551503697011e-07+0j) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.002462917007133933+0j) [X1 Z2 Z3 Z4 Z6 X7] +
(0.0007156734248908893+0j) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11] +
(-3.0767325321484006e-07+0j) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2919694863380308e-07+0j) [X1 Z2 Z3 X5] +
(0.0016095313817213858+0j) [X1 Z2 Z3 Z5 Z6 X7] +
(-7.141625221155571e-05+0j) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-2.6667317550582486e-07+0j) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.7379332625590598e-07+0j) [X1 Z2 Z4 X5] +
(0.0016676041811440625+0j) [X1 Z2 Z4 Z5 Z6 X7] +
(-0.0014528843214169026+0j) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(4.6704023903330105e-07+0j) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0032769719312316786+0j) [X1 X3] +
(3.606071868267355e-07+0j) [X1 Z3 Z4 X5] +
(0.003961560792496537+0j) [X1 Z3 Z4 Z5 Z6 X7] +
(0.00018787053389551813+0j) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.6569309315286678e-07+0j) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.9332412772834116e-07+0j) [Y1 X2 X3 Y4] +
(-0.002293956611352475+0j) [Y1 X2 X3 Z4 Z5 Y6] +
(-0.0016407548553124208+0j) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(3.013471458804342e-07+0j) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004424855449441847+0j) [Y1 X2 X4 Y5] +
(-8.09163719850359e-07+0j) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.005733569747311876+0j) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(4.523389677425367e-07+0j) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10] +
(0.0034841573002178834+0j) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.00468490338815519+0j) [Y1 X2 X6 Y7] +
(0.0051144738316603955+0j) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11] +
(-7.560692464353301e-07+0j) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.00466862031877631+0j) [Y1 X2 X7 Z8 Z9 Y10] +
(7.189990974896306e-07+0j) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381029+0j) [Y1 X2 X8 Y9] +
(-0.0017992194936630212+0j) [Y1 X2 X10 Y11] +
(-5.287660624617383e-07+0j) [Y1 X2 X10 Z11 Z12 Y13] +
(5.471647744557047e-07+0j) [Y1 X2 X11 Y12] +
(-0.004575007626639208+0j) [Y1 X2 X12 Y13] +
(1.9332412772834116e-07+0j) [Y1 Y2 X3 X4] +
(0.002293956611352475+0j) [Y1 Y2 X3 Z4 Z5 X6] +
(0.0016407548553124208+0j) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-3.013471458804342e-07+0j) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441847+0j) [Y1 Y2 Y4 Y5] +
(-8.09163719850359e-07+0j) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.005733569747311876+0j) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-4.523389677425367e-07+0j) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10] +
(-0.0034841573002178834+0j) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.00468490338815519+0j) [Y1 Y2 Y6 Y7] +
(0.0051144738316603955+0j) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11] +
(-7.560692464353301e-07+0j) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.00466862031877631+0j) [Y1 Y2 X7 Z8 Z9 X10] +
(-7.189990974896306e-07+0j) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381029+0j) [Y1 Y2 Y8 Y9] +
(-0.0017992194936630212+0j) [Y1 Y2 Y10 Y11] +
(-5.287660624617383e-07+0j) [Y1 Y2 Y10 Z11 Z12 Y13] +
(-5.471647744557047e-07+0j) [Y1 Y2 X11 X12] +
(-0.004575007626639208+0j) [Y1 Y2 Y12 Y13] +
(-3.5682475210782223e-07+0j) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.002249412447093991+0j) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.00044585351288408716+0j) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11] +
(-1.9742253790470122e-08+0j) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.0474716554238044e-08+0j) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.1250703257977226+0j) [Y1 Z2 Y3] +
(-1.380778148193924e-07+0j) [Y1 Z2 Y3 X4 Z5 X6] +
(-3.3767393083651703e-07+0j) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0018638942824587411+0j) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.380778148193924e-07+0j) [Y1 Z2 Y3 Y4 Z5 Y6] +
(-3.3767393083651703e-07+0j) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0018638942824587411+0j) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0011059037691896955+0j) [Y1 Z2 Y3 Z4] +
(-1.380778148193924e-07+0j) [Y1 Z2 Y3 X5 Z6 X7] +
(-7.900128985790538e-07+0j) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.005348051582676623+0j) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.551053917739348e-07+0j) [Y1 Z2 Y3 Y5 Z6 Y7] +
(-1.146837650686876e-06+0j) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.007597464029770616+0j) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005530759218631541+0j) [Y1 Z2 Y3 Z5] +
(0.0005940221543005461+0j) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10] +
(-8.37977324328611e-08+0j) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0005940221543005461+0j) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10] +
(-8.37977324328611e-08+0j) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0033476175306662017+0j) [Y1 Z2 Y3 Z6] +
(0.005262642473076857+0j) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11] +
(-8.074305985408966e-07+0j) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0057084959859609406+0j) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11] +
-1.9742253791047746e-08j [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.352332102497864e-07+0j) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.008032520918821392+0j) [Y1 Z2 Y3 Z7] +
(0.0029297686747510676+0j) [Y1 Z2 Y3 Z8] +
(0.011055020596132097+0j) [Y1 Z2 Y3 Z9] +
(-1.1076325599347869e-07+0j) [Y1 Z2 Y3 X10 Z11 X12] +
(-1.1076325599347869e-07+0j) [Y1 Z2 Y3 Y10 Z11 Y12] +
(0.0017560707018412598+0j) [Y1 Z2 Y3 Z10] +
(-6.556281914498774e-07+0j) [Y1 Z2 Y3 X11 Z12 X13] +
(-6.418291574545231e-07+0j) [Y1 Z2 Y3 Y11 Z12 Y13] +
(0.003555290195504281+0j) [Y1 Z2 Y3 Z11] +
(0.002326230623158096+0j) [Y1 Z2 Y3 Z12] +
(0.006901238249797304+0j) [Y1 Z2 Y3 Z13] +
(0.0007870896771024451+0j) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10] +
(-1.8394209156697395e-07+0j) [Y1 Z2 Z3 X4 X6 Y7] +
(-0.0012223378081538318+0j) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.00019400857029755713+0j) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12] +
(-2.3713289481418972e-07+0j) [Y1 Z2 Z3 X4 X8 Y9] +
(8.057446594612721e-08+0j) [Y1 Z2 Z3 X4 X10 Y11] +
(-0.000958165583669648+0j) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13] +
(-0.0017278753941369718+0j) [Y1 Z2 Z3 X4 X11 Y12] +
(-3.0868265654486404e-07+0j) [Y1 Z2 Z3 X4 X12 Y13] +
(-0.0007870896771024451+0j) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10] +
(-1.8394209156697395e-07+0j) [Y1 Z2 Z3 Y4 Y6 Y7] +
(-0.0012223378081538318+0j) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.00019400857029755713+0j) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12] +
(-2.3713289481418972e-07+0j) [Y1 Z2 Z3 Y4 Y8 Y9] +
(8.057446594612721e-08+0j) [Y1 Z2 Z3 Y4 Y10 Y11] +
(-0.000958165583669648+0j) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13] +
(0.0017278753941369718+0j) [Y1 Z2 Z3 Y4 X11 X12] +
(-3.0868265654486404e-07+0j) [Y1 Z2 Z3 Y4 Y12 Y13] +
(-0.001028329237856275+0j) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0026860409778066193+0j) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13] +
(3.202076880447948e-06+0j) [Y1 Z2 Z3 Z4 Y5] +
(4.0922506159816434e-07+0j) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10] +
(0.0023949726397980205+0j) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.0922506159816434e-07+0j) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10] +
(0.0023949726397980205+0j) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.6849150951351447e-07+0j) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11] +
(0.0022009640695004637+0j) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.4445978542416875e-07+0j) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11] +
(0.001172634831644189+0j) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.8394209156697395e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z7] +
(8.649310135731738e-08+0j) [Y1 Z2 Z3 Z4 Y5 Z8] +
(3.236259961715071e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z9] +
(0.002261966062482341+0j) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12] +
(0.002261966062482341+0j) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12] +
(-5.92745308233278e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z10] +
(0.003989841456619313+0j) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13] +
(0.0013038004788126934+0j) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13] +
(-6.733197741794053e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z11] +
(9.306536651743931e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z12] +
(1.239336321719257e-06+0j) [Y1 Z2 Z3 Z4 Y5 Z13] +
(0.0005192743499487665+0j) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10] +
(-1.8505641929368898e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.0033566705638328927+0j) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9] +
(-0.00013840177303551388+0j) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11] +
(-4.997018421848276e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13] +
(6.175246206951864e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12] +
(-0.0032675138544235576+0j) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13] +
(-0.0005192743499487665+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10] +
(1.8505641929368898e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-0.0033566705638328927+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9] +
(-0.00013840177303551388+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11] +
(-4.997018421848276e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13] +
(-6.175246206951864e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12] +
(-0.0032675138544235576+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13] +
(1.2004287493969867e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13] +
(0.04274327701378341+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7] +
(0.0012803060973496806+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8] +
(0.004636976661182572+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9] +
(7.246974425083315e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12] +
(7.246974425083315e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12] +
(0.005241535382803877+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10] +
(1.07172821813145e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13] +
(2.3120943051453685e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13] +
(0.0053799371558393904+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11] +
(0.001043524653490765+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12] +
(0.004311038507914323+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13] +
(0.003876470899336945+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10] +
(-7.540341413482891e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-0.003876470899336945+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10] +
(7.540341413482891e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-0.0029841661681219342+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(-0.0029841661681219342+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(0.07165035181002839+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.0012366478019245426+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(0.004220813970046477+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(-1.3987009016643125e-05+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(8.949476486905753e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(-7.661347213055165e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(-0.0021413612231015954+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11] +
(-6.876621658182731e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(0.005408954422410002+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11] +
(-1.0444941297975267e-06+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(0.0015324835230730574+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11] +
(-2.904599884492375e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(0.005286546538226896+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11] +
(-9.956079229907e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0027790267990255306+0j) [Y1 Z2 Z3 Z4 Z5 Y7] +
(0.00476727218827813+0j) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11] +
(-8.10551503697011e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.002462917007133933+0j) [Y1 Z2 Z3 Z4 Z6 Y7] +
(0.0007156734248908893+0j) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11] +
(-3.0767325321484006e-07+0j) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.2919694863380308e-07+0j) [Y1 Z2 Z3 Y5] +
(0.0016095313817213858+0j) [Y1 Z2 Z3 Z5 Z6 Y7] +
(-7.141625221155571e-05+0j) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-2.6667317550582486e-07+0j) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.7379332625590598e-07+0j) [Y1 Z2 Z4 Y5] +
(0.0016676041811440625+0j) [Y1 Z2 Z4 Z5 Z6 Y7] +
(-0.0014528843214169026+0j) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(4.6704023903330105e-07+0j) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0032769719312316786+0j) [Y1 Y3] +
(3.606071868267355e-07+0j) [Y1 Z3 Z4 Y5] +
(0.003961560792496537+0j) [Y1 Z3 Z4 Z5 Z6 Y7] +
(0.00018787053389551813+0j) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.6569309315286678e-07+0j) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(12.412630742111777+0j) [Z1] +
(-1.1908508084963526e-06+0j) [Z1 X2 Z3 X4] +
(-0.03276765782329057+0j) [Z1 X2 Z3 Z4 Z5 X6] +
(-0.07635021950635017+0j) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.5809603692663747e-05+0j) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.1908508084963526e-06+0j) [Z1 Y2 Z3 Y4] +
(-0.03276765782329057+0j) [Z1 Y2 Z3 Z4 Z5 Y6] +
(-0.07635021950635017+0j) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.5809603692663747e-05+0j) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-8.337746755477111e-07+0j) [Z1 X3 Z4 X5] +
(-0.027115036845273204+0j) [Z1 X3 Z4 Z5 Z6 X7] +
(-0.0675238509921403+0j) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.401710973499001e-05+0j) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.337746755477111e-07+0j) [Z1 Y3 Z4 Y5] +
(-0.027115036845273204+0j) [Z1 Y3 Z4 Z5 Z6 Y7] +
(-0.0675238509921403+0j) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.401710973499001e-05+0j) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.3440815567603924e-06+0j) [Z1 X4 Z5 X6] +
(-1.6103585305167606e-05+0j) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.09065144207036474+0j) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-3.3440815567603924e-06+0j) [Z1 Y4 Z5 Y6] +
(-1.6103585305167606e-05+0j) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.09065144207036474+0j) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.19936354537360823+0j) [Z1 Z4] +
(-3.099349243855689e-06+0j) [Z1 X5 Z6 X7] +
(-1.5316808794754993e-05+0j) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.08684737589863621+0j) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.099349243855689e-06+0j) [Z1 Y5 Z6 Y7] +
(-1.5316808794754993e-05+0j) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.08684737589863621+0j) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.19661770890342142+0j) [Z1 Z5] +
(0.05600733087780772+0j) [Z1 X6 Z7 Z8 Z9 X10] +
(-6.481851833144925e-06+0j) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.05600733087780772+0j) [Z1 Y6 Z7 Z8 Z9 Y10] +
(-6.481851833144925e-06+0j) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.2485348337131425+0j) [Z1 Z6] +
(0.056084681246613644+0j) [Z1 X7 Z8 Z9 Z10 X11] +
(-6.6522096687127125e-06+0j) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.056084681246613644+0j) [Z1 Y7 Z8 Z9 Z10 Y11] +
(-6.6522096687127125e-06+0j) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.24164663936017194+0j) [Z1 Z7] +
(0.2788345442672341+0j) [Z1 Z8] +
(0.2723251830660568+0j) [Z1 Z9] +
(-1.6148794138026357e-06+0j) [Z1 X10 Z11 X12] +
(-1.6148794138026357e-06+0j) [Z1 Y10 Z11 Y12] +
(0.2007286646044179+0j) [Z1 Z10] +
(-2.1776646049405766e-06+0j) [Z1 X11 Z12 X13] +
(-2.1776646049405766e-06+0j) [Z1 Y11 Z12 Y13] +
(0.1929972393536426+0j) [Z1 Z11] +
(0.21631037498631817+0j) [Z1 Z12] +
(0.21102659849791522+0j) [Z1 Z13] +
(-0.03583956795335344+0j) [X2 X3 Y4 Y5] +
(-2.1990516183690528e-07+0j) [X2 X3 Y4 Z5 Z6 Y7] +
(-2.3609563202538108e-06+0j) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.010311482489831793+0j) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.1990516183690528e-07+0j) [X2 X3 X5 X6] +
(-2.3609563202538108e-06+0j) [X2 X3 X5 Z6 Z7 Z8 Z9 X10] +
(-0.010311482489831793+0j) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.031143817988967086+0j) [X2 X3 Y6 Y7] +
(0.005368659358109539+0j) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11] +
(9.209350657254084e-08+0j) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005368659358109539+0j) [X2 X3 X7 Z8 Z9 X10] +
(9.209350657254084e-08+0j) [X2 X3 X7 Z8 Z9 Z10 Z11 X12] +
(-0.036194123559042585+0j) [X2 X3 Y8 Y9] +
(-0.025384657508457423+0j) [X2 X3 Y10 Y11] +
(2.1726691014844186e-06+0j) [X2 X3 Y10 Z11 Z12 Y13] +
(2.1726691014844186e-06+0j) [X2 X3 X11 X12] +
(-0.015577208063976448+0j) [X2 X3 Y12 Y13] +
(0.03583956795335344+0j) [X2 Y3 Y4 X5] +
(2.1990516183690528e-07+0j) [X2 Y3 Y4 Z5 Z6 X7] +
(2.3609563202538108e-06+0j) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.010311482489831793+0j) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.1990516183690528e-07+0j) [X2 Y3 Y5 X6] +
(-2.3609563202538108e-06+0j) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.010311482489831793+0j) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.031143817988967086+0j) [X2 Y3 Y6 X7] +
(-0.005368659358109539+0j) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11] +
(-9.209350657254084e-08+0j) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005368659358109539+0j) [X2 Y3 Y7 Z8 Z9 X10] +
(9.209350657254084e-08+0j) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12] +
(0.036194123559042585+0j) [X2 Y3 Y8 X9] +
(0.025384657508457423+0j) [X2 Y3 Y10 X11] +
(-2.1726691014844186e-06+0j) [X2 Y3 Y10 Z11 Z12 X13] +
(2.1726691014844186e-06+0j) [X2 Y3 Y11 X12] +
(0.015577208063976448+0j) [X2 Y3 Y12 X13] +
(-3.887051673669065e-06+0j) [X2 Z3 X4] +
(-0.005143391768825153+0j) [X2 Z3 X4 X5 Z6 X7] +
(-0.009841749246962623+0j) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.988511706487063e-06+0j) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.005143391768825153+0j) [X2 Z3 X4 Y5 Z6 Y7] +
(-0.009841749246962623+0j) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.988511706487063e-06+0j) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.764994118309244e-07+0j) [X2 Z3 X4 Z5] +
(1.689348951418275e-06+0j) [X2 Z3 X4 X6 Z7 Z8 Z9 X10] +
(0.010757563953908957+0j) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.537178096553528e-08+0j) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10] +
(4.205548411219395e-05+0j) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.593534391229091e-07+0j) [X2 Z3 X4 Z6] +
(3.2118420190033097e-06+0j) [X2 Z3 X4 X7 Z8 Z9 Z10 X11] +
(0.019299560579363786+0j) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.2118420190033097e-06+0j) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11] +
(0.019299560579363786+0j) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.195489010083836e-06+0j) [X2 Z3 X4 Z7] +
(2.1868423776798165e-07+0j) [X2 Z3 X4 Z8] +
(-5.770052995666779e-07+0j) [X2 Z3 X4 Z9] +
(0.015588250102380172+0j) [X2 Z3 X4 X10 Z11 X12] +
(0.005324835234221677+0j) [X2 Z3 X4 Y10 Z11 Y12] +
(-3.1586564318930577e-06+0j) [X2 Z3 X4 Z10] +
(0.024353077678068928+0j) [X2 Z3 X4 X11 Z12 X13] +
(0.024353077678068928+0j) [X2 Z3 X4 Y11 Z12 Y13] +
(-7.80170750019265e-06+0j) [X2 Z3 X4 Z11] +
(3.5390541844206886e-06+0j) [X2 Z3 X4 Z12] +
(8.814937306443681e-06+0j) [X2 Z3 X4 Z13] +
(1.6288532434660635e-06+0j) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10] +
(0.010715508469796763+0j) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.010263414868158495+0j) [X2 Z3 Y4 Y10 Z11 X12] +
(-1.454842449206745e-06+0j) [X2 Z3 Z4 X5 Y6 Y7] +
(-3.151346311051097e-06+0j) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11] +
(-0.019257505095251592+0j) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.522493067585034e-06+0j) [X2 Z3 Z4 X5 X7 Z8 Z9 X10] +
(-0.008541996625454832+0j) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-7.956895373346593e-07+0j) [X2 Z3 Z4 X5 Y8 Y9] +
(-4.643051068299593e-06+0j) [X2 Z3 Z4 X5 Y10 Y11] +
(-0.019028242443847244+0j) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13] +
(-0.008764827575688751+0j) [X2 Z3 Z4 X5 X11 X12] +
(5.275883122022994e-06+0j) [X2 Z3 Z4 X5 Y12 Y13] +
(1.454842449206745e-06+0j) [X2 Z3 Z4 Y5 Y6 X7] +
(3.151346311051097e-06+0j) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11] +
(0.019257505095251592+0j) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.522493067585034e-06+0j) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10] +
(-0.008541996625454832+0j) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(7.956895373346593e-07+0j) [X2 Z3 Z4 Y5 Y8 X9] +
(4.643051068299593e-06+0j) [X2 Z3 Z4 Y5 Y10 X11] +
(0.019028242443847244+0j) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13] +
(-0.008764827575688751+0j) [X2 Z3 Z4 Y5 Y11 X12] +
(-5.275883122022994e-06+0j) [X2 Z3 Z4 Y5 Y12 X13] +
(-0.12133276911042346+0j) [X2 Z3 Z4 Z5 X6] +
(-0.008469978791023972+0j) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(2.686381543319587e-07+0j) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.008469978791023972+0j) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(2.686381543319587e-07+0j) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.025996177598021156+0j) [X2 Z3 Z4 Z5 X6 Z7] +
(-0.005805188989826925+0j) [X2 Z3 Z4 Z5 X6 Z8] +
(-0.017561202409646183+0j) [X2 Z3 Z4 Z5 X6 Z9] +
(-7.988770289333041e-07+0j) [X2 Z3 Z4 Z5 X6 X10 Z11 X12] +
(-3.427323108739172e-07+0j) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12] +
(-0.0008145313270956959+0j) [X2 Z3 Z4 Z5 X6 Z10] +
(2.745518400247564e-06+0j) [X2 Z3 Z4 Z5 X6 X11 Z12 X13] +
(2.745518400247564e-06+0j) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13] +
(0.014411099430130869+0j) [X2 Z3 Z4 Z5 X6 Z11] +
(0.0006650070219498878+0j) [X2 Z3 Z4 Z5 X6 Z12] +
(-0.0034937903598901733+0j) [X2 Z3 Z4 Z5 X6 Z13] +
(-4.56144718059387e-07+0j) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12] +
(-0.011756013419819258+0j) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9] +
(0.015225630757226563+0j) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11] +
(-3.0882507111214816e-06+0j) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(-3.544395429180869e-06+0j) [X2 Z3 Z4 Z5 Z6 X7 X11 X12] +
(-0.004158797381840062+0j) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13] +
(0.011756013419819258+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9] +
(-0.015225630757226563+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11] +
(3.0882507111214816e-06+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(-3.544395429180869e-06+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12] +
(0.004158797381840062+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13] +
(0.014603704729162096+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(-2.874299071311966e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.014603704729162096+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(-2.874299071311966e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(-0.28164257767022904+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-1.300294656192336e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(-1.300294656192336e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(-0.024282117354693072+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(-0.01953805031131475+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(-0.017091553155898904+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(0.0024464971554158488+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(-0.0024464971554158488+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(5.7759505271129055e-05+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(2.883676576005985e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(5.146496327371224e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(3.846201671178887e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(-0.03935916802205308+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10] +
(7.979825793227384e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(-0.02475546329289098+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10] +
(5.10552672191542e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(-0.021433810721600863+0j) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10] +
(5.159350501931196e-06+0j) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(-0.029903789512624835+0j) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10] +
(5.427988656263154e-06+0j) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0016638798784908144+0j) [X2 Z3 Z4 X6] +
(-0.01888903030494289+0j) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10] +
(2.94735601151092e-06+0j) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.0034795118903343373+0j) [X2 Z3 Z5 X6] +
(-0.02873077955190552+0j) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10] +
(5.9358677179979826e-06+0j) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.6021167407470289e-06+0j) [X2 X4] +
(0.0004956762314916132+0j) [X2 Z4 Z5 X6] +
(-0.03560837898831257+0j) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(7.253273347984072e-06+0j) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.03583956795335344+0j) [Y2 X3 X4 Y5] +
(2.1990516183690528e-07+0j) [Y2 X3 X4 Z5 Z6 Y7] +
(2.3609563202538108e-06+0j) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.010311482489831793+0j) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.1990516183690528e-07+0j) [Y2 X3 X5 Y6] +
(-2.3609563202538108e-06+0j) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.010311482489831793+0j) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.031143817988967086+0j) [Y2 X3 X6 Y7] +
(-0.005368659358109539+0j) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11] +
(-9.209350657254084e-08+0j) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005368659358109539+0j) [Y2 X3 X7 Z8 Z9 Y10] +
(9.209350657254084e-08+0j) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12] +
(0.036194123559042585+0j) [Y2 X3 X8 Y9] +
(0.025384657508457423+0j) [Y2 X3 X10 Y11] +
(-2.1726691014844186e-06+0j) [Y2 X3 X10 Z11 Z12 Y13] +
(2.1726691014844186e-06+0j) [Y2 X3 X11 Y12] +
(0.015577208063976448+0j) [Y2 X3 X12 Y13] +
(-0.03583956795335344+0j) [Y2 Y3 X4 X5] +
(-2.1990516183690528e-07+0j) [Y2 Y3 X4 Z5 Z6 X7] +
(-2.3609563202538108e-06+0j) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.010311482489831793+0j) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.1990516183690528e-07+0j) [Y2 Y3 Y5 Y6] +
(-2.3609563202538108e-06+0j) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.010311482489831793+0j) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.031143817988967086+0j) [Y2 Y3 X6 X7] +
(0.005368659358109539+0j) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11] +
(9.209350657254084e-08+0j) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005368659358109539+0j) [Y2 Y3 Y7 Z8 Z9 Y10] +
(9.209350657254084e-08+0j) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.036194123559042585+0j) [Y2 Y3 X8 X9] +
(-0.025384657508457423+0j) [Y2 Y3 X10 X11] +
(2.1726691014844186e-06+0j) [Y2 Y3 X10 Z11 Z12 X13] +
(2.1726691014844186e-06+0j) [Y2 Y3 Y11 Y12] +
(-0.015577208063976448+0j) [Y2 Y3 X12 X13] +
(1.6288532434660635e-06+0j) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10] +
(0.010715508469796763+0j) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.010263414868158495+0j) [Y2 Z3 X4 X10 Z11 Y12] +
(-3.887051673669065e-06+0j) [Y2 Z3 Y4] +
(-0.005143391768825153+0j) [Y2 Z3 Y4 X5 Z6 X7] +
(-0.009841749246962623+0j) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.988511706487063e-06+0j) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.005143391768825153+0j) [Y2 Z3 Y4 Y5 Z6 Y7] +
(-0.009841749246962623+0j) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.988511706487063e-06+0j) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.764994118309244e-07+0j) [Y2 Z3 Y4 Z5] +
(4.537178096553528e-08+0j) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10] +
(4.205548411219395e-05+0j) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.689348951418275e-06+0j) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10] +
(0.010757563953908957+0j) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.593534391229091e-07+0j) [Y2 Z3 Y4 Z6] +
(3.2118420190033097e-06+0j) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11] +
(0.019299560579363786+0j) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.2118420190033097e-06+0j) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11] +
(0.019299560579363786+0j) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.195489010083836e-06+0j) [Y2 Z3 Y4 Z7] +
(2.1868423776798165e-07+0j) [Y2 Z3 Y4 Z8] +
(-5.770052995666779e-07+0j) [Y2 Z3 Y4 Z9] +
(0.005324835234221677+0j) [Y2 Z3 Y4 X10 Z11 X12] +
(0.015588250102380172+0j) [Y2 Z3 Y4 Y10 Z11 Y12] +
(-3.1586564318930577e-06+0j) [Y2 Z3 Y4 Z10] +
(0.024353077678068928+0j) [Y2 Z3 Y4 X11 Z12 X13] +
(0.024353077678068928+0j) [Y2 Z3 Y4 Y11 Z12 Y13] +
(-7.80170750019265e-06+0j) [Y2 Z3 Y4 Z11] +
(3.5390541844206886e-06+0j) [Y2 Z3 Y4 Z12] +
(8.814937306443681e-06+0j) [Y2 Z3 Y4 Z13] +
(1.454842449206745e-06+0j) [Y2 Z3 Z4 X5 X6 Y7] +
(3.151346311051097e-06+0j) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11] +
(0.019257505095251592+0j) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.522493067585034e-06+0j) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10] +
(-0.008541996625454832+0j) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(7.956895373346593e-07+0j) [Y2 Z3 Z4 X5 X8 Y9] +
(4.643051068299593e-06+0j) [Y2 Z3 Z4 X5 X10 Y11] +
(0.019028242443847244+0j) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13] +
(-0.008764827575688751+0j) [Y2 Z3 Z4 X5 X11 Y12] +
(-5.275883122022994e-06+0j) [Y2 Z3 Z4 X5 X12 Y13] +
(-1.454842449206745e-06+0j) [Y2 Z3 Z4 Y5 X6 X7] +
(-3.151346311051097e-06+0j) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11] +
(-0.019257505095251592+0j) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.522493067585034e-06+0j) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10] +
(-0.008541996625454832+0j) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895373346593e-07+0j) [Y2 Z3 Z4 Y5 X8 X9] +
(-4.643051068299593e-06+0j) [Y2 Z3 Z4 Y5 X10 X11] +
(-0.019028242443847244+0j) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13] +
(-0.008764827575688751+0j) [Y2 Z3 Z4 Y5 Y11 Y12] +
(5.275883122022994e-06+0j) [Y2 Z3 Z4 Y5 X12 X13] +
(-4.56144718059387e-07+0j) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12] +
(-0.12133276911042346+0j) [Y2 Z3 Z4 Z5 Y6] +
(-0.008469978791023972+0j) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(2.686381543319587e-07+0j) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.008469978791023972+0j) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(2.686381543319587e-07+0j) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.025996177598021156+0j) [Y2 Z3 Z4 Z5 Y6 Z7] +
(-0.005805188989826925+0j) [Y2 Z3 Z4 Z5 Y6 Z8] +
(-0.017561202409646183+0j) [Y2 Z3 Z4 Z5 Y6 Z9] +
(-3.427323108739172e-07+0j) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12] +
(-7.988770289333041e-07+0j) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12] +
(-0.0008145313270956959+0j) [Y2 Z3 Z4 Z5 Y6 Z10] +
(2.745518400247564e-06+0j) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13] +
(2.745518400247564e-06+0j) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13] +
(0.014411099430130869+0j) [Y2 Z3 Z4 Z5 Y6 Z11] +
(0.0006650070219498878+0j) [Y2 Z3 Z4 Z5 Y6 Z12] +
(-0.0034937903598901733+0j) [Y2 Z3 Z4 Z5 Y6 Z13] +
(0.011756013419819258+0j) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9] +
(-0.015225630757226563+0j) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11] +
(3.0882507111214816e-06+0j) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(-3.544395429180869e-06+0j) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12] +
(0.004158797381840062+0j) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13] +
(-0.011756013419819258+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9] +
(0.015225630757226563+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11] +
(-3.0882507111214816e-06+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(-3.544395429180869e-06+0j) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12] +
(-0.004158797381840062+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13] +
(0.014603704729162096+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(-2.874299071311966e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.014603704729162096+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(-2.874299071311966e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(-0.28164257767022904+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-1.300294656192336e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(-1.300294656192336e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(-0.024282117354693072+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(-0.01953805031131475+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(-0.017091553155898904+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(-0.0024464971554158488+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(0.0024464971554158488+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(5.7759505271129055e-05+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.883676576005985e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(5.146496327371224e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(3.846201671178887e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(-0.03935916802205308+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10] +
(7.979825793227384e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(-0.02475546329289098+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10] +
(5.10552672191542e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(-0.021433810721600863+0j) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10] +
(5.159350501931196e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(-0.029903789512624835+0j) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10] +
(5.427988656263154e-06+0j) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0016638798784908144+0j) [Y2 Z3 Z4 Y6] +
(-0.01888903030494289+0j) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10] +
(2.94735601151092e-06+0j) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.0034795118903343373+0j) [Y2 Z3 Z5 Y6] +
(-0.02873077955190552+0j) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10] +
(5.9358677179979826e-06+0j) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.6021167407470289e-06+0j) [Y2 Y4] +
(0.0004956762314916132+0j) [Y2 Z4 Z5 Y6] +
(-0.03560837898831257+0j) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(7.253273347984072e-06+0j) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.653894222683171+0j) [Z2] +
(1.6021167407470289e-06+0j) [Z2 X3 Z4 X5] +
(0.0004956762314916131+0j) [Z2 X3 Z4 Z5 Z6 X7] +
(-0.03560837898831257+0j) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(7.253273347984072e-06+0j) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.6021167407470289e-06+0j) [Z2 Y3 Z4 Y5] +
(0.0004956762314916131+0j) [Z2 Y3 Z4 Z5 Z6 Y7] +
(-0.03560837898831257+0j) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(7.253273347984072e-06+0j) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.18189085790751341+0j) [Z2 Z3] +
(-9.509249752659195e-07+0j) [Z2 X4 Z5 X6] +
(-4.72884314709957e-06+0j) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.024591860883829912+0j) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-9.509249752659195e-07+0j) [Z2 Y4 Z5 Y6] +
(-4.72884314709957e-06+0j) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.024591860883829912+0j) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.12495807739503205+0j) [Z2 Z4] +
(-1.1708301371028247e-06+0j) [Z2 X5 Z6 X7] +
(-7.0897994673533815e-06+0j) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.03490334337366171+0j) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.1708301371028247e-06+0j) [Z2 Y5 Z6 Y7] +
(-7.0897994673533815e-06+0j) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.03490334337366171+0j) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.16079764534838548+0j) [Z2 Z5] +
(0.019020423173039955+0j) [Z2 X6 Z7 Z8 Z9 X10] +
(-2.1032156043935615e-06+0j) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.019020423173039955+0j) [Z2 Y6 Z7 Z8 Z9 Y10] +
(-2.1032156043935615e-06+0j) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.13739104762683205+0j) [Z2 Z6] +
(0.0243890825311495+0j) [Z2 X7 Z8 Z9 Z10 X11] +
(-2.0111220978210207e-06+0j) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0243890825311495+0j) [Z2 Y7 Z8 Z9 Z10 Y11] +
(-2.0111220978210207e-06+0j) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.16853486561579917+0j) [Z2 Z7] +
(0.15071408121008273+0j) [Z2 Z8] +
(0.18690820476912534+0j) [Z2 Z9] +
(-1.0632283422650166e-06+0j) [Z2 X10 Z11 X12] +
(-1.0632283422650166e-06+0j) [Z2 Y10 Z11 Y12] +
(0.12799502492468404+0j) [Z2 Z10] +
(1.1094407592194015e-06+0j) [Z2 X11 Z12 X13] +
(1.1094407592194015e-06+0j) [Z2 Y11 Z12 Y13] +
(0.15337968243314146+0j) [Z2 Z11] +
(0.14011289865354798+0j) [Z2 Z12] +
(0.15569010671752442+0j) [Z2 Z13] +
(0.005143391768825153+0j) [X3 X4 Y5 Y6] +
(0.009841749246962623+0j) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10] +
(-2.988511706487063e-06+0j) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.4548424492067448e-06+0j) [X3 X4 X6 X7] +
(-1.522493067585034e-06+0j) [X3 X4 X6 Z7 Z8 Z9 Z10 X11] +
(-0.008541996625454832+0j) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.151346311051097e-06+0j) [X3 X4 Y7 Z8 Z9 Y10] +
(-0.019257505095251592+0j) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895373346593e-07+0j) [X3 X4 X8 X9] +
(-4.643051068299592e-06+0j) [X3 X4 X10 X11] +
(-0.008764827575688751+0j) [X3 X4 X10 Z11 Z12 X13] +
(-0.019028242443847244+0j) [X3 X4 Y11 Y12] +
(5.275883122022993e-06+0j) [X3 X4 X12 X13] +
(-0.005143391768825153+0j) [X3 Y4 Y5 X6] +
(-0.009841749246962623+0j) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10] +
(2.988511706487063e-06+0j) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.4548424492067448e-06+0j) [X3 Y4 Y6 X7] +
(-1.522493067585034e-06+0j) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11] +
(-0.008541996625454832+0j) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.151346311051097e-06+0j) [X3 Y4 Y7 Z8 Z9 X10] +
(0.019257505095251592+0j) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12] +
(-7.956895373346593e-07+0j) [X3 Y4 Y8 X9] +
(-4.643051068299592e-06+0j) [X3 Y4 Y10 X11] +
(-0.008764827575688751+0j) [X3 Y4 Y10 Z11 Z12 X13] +
(0.019028242443847244+0j) [X3 Y4 Y11 X12] +
(5.275883122022993e-06+0j) [X3 Y4 Y12 X13] +
(-3.8870516736690645e-06+0j) [X3 Z4 X5] +
(3.2118420190033097e-06+0j) [X3 Z4 X5 X6 Z7 Z8 Z9 X10] +
(0.019299560579363786+0j) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.2118420190033097e-06+0j) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10] +
(0.019299560579363786+0j) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.195489010083836e-06+0j) [X3 Z4 X5 Z6] +
(1.689348951418275e-06+0j) [X3 Z4 X5 X7 Z8 Z9 Z10 X11] +
(0.010757563953908957+0j) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.537178096553528e-08+0j) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11] +
(4.205548411219395e-05+0j) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.593534391229091e-07+0j) [X3 Z4 X5 Z7] +
(-5.770052995666779e-07+0j) [X3 Z4 X5 Z8] +
(2.1868423776798165e-07+0j) [X3 Z4 X5 Z9] +
(0.024353077678068928+0j) [X3 Z4 X5 X10 Z11 X12] +
(0.024353077678068928+0j) [X3 Z4 X5 Y10 Z11 Y12] +
(-7.80170750019265e-06+0j) [X3 Z4 X5 Z10] +
(0.015588250102380172+0j) [X3 Z4 X5 X11 Z12 X13] +
(0.005324835234221677+0j) [X3 Z4 X5 Y11 Z12 Y13] +
(-3.1586564318930577e-06+0j) [X3 Z4 X5 Z11] +
(8.814937306443681e-06+0j) [X3 Z4 X5 Z12] +
(3.5390541844206886e-06+0j) [X3 Z4 X5 Z13] +
(1.6288532434660635e-06+0j) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11] +
(0.010715508469796763+0j) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.010263414868158495+0j) [X3 Z4 Y5 Y11 Z12 X13] +
(0.00846997879102397+0j) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10] +
(-2.686381543319587e-07+0j) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.011756013419819258+0j) [X3 Z4 Z5 X6 X8 X9] +
(0.015225630757226563+0j) [X3 Z4 Z5 X6 X10 X11] +
(-3.544395429180869e-06+0j) [X3 Z4 Z5 X6 X10 Z11 Z12 X13] +
(-3.0882507111214816e-06+0j) [X3 Z4 Z5 X6 Y11 Y12] +
(-0.004158797381840062+0j) [X3 Z4 Z5 X6 X12 X13] +
(-0.00846997879102397+0j) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10] +
(2.686381543319587e-07+0j) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.011756013419819258+0j) [X3 Z4 Z5 Y6 Y8 X9] +
(0.015225630757226563+0j) [X3 Z4 Z5 Y6 Y10 X11] +
(-3.544395429180869e-06+0j) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13] +
(3.0882507111214816e-06+0j) [X3 Z4 Z5 Y6 Y11 X12] +
(-0.004158797381840062+0j) [X3 Z4 Z5 Y6 Y12 X13] +
(-0.1213327691104235+0j) [X3 Z4 Z5 Z6 X7] +
(-0.017561202409646183+0j) [X3 Z4 Z5 Z6 X7 Z8] +
(-0.005805188989826925+0j) [X3 Z4 Z5 Z6 X7 Z9] +
(2.745518400247564e-06+0j) [X3 Z4 Z5 Z6 X7 X10 Z11 X12] +
(2.745518400247564e-06+0j) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12] +
(0.014411099430130869+0j) [X3 Z4 Z5 Z6 X7 Z10] +
(-7.988770289333041e-07+0j) [X3 Z4 Z5 Z6 X7 X11 Z12 X13] +
(-3.427323108739172e-07+0j) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13] +
(-0.0008145313270956959+0j) [X3 Z4 Z5 Z6 X7 Z11] +
(-0.0034937903598901733+0j) [X3 Z4 Z5 Z6 X7 Z12] +
(0.0006650070219498878+0j) [X3 Z4 Z5 Z6 X7 Z13] +
(-4.56144718059387e-07+0j) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13] +
(-0.014603704729162096+0j) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10] +
(2.874299071311966e-06+0j) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(0.014603704729162096+0j) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10] +
(-2.874299071311966e-06+0j) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(1.300294656192336e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12] +
(0.0024464971554158488+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-1.300294656192336e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12] +
(0.0024464971554158488+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(-0.28164257767022916+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.017091553155898904+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(-0.01953805031131475+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(5.775950527112907e-05+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(2.883676576005985e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(3.846201671178887e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(-0.024282117354693072+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11] +
(5.146496327371224e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(-0.02475546329289098+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11] +
(5.10552672191542e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(-0.03935916802205308+0j) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11] +
(7.979825793227384e-06+0j) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(-0.029903789512624835+0j) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11] +
(5.427988656263154e-06+0j) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.025996177598021156+0j) [X3 Z4 Z5 X7] +
(-0.021433810721600863+0j) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11] +
(5.159350501931196e-06+0j) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0034795118903343373+0j) [X3 Z4 Z6 X7] +
(-0.02873077955190552+0j) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11] +
(5.9358677179979826e-06+0j) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.764994118309244e-07+0j) [X3 X5] +
(0.0016638798784908144+0j) [X3 Z5 Z6 X7] +
(-0.01888903030494289+0j) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.94735601151092e-06+0j) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.005143391768825153+0j) [Y3 X4 X5 Y6] +
(-0.009841749246962623+0j) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10] +
(2.988511706487063e-06+0j) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.4548424492067448e-06+0j) [Y3 X4 X6 Y7] +
(-1.522493067585034e-06+0j) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11] +
(-0.008541996625454832+0j) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.151346311051097e-06+0j) [Y3 X4 X7 Z8 Z9 Y10] +
(0.019257505095251592+0j) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895373346593e-07+0j) [Y3 X4 X8 Y9] +
(-4.643051068299592e-06+0j) [Y3 X4 X10 Y11] +
(-0.008764827575688751+0j) [Y3 X4 X10 Z11 Z12 Y13] +
(0.019028242443847244+0j) [Y3 X4 X11 Y12] +
(5.275883122022993e-06+0j) [Y3 X4 X12 Y13] +
(0.005143391768825153+0j) [Y3 Y4 X5 X6] +
(0.009841749246962623+0j) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10] +
(-2.988511706487063e-06+0j) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.4548424492067448e-06+0j) [Y3 Y4 Y6 Y7] +
(-1.522493067585034e-06+0j) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11] +
(-0.008541996625454832+0j) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.151346311051097e-06+0j) [Y3 Y4 X7 Z8 Z9 X10] +
(-0.019257505095251592+0j) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12] +
(-7.956895373346593e-07+0j) [Y3 Y4 Y8 Y9] +
(-4.643051068299592e-06+0j) [Y3 Y4 Y10 Y11] +
(-0.008764827575688751+0j) [Y3 Y4 Y10 Z11 Z12 Y13] +
(-0.019028242443847244+0j) [Y3 Y4 X11 X12] +
(5.275883122022993e-06+0j) [Y3 Y4 Y12 Y13] +
(1.6288532434660635e-06+0j) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11] +
(0.010715508469796763+0j) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.010263414868158495+0j) [Y3 Z4 X5 X11 Z12 Y13] +
(-3.8870516736690645e-06+0j) [Y3 Z4 Y5] +
(3.2118420190033097e-06+0j) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10] +
(0.019299560579363786+0j) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.2118420190033097e-06+0j) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10] +
(0.019299560579363786+0j) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.195489010083836e-06+0j) [Y3 Z4 Y5 Z6] +
(4.537178096553528e-08+0j) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11] +
(4.205548411219395e-05+0j) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.689348951418275e-06+0j) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11] +
(0.010757563953908957+0j) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.593534391229091e-07+0j) [Y3 Z4 Y5 Z7] +
(-5.770052995666779e-07+0j) [Y3 Z4 Y5 Z8] +
(2.1868423776798165e-07+0j) [Y3 Z4 Y5 Z9] +
(0.024353077678068928+0j) [Y3 Z4 Y5 X10 Z11 X12] +
(0.024353077678068928+0j) [Y3 Z4 Y5 Y10 Z11 Y12] +
(-7.80170750019265e-06+0j) [Y3 Z4 Y5 Z10] +
(0.005324835234221677+0j) [Y3 Z4 Y5 X11 Z12 X13] +
(0.015588250102380172+0j) [Y3 Z4 Y5 Y11 Z12 Y13] +
(-3.1586564318930577e-06+0j) [Y3 Z4 Y5 Z11] +
(8.814937306443681e-06+0j) [Y3 Z4 Y5 Z12] +
(3.5390541844206886e-06+0j) [Y3 Z4 Y5 Z13] +
(-0.00846997879102397+0j) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10] +
(2.686381543319587e-07+0j) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.011756013419819258+0j) [Y3 Z4 Z5 X6 X8 Y9] +
(0.015225630757226563+0j) [Y3 Z4 Z5 X6 X10 Y11] +
(-3.544395429180869e-06+0j) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13] +
(3.0882507111214816e-06+0j) [Y3 Z4 Z5 X6 X11 Y12] +
(-0.004158797381840062+0j) [Y3 Z4 Z5 X6 X12 Y13] +
(0.00846997879102397+0j) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10] +
(-2.686381543319587e-07+0j) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-0.011756013419819258+0j) [Y3 Z4 Z5 Y6 Y8 Y9] +
(0.015225630757226563+0j) [Y3 Z4 Z5 Y6 Y10 Y11] +
(-3.544395429180869e-06+0j) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13] +
(-3.0882507111214816e-06+0j) [Y3 Z4 Z5 Y6 X11 X12] +
(-0.004158797381840062+0j) [Y3 Z4 Z5 Y6 Y12 Y13] +
(-4.56144718059387e-07+0j) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13] +
(-0.1213327691104235+0j) [Y3 Z4 Z5 Z6 Y7] +
(-0.017561202409646183+0j) [Y3 Z4 Z5 Z6 Y7 Z8] +
(-0.005805188989826925+0j) [Y3 Z4 Z5 Z6 Y7 Z9] +
(2.745518400247564e-06+0j) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12] +
(2.745518400247564e-06+0j) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12] +
(0.014411099430130869+0j) [Y3 Z4 Z5 Z6 Y7 Z10] +
(-3.427323108739172e-07+0j) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13] +
(-7.988770289333041e-07+0j) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13] +
(-0.0008145313270956959+0j) [Y3 Z4 Z5 Z6 Y7 Z11] +
(-0.0034937903598901733+0j) [Y3 Z4 Z5 Z6 Y7 Z12] +
(0.0006650070219498878+0j) [Y3 Z4 Z5 Z6 Y7 Z13] +
(0.014603704729162096+0j) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10] +
(-2.874299071311966e-06+0j) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-0.014603704729162096+0j) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10] +
(2.874299071311966e-06+0j) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-1.300294656192336e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12] +
(0.0024464971554158488+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(1.300294656192336e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12] +
(0.0024464971554158488+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(-0.28164257767022916+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.017091553155898904+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(-0.01953805031131475+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(5.775950527112907e-05+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.883676576005985e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(3.846201671178887e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(-0.024282117354693072+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11] +
(5.146496327371224e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(-0.02475546329289098+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11] +
(5.10552672191542e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(-0.03935916802205308+0j) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11] +
(7.979825793227384e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(-0.029903789512624835+0j) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11] +
(5.427988656263154e-06+0j) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.025996177598021156+0j) [Y3 Z4 Z5 Y7] +
(-0.021433810721600863+0j) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11] +
(5.159350501931196e-06+0j) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0034795118903343373+0j) [Y3 Z4 Z6 Y7] +
(-0.02873077955190552+0j) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11] +
(5.9358677179979826e-06+0j) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.764994118309244e-07+0j) [Y3 Y5] +
(0.0016638798784908144+0j) [Y3 Z5 Z6 Y7] +
(-0.01888903030494289+0j) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.94735601151092e-06+0j) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.653894222683171+0j) [Z3] +
(-1.1708301371028247e-06+0j) [Z3 X4 Z5 X6] +
(-7.0897994673533815e-06+0j) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.03490334337366171+0j) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.1708301371028247e-06+0j) [Z3 Y4 Z5 Y6] +
(-7.0897994673533815e-06+0j) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.03490334337366171+0j) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.16079764534838548+0j) [Z3 Z4] +
(-9.509249752659195e-07+0j) [Z3 X5 Z6 X7] +
(-4.72884314709957e-06+0j) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.024591860883829912+0j) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-9.509249752659195e-07+0j) [Z3 Y5 Z6 Y7] +
(-4.72884314709957e-06+0j) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.024591860883829912+0j) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.12495807739503205+0j) [Z3 Z5] +
(0.0243890825311495+0j) [Z3 X6 Z7 Z8 Z9 X10] +
(-2.0111220978210207e-06+0j) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0243890825311495+0j) [Z3 Y6 Z7 Z8 Z9 Y10] +
(-2.0111220978210207e-06+0j) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.16853486561579917+0j) [Z3 Z6] +
(0.019020423173039955+0j) [Z3 X7 Z8 Z9 Z10 X11] +
(-2.1032156043935615e-06+0j) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.019020423173039955+0j) [Z3 Y7 Z8 Z9 Z10 Y11] +
(-2.1032156043935615e-06+0j) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.13739104762683205+0j) [Z3 Z7] +
(0.18690820476912534+0j) [Z3 Z8] +
(0.15071408121008273+0j) [Z3 Z9] +
(1.1094407592194015e-06+0j) [Z3 X10 Z11 X12] +
(1.1094407592194015e-06+0j) [Z3 Y10 Z11 Y12] +
(0.15337968243314146+0j) [Z3 Z10] +
(-1.0632283422650166e-06+0j) [Z3 X11 Z12 X13] +
(-1.0632283422650166e-06+0j) [Z3 Y11 Z12 Y13] +
(0.12799502492468404+0j) [Z3 Z11] +
(0.15569010671752442+0j) [Z3 Z12] +
(0.14011289865354798+0j) [Z3 Z13] +
(-0.011982389010247969+0j) [X4 X5 Y6 Y7] +
(-0.007306759928832994+0j) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11] +
(-2.8882935965214123e-07+0j) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.007306759928832994+0j) [X4 X5 X7 Z8 Z9 X10] +
(-2.8882935965214123e-07+0j) [X4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-0.007156934919856931+0j) [X4 X5 Y8 Y9] +
(-0.01768006795248153+0j) [X4 X5 Y10 Y11] +
(-3.694513294251829e-06+0j) [X4 X5 Y10 Z11 Z12 Y13] +
(-3.694513294251829e-06+0j) [X4 X5 X11 X12] +
(-0.03831467029480385+0j) [X4 X5 Y12 Y13] +
(0.011982389010247969+0j) [X4 Y5 Y6 X7] +
(0.007306759928832994+0j) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11] +
(2.8882935965214123e-07+0j) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.007306759928832994+0j) [X4 Y5 Y7 Z8 Z9 X10] +
(-2.8882935965214123e-07+0j) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(0.007156934919856931+0j) [X4 Y5 Y8 X9] +
(0.01768006795248153+0j) [X4 Y5 Y10 X11] +
(3.694513294251829e-06+0j) [X4 Y5 Y10 Z11 Z12 X13] +
(-3.694513294251829e-06+0j) [X4 Y5 Y11 X12] +
(0.03831467029480385+0j) [X4 Y5 Y12 X13] +
(-1.226048498979666e-05+0j) [X4 Z5 X6] +
(-1.2283337826086257e-06+0j) [X4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(0.0002463643756957566+0j) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2283337826086257e-06+0j) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(0.0002463643756957566+0j) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.8540608581159758e-06+0j) [X4 Z5 X6 Z7] +
(-1.3980449082340533e-06+0j) [X4 Z5 X6 Z8] +
(-1.881850183365535e-06+0j) [X4 Z5 X6 Z9] +
(0.007960880725921571+0j) [X4 Z5 X6 X10 Z11 X12] +
(-0.0009298507967730426+0j) [X4 Z5 X6 Y10 Z11 Y12] +
(-1.6923978285456259e-06+0j) [X4 Z5 X6 Z10] +
(-0.012215040997613964+0j) [X4 Z5 X6 X11 Z12 X13] +
(-0.012215040997613964+0j) [X4 Z5 X6 Y11 Z12 Y13] +
(4.281913884736485e-06+0j) [X4 Z5 X6 Z11] +
(-4.588855155597218e-06+0j) [X4 Z5 X6 Z13] +
(0.008890731522694614+0j) [X4 Z5 Y6 Y10 Z11 X12] +
(-4.838052751314816e-07+0j) [X4 Z5 Z6 X7 Y8 Y9] +
(5.974311713282112e-06+0j) [X4 Z5 Z6 X7 Y10 Y11] +
(0.01128519020084092+0j) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(0.020175921723535536+0j) [X4 Z5 Z6 X7 X11 X12] +
(-4.556569217927557e-06+0j) [X4 Z5 Z6 X7 Y12 Y13] +
(4.838052751314816e-07+0j) [X4 Z5 Z6 Y7 Y8 X9] +
(-5.974311713282112e-06+0j) [X4 Z5 Z6 Y7 Y10 X11] +
(-0.01128519020084092+0j) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(0.020175921723535536+0j) [X4 Z5 Z6 Y7 Y11 X12] +
(4.556569217927557e-06+0j) [X4 Z5 Z6 Y7 Y12 X13] +
(1.330473188659669e-06+0j) [X4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(0.0059237983365613405+0j) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(1.330473188659669e-06+0j) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(0.0059237983365613405+0j) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(-6.631277928185392e-05+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.016024603689179514+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(-0.016024603689179514+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(3.334331289302993e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(-4.734622038533115e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(-9.80610277493624e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(-5.071480736403124e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(5.071480736403124e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-0.3693708936615617+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.023145130929529023+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-0.009612634606847293+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(-0.025637238296026807+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(-8.77481786430822e-06+0j) [X4 Z5 Z6 Z7 Z8 X10] +
(-0.047642612176383055+0j) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(-7.4443446756485515e-06+0j) [X4 Z5 Z6 Z7 Z9 X10] +
(-0.04171881383982171+0j) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(-6.290028432884958e-06+0j) [X4 Z5 Z6 Z8 Z9 X10] +
(-0.039564416322893245+0j) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(-7.518362215493584e-06+0j) [X4 Z5 Z7 Z8 Z9 X10] +
(-0.039318051947197494+0j) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(-5.929765816798197e-07+0j) [X4 X6] +
(-4.253224225545685e-06+0j) [X4 Z6 Z7 Z8 Z9 X10] +
(-0.022528440196012908+0j) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.011982389010247969+0j) [Y4 X5 X6 Y7] +
(0.007306759928832994+0j) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11] +
(2.8882935965214123e-07+0j) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.007306759928832994+0j) [Y4 X5 X7 Z8 Z9 Y10] +
(-2.8882935965214123e-07+0j) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(0.007156934919856931+0j) [Y4 X5 X8 Y9] +
(0.01768006795248153+0j) [Y4 X5 X10 Y11] +
(3.694513294251829e-06+0j) [Y4 X5 X10 Z11 Z12 Y13] +
(-3.694513294251829e-06+0j) [Y4 X5 X11 Y12] +
(0.03831467029480385+0j) [Y4 X5 X12 Y13] +
(-0.011982389010247969+0j) [Y4 Y5 X6 X7] +
(-0.007306759928832994+0j) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11] +
(-2.8882935965214123e-07+0j) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.007306759928832994+0j) [Y4 Y5 Y7 Z8 Z9 Y10] +
(-2.8882935965214123e-07+0j) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.007156934919856931+0j) [Y4 Y5 X8 X9] +
(-0.01768006795248153+0j) [Y4 Y5 X10 X11] +
(-3.694513294251829e-06+0j) [Y4 Y5 X10 Z11 Z12 X13] +
(-3.694513294251829e-06+0j) [Y4 Y5 Y11 Y12] +
(-0.03831467029480385+0j) [Y4 Y5 X12 X13] +
(0.008890731522694614+0j) [Y4 Z5 X6 X10 Z11 Y12] +
(-1.226048498979666e-05+0j) [Y4 Z5 Y6] +
(-1.2283337826086257e-06+0j) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(0.0002463643756957566+0j) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2283337826086257e-06+0j) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(0.0002463643756957566+0j) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.8540608581159758e-06+0j) [Y4 Z5 Y6 Z7] +
(-1.3980449082340533e-06+0j) [Y4 Z5 Y6 Z8] +
(-1.881850183365535e-06+0j) [Y4 Z5 Y6 Z9] +
(-0.0009298507967730426+0j) [Y4 Z5 Y6 X10 Z11 X12] +
(0.007960880725921571+0j) [Y4 Z5 Y6 Y10 Z11 Y12] +
(-1.6923978285456259e-06+0j) [Y4 Z5 Y6 Z10] +
(-0.012215040997613964+0j) [Y4 Z5 Y6 X11 Z12 X13] +
(-0.012215040997613964+0j) [Y4 Z5 Y6 Y11 Z12 Y13] +
(4.281913884736485e-06+0j) [Y4 Z5 Y6 Z11] +
(-4.588855155597218e-06+0j) [Y4 Z5 Y6 Z13] +
(4.838052751314816e-07+0j) [Y4 Z5 Z6 X7 X8 Y9] +
(-5.974311713282112e-06+0j) [Y4 Z5 Z6 X7 X10 Y11] +
(-0.01128519020084092+0j) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(0.020175921723535536+0j) [Y4 Z5 Z6 X7 X11 Y12] +
(4.556569217927557e-06+0j) [Y4 Z5 Z6 X7 X12 Y13] +
(-4.838052751314816e-07+0j) [Y4 Z5 Z6 Y7 X8 X9] +
(5.974311713282112e-06+0j) [Y4 Z5 Z6 Y7 X10 X11] +
(0.01128519020084092+0j) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(0.020175921723535536+0j) [Y4 Z5 Z6 Y7 Y11 Y12] +
(-4.556569217927557e-06+0j) [Y4 Z5 Z6 Y7 X12 X13] +
(1.330473188659669e-06+0j) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(0.0059237983365613405+0j) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(1.330473188659669e-06+0j) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(0.0059237983365613405+0j) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(-6.631277928185392e-05+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.016024603689179514+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(-0.016024603689179514+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(3.334331289302993e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(-4.734622038533115e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(-9.80610277493624e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(5.071480736403124e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-5.071480736403124e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-0.3693708936615617+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.023145130929529023+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-0.009612634606847293+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(-0.025637238296026807+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(-8.77481786430822e-06+0j) [Y4 Z5 Z6 Z7 Z8 Y10] +
(-0.047642612176383055+0j) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(-7.4443446756485515e-06+0j) [Y4 Z5 Z6 Z7 Z9 Y10] +
(-0.04171881383982171+0j) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(-6.290028432884958e-06+0j) [Y4 Z5 Z6 Z8 Z9 Y10] +
(-0.039564416322893245+0j) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(-7.518362215493584e-06+0j) [Y4 Z5 Z7 Z8 Z9 Y10] +
(-0.039318051947197494+0j) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(-5.929765816798197e-07+0j) [Y4 Y6] +
(-4.253224225545685e-06+0j) [Y4 Z6 Z7 Z8 Z9 Y10] +
(-0.022528440196012908+0j) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.2034402289145631+0j) [Z4] +
(-5.929765816798197e-07+0j) [Z4 X5 Z6 X7] +
(-4.253224225545685e-06+0j) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.022528440196012908+0j) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-5.929765816798197e-07+0j) [Z4 Y5 Z6 Y7] +
(-4.253224225545685e-06+0j) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.022528440196012908+0j) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.15755314797985667+0j) [Z4 Z5] +
(0.018266834869375567+0j) [Z4 X6 Z7 Z8 Z9 X10] +
(-1.654117476803e-06+0j) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.018266834869375567+0j) [Z4 Y6 Z7 Z8 Z9 Y10] +
(-1.654117476803e-06+0j) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.13701191674040739+0j) [Z4 Z6] +
(0.010960074940542575+0j) [Z4 X7 Z8 Z9 Z10 X11] +
(-1.942946836455141e-06+0j) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.010960074940542575+0j) [Z4 Y7 Z8 Z9 Z10 Y11] +
(-1.942946836455141e-06+0j) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.14899430575065536+0j) [Z4 Z7] +
(0.15676396176430982+0j) [Z4 Z9] +
(1.8782101246534046e-06+0j) [Z4 X10 Z11 X12] +
(1.8782101246534046e-06+0j) [Z4 Y10 Z11 Y12] +
(0.12489990917237599+0j) [Z4 Z10] +
(-1.8163031695984252e-06+0j) [Z4 X11 Z12 X13] +
(-1.8163031695984252e-06+0j) [Z4 Y11 Z12 Y13] +
(0.11383573679388653+0j) [Z4 Z12] +
(0.15215040708869038+0j) [Z4 Z13] +
(1.2283337826086257e-06+0j) [X5 X6 Y7 Z8 Z9 Y10] +
(-0.0002463643756957566+0j) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-4.838052751314816e-07+0j) [X5 X6 X8 X9] +
(5.974311713282111e-06+0j) [X5 X6 X10 X11] +
(0.020175921723535536+0j) [X5 X6 X10 Z11 Z12 X13] +
(0.01128519020084092+0j) [X5 X6 Y11 Y12] +
(-4.556569217927557e-06+0j) [X5 X6 X12 X13] +
(-1.2283337826086257e-06+0j) [X5 Y6 Y7 Z8 Z9 X10] +
(0.0002463643756957566+0j) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-4.838052751314816e-07+0j) [X5 Y6 Y8 X9] +
(5.974311713282111e-06+0j) [X5 Y6 Y10 X11] +
(0.020175921723535536+0j) [X5 Y6 Y10 Z11 Z12 X13] +
(-0.01128519020084092+0j) [X5 Y6 Y11 X12] +
(-4.556569217927557e-06+0j) [X5 Y6 Y12 X13] +
(-1.2260484989796663e-05+0j) [X5 Z6 X7] +
(-1.881850183365535e-06+0j) [X5 Z6 X7 Z8] +
(-1.3980449082340533e-06+0j) [X5 Z6 X7 Z9] +
(-0.012215040997613964+0j) [X5 Z6 X7 X10 Z11 X12] +
(-0.012215040997613964+0j) [X5 Z6 X7 Y10 Z11 Y12] +
(4.281913884736485e-06+0j) [X5 Z6 X7 Z10] +
(0.007960880725921571+0j) [X5 Z6 X7 X11 Z12 X13] +
(-0.0009298507967730426+0j) [X5 Z6 X7 Y11 Z12 Y13] +
(-1.6923978285456259e-06+0j) [X5 Z6 X7 Z11] +
(-4.588855155597218e-06+0j) [X5 Z6 X7 Z12] +
(0.008890731522694614+0j) [X5 Z6 Y7 Y11 Z12 X13] +
(-1.330473188659669e-06+0j) [X5 Z6 Z7 X8 Y9 Y10] +
(-0.0059237983365613405+0j) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(1.330473188659669e-06+0j) [X5 Z6 Z7 Y8 Y9 X10] +
(0.0059237983365613405+0j) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(0.016024603689179514+0j) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12] +
(-5.071480736403124e-06+0j) [X5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-0.016024603689179514+0j) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12] +
(-5.071480736403124e-06+0j) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(-6.631277928185386e-05+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-9.80610277493624e-06+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(-4.734622038533115e-06+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(-0.3693708936615617+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.023145130929529023+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(-0.025637238296026807+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(3.3343312893029925e-06+0j) [X5 Z6 Z7 Z8 Z9 X11] +
(-0.009612634606847293+0j) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(-7.4443446756485515e-06+0j) [X5 Z6 Z7 Z8 Z10 X11] +
(-0.04171881383982171+0j) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(-8.77481786430822e-06+0j) [X5 Z6 Z7 Z9 Z10 X11] +
(-0.047642612176383055+0j) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(-7.518362215493584e-06+0j) [X5 Z6 Z8 Z9 Z10 X11] +
(-0.039318051947197494+0j) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.854060858115976e-06+0j) [X5 X7] +
(-6.290028432884958e-06+0j) [X5 Z7 Z8 Z9 Z10 X11] +
(-0.039564416322893245+0j) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2283337826086257e-06+0j) [Y5 X6 X7 Z8 Z9 Y10] +
(0.0002463643756957566+0j) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-4.838052751314816e-07+0j) [Y5 X6 X8 Y9] +
(5.974311713282111e-06+0j) [Y5 X6 X10 Y11] +
(0.020175921723535536+0j) [Y5 X6 X10 Z11 Z12 Y13] +
(-0.01128519020084092+0j) [Y5 X6 X11 Y12] +
(-4.556569217927557e-06+0j) [Y5 X6 X12 Y13] +
(1.2283337826086257e-06+0j) [Y5 Y6 X7 Z8 Z9 X10] +
(-0.0002463643756957566+0j) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-4.838052751314816e-07+0j) [Y5 Y6 Y8 Y9] +
(5.974311713282111e-06+0j) [Y5 Y6 Y10 Y11] +
(0.020175921723535536+0j) [Y5 Y6 Y10 Z11 Z12 Y13] +
(0.01128519020084092+0j) [Y5 Y6 X11 X12] +
(-4.556569217927557e-06+0j) [Y5 Y6 Y12 Y13] +
(0.008890731522694614+0j) [Y5 Z6 X7 X11 Z12 Y13] +
(-1.2260484989796663e-05+0j) [Y5 Z6 Y7] +
(-1.881850183365535e-06+0j) [Y5 Z6 Y7 Z8] +
(-1.3980449082340533e-06+0j) [Y5 Z6 Y7 Z9] +
(-0.012215040997613964+0j) [Y5 Z6 Y7 X10 Z11 X12] +
(-0.012215040997613964+0j) [Y5 Z6 Y7 Y10 Z11 Y12] +
(4.281913884736485e-06+0j) [Y5 Z6 Y7 Z10] +
(-0.0009298507967730426+0j) [Y5 Z6 Y7 X11 Z12 X13] +
(0.007960880725921571+0j) [Y5 Z6 Y7 Y11 Z12 Y13] +
(-1.6923978285456259e-06+0j) [Y5 Z6 Y7 Z11] +
(-4.588855155597218e-06+0j) [Y5 Z6 Y7 Z12] +
(1.330473188659669e-06+0j) [Y5 Z6 Z7 X8 X9 Y10] +
(0.0059237983365613405+0j) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-1.330473188659669e-06+0j) [Y5 Z6 Z7 Y8 X9 X10] +
(-0.0059237983365613405+0j) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-0.016024603689179514+0j) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12] +
(-5.071480736403124e-06+0j) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(0.016024603689179514+0j) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12] +
(-5.071480736403124e-06+0j) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(-6.631277928185386e-05+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-9.80610277493624e-06+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(-4.734622038533115e-06+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(-0.3693708936615617+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.023145130929529023+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(-0.025637238296026807+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(3.3343312893029925e-06+0j) [Y5 Z6 Z7 Z8 Z9 Y11] +
(-0.009612634606847293+0j) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(-7.4443446756485515e-06+0j) [Y5 Z6 Z7 Z8 Z10 Y11] +
(-0.04171881383982171+0j) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(-8.77481786430822e-06+0j) [Y5 Z6 Z7 Z9 Z10 Y11] +
(-0.047642612176383055+0j) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(-7.518362215493584e-06+0j) [Y5 Z6 Z8 Z9 Z10 Y11] +
(-0.039318051947197494+0j) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.854060858115976e-06+0j) [Y5 Y7] +
(-6.290028432884958e-06+0j) [Y5 Z7 Z8 Z9 Z10 Y11] +
(-0.039564416322893245+0j) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.2034402289145625+0j) [Z5] +
(0.010960074940542575+0j) [Z5 X6 Z7 Z8 Z9 X10] +
(-1.942946836455141e-06+0j) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.010960074940542575+0j) [Z5 Y6 Z7 Z8 Z9 Y10] +
(-1.942946836455141e-06+0j) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.14899430575065536+0j) [Z5 Z6] +
(0.018266834869375567+0j) [Z5 X7 Z8 Z9 Z10 X11] +
(-1.654117476803e-06+0j) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.018266834869375567+0j) [Z5 Y7 Z8 Z9 Z10 Y11] +
(-1.654117476803e-06+0j) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.13701191674040739+0j) [Z5 Z7] +
(0.15676396176430982+0j) [Z5 Z8] +
(-1.8163031695984252e-06+0j) [Z5 X10 Z11 X12] +
(-1.8163031695984252e-06+0j) [Z5 Y10 Z11 Y12] +
(1.8782101246534046e-06+0j) [Z5 X11 Z12 X13] +
(1.8782101246534046e-06+0j) [Z5 Y11 Z12 Y13] +
(0.12489990917237599+0j) [Z5 Z11] +
(0.15215040708869038+0j) [Z5 Z12] +
(0.11383573679388653+0j) [Z5 Z13] +
(-0.013873381748426093+0j) [X6 X7 Y8 Y9] +
(-0.01782514099578651+0j) [X6 X7 Y10 Y11] +
(-1.0358477602844546e-06+0j) [X6 X7 Y10 Z11 Z12 Y13] +
(-1.0358477602844546e-06+0j) [X6 X7 X11 X12] +
(-0.017366118994651413+0j) [X6 X7 Y12 Y13] +
(0.013873381748426093+0j) [X6 Y7 Y8 X9] +
(0.01782514099578651+0j) [X6 Y7 Y10 X11] +
(1.0358477602844546e-06+0j) [X6 Y7 Y10 Z11 Z12 X13] +
(-1.0358477602844546e-06+0j) [X6 Y7 Y11 X12] +
(0.017366118994651413+0j) [X6 Y7 Y12 X13] +
(0.0002921986261110524+0j) [X6 Z7 X8 X9 Z10 X11] +
(-3.3281393508579645e-07+0j) [X6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.0002921986261110524+0j) [X6 Z7 X8 Y9 Z10 Y11] +
(-3.3281393508579645e-07+0j) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(0.2284810656491893+0j) [X6 Z7 Z8 Z9 X10] +
(3.313145500133372e-06+0j) [X6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(3.313145500133372e-06+0j) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(0.01130727400884824+0j) [X6 Z7 Z8 Z9 X10 Z11] +
(0.025104957138844565+0j) [X6 Z7 Z8 Z9 X10 Z12] +
(0.01054042590767158+0j) [X6 Z7 Z8 Z9 X10 Z13] +
(-0.014564531231172986+0j) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(0.014564531231172986+0j) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-2.595086006668199e-05+0j) [X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.183932559493417e-06+0j) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-6.524373848270057e-06+0j) [X6 Z7 Z8 Z9 Z10 X12] +
(-3.2112283481366846e-06+0j) [X6 Z7 Z8 Z9 Z11 X12] +
(0.029812424517345806+0j) [X6 Z7 Z8 X10] +
(-3.277483195095374e-06+0j) [X6 Z7 Z8 Z10 Z11 X12] +
(0.030104623143456855+0j) [X6 Z7 Z9 X10] +
(-3.6102971301811705e-06+0j) [X6 Z7 Z9 Z10 Z11 X12] +
(0.030787505389143953+0j) [X6 Z8 Z9 X10] +
(-3.7696594515755945e-06+0j) [X6 Z8 Z9 Z10 Z11 X12] +
(0.013873381748426093+0j) [Y6 X7 X8 Y9] +
(0.01782514099578651+0j) [Y6 X7 X10 Y11] +
(1.0358477602844546e-06+0j) [Y6 X7 X10 Z11 Z12 Y13] +
(-1.0358477602844546e-06+0j) [Y6 X7 X11 Y12] +
(0.017366118994651413+0j) [Y6 X7 X12 Y13] +
(-0.013873381748426093+0j) [Y6 Y7 X8 X9] +
(-0.01782514099578651+0j) [Y6 Y7 X10 X11] +
(-1.0358477602844546e-06+0j) [Y6 Y7 X10 Z11 Z12 X13] +
(-1.0358477602844546e-06+0j) [Y6 Y7 Y11 Y12] +
(-0.017366118994651413+0j) [Y6 Y7 X12 X13] +
(0.0002921986261110524+0j) [Y6 Z7 Y8 X9 Z10 X11] +
(-3.3281393508579645e-07+0j) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.0002921986261110524+0j) [Y6 Z7 Y8 Y9 Z10 Y11] +
(-3.3281393508579645e-07+0j) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(0.2284810656491893+0j) [Y6 Z7 Z8 Z9 Y10] +
(3.313145500133372e-06+0j) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(3.313145500133372e-06+0j) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(0.01130727400884824+0j) [Y6 Z7 Z8 Z9 Y10 Z11] +
(0.025104957138844565+0j) [Y6 Z7 Z8 Z9 Y10 Z12] +
(0.01054042590767158+0j) [Y6 Z7 Z8 Z9 Y10 Z13] +
(0.014564531231172986+0j) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-0.014564531231172986+0j) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-2.595086006668199e-05+0j) [Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.183932559493417e-06+0j) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-6.524373848270057e-06+0j) [Y6 Z7 Z8 Z9 Z10 Y12] +
(-3.2112283481366846e-06+0j) [Y6 Z7 Z8 Z9 Z11 Y12] +
(0.029812424517345806+0j) [Y6 Z7 Z8 Y10] +
(-3.277483195095374e-06+0j) [Y6 Z7 Z8 Z10 Z11 Y12] +
(0.030104623143456855+0j) [Y6 Z7 Z9 Y10] +
(-3.6102971301811705e-06+0j) [Y6 Z7 Z9 Z10 Z11 Y12] +
(0.030787505389143953+0j) [Y6 Z8 Z9 Y10] +
(-3.7696594515755945e-06+0j) [Y6 Z8 Z9 Z10 Z11 Y12] +
(1.3096862988615428+0j) [Z6] +
(0.030787505389143953+0j) [Z6 X7 Z8 Z9 Z10 X11] +
(-3.7696594515755945e-06+0j) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.030787505389143953+0j) [Z6 Y7 Z8 Z9 Z10 Y11] +
(-3.7696594515755945e-06+0j) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.19392534613270176+0j) [Z6 Z7] +
(0.16756653265461252+0j) [Z6 Z8] +
(0.18143991440303864+0j) [Z6 Z9] +
(-1.8551201214294811e-06+0j) [Z6 X10 Z11 X12] +
(-1.8551201214294811e-06+0j) [Z6 Y10 Z11 Y12] +
(0.1195243896468267+0j) [Z6 Z10] +
(-2.8909678817139355e-06+0j) [Z6 X11 Z12 X13] +
(-2.8909678817139355e-06+0j) [Z6 Y11 Z12 Y13] +
(0.13734953064261318+0j) [Z6 Z11] +
(0.13401715261963693+0j) [Z6 Z12] +
(0.15138327161428833+0j) [Z6 Z13] +
(-0.0002921986261110524+0j) [X7 X8 Y9 Y10] +
(3.3281393508579645e-07+0j) [X7 X8 Y9 Z10 Z11 Y12] +
(0.0002921986261110524+0j) [X7 Y8 Y9 X10] +
(-3.3281393508579645e-07+0j) [X7 Y8 Y9 Z10 Z11 X12] +
(-3.3131455001333725e-06+0j) [X7 Z8 Z9 X10 Y11 Y12] +
(-0.014564531231172986+0j) [X7 Z8 Z9 X10 X12 X13] +
(3.3131455001333725e-06+0j) [X7 Z8 Z9 Y10 Y11 X12] +
(-0.014564531231172986+0j) [X7 Z8 Z9 Y10 Y12 X13] +
(0.22848106564918932+0j) [X7 Z8 Z9 Z10 X11] +
(0.01054042590767158+0j) [X7 Z8 Z9 Z10 X11 Z12] +
(0.025104957138844565+0j) [X7 Z8 Z9 Z10 X11 Z13] +
(-2.5950860066681984e-05+0j) [X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.183932559493417e-06+0j) [X7 Z8 Z9 Z10 Z11 X13] +
(-3.2112283481366846e-06+0j) [X7 Z8 Z9 Z10 Z12 X13] +
(0.01130727400884824+0j) [X7 Z8 Z9 X11] +
(-6.524373848270057e-06+0j) [X7 Z8 Z9 Z11 Z12 X13] +
(0.030104623143456855+0j) [X7 Z8 Z10 X11] +
(-3.6102971301811705e-06+0j) [X7 Z8 Z10 Z11 Z12 X13] +
(0.029812424517345806+0j) [X7 Z9 Z10 X11] +
(-3.277483195095374e-06+0j) [X7 Z9 Z10 Z11 Z12 X13] +
(0.0002921986261110524+0j) [Y7 X8 X9 Y10] +
(-3.3281393508579645e-07+0j) [Y7 X8 X9 Z10 Z11 Y12] +
(-0.0002921986261110524+0j) [Y7 Y8 X9 X10] +
(3.3281393508579645e-07+0j) [Y7 Y8 X9 Z10 Z11 X12] +
(3.3131455001333725e-06+0j) [Y7 Z8 Z9 X10 X11 Y12] +
(-0.014564531231172986+0j) [Y7 Z8 Z9 X10 X12 Y13] +
(-3.3131455001333725e-06+0j) [Y7 Z8 Z9 Y10 X11 X12] +
(-0.014564531231172986+0j) [Y7 Z8 Z9 Y10 Y12 Y13] +
(0.22848106564918932+0j) [Y7 Z8 Z9 Z10 Y11] +
(0.01054042590767158+0j) [Y7 Z8 Z9 Z10 Y11 Z12] +
(0.025104957138844565+0j) [Y7 Z8 Z9 Z10 Y11 Z13] +
(-2.5950860066681984e-05+0j) [Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(4.183932559493417e-06+0j) [Y7 Z8 Z9 Z10 Z11 Y13] +
(-3.2112283481366846e-06+0j) [Y7 Z8 Z9 Z10 Z12 Y13] +
(0.01130727400884824+0j) [Y7 Z8 Z9 Y11] +
(-6.524373848270057e-06+0j) [Y7 Z8 Z9 Z11 Z12 Y13] +
(0.030104623143456855+0j) [Y7 Z8 Z10 Y11] +
(-3.6102971301811705e-06+0j) [Y7 Z8 Z10 Z11 Z12 Y13] +
(0.029812424517345806+0j) [Y7 Z9 Z10 Y11] +
(-3.277483195095374e-06+0j) [Y7 Z9 Z10 Z11 Z12 Y13] +
(1.3096862988615423+0j) [Z7] +
(0.18143991440303864+0j) [Z7 Z8] +
(0.16756653265461252+0j) [Z7 Z9] +
(-2.8909678817139355e-06+0j) [Z7 X10 Z11 X12] +
(-2.8909678817139355e-06+0j) [Z7 Y10 Z11 Y12] +
(0.13734953064261318+0j) [Z7 Z10] +
(-1.8551201214294811e-06+0j) [Z7 X11 Z12 X13] +
(-1.8551201214294811e-06+0j) [Z7 Y11 Z12 Y13] +
(0.1195243896468267+0j) [Z7 Z11] +
(0.15138327161428833+0j) [Z7 Z12] +
(0.13401715261963693+0j) [Z7 Z13] +
(-0.009560705729135944+0j) [X8 X9 Y10 Y11] +
(6.628614201475733e-07+0j) [X8 X9 Y10 Z11 Z12 Y13] +
(6.628614201475733e-07+0j) [X8 X9 X11 X12] +
(-0.00608782248056186+0j) [X8 X9 Y12 Y13] +
(0.009560705729135944+0j) [X8 Y9 Y10 X11] +
(-6.628614201475733e-07+0j) [X8 Y9 Y10 Z11 Z12 X13] +
(6.628614201475733e-07+0j) [X8 Y9 Y11 X12] +
(0.00608782248056186+0j) [X8 Y9 Y12 X13] +
(0.009560705729135944+0j) [Y8 X9 X10 Y11] +
(-6.628614201475733e-07+0j) [Y8 X9 X10 Z11 Z12 Y13] +
(6.628614201475733e-07+0j) [Y8 X9 X11 Y12] +
(0.00608782248056186+0j) [Y8 X9 X12 Y13] +
(-0.009560705729135944+0j) [Y8 Y9 X10 X11] +
(6.628614201475733e-07+0j) [Y8 Y9 X10 Z11 Z12 X13] +
(6.628614201475733e-07+0j) [Y8 Y9 Y11 Y12] +
(-0.00608782248056186+0j) [Y8 Y9 X12 X13] +
(1.3693525634718184+0j) [Z8] +
(0.2200397733437608+0j) [Z8 Z9] +
(-1.597317197761541e-06+0j) [Z8 X10 Z11 X12] +
(-1.597317197761541e-06+0j) [Z8 Y10 Z11 Y12] +
(0.1376687264585258+0j) [Z8 Z10] +
(-9.344557776139677e-07+0j) [Z8 X11 Z12 X13] +
(-9.344557776139677e-07+0j) [Z8 Y11 Z12 Y13] +
(0.1472294321876617+0j) [Z8 Z11] +
(0.1497348680349692+0j) [Z8 Z12] +
(0.15582269051553105+0j) [Z8 Z13] +
(1.3693525634718184+0j) [Z9] +
(-9.344557776139677e-07+0j) [Z9 X10 Z11 X12] +
(-9.344557776139677e-07+0j) [Z9 Y10 Z11 Y12] +
(0.1472294321876617+0j) [Z9 Z10] +
(-1.597317197761541e-06+0j) [Z9 X11 Z12 X13] +
(-1.597317197761541e-06+0j) [Z9 Y11 Z12 Y13] +
(0.1376687264585258+0j) [Z9 Z11] +
(0.15582269051553105+0j) [Z9 Z12] +
(0.1497348680349692+0j) [Z9 Z13] +
(-0.028685183716105823+0j) [X10 X11 Y12 Y13] +
(0.028685183716105823+0j) [X10 Y11 Y12 X13] +
(-1.0722312157607353e-05+0j) [X10 Z11 X12] +
(7.954413176044677e-06+0j) [X10 Z11 X12 Z13] +
(-8.194261372000181e-06+0j) [X10 X12] +
(0.028685183716105823+0j) [Y10 X11 X12 Y13] +
(-0.028685183716105823+0j) [Y10 Y11 X12 X13] +
(-1.0722312157607353e-05+0j) [Y10 Z11 Y12] +
(7.954413176044677e-06+0j) [Y10 Z11 Y12 Z13] +
(-8.194261372000181e-06+0j) [Y10 Y12] +
(0.7829661725950203+0j) [Z10] +
(-8.19426137200018e-06+0j) [Z10 X11 Z12 X13] +
(-8.19426137200018e-06+0j) [Z10 Y11 Z12 Y13] +
(0.149263551473889+0j) [Z10 Z11] +
(0.11270386920332216+0j) [Z10 Z12] +
(0.141389052919428+0j) [Z10 Z13] +
(-1.0722312157607353e-05+0j) [X11 Z12 X13] +
(7.954413176044677e-06+0j) [X11 X13] +
(-1.0722312157607353e-05+0j) [Y11 Z12 Y13] +
(7.954413176044677e-06+0j) [Y11 Y13] +
(0.7829661725950203+0j) [Z11] +
(0.141389052919428+0j) [Z11 Z12] +
(0.11270386920332216+0j) [Z11 Z13] +
(0.8084581961720486+0j) [Z12] +
(0.15435748657223614+0j) [Z12 Z13] +
(0.8084581961720488+0j) [Z13]
  (-46.463906788688966) [I0]
+ (0.7829661725950179) [Z10]
+ (0.7829661725950182) [Z11]
+ (0.8084581961720467) [Z12]
+ (0.8084581961720472) [Z13]
+ (1.2034402289145631) [Z4]
+ (1.2034402289145634) [Z5]
+ (1.3096862988615412) [Z7]
+ (1.3096862988615419) [Z6]
+ (1.3693525634718173) [Z8]
+ (1.3693525634718184) [Z9]
+ (1.6538942226831703) [Z3]
+ (1.6538942226831705) [Z2]
+ (12.41263074211178) [Z0]
+ (12.41263074211178) [Z1]
+ (-8.194261372114766e-06) [Y10 Y12]
+ (-8.194261372114766e-06) [X10 X12]
+ (-1.8540608580678247e-06) [Y5 Y7]
+ (-1.8540608580678247e-06) [X5 X7]
+ (-7.76499411874337e-07) [Y3 Y5]
+ (-7.76499411874337e-07) [X3 X5]
+ (-5.929765816690454e-07) [Y4 Y6]
+ (-5.929765816690454e-07) [X4 X6]
+ (1.60211674058712e-06) [Y2 Y4]
+ (1.60211674058712e-06) [X2 X4]
+ (7.954413176230292e-06) [Y11 Y13]
+ (7.954413176230292e-06) [X11 X13]
+ (0.0032769719312316665) [Y1 Y3]
+ (0.0032769719312316665) [X1 X3]
+ (0.10433064780651444) [Y0 Y2]
+ (0.10433064780651444) [X0 X2]
+ (0.11270386920332207) [Z10 Z12]
+ (0.11270386920332207) [Z11 Z13]
+ (0.11383573679388657) [Z4 Z12]
+ (0.11383573679388657) [Z5 Z13]
+ (0.11952438964682666) [Z6 Z10]
+ (0.11952438964682666) [Z7 Z11]
+ (0.12489990917237598) [Z4 Z10]
+ (0.12489990917237598) [Z5 Z11]
+ (0.12495807739503215) [Z2 Z4]
+ (0.12495807739503215) [Z3 Z5]
+ (0.12799502492468398) [Z2 Z10]
+ (0.12799502492468398) [Z3 Z11]
+ (0.13401715261963698) [Z6 Z12]
+ (0.13401715261963698) [Z7 Z13]
+ (0.1370119167404075) [Z4 Z6]
+ (0.1370119167404075) [Z5 Z7]
+ (0.13734953064261313) [Z6 Z11]
+ (0.13734953064261313) [Z7 Z10]
+ (0.1373910476268322) [Z2 Z6]
+ (0.1373910476268322) [Z3 Z7]
+ (0.13766872645852574) [Z8 Z10]
+ (0.13766872645852574) [Z9 Z11]
+ (0.14011289865354803) [Z2 Z12]
+ (0.14011289865354803) [Z3 Z13]
+ (0.141389052919428) [Z10 Z13]
+ (0.141389052919428) [Z11 Z12]
+ (0.14257997712485748) [Z4 Z11]
+ (0.14257997712485748) [Z5 Z10]
+ (0.14722943218766166) [Z8 Z11]
+ (0.14722943218766166) [Z9 Z10]
+ (0.14899430575065545) [Z4 Z7]
+ (0.14899430575065545) [Z5 Z6]
+ (0.1492635514738889) [Z10 Z11]
+ (0.149607026844453) [Z4 Z8]
+ (0.149607026844453) [Z5 Z9]
+ (0.14973486803496922) [Z8 Z12]
+ (0.14973486803496922) [Z9 Z13]
+ (0.15071408121008284) [Z2 Z8]
+ (0.15071408121008284) [Z3 Z9]
+ (0.15138327161428833) [Z6 Z13]
+ (0.15138327161428833) [Z7 Z12]
+ (0.15215040708869043) [Z4 Z13]
+ (0.15215040708869043) [Z5 Z12]
+ (0.15337968243314135) [Z2 Z11]
+ (0.15337968243314135) [Z3 Z10]
+ (0.15435748657223625) [Z12 Z13]
+ (0.15569010671752448) [Z2 Z13]
+ (0.15569010671752448) [Z3 Z12]
+ (0.15582269051553105) [Z8 Z13]
+ (0.15582269051553105) [Z9 Z12]
+ (0.15676396176430996) [Z4 Z9]
+ (0.15676396176430996) [Z5 Z8]
+ (0.15755314797985667) [Z4 Z5]
+ (0.16079764534838556) [Z2 Z5]
+ (0.16079764534838556) [Z3 Z4]
+ (0.1675665326546127) [Z6 Z8]
+ (0.1675665326546127) [Z7 Z9]
+ (0.1685348656157993) [Z2 Z7]
+ (0.1685348656157993) [Z3 Z6]
+ (0.18143991440303875) [Z6 Z9]
+ (0.18143991440303875) [Z7 Z8]
+ (0.1818908579075135) [Z2 Z3]
+ (0.19299723935364244) [Z0 Z10]
+ (0.19299723935364244) [Z1 Z11]
+ (0.19392534613270188) [Z6 Z7]
+ (0.1966177089034217) [Z0 Z4]
+ (0.1966177089034217) [Z1 Z5]
+ (0.19936354537360854) [Z0 Z5]
+ (0.19936354537360854) [Z1 Z4]
+ (0.20072866460441774) [Z0 Z11]
+ (0.20072866460441774) [Z1 Z10]
+ (0.21102659849791527) [Z0 Z12]
+ (0.21102659849791527) [Z1 Z13]
+ (0.21631037498631825) [Z0 Z13]
+ (0.21631037498631825) [Z1 Z12]
+ (0.23671080783830445) [Z0 Z2]
+ (0.23671080783830445) [Z1 Z3]
+ (0.24164663936017225) [Z0 Z6]
+ (0.24164663936017225) [Z1 Z7]
+ (0.2485348337131428) [Z0 Z7]
+ (0.2485348337131428) [Z1 Z6]
+ (0.25129445674591716) [Z0 Z3]
+ (0.25129445674591716) [Z1 Z2]
+ (0.27232518306605713) [Z0 Z8]
+ (0.27232518306605713) [Z1 Z9]
+ (0.2788345442672344) [Z0 Z9]
+ (0.2788345442672344) [Z1 Z8]
+ (1.1861763734860524) [Z0 Z1]
+ (-1.2260484989462601e-05) [Y4 Z5 Y6]
+ (-1.2260484989462601e-05) [X4 Z5 X6]
+ (-1.2260484989462596e-05) [Y5 Z6 Y7]
+ (-1.2260484989462596e-05) [X5 Z6 X7]
+ (-1.0722312157448436e-05) [Y10 Z11 Y12]
+ (-1.0722312157448436e-05) [X10 Z11 X12]
+ (-1.0722312157448431e-05) [Y11 Z12 Y13]
+ (-1.0722312157448431e-05) [X11 Z12 X13]
+ (-3.887051673943381e-06) [Y2 Z3 Y4]
+ (-3.887051673943381e-06) [X2 Z3 X4]
+ (-3.88705167394338e-06) [Y3 Z4 Y5]
+ (-3.88705167394338e-06) [X3 Z4 X5]
+ (0.12507032579772115) [Y1 Z2 Y3]
+ (0.12507032579772115) [X1 Z2 X3]
+ (0.12507032579772123) [Y0 Z1 Y2]
+ (0.12507032579772123) [X0 Z1 X2]
+ (-0.03831467029480388) [Y4 Y5 X12 X13]
+ (-0.03831467029480388) [X4 X5 Y12 Y13]
+ (-0.03619412355904263) [Y2 Y3 X8 X9]
+ (-0.03619412355904263) [X2 X3 Y8 Y9]
+ (-0.03583956795335342) [Y2 Y3 X4 X5]
+ (-0.03583956795335342) [X2 X3 Y4 Y5]
+ (-0.031143817988967117) [Y2 Y3 X6 X7]
+ (-0.031143817988967117) [X2 X3 Y6 Y7]
+ (-0.02868518371610595) [Y10 Y11 X12 X13]
+ (-0.02868518371610595) [X10 X11 Y12 Y13]
+ (-0.025996177598021156) [Y3 Z4 Z5 Y7]
+ (-0.025996177598021156) [X3 Z4 Z5 X7]
+ (-0.025384657508457368) [Y2 Y3 X10 X11]
+ (-0.025384657508457368) [X2 X3 Y10 Y11]
+ (-0.01902824244384726) [Y3 Y4 X11 X12]
+ (-0.01902824244384726) [X3 X4 Y11 Y12]
+ (-0.017825140995786467) [Y6 Y7 X10 X11]
+ (-0.017825140995786467) [X6 X7 Y10 Y11]
+ (-0.017680067952481514) [Y4 Y5 X10 X11]
+ (-0.017680067952481514) [X4 X5 Y10 Y11]
+ (-0.017366118994651365) [Y6 Y7 X12 X13]
+ (-0.017366118994651365) [X6 X7 Y12 Y13]
+ (-0.01557720806397644) [Y2 Y3 X12 X13]
+ (-0.01557720806397644) [X2 X3 Y12 Y13]
+ (-0.014583648907612726) [Y0 Y1 X2 X3]
+ (-0.014583648907612726) [X0 X1 Y2 Y3]
+ (-0.013873381748426089) [Y6 Y7 X8 X9]
+ (-0.013873381748426089) [X6 X7 Y8 Y9]
+ (-0.011982389010247941) [Y4 Y5 X6 X7]
+ (-0.011982389010247941) [X4 X5 Y6 Y7]
+ (-0.011285190200840903) [Y5 X6 X11 Y12]
+ (-0.011285190200840903) [X5 Y6 Y11 X12]
+ (-0.009560705729135926) [Y8 Y9 X10 X11]
+ (-0.009560705729135926) [X8 X9 Y10 Y11]
+ (-0.008125251921381043) [Y1 X2 X8 Y9]
+ (-0.008125251921381043) [Y1 Y2 Y8 Y9]
+ (-0.008125251921381043) [X1 X2 X8 X9]
+ (-0.008125251921381043) [X1 Y2 Y8 X9]
+ (-0.007731425250775297) [Y0 Y1 X10 X11]
+ (-0.007731425250775297) [X0 X1 Y10 Y11]
+ (-0.007156934919856946) [Y4 Y5 X8 X9]
+ (-0.007156934919856946) [X4 X5 Y8 Y9]
+ (-0.006888194352970577) [Y0 Y1 X6 X7]
+ (-0.006888194352970577) [X0 X1 Y6 Y7]
+ (-0.006509361201177248) [Y0 Y1 X8 X9]
+ (-0.006509361201177248) [X0 X1 Y8 Y9]
+ (-0.006087822480561852) [Y8 Y9 X12 X13]
+ (-0.006087822480561852) [X8 X9 Y12 Y13]
+ (-0.005283776488402964) [Y0 Y1 X12 X13]
+ (-0.005283776488402964) [X0 X1 Y12 Y13]
+ (-0.005143391768825094) [Y3 X4 X5 Y6]
+ (-0.005143391768825094) [X3 Y4 Y5 X6]
+ (-0.004684903388155215) [Y1 X2 X6 Y7]
+ (-0.004684903388155215) [Y1 Y2 Y6 Y7]
+ (-0.004684903388155215) [X1 X2 X6 X7]
+ (-0.004684903388155215) [X1 Y2 Y6 X7]
+ (-0.004575007626639207) [Y1 X2 X12 Y13]
+ (-0.004575007626639207) [Y1 Y2 Y12 Y13]
+ (-0.004575007626639207) [X1 X2 X12 X13]
+ (-0.004575007626639207) [X1 Y2 Y12 X13]
+ (-0.0044248554494418614) [Y1 X2 X4 Y5]
+ (-0.0044248554494418614) [Y1 Y2 Y4 Y5]
+ (-0.0044248554494418614) [X1 X2 X4 X5]
+ (-0.0044248554494418614) [X1 Y2 Y4 X5]
+ (-0.00347951189033433) [Y2 Z3 Z5 Y6]
+ (-0.00347951189033433) [X2 Z3 Z5 X6]
+ (-0.00347951189033433) [Y3 Z4 Z6 Y7]
+ (-0.00347951189033433) [X3 Z4 Z6 X7]
+ (-0.002745836470186818) [Y0 Y1 X4 X5]
+ (-0.002745836470186818) [X0 X1 Y4 Y5]
+ (-0.0017992194936630199) [Y1 X2 X10 Y11]
+ (-0.0017992194936630199) [Y1 Y2 Y10 Y11]
+ (-0.0017992194936630199) [X1 X2 X10 X11]
+ (-0.0017992194936630199) [X1 Y2 Y10 X11]
+ (-0.0002921986261110458) [Y7 Y8 X9 X10]
+ (-0.0002921986261110458) [X7 X8 Y9 Y10]
+ (-8.194261372114766e-06) [Z10 Y11 Z12 Y13]
+ (-8.194261372114766e-06) [Z10 X11 Z12 X13]
+ (-7.801707500429111e-06) [Y2 Z3 Y4 Z11]
+ (-7.801707500429111e-06) [X2 Z3 X4 Z11]
+ (-7.801707500429111e-06) [Y3 Z4 Y5 Z10]
+ (-7.801707500429111e-06) [X3 Z4 X5 Z10]
+ (-4.6430510683925655e-06) [Y3 X4 X10 Y11]
+ (-4.6430510683925655e-06) [Y3 Y4 Y10 Y11]
+ (-4.6430510683925655e-06) [X3 X4 X10 X11]
+ (-4.6430510683925655e-06) [X3 Y4 Y10 X11]
+ (-4.588855155670244e-06) [Y4 Z5 Y6 Z13]
+ (-4.588855155670244e-06) [X4 Z5 X6 Z13]
+ (-4.588855155670244e-06) [Y5 Z6 Y7 Z12]
+ (-4.588855155670244e-06) [X5 Z6 X7 Z12]
+ (-4.556569218031597e-06) [Y5 X6 X12 Y13]
+ (-4.556569218031597e-06) [Y5 Y6 Y12 Y13]
+ (-4.556569218031597e-06) [X5 X6 X12 X13]
+ (-4.556569218031597e-06) [X5 Y6 Y12 X13]
+ (-3.694513294333329e-06) [Y4 X5 X11 Y12]
+ (-3.694513294333329e-06) [Y4 Y5 Y11 Y12]
+ (-3.694513294333329e-06) [X4 X5 X11 X12]
+ (-3.694513294333329e-06) [X4 Y5 Y11 X12]
+ (-3.344081556694598e-06) [Z0 Y5 Z6 Y7]
+ (-3.344081556694598e-06) [Z0 X5 Z6 X7]
+ (-3.344081556694598e-06) [Z1 Y4 Z5 Y6]
+ (-3.344081556694598e-06) [Z1 X4 Z5 X6]
+ (-3.1586564320365455e-06) [Y2 Z3 Y4 Z10]
+ (-3.1586564320365455e-06) [X2 Z3 X4 Z10]
+ (-3.1586564320365455e-06) [Y3 Z4 Y5 Z11]
+ (-3.1586564320365455e-06) [X3 Z4 X5 Z11]
+ (-3.099349243796988e-06) [Z0 Y4 Z5 Y6]
+ (-3.099349243796988e-06) [Z0 X4 Z5 X6]
+ (-3.099349243796988e-06) [Z1 Y5 Z6 Y7]
+ (-3.099349243796988e-06) [Z1 X5 Z6 X7]
+ (-2.8909678816874377e-06) [Z6 Y11 Z12 Y13]
+ (-2.8909678816874377e-06) [Z6 X11 Z12 X13]
+ (-2.8909678816874377e-06) [Z7 Y10 Z11 Y12]
+ (-2.8909678816874377e-06) [Z7 X10 Z11 X12]
+ (-2.177664604896869e-06) [Z0 Y10 Z11 Y12]
+ (-2.177664604896869e-06) [Z0 X10 Z11 X12]
+ (-2.177664604896869e-06) [Z1 Y11 Z12 Y13]
+ (-2.177664604896869e-06) [Z1 X11 Z12 X13]
+ (-1.8818501833261123e-06) [Y4 Z5 Y6 Z9]
+ (-1.8818501833261123e-06) [X4 Z5 X6 Z9]
+ (-1.8818501833261123e-06) [Y5 Z6 Y7 Z8]
+ (-1.8818501833261123e-06) [X5 Z6 X7 Z8]
+ (-1.8551201214416778e-06) [Z6 Y10 Z11 Y12]
+ (-1.8551201214416778e-06) [Z6 X10 Z11 X12]
+ (-1.8551201214416778e-06) [Z7 Y11 Z12 Y13]
+ (-1.8551201214416778e-06) [Z7 X11 Z12 X13]
+ (-1.854060858067825e-06) [Y4 Z5 Y6 Z7]
+ (-1.854060858067825e-06) [X4 Z5 X6 Z7]
+ (-1.816303169604883e-06) [Z4 Y11 Z12 Y13]
+ (-1.816303169604883e-06) [Z4 X11 Z12 X13]
+ (-1.816303169604883e-06) [Z5 Y10 Z11 Y12]
+ (-1.816303169604883e-06) [Z5 X10 Z11 X12]
+ (-1.6923978285816957e-06) [Y4 Z5 Y6 Z10]
+ (-1.6923978285816957e-06) [X4 Z5 X6 Z10]
+ (-1.6923978285816957e-06) [Y5 Z6 Y7 Z11]
+ (-1.6923978285816957e-06) [X5 Z6 X7 Z11]
+ (-1.6148794137510537e-06) [Z0 Y11 Z12 Y13]
+ (-1.6148794137510537e-06) [Z0 X11 Z12 X13]
+ (-1.6148794137510537e-06) [Z1 Y10 Z11 Y12]
+ (-1.6148794137510537e-06) [Z1 X10 Z11 X12]
+ (-1.5973171977483997e-06) [Z8 Y10 Z11 Y12]
+ (-1.5973171977483997e-06) [Z8 X10 Z11 X12]
+ (-1.5973171977483997e-06) [Z9 Y11 Z12 Y13]
+ (-1.5973171977483997e-06) [Z9 X11 Z12 X13]
+ (-1.4548424491594855e-06) [Y3 X4 X6 Y7]
+ (-1.4548424491594855e-06) [Y3 Y4 Y6 Y7]
+ (-1.4548424491594855e-06) [X3 X4 X6 X7]
+ (-1.4548424491594855e-06) [X3 Y4 Y6 X7]
+ (-1.398044908211719e-06) [Y4 Z5 Y6 Z8]
+ (-1.398044908211719e-06) [X4 Z5 X6 Z8]
+ (-1.398044908211719e-06) [Y5 Z6 Y7 Z9]
+ (-1.398044908211719e-06) [X5 Z6 X7 Z9]
+ (-1.1954890101097644e-06) [Y2 Z3 Y4 Z7]
+ (-1.1954890101097644e-06) [X2 Z3 X4 Z7]
+ (-1.1954890101097644e-06) [Y3 Z4 Y5 Z6]
+ (-1.1954890101097644e-06) [X3 Z4 X5 Z6]
+ (-1.1908508085500005e-06) [Z0 Y3 Z4 Y5]
+ (-1.1908508085500005e-06) [Z0 X3 Z4 X5]
+ (-1.1908508085500005e-06) [Z1 Y2 Z3 Y4]
+ (-1.1908508085500005e-06) [Z1 X2 Z3 X4]
+ (-1.1708301370805298e-06) [Z2 Y5 Z6 Y7]
+ (-1.1708301370805298e-06) [Z2 X5 Z6 X7]
+ (-1.1708301370805298e-06) [Z3 Y4 Z5 Y6]
+ (-1.1708301370805298e-06) [Z3 X4 Z5 X6]
+ (-1.0632283423128846e-06) [Z2 Y10 Z11 Y12]
+ (-1.0632283423128846e-06) [Z2 X10 Z11 X12]
+ (-1.0632283423128846e-06) [Z3 Y11 Z12 Y13]
+ (-1.0632283423128846e-06) [Z3 X11 Z12 X13]
+ (-1.03584776024576e-06) [Y6 X7 X11 Y12]
+ (-1.03584776024576e-06) [Y6 Y7 Y11 Y12]
+ (-1.03584776024576e-06) [X6 X7 X11 X12]
+ (-1.03584776024576e-06) [X6 Y7 Y11 X12]
+ (-9.509249752289128e-07) [Z2 Y4 Z5 Y6]
+ (-9.509249752289128e-07) [Z2 X4 Z5 X6]
+ (-9.509249752289128e-07) [Z3 Y5 Z6 Y7]
+ (-9.509249752289128e-07) [Z3 X5 Z6 X7]
+ (-9.344557775811811e-07) [Z8 Y11 Z12 Y13]
+ (-9.344557775811811e-07) [Z8 X11 Z12 X13]
+ (-9.344557775811811e-07) [Z9 Y10 Z11 Y12]
+ (-9.344557775811811e-07) [Z9 X10 Z11 X12]
+ (-8.337746756090892e-07) [Z0 Y2 Z3 Y4]
+ (-8.337746756090892e-07) [Z0 X2 Z3 X4]
+ (-8.337746756090892e-07) [Z1 Y3 Z4 Y5]
+ (-8.337746756090892e-07) [Z1 X3 Z4 X5]
+ (-7.956895373018767e-07) [Y3 X4 X8 Y9]
+ (-7.956895373018767e-07) [Y3 Y4 Y8 Y9]
+ (-7.956895373018767e-07) [X3 X4 X8 X9]
+ (-7.956895373018767e-07) [X3 Y4 Y8 X9]
+ (-7.76499411874337e-07) [Y2 Z3 Y4 Z5]
+ (-7.76499411874337e-07) [X2 Z3 X4 Z5]
+ (-5.929765816690454e-07) [Z4 Y5 Z6 Y7]
+ (-5.929765816690454e-07) [Z4 X5 Z6 X7]
+ (-5.770052996122635e-07) [Y2 Z3 Y4 Z9]
+ (-5.770052996122635e-07) [X2 Z3 X4 Z9]
+ (-5.770052996122635e-07) [Y3 Z4 Y5 Z8]
+ (-5.770052996122635e-07) [X3 Z4 X5 Z8]
+ (-5.471647744592682e-07) [Y1 Y2 X11 X12]
+ (-5.471647744592682e-07) [X1 X2 Y11 Y12]
+ (-4.838052751143931e-07) [Y5 X6 X8 Y9]
+ (-4.838052751143931e-07) [Y5 Y6 Y8 Y9]
+ (-4.838052751143931e-07) [X5 X6 X8 X9]
+ (-4.838052751143931e-07) [X5 Y6 Y8 X9]
+ (-3.5707613294091137e-07) [Y0 X1 X3 Y4]
+ (-3.5707613294091137e-07) [Y0 Y1 Y3 Y4]
+ (-3.5707613294091137e-07) [X0 X1 X3 X4]
+ (-3.5707613294091137e-07) [X0 Y1 Y3 X4]
+ (-2.4473231289760994e-07) [Y0 X1 X5 Y6]
+ (-2.4473231289760994e-07) [Y0 Y1 Y5 Y6]
+ (-2.4473231289760994e-07) [X0 X1 X5 X6]
+ (-2.4473231289760994e-07) [X0 Y1 Y5 X6]
+ (-2.1990516185161713e-07) [Y2 X3 X5 Y6]
+ (-2.1990516185161713e-07) [Y2 Y3 Y5 Y6]
+ (-2.1990516185161713e-07) [X2 X3 X5 X6]
+ (-2.1990516185161713e-07) [X2 Y3 Y5 X6]
+ (-1.9332412771928585e-07) [Y1 X2 X3 Y4]
+ (-1.9332412771928585e-07) [X1 Y2 Y3 X4]
+ (-1.2919694861001657e-07) [Y1 Z2 Z3 Y5]
+ (-1.2919694861001657e-07) [X1 Z2 Z3 X5]
+ (1.7379332626061525e-07) [Y0 Z1 Z3 Y4]
+ (1.7379332626061525e-07) [X0 Z1 Z3 X4]
+ (1.7379332626061525e-07) [Y1 Z2 Z4 Y5]
+ (1.7379332626061525e-07) [X1 Z2 Z4 X5]
+ (1.9332412771928585e-07) [Y1 Y2 X3 X4]
+ (1.9332412771928585e-07) [X1 X2 Y3 Y4]
+ (2.186842376896132e-07) [Y2 Z3 Y4 Z8]
+ (2.186842376896132e-07) [X2 Z3 X4 Z8]
+ (2.186842376896132e-07) [Y3 Z4 Y5 Z9]
+ (2.186842376896132e-07) [X3 Z4 X5 Z9]
+ (2.5935343904972114e-07) [Y2 Z3 Y4 Z6]
+ (2.5935343904972114e-07) [X2 Z3 X4 Z6]
+ (2.5935343904972114e-07) [Y3 Z4 Y5 Z7]
+ (2.5935343904972114e-07) [X3 Z4 X5 Z7]
+ (3.606071868267709e-07) [Y0 Z1 Z2 Y4]
+ (3.606071868267709e-07) [X0 Z1 Z2 X4]
+ (3.606071868267709e-07) [Y1 Z3 Z4 Y5]
+ (3.606071868267709e-07) [X1 Z3 Z4 X5]
+ (5.471647744592682e-07) [Y1 X2 X11 Y12]
+ (5.471647744592682e-07) [X1 Y2 Y11 X12]
+ (5.62785191145815e-07) [Y0 X1 X11 Y12]
+ (5.62785191145815e-07) [Y0 Y1 Y11 Y12]
+ (5.62785191145815e-07) [X0 X1 X11 X12]
+ (5.62785191145815e-07) [X0 Y1 Y11 X12]
+ (6.628614201672186e-07) [Y8 X9 X11 Y12]
+ (6.628614201672186e-07) [Y8 Y9 Y11 Y12]
+ (6.628614201672186e-07) [X8 X9 X11 X12]
+ (6.628614201672186e-07) [X8 Y9 Y11 X12]
+ (1.1094407591836322e-06) [Z2 Y11 Z12 Y13]
+ (1.1094407591836322e-06) [Z2 X11 Z12 X13]
+ (1.1094407591836322e-06) [Z3 Y10 Z11 Y12]
+ (1.1094407591836322e-06) [Z3 X10 Z11 X12]
+ (1.60211674058712e-06) [Z2 Y3 Z4 Y5]
+ (1.60211674058712e-06) [Z2 X3 Z4 X5]
+ (1.8782101247284464e-06) [Z4 Y10 Z11 Y12]
+ (1.8782101247284464e-06) [Z4 X10 Z11 X12]
+ (1.8782101247284464e-06) [Z5 Y11 Z12 Y13]
+ (1.8782101247284464e-06) [Z5 X11 Z12 X13]
+ (2.172669101496517e-06) [Y2 X3 X11 Y12]
+ (2.172669101496517e-06) [Y2 Y3 Y11 Y12]
+ (2.172669101496517e-06) [X2 X3 X11 X12]
+ (2.172669101496517e-06) [X2 Y3 Y11 X12]
+ (3.1174479463578105e-06) [Y0 Z2 Z3 Y4]
+ (3.1174479463578105e-06) [X0 Z2 Z3 X4]
+ (3.5390541844374556e-06) [Y2 Z3 Y4 Z12]
+ (3.5390541844374556e-06) [X2 Z3 X4 Z12]
+ (3.5390541844374556e-06) [Y3 Z4 Y5 Z13]
+ (3.5390541844374556e-06) [X3 Z4 X5 Z13]
+ (4.281913884792363e-06) [Y4 Z5 Y6 Z11]
+ (4.281913884792363e-06) [X4 Z5 X6 Z11]
+ (4.281913884792363e-06) [Y5 Z6 Y7 Z10]
+ (4.281913884792363e-06) [X5 Z6 X7 Z10]
+ (5.275883122074964e-06) [Y3 X4 X12 Y13]
+ (5.275883122074964e-06) [Y3 Y4 Y12 Y13]
+ (5.275883122074964e-06) [X3 X4 X12 X13]
+ (5.275883122074964e-06) [X3 Y4 Y12 X13]
+ (5.97431171337406e-06) [Y5 X6 X10 Y11]
+ (5.97431171337406e-06) [Y5 Y6 Y10 Y11]
+ (5.97431171337406e-06) [X5 X6 X10 X11]
+ (5.97431171337406e-06) [X5 Y6 Y10 X11]
+ (7.954413176230292e-06) [Y10 Z11 Y12 Z13]
+ (7.954413176230292e-06) [X10 Z11 X12 Z13]
+ (8.81493730651242e-06) [Y2 Z3 Y4 Z13]
+ (8.81493730651242e-06) [X2 Z3 X4 Z13]
+ (8.81493730651242e-06) [Y3 Z4 Y5 Z12]
+ (8.81493730651242e-06) [X3 Z4 X5 Z12]
+ (0.0002921986261110458) [Y7 X8 X9 Y10]
+ (0.0002921986261110458) [X7 Y8 Y9 X10]
+ (0.0004956762314916296) [Y2 Z4 Z5 Y6]
+ (0.0004956762314916296) [X2 Z4 Z5 X6]
+ (0.0011059037691896853) [Y0 Z1 Y2 Z5]
+ (0.0011059037691896853) [X0 Z1 X2 Z5]
+ (0.0011059037691896853) [Y1 Z2 Y3 Z4]
+ (0.0011059037691896853) [X1 Z2 X3 Z4]
+ (0.0016638798784907626) [Y2 Z3 Z4 Y6]
+ (0.0016638798784907626) [X2 Z3 Z4 X6]
+ (0.0016638798784907626) [Y3 Z5 Z6 Y7]
+ (0.0016638798784907626) [X3 Z5 Z6 X7]
+ (0.0017560707018412312) [Y0 Z1 Y2 Z11]
+ (0.0017560707018412312) [X0 Z1 X2 Z11]
+ (0.0017560707018412312) [Y1 Z2 Y3 Z10]
+ (0.0017560707018412312) [X1 Z2 X3 Z10]
+ (0.002326230623158082) [Y0 Z1 Y2 Z13]
+ (0.002326230623158082) [X0 Z1 X2 Z13]
+ (0.002326230623158082) [Y1 Z2 Y3 Z12]
+ (0.002326230623158082) [X1 Z2 X3 Z12]
+ (0.002745836470186818) [Y0 X1 X4 Y5]
+ (0.002745836470186818) [X0 Y1 Y4 X5]
+ (0.002929768674751051) [Y0 Z1 Y2 Z9]
+ (0.002929768674751051) [X0 Z1 X2 Z9]
+ (0.002929768674751051) [Y1 Z2 Y3 Z8]
+ (0.002929768674751051) [X1 Z2 X3 Z8]
+ (0.0032769719312316665) [Y0 Z1 Y2 Z3]
+ (0.0032769719312316665) [X0 Z1 X2 Z3]
+ (0.003347617530666188) [Y0 Z1 Y2 Z7]
+ (0.003347617530666188) [X0 Z1 X2 Z7]
+ (0.003347617530666188) [Y1 Z2 Y3 Z6]
+ (0.003347617530666188) [X1 Z2 X3 Z6]
+ (0.003555290195504251) [Y0 Z1 Y2 Z10]
+ (0.003555290195504251) [X0 Z1 X2 Z10]
+ (0.003555290195504251) [Y1 Z2 Y3 Z11]
+ (0.003555290195504251) [X1 Z2 X3 Z11]
+ (0.005143391768825094) [Y3 Y4 X5 X6]
+ (0.005143391768825094) [X3 X4 Y5 Y6]
+ (0.005283776488402964) [Y0 X1 X12 Y13]
+ (0.005283776488402964) [X0 Y1 Y12 X13]
+ (0.005530759218631547) [Y0 Z1 Y2 Z4]
+ (0.005530759218631547) [X0 Z1 X2 Z4]
+ (0.005530759218631547) [Y1 Z2 Y3 Z5]
+ (0.005530759218631547) [X1 Z2 X3 Z5]
+ (0.006087822480561852) [Y8 X9 X12 Y13]
+ (0.006087822480561852) [X8 Y9 Y12 X13]
+ (0.006509361201177248) [Y0 X1 X8 Y9]
+ (0.006509361201177248) [X0 Y1 Y8 X9]
+ (0.006888194352970577) [Y0 X1 X6 Y7]
+ (0.006888194352970577) [X0 Y1 Y6 X7]
+ (0.0069012382497972875) [Y0 Z1 Y2 Z12]
+ (0.0069012382497972875) [X0 Z1 X2 Z12]
+ (0.0069012382497972875) [Y1 Z2 Y3 Z13]
+ (0.0069012382497972875) [X1 Z2 X3 Z13]
+ (0.007156934919856946) [Y4 X5 X8 Y9]
+ (0.007156934919856946) [X4 Y5 Y8 X9]
+ (0.007731425250775297) [Y0 X1 X10 Y11]
+ (0.007731425250775297) [X0 Y1 Y10 X11]
+ (0.008032520918821404) [Y0 Z1 Y2 Z6]
+ (0.008032520918821404) [X0 Z1 X2 Z6]
+ (0.008032520918821404) [Y1 Z2 Y3 Z7]
+ (0.008032520918821404) [X1 Z2 X3 Z7]
+ (0.009560705729135926) [Y8 X9 X10 Y11]
+ (0.009560705729135926) [X8 Y9 Y10 X11]
+ (0.011055020596132094) [Y0 Z1 Y2 Z8]
+ (0.011055020596132094) [X0 Z1 X2 Z8]
+ (0.011055020596132094) [Y1 Z2 Y3 Z9]
+ (0.011055020596132094) [X1 Z2 X3 Z9]
+ (0.011285190200840903) [Y5 Y6 X11 X12]
+ (0.011285190200840903) [X5 X6 Y11 Y12]
+ (0.011307274008848199) [Y7 Z8 Z9 Y11]
+ (0.011307274008848199) [X7 Z8 Z9 X11]
+ (0.011982389010247941) [Y4 X5 X6 Y7]
+ (0.011982389010247941) [X4 Y5 Y6 X7]
+ (0.013873381748426089) [Y6 X7 X8 Y9]
+ (0.013873381748426089) [X6 Y7 Y8 X9]
+ (0.014583648907612726) [Y0 X1 X2 Y3]
+ (0.014583648907612726) [X0 Y1 Y2 X3]
+ (0.01557720806397644) [Y2 X3 X12 Y13]
+ (0.01557720806397644) [X2 Y3 Y12 X13]
+ (0.017366118994651365) [Y6 X7 X12 Y13]
+ (0.017366118994651365) [X6 Y7 Y12 X13]
+ (0.017680067952481514) [Y4 X5 X10 Y11]
+ (0.017680067952481514) [X4 Y5 Y10 X11]
+ (0.017825140995786467) [Y6 X7 X10 Y11]
+ (0.017825140995786467) [X6 Y7 Y10 X11]
+ (0.01902824244384726) [Y3 X4 X11 Y12]
+ (0.01902824244384726) [X3 Y4 Y11 X12]
+ (0.025384657508457368) [Y2 X3 X10 Y11]
+ (0.025384657508457368) [X2 Y3 Y10 X11]
+ (0.02868518371610595) [Y10 X11 X12 Y13]
+ (0.02868518371610595) [X10 Y11 Y12 X13]
+ (0.029812424517345823) [Y6 Z7 Z8 Y10]
+ (0.029812424517345823) [X6 Z7 Z8 X10]
+ (0.029812424517345823) [Y7 Z9 Z10 Y11]
+ (0.029812424517345823) [X7 Z9 Z10 X11]
+ (0.03010462314345687) [Y6 Z7 Z9 Y10]
+ (0.03010462314345687) [X6 Z7 Z9 X10]
+ (0.03010462314345687) [Y7 Z8 Z10 Y11]
+ (0.03010462314345687) [X7 Z8 Z10 X11]
+ (0.030787505389143963) [Y6 Z8 Z9 Y10]
+ (0.030787505389143963) [X6 Z8 Z9 X10]
+ (0.031143817988967117) [Y2 X3 X6 Y7]
+ (0.031143817988967117) [X2 Y3 Y6 X7]
+ (0.03583956795335342) [Y2 X3 X4 Y5]
+ (0.03583956795335342) [X2 Y3 Y4 X5]
+ (0.03619412355904263) [Y2 X3 X8 Y9]
+ (0.03619412355904263) [X2 Y3 Y8 X9]
+ (0.03831467029480388) [Y4 X5 X12 Y13]
+ (0.03831467029480388) [X4 Y5 Y12 X13]
+ (0.10433064780651444) [Z0 Y1 Z2 Y3]
+ (0.10433064780651444) [Z0 X1 Z2 X3]
+ (-0.12133276911042327) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133276911042327) [X3 Z4 Z5 Z6 X7]
+ (-0.12133276911042326) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042326) [X2 Z3 Z4 Z5 X6]
+ (3.2020768811879885e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.2020768811879885e-06) [X1 Z2 Z3 Z4 X5]
+ (3.202076881187989e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.202076881187989e-06) [X0 Z1 Z2 Z3 X4]
+ (0.2284810656491882) [Y7 Z8 Z9 Z10 Y11]
+ (0.2284810656491882) [X7 Z8 Z9 Z10 X11]
+ (0.22848106564918824) [Y6 Z7 Z8 Z9 Y10]
+ (0.22848106564918824) [X6 Z7 Z8 Z9 X10]
+ (-0.032767657823290455) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.032767657823290455) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.032767657823290455) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.032767657823290455) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273097) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.027115036845273097) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273097) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.027115036845273097) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.025996177598021156) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.025996177598021156) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.017561202409646127) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.017561202409646127) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.017561202409646127) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.017561202409646127) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.014564531231172989) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564531231172989) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564531231172989) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564531231172989) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.012215040997613941) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.012215040997613941) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.012215040997613941) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.012215040997613941) [X4 Z5 X6 X11 Z12 X13]
+ (-0.012215040997613941) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.012215040997613941) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.012215040997613941) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.012215040997613941) [X5 Z6 X7 X10 Z11 X12]
+ (-0.011756013419819229) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.011756013419819229) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.011756013419819229) [X3 Z4 Z5 X6 X8 X9]
+ (-0.011756013419819229) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.008764827575688746) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.008764827575688746) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.008764827575688746) [X2 Z3 Z4 X5 X11 X12]
+ (-0.008764827575688746) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.008764827575688746) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.008764827575688746) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.008764827575688746) [X3 X4 X10 Z11 Z12 X13]
+ (-0.008764827575688746) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.008125251921381043) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125251921381043) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.007306759928832953) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306759928832953) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306759928832953) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306759928832953) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005805188989826897) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.005805188989826897) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.005805188989826897) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.005805188989826897) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.005652620978017359) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.005652620978017359) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.005652620978017359) [X0 X1 X3 Z4 Z5 X6]
+ (-0.005652620978017359) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.005143391768825093) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.005143391768825093) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.005143391768825093) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.005143391768825093) [X2 Z3 X4 X5 Z6 X7]
+ (-0.004684903388155216) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.004684903388155216) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668620318776311) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668620318776311) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.0044248554494418614) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.0044248554494418614) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.004158797381840054) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.004158797381840054) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.004158797381840054) [X3 Z4 Z5 X6 X12 X13]
+ (-0.004158797381840054) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.0034937903598901312) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.0034937903598901312) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.0034937903598901312) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.0034937903598901312) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.002779026799025525) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.002779026799025525) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.0022939566113524736) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.0022939566113524736) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.0017992194936630199) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.0017992194936630199) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278753941369642) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.0017278753941369642) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298507967730374) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298507967730374) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298507967730374) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298507967730374) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533856254125499) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533856254125499) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008145313270956685) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008145313270956685) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008145313270956685) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008145313270956685) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.735036880590427e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735036880590427e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735036880590427e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735036880590427e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.774817864415375e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.774817864415375e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.774817864415375e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.774817864415375e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.5183622155696954e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.5183622155696954e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.5183622155696954e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.5183622155696954e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.444344675751336e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.444344675751336e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.444344675751336e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.444344675751336e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.524373848441078e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.524373848441078e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.524373848441078e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.524373848441078e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.290028432996987e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.290028432996987e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.290028432996987e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.290028432996987e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.974311713374059e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.974311713374059e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.275883122074963e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.275883122074963e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.643051068392567e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.643051068392567e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.556569218031597e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.556569218031597e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.253224225560565e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.253224225560565e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.769659451758355e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.769659451758355e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.6945132943333296e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.6945132943333296e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.6102971303535964e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.6102971303535964e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.6102971303535964e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.6102971303535964e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.313145500194655e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.313145500194655e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.2774831952720154e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.2774831952720154e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.2774831952720154e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.2774831952720154e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.211228348246422e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.211228348246422e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.211228348246422e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.211228348246422e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.1513463111204906e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.1513463111204906e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.0882507112013755e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.0882507112013755e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.1726691014965167e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.1726691014965167e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.4548424491594857e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.4548424491594857e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.3304731886640386e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.3304731886640386e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2283337825727083e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2283337825727083e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.03584776024576e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.03584776024576e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956895373018765e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956895373018765e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733197741891969e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733197741891969e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733197741891969e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733197741891969e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.628614201672188e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.628614201672188e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.556281914479791e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.556281914479791e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.556281914479791e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.556281914479791e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.418291574499632e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.418291574499632e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.418291574499632e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.418291574499632e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.927453082404589e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927453082404589e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927453082404589e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927453082404589e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.62785191145815e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.62785191145815e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660624623611e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660624623611e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660624623611e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660624623611e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660624623611e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660624623611e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660624623611e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660624623611e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.838052751143931e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.838052751143931e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.5707613294091137e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.5707613294091137e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.3281393508158056e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.3281393508158056e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.0868265652697486e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.0868265652697486e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.0868265652697486e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.0868265652697486e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.4473231289760994e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.4473231289760994e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.3713289480192317e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.3713289480192317e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.3713289480192317e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.3713289480192317e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.1990516185161713e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.1990516185161713e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.9332412771928585e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.9332412771928585e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.9332412771928585e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.9332412771928585e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.8394209155761683e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.8394209155761683e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.8394209155761683e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.8394209155761683e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.5510539177014126e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.5510539177014126e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.5510539177014126e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.5510539177014126e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.3807781481720385e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.3807781481720385e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.3807781481720385e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.3807781481720385e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.3807781481720385e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.3807781481720385e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.3807781481720385e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.3807781481720385e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.3807781481720385e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.3807781481720385e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.3807781481720385e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.3807781481720385e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.2919694861001657e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.2919694861001657e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.107632559881565e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.107632559881565e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.107632559881565e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.107632559881565e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.107632559881565e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.107632559881565e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.107632559881565e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.107632559881565e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.057446594873812e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057446594873812e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057446594873812e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057446594873812e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649310137902584e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649310137902584e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649310137902584e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649310137902584e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.8394209155761683e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.8394209155761683e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.8394209155761683e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.8394209155761683e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.1990516185161713e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.1990516185161713e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.4473231289760994e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.4473231289760994e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.23625996180949e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.23625996180949e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.23625996180949e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.23625996180949e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.3281393508158056e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.3281393508158056e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.5707613294091137e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.5707613294091137e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.838052751143931e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.838052751143931e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.62785191145815e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.62785191145815e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.628614201672188e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.628614201672188e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956895373018765e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956895373018765e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.306536652059584e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.306536652059584e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.306536652059584e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.306536652059584e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.03584776024576e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.03584776024576e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2283337825727083e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2283337825727083e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.2393363217329332e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393363217329332e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393363217329332e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393363217329332e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.3304731886640386e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.3304731886640386e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.4548424491594857e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.4548424491594857e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.1726691014965167e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.1726691014965167e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.0882507112013755e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.0882507112013755e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.1174479463578105e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.1174479463578105e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.1513463111204906e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.1513463111204906e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.313145500194655e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.313145500194655e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.334331289418592e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.334331289418592e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.6945132943333296e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.6945132943333296e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.183932559487133e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.183932559487133e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.556569218031597e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.556569218031597e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.643051068392567e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.643051068392567e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.275883122074963e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.275883122074963e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.974311713374059e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.974311713374059e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.00029219862611104574) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.00029219862611104574) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.00029219862611104574) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.00029219862611104574) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004956762314916296) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004956762314916296) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650070219499229) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650070219499229) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650070219499229) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650070219499229) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533856254125499) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533856254125499) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.0016095313817213915) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.0016095313817213915) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.0016095313817213915) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.0016095313817213915) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0016676041811440724) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.0016676041811440724) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.0016676041811440724) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.0016676041811440724) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.0017278753941369642) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.0017278753941369642) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.0017992194936630199) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.0017992194936630199) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.0022939566113524736) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.0022939566113524736) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.0024629170071339412) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.0024629170071339412) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.0024629170071339412) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.0024629170071339412) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.003961560792496546) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.003961560792496546) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.003961560792496546) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.003961560792496546) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.0044248554494418614) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.0044248554494418614) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004668620318776311) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668620318776311) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.004684903388155216) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.004684903388155216) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005324835234221674) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.005324835234221674) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.005324835234221674) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.005324835234221674) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.005368659358109569) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368659358109569) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368659358109569) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368659358109569) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.007960880725921573) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.007960880725921573) [X4 Z5 X6 X10 Z11 X12]
+ (0.007960880725921573) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.007960880725921573) [X5 Z6 X7 X11 Z12 X13]
+ (0.008125251921381043) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125251921381043) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.008890731522694609) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.008890731522694609) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.008890731522694609) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.008890731522694609) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.010263414868158512) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.010263414868158512) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.010263414868158512) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.010263414868158512) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.010540425907671529) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.010540425907671529) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.010540425907671529) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.010540425907671529) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010960074940542632) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010960074940542632) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010960074940542632) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010960074940542632) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307274008848199) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307274008848199) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.01441109943013091) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.01441109943013091) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.01441109943013091) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.01441109943013091) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.015225630757226577) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.015225630757226577) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.015225630757226577) [X3 Z4 Z5 X6 X10 X11]
+ (0.015225630757226577) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.015588250102380186) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.015588250102380186) [X2 Z3 X4 X10 Z11 X12]
+ (0.015588250102380186) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.015588250102380186) [X3 Z4 X5 X11 Z12 X13]
+ (0.018266834869375585) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.018266834869375585) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.018266834869375585) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.018266834869375585) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.019020423173039983) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.019020423173039983) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.019020423173039983) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.019020423173039983) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.020175921723535512) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.020175921723535512) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.020175921723535512) [X4 Z5 Z6 X7 X11 X12]
+ (0.020175921723535512) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.020175921723535512) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.020175921723535512) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.020175921723535512) [X5 X6 X10 Z11 Z12 X13]
+ (0.020175921723535512) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.024353077678068935) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.024353077678068935) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.024353077678068935) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.024353077678068935) [X2 Z3 X4 X11 Z12 X13]
+ (0.024353077678068935) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.024353077678068935) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.024353077678068935) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.024353077678068935) [X3 Z4 X5 X10 Z11 X12]
+ (0.024389082531149554) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.024389082531149554) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.024389082531149554) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.024389082531149554) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.025104957138844516) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.025104957138844516) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.025104957138844516) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.025104957138844516) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.030787505389143963) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.030787505389143963) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.045879470781298115) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.045879470781298115) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.05600733087780778) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.05600733087780778) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.05600733087780778) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.05600733087780778) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.056084681246613685) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.056084681246613685) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.056084681246613685) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.056084681246613685) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.631277928248335e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631277928248335e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.631277928248334e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631277928248334e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.595086006791916e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.595086006791916e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.5950860067919147e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.5950860067919147e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0427432770137838) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.0427432770137838) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (0.0427432770137838) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.0427432770137838) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.04764261217638312) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.04764261217638312) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.04764261217638312) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.04764261217638312) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.041718813839821775) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.041718813839821775) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.041718813839821775) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.041718813839821775) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.03956441632289337) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.03956441632289337) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.03956441632289337) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03956441632289337) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03935916802205308) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.03935916802205308) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.03935916802205308) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.03935916802205308) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.03931805194719757) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03931805194719757) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.03931805194719757) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03931805194719757) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03560837898831259) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.03560837898831259) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.029903789512624835) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.029903789512624835) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.029903789512624835) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.029903789512624835) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.028730779551905547) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.028730779551905547) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.028730779551905547) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.028730779551905547) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.02563723829602682) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.02563723829602682) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.02563723829602682) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.02563723829602682) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.024755463292890988) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.024755463292890988) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.024755463292890988) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.024755463292890988) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.024282117354692975) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.024282117354692975) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.023145130929529006) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.023145130929529006) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.022528440196013043) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.022528440196013043) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.021433810721600936) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.021433810721600936) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.021433810721600936) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.021433810721600936) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251565) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.019257505095251565) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.01902824244384726) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.01902824244384726) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.018889030304942947) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.018889030304942947) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.018889030304942947) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.018889030304942947) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.016024603689179562) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.016024603689179562) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.015225630757226579) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225630757226579) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.0146037047291621) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.0146037047291621) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564531231172987) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564531231172987) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.011756013419819229) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.011756013419819229) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.011285190200840903) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.011285190200840903) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.009841749246962597) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009841749246962597) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.009612634606847257) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612634606847257) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612634606847257) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612634606847257) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.008469978791023895) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.008469978791023895) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.007306759928832951) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.007306759928832951) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005923798336561342) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.005923798336561342) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.00565262097801736) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.00565262097801736) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368659358109569) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368659358109569) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.004158797381840054) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.004158797381840054) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0033566705638328857) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.0033566705638328857) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.0033566705638328857) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.0033566705638328857) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.0032675138544235507) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.0032675138544235507) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.0032675138544235507) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.0032675138544235507) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.002779026799025525) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.002779026799025525) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.0026860409778066132) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.0026860409778066132) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.0026860409778066132) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.0026860409778066132) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.002293956611352473) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.002293956611352473) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.002293956611352473) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.002293956611352473) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.0009581655836696486) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.0009581655836696486) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0009581655836696486) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.0009581655836696486) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.0009581655836696486) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.0009581655836696486) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0009581655836696486) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.0009581655836696486) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.00024636437569580184) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00024636437569580184) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0001384017730354993) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.0001384017730354993) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.0001384017730354993) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.0001384017730354993) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.735036880590427e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735036880590427e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.610358530540232e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.610358530540232e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.610358530540232e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.610358530540232e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808794983353e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.5316808794983353e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808794983353e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5316808794983353e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.806102775041242e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.806102775041242e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.806102775041242e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.806102775041242e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.089799467389211e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.089799467389211e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.089799467389211e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.089799467389211e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.6522096690326225e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.6522096690326225e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.6522096690326225e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.6522096690326225e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851833471253e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.481851833471253e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851833471253e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.481851833471253e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.071480736432727e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.071480736432727e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.071480736432727e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.071480736432727e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.7346220386085145e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.7346220386085145e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.7346220386085145e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.7346220386085145e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.728843147105625e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.728843147105625e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.728843147105625e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.728843147105625e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.253224225560565e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.253224225560565e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.769659451758355e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.769659451758355e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.544395429229459e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.544395429229459e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.544395429229459e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.544395429229459e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.544395429229459e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.544395429229459e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.544395429229459e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.544395429229459e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.360956320283586e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.360956320283586e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.360956320283586e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.360956320283586e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.1032156045435596e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.1032156045435596e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.1032156045435596e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.1032156045435596e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220980444315e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.0111220980444315e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220980444315e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.0111220980444315e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9429468365254308e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.9429468365254308e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9429468365254308e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9429468365254308e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174769326956e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.6541174769326956e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174769326956e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.6541174769326956e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.522493067621554e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.522493067621554e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.522493067621554e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.522493067621554e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.522493067621554e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.522493067621554e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.522493067621554e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.522493067621554e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.2283337825727083e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337825727083e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.2283337825727083e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337825727083e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.988770289126634e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988770289126634e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988770289126634e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988770289126634e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.867765104189646e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104189646e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104189646e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.867765104189646e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.189990975103717e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.189990975103717e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246206993777e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246206993777e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.471647744592682e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.471647744592682e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.561447180280834e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.561447180280834e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.561447180280834e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.561447180280834e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.5233896775354433e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.5233896775354433e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.427323108845797e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.427323108845797e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.427323108845797e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.427323108845797e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.328139350815805e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350815805e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.328139350815805e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350815805e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.0868265652697486e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.0868265652697486e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.8882935959273505e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935959273505e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935959273505e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.8882935959273505e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.3713289480192317e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.3713289480192317e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.8394209155761683e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.8394209155761683e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.057446594873812e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057446594873812e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.5371780954898e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.5371780954898e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.5371780954898e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.5371780954898e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.057446594873812e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057446594873812e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.209350649912805e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350649912805e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350649912805e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.209350649912805e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.703578355613688e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.703578355613688e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.703578355613688e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.703578355613688e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.8394209155761683e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.8394209155761683e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.3713289480192317e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.3713289480192317e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.0868265652697486e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.0868265652697486e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.5233896775354433e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.5233896775354433e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.471647744592682e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.471647744592682e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246206993777e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246206993777e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189990975103717e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.189990975103717e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.3304731886640386e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.3304731886640386e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.3304731886640386e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.3304731886640386e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288532434989368e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288532434989368e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288532434989368e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288532434989368e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.6893489514375483e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.6893489514375483e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.6893489514375483e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.6893489514375483e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.7455184003167955e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.7455184003167955e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.7455184003167955e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.7455184003167955e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.7455184003167955e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.7455184003167955e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.7455184003167955e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.7455184003167955e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.211842019059102e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.211842019059102e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.211842019059102e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.211842019059102e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.211842019059102e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.211842019059102e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.211842019059102e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.211842019059102e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.3131455001946553e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.3131455001946553e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.3131455001946553e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.3131455001946553e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.334331289418592e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.334331289418592e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.183932559487133e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.183932559487133e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.735036880590427e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735036880590427e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00024636437569580184) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00024636437569580184) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00044585351288408325) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.00044585351288408325) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.00044585351288408325) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.00044585351288408325) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940221543005389) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005389) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005389) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005389) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005389) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005389) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940221543005389) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005389) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.00085338562541255) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.00085338562541255) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.00085338562541255) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.00085338562541255) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0010435246534907707) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435246534907707) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435246534907707) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435246534907707) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.001280306097349688) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.001280306097349688) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.001280306097349688) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.001280306097349688) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.001303800478812699) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.001303800478812699) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.001303800478812699) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.001303800478812699) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.0022619660624823477) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.0022619660624823477) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.0022619660624823477) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.0022619660624823477) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.0022619660624823477) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.0022619660624823477) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.0022619660624823477) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.0022619660624823477) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.003989841456619312) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.003989841456619312) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.003989841456619312) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.003989841456619312) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.004158797381840054) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.004158797381840054) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.004311038507914321) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.004311038507914321) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.004311038507914321) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.004311038507914321) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.004636976661182574) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.004636976661182574) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.004636976661182574) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.004636976661182574) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.005241535382803877) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.005241535382803877) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.005241535382803877) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.005241535382803877) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.00526264247307685) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.00526264247307685) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.00526264247307685) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.00526264247307685) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368659358109569) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368659358109569) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005379937155839376) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.005379937155839376) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.005379937155839376) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.005379937155839376) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.00565262097801736) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.00565262097801736) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.005708495985960933) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708495985960933) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708495985960933) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708495985960933) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.005923798336561342) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.005923798336561342) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.007306759928832951) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.007306759928832951) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469978791023895) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.008469978791023895) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.009841749246962597) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.009841749246962597) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.011285190200840903) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.011285190200840903) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.011756013419819229) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.011756013419819229) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.014564531231172987) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564531231172987) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.0146037047291621) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.0146037047291621) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.015225630757226579) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225630757226579) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.016024603689179562) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.016024603689179562) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.01902824244384726) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.01902824244384726) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.019257505095251565) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.019257505095251565) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.045879470781298115) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.045879470781298115) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.3693708936615614) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.3693708936615614) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.3693708936615614) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.3693708936615614) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.281642577670229) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.281642577670229) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.2816425776702289) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.2816425776702289) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.09065144207036488) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.09065144207036488) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.09065144207036488) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.09065144207036488) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863634) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684737589863634) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863634) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684737589863634) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635021950635018) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635021950635018) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635021950635018) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635021950635018) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214031) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.06752385099214031) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214031) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.06752385099214031) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03560837898831259) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.03560837898831259) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03490334337366182) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03490334337366182) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03490334337366182) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03490334337366182) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883830002) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.024591860883830002) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883830002) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.024591860883830002) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.024282117354692975) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.024282117354692975) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.02314513092952901) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.02314513092952901) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.022528440196013043) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.022528440196013043) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019538050311314725) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.019538050311314725) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.019538050311314725) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.019538050311314725) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.017091553155898862) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.017091553155898862) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.017091553155898862) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.017091553155898862) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.016024603689179562) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.016024603689179562) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.016024603689179562) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.016024603689179562) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.010311482489831816) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831816) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831816) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311482489831816) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.009841749246962597) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962597) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.009841749246962597) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962597) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209874) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209874) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209874) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008826368514209874) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008541996625454818) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454818) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454818) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454818) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454818) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454818) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454818) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008541996625454818) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008469978791023895) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023895) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.008469978791023895) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023895) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.004668620318776311) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668620318776311) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.003876470899336949) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.003876470899336949) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.0038040661717285472) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0038040661717285472) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0038040661717285472) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0038040661717285472) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178904) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178904) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003356670563832886) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.003356670563832886) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.0032675138544235507) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.0032675138544235507) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.002141361223101621) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.002141361223101621) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.0017278753941369642) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.0017278753941369642) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.0016407548553124204) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0016407548553124204) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214169332) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0014528843214169332) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214169332) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0014528843214169332) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.000787089677102449) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.000787089677102449) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192743499487808) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192743499487808) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.00019400857029755684) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00019400857029755684) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0001384017730354993) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.0001384017730354993) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.141625221157405e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.141625221157405e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.141625221157405e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.141625221157405e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.071480736432727e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.071480736432727e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.1513463111204906e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.1513463111204906e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0882507112013755e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.0882507112013755e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.9885117064553473e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.9885117064553473e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.874299071361201e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.874299071361201e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.360956320283586e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.360956320283586e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.3002946562455766e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3002946562455766e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.146837650709554e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.146837650709554e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.146837650709554e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.146837650709554e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.352332102780494e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.352332102780494e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.352332102780494e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.352332102780494e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091637198677806e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637198677806e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637198677806e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637198677806e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637198677806e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637198677806e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637198677806e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091637198677806e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.074305985654853e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.074305985654853e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.074305985654853e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.074305985654853e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128985953176e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128985953176e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128985953176e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128985953176e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867765104189646e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.867765104189646e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560692464608393e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692464608393e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692464608393e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692464608393e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692464608393e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692464608393e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692464608393e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560692464608393e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.997018422013047e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.997018422013047e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.997018422013047e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.997018422013047e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.997018422013047e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.997018422013047e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.997018422013047e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.997018422013047e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.568247521142363e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.568247521142363e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.568247521142363e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.568247521142363e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393084177345e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393084177345e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393084177345e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393084177345e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393084177345e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393084177345e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.3767393084177345e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393084177345e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.8882935959273505e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.8882935959273505e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6863815440518237e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.6863815440518237e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.703578355613688e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.703578355613688e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.209350649912805e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.209350649912805e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773243616198e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773243616198e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773243616198e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773243616198e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773243616198e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773243616198e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773243616198e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773243616198e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253792208115e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253792208115e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9742253792208115e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9742253792208115e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0474716554700008e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.0474716554700008e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.0474716554700008e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.0474716554700008e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.209350649912805e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.209350649912805e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0717282182541914e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0717282182541914e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0717282182541914e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0717282182541914e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.200428749384988e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.200428749384988e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.200428749384988e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.200428749384988e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.703578355613688e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.703578355613688e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.3120943051671605e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.3120943051671605e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.3120943051671605e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.3120943051671605e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.6863815440518237e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815440518237e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.8882935959273505e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.8882935959273505e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.0922506161191184e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506161191184e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.0922506161191184e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506161191184e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.0922506161191184e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506161191184e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.0922506161191184e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506161191184e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.4445978542687397e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.4445978542687397e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.4445978542687397e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.4445978542687397e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.684915095214079e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.684915095214079e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.684915095214079e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.684915095214079e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246974425247969e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246974425247969e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246974425247969e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246974425247969e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246974425247969e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246974425247969e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246974425247969e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246974425247969e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.867765104189646e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.867765104189646e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3002946562455766e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3002946562455766e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.360956320283586e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.360956320283586e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.874299071361201e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.874299071361201e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.8836765760439955e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.8836765760439955e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.9473560115928563e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9473560115928563e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.9473560115928563e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9473560115928563e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117064553473e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9885117064553473e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.0882507112013755e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.0882507112013755e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.1513463111204906e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.1513463111204906e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.846201671229958e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846201671229958e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846201671229958e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846201671229958e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.071480736432727e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.071480736432727e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.105526721944129e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.105526721944129e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.105526721944129e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.105526721944129e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146496327475534e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146496327475534e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146496327475534e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146496327475534e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159350501945794e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159350501945794e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159350501945794e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159350501945794e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427988656350976e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427988656350976e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427988656350976e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427988656350976e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935867718048203e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935867718048203e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935867718048203e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935867718048203e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348015924e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.253273348015924e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979825793305328e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979825793305328e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979825793305328e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979825793305328e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.20554841121714e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.20554841121714e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.20554841121714e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.20554841121714e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0001384017730354993) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.0001384017730354993) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.00018787053389548747) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.00018787053389548747) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.00018787053389548747) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00018787053389548747) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00019400857029755684) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00019400857029755684) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00024636437569580184) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569580184) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00024636437569580184) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569580184) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192743499487808) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192743499487808) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156734248908749) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156734248908749) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156734248908749) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156734248908749) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.000787089677102449) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.000787089677102449) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.0015324835230730183) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.0015324835230730183) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.0015324835230730183) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.0015324835230730183) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.0016407548553124204) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0016407548553124204) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0017278753941369642) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.0017278753941369642) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.002446497155415862) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.002446497155415862) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.002446497155415862) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.002446497155415862) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.0032675138544235507) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.0032675138544235507) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.003356670563832886) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.003356670563832886) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.0034841573002178904) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0034841573002178904) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.003876470899336949) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.003876470899336949) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.004668620318776311) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668620318776311) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278094) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.004767272188278094) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.004767272188278094) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278094) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.005286546538226875) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.005286546538226875) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.005286546538226875) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.005286546538226875) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.005408954422409967) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.005408954422409967) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.005408954422409967) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.005408954422409967) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.005923798336561342) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561342) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.005923798336561342) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561342) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.010715508469796747) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010715508469796747) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010715508469796747) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010715508469796747) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01075756395390892) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01075756395390892) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.01075756395390892) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01075756395390892) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0146037047291621) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.0146037047291621) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.0146037047291621) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.0146037047291621) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.019299560579363734) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363734) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363734) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363734) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363734) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363734) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.019299560579363734) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363734) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.05859198873386201) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.05859198873386201) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.775950527184045e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.775950527184045e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.7759505271840454e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.7759505271840454e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.07165035181002664) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07165035181002664) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.07165035181002673) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.07165035181002673) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251565) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019257505095251565) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010311482489831818) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311482489831818) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008826368514209875) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209875) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.007597464029770622) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.007597464029770622) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.007597464029770622) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.007597464029770622) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311883) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311883) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311883) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311883) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311883) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311883) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311883) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311883) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0053480515826766295) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0053480515826766295) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0053480515826766295) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0053480515826766295) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0038040661717285472) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0038040661717285472) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219377) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.0029841661681219377) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.0029841661681219377) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.0029841661681219377) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.002446497155415862) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.002446497155415862) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.002249412447093993) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.002249412447093993) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.002249412447093993) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002249412447093993) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0021413612231016214) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.0021413612231016214) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.0018638942824587398) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587398) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587398) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587398) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587398) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587398) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0018638942824587398) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587398) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0016407548553124204) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124204) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0016407548553124204) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124204) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0012223378081538318) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538318) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538318) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538318) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538318) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538318) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538318) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0012223378081538318) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.001028329237856275) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.001028329237856275) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.001028329237856275) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.001028329237856275) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1463061452966016e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1463061452966016e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.874299071361201e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.874299071361201e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.874299071361201e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.874299071361201e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3002946562455766e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3002946562455766e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3002946562455766e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3002946562455766e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.0444941297716266e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.0444941297716266e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.0444941297716266e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.0444941297716266e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.956079229726862e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.956079229726862e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.956079229726862e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.956079229726862e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105515036750275e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105515036750275e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105515036750275e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105515036750275e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661347212834756e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661347212834756e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661347212834756e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661347212834756e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341413648239e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341413648239e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189990975103717e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.189990975103717e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.87662165789879e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.87662165789879e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.87662165789879e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.87662165789879e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246206993777e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246206993777e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.5233896775354433e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.5233896775354433e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0767325318044427e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.0767325318044427e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0767325318044427e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.0767325318044427e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.013471458861918e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.013471458861918e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.904599884068026e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.904599884068026e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.904599884068026e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.904599884068026e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.666731754497526e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.666731754497526e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.666731754497526e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.666731754497526e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.850564192976586e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.850564192976586e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.656930931757105e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.656930931757105e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.656930931757105e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.656930931757105e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.850564192976586e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.850564192976586e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815440518237e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815440518237e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.6863815440518237e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815440518237e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458861918e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.013471458861918e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.5233896775354433e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.5233896775354433e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.6704023906190233e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.6704023906190233e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.6704023906190233e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.6704023906190233e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246206993777e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246206993777e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189990975103717e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.189990975103717e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341413648239e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341413648239e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949476487504514e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949476487504514e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.7924939576920568e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939576920568e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939576920568e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.7924939576920568e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.883676576043995e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.883676576043995e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.9885117064553473e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117064553473e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117064553473e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117064553473e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348015924e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253273348015924e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4017109735115493e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4017109735115493e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4017109735115493e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4017109735115493e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.580960369280755e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.580960369280755e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.580960369280755e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.580960369280755e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192743499487808) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487808) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192743499487808) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487808) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.0007870896771024489) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024489) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024489) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024489) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0011726348316441885) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0011726348316441885) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0011726348316441885) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0011726348316441885) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012366478019245144) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.0012366478019245144) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.0012366478019245144) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.0012366478019245144) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.0022009640695004637) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0022009640695004637) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0022009640695004637) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0022009640695004637) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00239497263979802) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00239497263979802) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00239497263979802) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00239497263979802) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00239497263979802) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.00239497263979802) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.00239497263979802) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.00239497263979802) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002446497155415862) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.002446497155415862) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.0038040661717285472) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0038040661717285472) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.003876470899336949) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.003876470899336949) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.003876470899336949) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.003876470899336949) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.0042208139700464515) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.0042208139700464515) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.0042208139700464515) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.0042208139700464515) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.008826368514209875) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.008826368514209875) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.010311482489831818) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311482489831818) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257505095251565) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019257505095251565) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.05859198873386201) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.05859198873386201) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3987009015808716e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.3987009015808716e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.3987009015808716e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3987009015808716e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178904) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178904) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219373) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.0029841661681219373) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.00019400857029755684) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00019400857029755684) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061452966016e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061452966016e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7924939576920566e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7924939576920566e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540341413648239e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413648239e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540341413648239e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413648239e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.850564192976586e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.850564192976586e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.850564192976586e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.850564192976586e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458861918e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471458861918e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458861918e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471458861918e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949476487504514e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949476487504514e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.7924939576920566e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7924939576920566e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029755684) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029755684) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0029841661681219373) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.0029841661681219373) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.0034841573002178904) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0034841573002178904) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
  (-73.13873231352527) [I0]
+ (-0.18066792656583225) [Z6]
+ (-0.18066792656583225) [Z7]
+ (-0.15961432501809744) [Z5]
+ (-0.1596143250180974) [Z4]
+ (0.17419956155055877) [Z2]
+ (0.17419956155055877) [Z3]
+ (0.22757269005453667) [Z0]
+ (0.22757269005453676) [Z1]
+ (-8.194261372246857e-06) [Y4 Y6]
+ (-8.194261372246857e-06) [X4 X6]
+ (7.954413176027687e-06) [Y5 Y7]
+ (7.954413176027687e-06) [X5 X7]
+ (0.1127038692033218) [Z4 Z6]
+ (0.1127038692033218) [Z5 Z7]
+ (0.1195243896468266) [Z0 Z4]
+ (0.1195243896468266) [Z1 Z5]
+ (0.1340171526196368) [Z0 Z6]
+ (0.1340171526196368) [Z1 Z7]
+ (0.13734953064261313) [Z0 Z5]
+ (0.13734953064261313) [Z1 Z4]
+ (0.13766872645852557) [Z2 Z4]
+ (0.13766872645852557) [Z3 Z5]
+ (0.1413890529194277) [Z4 Z7]
+ (0.1413890529194277) [Z5 Z6]
+ (0.14722943218766146) [Z2 Z5]
+ (0.14722943218766146) [Z3 Z4]
+ (0.14926355147388864) [Z4 Z5]
+ (0.149734868034969) [Z2 Z6]
+ (0.149734868034969) [Z3 Z7]
+ (0.15138327161428822) [Z0 Z7]
+ (0.15138327161428822) [Z1 Z6]
+ (0.15435748657223586) [Z6 Z7]
+ (0.15582269051553083) [Z2 Z7]
+ (0.15582269051553083) [Z3 Z6]
+ (0.16756653265461274) [Z0 Z2]
+ (0.16756653265461274) [Z1 Z3]
+ (0.18143991440303883) [Z0 Z3]
+ (0.18143991440303883) [Z1 Z2]
+ (0.19392534613270207) [Z0 Z1]
+ (-7.037887510555748e-06) [Y5 Z6 Y7]
+ (-7.037887510555748e-06) [X5 Z6 X7]
+ (-7.0378875105557466e-06) [Y4 Z5 Y6]
+ (-7.0378875105557466e-06) [X4 Z5 X6]
+ (-0.028685183716105876) [Y4 Y5 X6 X7]
+ (-0.028685183716105876) [X4 X5 Y6 Y7]
+ (-0.017825140995786512) [Y0 Y1 X4 X5]
+ (-0.017825140995786512) [X0 X1 Y4 Y5]
+ (-0.017366118994651392) [Y0 Y1 X6 X7]
+ (-0.017366118994651392) [X0 X1 Y6 Y7]
+ (-0.013873381748426089) [Y0 Y1 X2 X3]
+ (-0.013873381748426089) [X0 X1 Y2 Y3]
+ (-0.009560705729135905) [Y2 Y3 X4 X5]
+ (-0.009560705729135905) [X2 X3 Y4 Y5]
+ (-0.00608782248056184) [Y2 Y3 X6 X7]
+ (-0.00608782248056184) [X2 X3 Y6 Y7]
+ (-0.000292198626111035) [Y1 Y2 X3 X4]
+ (-0.000292198626111035) [X1 X2 Y3 Y4]
+ (-8.194261372246856e-06) [Z4 Y5 Z6 Y7]
+ (-8.194261372246856e-06) [Z4 X5 Z6 X7]
+ (-2.8909678817524526e-06) [Z0 Y5 Z6 Y7]
+ (-2.8909678817524526e-06) [Z0 X5 Z6 X7]
+ (-2.8909678817524526e-06) [Z1 Y4 Z5 Y6]
+ (-2.8909678817524526e-06) [Z1 X4 Z5 X6]
+ (-1.8551201215903856e-06) [Z0 Y4 Z5 Y6]
+ (-1.8551201215903856e-06) [Z0 X4 Z5 X6]
+ (-1.8551201215903856e-06) [Z1 Y5 Z6 Y7]
+ (-1.8551201215903856e-06) [Z1 X5 Z6 X7]
+ (-1.5973171978515882e-06) [Z2 Y4 Z5 Y6]
+ (-1.5973171978515882e-06) [Z2 X4 Z5 X6]
+ (-1.5973171978515882e-06) [Z3 Y5 Z6 Y7]
+ (-1.5973171978515882e-06) [Z3 X5 Z6 X7]
+ (-1.0358477601620668e-06) [Y0 X1 X5 Y6]
+ (-1.0358477601620668e-06) [Y0 Y1 Y5 Y6]
+ (-1.0358477601620668e-06) [X0 X1 X5 X6]
+ (-1.0358477601620668e-06) [X0 Y1 Y5 X6]
+ (-9.344557776930823e-07) [Z2 Y5 Z6 Y7]
+ (-9.344557776930823e-07) [Z2 X5 Z6 X7]
+ (-9.344557776930823e-07) [Z3 Y4 Z5 Y6]
+ (-9.344557776930823e-07) [Z3 X4 Z5 X6]
+ (6.62861420158506e-07) [Y2 X3 X5 Y6]
+ (6.62861420158506e-07) [Y2 Y3 Y5 Y6]
+ (6.62861420158506e-07) [X2 X3 X5 X6]
+ (6.62861420158506e-07) [X2 Y3 Y5 X6]
+ (7.954413176027687e-06) [Y4 Z5 Y6 Z7]
+ (7.954413176027687e-06) [X4 Z5 X6 Z7]
+ (0.000292198626111035) [Y1 X2 X3 Y4]
+ (0.000292198626111035) [X1 Y2 Y3 X4]
+ (0.00608782248056184) [Y2 X3 X6 Y7]
+ (0.00608782248056184) [X2 Y3 Y6 X7]
+ (0.009560705729135905) [Y2 X3 X4 Y5]
+ (0.009560705729135905) [X2 Y3 Y4 X5]
+ (0.011307274008848072) [Y1 Z2 Z3 Y5]
+ (0.011307274008848072) [X1 Z2 Z3 X5]
+ (0.013873381748426089) [Y0 X1 X2 Y3]
+ (0.013873381748426089) [X0 Y1 Y2 X3]
+ (0.017366118994651392) [Y0 X1 X6 Y7]
+ (0.017366118994651392) [X0 Y1 Y6 X7]
+ (0.017825140995786512) [Y0 X1 X4 Y5]
+ (0.017825140995786512) [X0 Y1 Y4 X5]
+ (0.028685183716105876) [Y4 X5 X6 Y7]
+ (0.028685183716105876) [X4 Y5 Y6 X7]
+ (0.029812424517345767) [Y0 Z1 Z2 Y4]
+ (0.029812424517345767) [X0 Z1 Z2 X4]
+ (0.029812424517345767) [Y1 Z3 Z4 Y5]
+ (0.029812424517345767) [X1 Z3 Z4 X5]
+ (0.0301046231434568) [Y0 Z1 Z3 Y4]
+ (0.0301046231434568) [X0 Z1 Z3 X4]
+ (0.0301046231434568) [Y1 Z2 Z4 Y5]
+ (0.0301046231434568) [X1 Z2 Z4 X5]
+ (0.030787505389143908) [Y0 Z2 Z3 Y4]
+ (0.030787505389143908) [X0 Z2 Z3 X4]
+ (0.04375263801066059) [Y0 Z1 Z2 Z3 Y4]
+ (0.04375263801066059) [X0 Z1 Z2 Z3 X4]
+ (0.04375263801066061) [Y1 Z2 Z3 Z4 Y5]
+ (0.04375263801066061) [X1 Z2 Z3 Z4 X5]
+ (-0.01456453123117301) [Y1 Z2 Z3 X4 X6 Y7]
+ (-0.01456453123117301) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-0.01456453123117301) [X1 Z2 Z3 X4 X6 X7]
+ (-0.01456453123117301) [X1 Z2 Z3 Y4 Y6 X7]
+ (-6.524373848510934e-06) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (-6.524373848510934e-06) [X0 Z1 Z2 Z3 Z4 X6]
+ (-6.524373848510934e-06) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (-6.524373848510934e-06) [X1 Z2 Z3 Z5 Z6 X7]
+ (-3.769659451907985e-06) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (-3.769659451907985e-06) [X0 Z2 Z3 Z4 Z5 X6]
+ (-3.610297130503592e-06) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (-3.610297130503592e-06) [X0 Z1 Z3 Z4 Z5 X6]
+ (-3.610297130503592e-06) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (-3.610297130503592e-06) [X1 Z2 Z4 Z5 Z6 X7]
+ (-3.313145500132593e-06) [Y1 Z2 Z3 Y4 X5 X6]
+ (-3.313145500132593e-06) [X1 Z2 Z3 X4 Y5 Y6]
+ (-3.2774831954421305e-06) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (-3.2774831954421305e-06) [X0 Z1 Z2 Z4 Z5 X6]
+ (-3.2774831954421305e-06) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (-3.2774831954421305e-06) [X1 Z3 Z4 Z5 Z6 X7]
+ (-3.2112283483783423e-06) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (-3.2112283483783423e-06) [X0 Z1 Z2 Z3 Z5 X6]
+ (-3.2112283483783423e-06) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (-3.2112283483783423e-06) [X1 Z2 Z3 Z4 Z6 X7]
+ (-1.035847760162067e-06) [Y0 Y1 X4 Z5 Z6 X7]
+ (-1.035847760162067e-06) [X0 X1 Y4 Z5 Z6 Y7]
+ (-6.628614201585062e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (-6.628614201585062e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (-3.328139350614614e-07) [Y1 X2 X3 Z4 Z5 Y6]
+ (-3.328139350614614e-07) [X1 Y2 Y3 Z4 Z5 X6]
+ (3.328139350614614e-07) [Y1 Y2 X3 Z4 Z5 X6]
+ (3.328139350614614e-07) [X1 X2 Y3 Z4 Z5 Y6]
+ (6.628614201585062e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (6.628614201585062e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (1.035847760162067e-06) [Y0 X1 X4 Z5 Z6 Y7]
+ (1.035847760162067e-06) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.313145500132593e-06) [Y1 Z2 Z3 X4 X5 Y6]
+ (3.313145500132593e-06) [X1 Z2 Z3 Y4 Y5 X6]
+ (4.183932559328342e-06) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (4.183932559328342e-06) [X1 Z2 Z3 Z4 Z5 X7]
+ (0.000292198626111035) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (0.000292198626111035) [Y0 Z1 Y2 X3 Z4 X5]
+ (0.000292198626111035) [X0 Z1 X2 Y3 Z4 Y5]
+ (0.000292198626111035) [X0 Z1 X2 X3 Z4 X5]
+ (0.010540425907671418) [Y0 Z1 Z2 Z3 Y4 Z7]
+ (0.010540425907671418) [X0 Z1 Z2 Z3 X4 Z7]
+ (0.010540425907671418) [Y1 Z2 Z3 Z4 Y5 Z6]
+ (0.010540425907671418) [X1 Z2 Z3 Z4 X5 Z6]
+ (0.01130727400884807) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (0.01130727400884807) [X0 Z1 Z2 Z3 X4 Z5]
+ (0.02510495713884443) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (0.02510495713884443) [X0 Z1 Z2 Z3 X4 Z6]
+ (0.02510495713884443) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (0.02510495713884443) [X1 Z2 Z3 Z4 X5 Z7]
+ (0.030787505389143908) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (0.030787505389143908) [Z0 X1 Z2 Z3 Z4 X5]
+ (-5.10539654943367e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-5.10539654943367e-06) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-5.105396549433663e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (-5.105396549433663e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (-0.01456453123117301) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-0.01456453123117301) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-3.769659451907985e-06) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-3.769659451907985e-06) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-3.328139350614614e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-3.328139350614614e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-3.328139350614614e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-3.328139350614614e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (3.3131455001325932e-06) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (3.3131455001325932e-06) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (3.3131455001325932e-06) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (3.3131455001325932e-06) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (4.183932559328342e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (4.183932559328342e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
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
(-46.46390678868891+0j) [] +
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
(-1.1908508084640527e-06+0j) [Z0 X3 Z4 X5] +
(-0.03276765782329055+0j) [Z0 X3 Z4 Z5 Z6 X7] +
(-0.07635021950635029+0j) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.5809603692086352e-05+0j) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.1908508084640527e-06+0j) [Z0 Y3 Z4 Y5] +
(-0.03276765782329055+0j) [Z0 Y3 Z4 Z5 Z6 Y7] +
(-0.07635021950635029+0j) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.5809603692086352e-05+0j) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
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
(0.27883454426723386+0j) [Z0 Z9] +
(-2.177664604637626e-06+0j) [Z0 X10 Z11 X12] +
(-2.177664604637626e-06+0j) [Z0 Y10 Z11 Y12] +
(0.19299723935364246+0j) [Z0 Z10] +
(-1.6148794135137072e-06+0j) [Z0 X11 Z12 X13] +
(-1.6148794135137072e-06+0j) [Z0 Y11 Z12 Y13] +
(0.20072866460441777+0j) [Z0 Z11] +
(0.21102659849791544+0j) [Z0 Z12] +
(0.2163103749863184+0j) [Z0 Z13] +
(1.9332412771118596e-07+0j) [X1 X2 Y3 Y4] +
(0.0022939566113524762+0j) [X1 X2 Y3 Z4 Z5 Y6] +
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
(-0.0022939566113524762+0j) [X1 Y2 Y3 Z4 Z5 X6] +
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
(-0.0007870896771024431+0j) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10] +
(-1.8394209155175266e-07+0j) [X1 Z2 Z3 X4 X6 X7] +
(-0.0012223378081538377+0j) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0001940085702975649+0j) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12] +
(-2.3713289480110134e-07+0j) [X1 Z2 Z3 X4 X8 X9] +
(8.057446594512792e-08+0j) [X1 Z2 Z3 X4 X10 X11] +
(-0.0009581655836696554+0j) [X1 Z2 Z3 X4 X10 Z11 Z12 X13] +
(0.0017278753941369677+0j) [X1 Z2 Z3 X4 Y11 Y12] +
(-3.086826565198475e-07+0j) [X1 Z2 Z3 X4 X12 X13] +
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
(-0.0022939566113524762+0j) [Y1 X2 X3 Z4 Z5 Y6] +
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
(0.0022939566113524762+0j) [Y1 Y2 X3 Z4 Z5 X6] +
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
(0.0007870896771024431+0j) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10] +
(-1.8394209155175266e-07+0j) [Y1 Z2 Z3 X4 X6 Y7] +
(-0.0012223378081538377+0j) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0001940085702975649+0j) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12] +
(-2.3713289480110134e-07+0j) [Y1 Z2 Z3 X4 X8 Y9] +
(8.057446594512792e-08+0j) [Y1 Z2 Z3 X4 X10 Y11] +
(-0.0009581655836696554+0j) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13] +
(-0.0017278753941369677+0j) [Y1 Z2 Z3 X4 X11 Y12] +
(-3.086826565198475e-07+0j) [Y1 Z2 Z3 X4 X12 Y13] +
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
(-8.337746755398009e-07+0j) [Z1 X3 Z4 X5] +
(-0.027115036845273183+0j) [Z1 X3 Z4 Z5 Z6 X7] +
(-0.06752385099214042+0j) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.4017109734481867e-05+0j) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.337746755398009e-07+0j) [Z1 Y3 Z4 Y5] +
(-0.027115036845273183+0j) [Z1 Y3 Z4 Z5 Z6 Y7] +
(-0.06752385099214042+0j) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.4017109734481867e-05+0j) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
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
(0.27883454426723386+0j) [Z1 Z8] +
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
(-0.03831467029480389+0j) [X4 X5 Y12 Y13] +
(0.01198238901024794+0j) [X4 Y5 Y6 X7] +
(0.007306759928832983+0j) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11] +
(2.888293596288883e-07+0j) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.007306759928832983+0j) [X4 Y5 Y7 Z8 Z9 X10] +
(-2.888293596288883e-07+0j) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(0.0071569349198569296+0j) [X4 Y5 Y8 X9] +
(0.01768006795248155+0j) [X4 Y5 Y10 X11] +
(3.6945132942591346e-06+0j) [X4 Y5 Y10 Z11 Z12 X13] +
(-3.6945132942591346e-06+0j) [X4 Y5 Y11 X12] +
(0.03831467029480389+0j) [X4 Y5 Y12 X13] +
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
(0.03831467029480389+0j) [Y4 X5 X12 Y13] +
(-0.01198238901024794+0j) [Y4 Y5 X6 X7] +
(-0.007306759928832983+0j) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11] +
(-2.888293596288883e-07+0j) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.007306759928832983+0j) [Y4 Y5 Y7 Z8 Z9 Y10] +
(-2.888293596288883e-07+0j) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.0071569349198569296+0j) [Y4 Y5 X8 X9] +
(-0.01768006795248155+0j) [Y4 Y5 X10 X11] +
(-3.6945132942591346e-06+0j) [Y4 Y5 X10 Z11 Z12 X13] +
(-3.6945132942591346e-06+0j) [Y4 Y5 Y11 Y12] +
(-0.03831467029480389+0j) [Y4 Y5 X12 X13] +
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
(0.1567639617643098+0j) [Z4 Z9] +
(1.8782101246652918e-06+0j) [Z4 X10 Z11 X12] +
(1.8782101246652918e-06+0j) [Z4 Y10 Z11 Y12] +
(0.124899909172376+0j) [Z4 Z10] +
(-1.8163031695938426e-06+0j) [Z4 X11 Z12 X13] +
(-1.8163031695938426e-06+0j) [Z4 Y11 Z12 Y13] +
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
(-1.8163031695938426e-06+0j) [Z5 X10 Z11 X12] +
(-1.8163031695938426e-06+0j) [Z5 Y10 Z11 Y12] +
(1.8782101246652918e-06+0j) [Z5 X11 Z12 X13] +
(1.8782101246652918e-06+0j) [Z5 Y11 Z12 Y13] +
(0.124899909172376+0j) [Z5 Z11] +
(0.15215040708869057+0j) [Z5 Z12] +
(0.11383573679388667+0j) [Z5 Z13] +
(-0.01387338174842612+0j) [X6 X7 Y8 Y9] +
(-0.017825140995786404+0j) [X6 X7 Y10 Y11] +
(-1.0358477601352874e-06+0j) [X6 X7 Y10 Z11 Z12 Y13] +
(-1.0358477601352874e-06+0j) [X6 X7 X11 X12] +
(-0.017366118994651406+0j) [X6 X7 Y12 Y13] +
(0.01387338174842612+0j) [X6 Y7 Y8 X9] +
(0.017825140995786404+0j) [X6 Y7 Y10 X11] +
(1.0358477601352874e-06+0j) [X6 Y7 Y10 Z11 Z12 X13] +
(-1.0358477601352874e-06+0j) [X6 Y7 Y11 X12] +
(0.017366118994651406+0j) [X6 Y7 Y12 X13] +
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
(0.017366118994651406+0j) [Y6 X7 X12 Y13] +
(-0.01387338174842612+0j) [Y6 Y7 X8 X9] +
(-0.017825140995786404+0j) [Y6 Y7 X10 X11] +
(-1.0358477601352874e-06+0j) [Y6 Y7 X10 Z11 Z12 X13] +
(-1.0358477601352874e-06+0j) [Y6 Y7 Y11 Y12] +
(-0.017366118994651406+0j) [Y6 Y7 X12 X13] +
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
  (-46.463906788689) [I0]
+ (0.7829661725950184) [Z10]
+ (0.7829661725950186) [Z11]
+ (0.8084581961720474) [Z12]
+ (0.8084581961720475) [Z13]
+ (1.2034402289145623) [Z4]
+ (1.2034402289145625) [Z5]
+ (1.3096862988615428) [Z6]
+ (1.3096862988615432) [Z7]
+ (1.3693525634718167) [Z8]
+ (1.369352563471817) [Z9]
+ (1.653894222683169) [Z2]
+ (1.6538942226831692) [Z3]
+ (12.412630742111777) [Z0]
+ (12.412630742111777) [Z1]
+ (-8.194261372433133e-06) [Y10 Y12]
+ (-8.194261372433133e-06) [X10 X12]
+ (-1.854060857852828e-06) [Y5 Y7]
+ (-1.854060857852828e-06) [X5 X7]
+ (-7.764994120102223e-07) [Y3 Y5]
+ (-7.764994120102223e-07) [X3 X5]
+ (-5.929765814229193e-07) [Y4 Y6]
+ (-5.929765814229193e-07) [X4 X6]
+ (1.602116740745163e-06) [Y2 Y4]
+ (1.602116740745163e-06) [X2 X4]
+ (7.954413176426263e-06) [Y11 Y13]
+ (7.954413176426263e-06) [X11 X13]
+ (0.0032769719312315164) [Y1 Y3]
+ (0.0032769719312315164) [X1 X3]
+ (0.10433064780651378) [Y0 Y2]
+ (0.10433064780651378) [X0 X2]
+ (0.11270386920332229) [Z10 Z12]
+ (0.11270386920332229) [Z11 Z13]
+ (0.11383573679388656) [Z4 Z12]
+ (0.11383573679388656) [Z5 Z13]
+ (0.11952438964682698) [Z6 Z10]
+ (0.11952438964682698) [Z7 Z11]
+ (0.1248999091723761) [Z4 Z10]
+ (0.1248999091723761) [Z5 Z11]
+ (0.12495807739503213) [Z2 Z4]
+ (0.12495807739503213) [Z3 Z5]
+ (0.1279950249246842) [Z2 Z10]
+ (0.1279950249246842) [Z3 Z11]
+ (0.13401715261963718) [Z6 Z12]
+ (0.13401715261963718) [Z7 Z13]
+ (0.13701191674040764) [Z4 Z6]
+ (0.13701191674040764) [Z5 Z7]
+ (0.1373495306426134) [Z6 Z11]
+ (0.1373495306426134) [Z7 Z10]
+ (0.13739104762683238) [Z2 Z6]
+ (0.13739104762683238) [Z3 Z7]
+ (0.13766872645852588) [Z8 Z10]
+ (0.13766872645852588) [Z9 Z11]
+ (0.14011289865354815) [Z2 Z12]
+ (0.14011289865354815) [Z3 Z13]
+ (0.14138905291942824) [Z10 Z13]
+ (0.14138905291942824) [Z11 Z12]
+ (0.14257997712485765) [Z4 Z11]
+ (0.14257997712485765) [Z5 Z10]
+ (0.14722943218766188) [Z8 Z11]
+ (0.14722943218766188) [Z9 Z10]
+ (0.1489943057506556) [Z4 Z7]
+ (0.1489943057506556) [Z5 Z6]
+ (0.14926355147388926) [Z10 Z11]
+ (0.14960702684445298) [Z4 Z8]
+ (0.14960702684445298) [Z5 Z9]
+ (0.1497348680349693) [Z8 Z12]
+ (0.1497348680349693) [Z9 Z13]
+ (0.1507140812100829) [Z2 Z8]
+ (0.1507140812100829) [Z3 Z9]
+ (0.1513832716142886) [Z6 Z13]
+ (0.1513832716142886) [Z7 Z12]
+ (0.15215040708869046) [Z4 Z13]
+ (0.15215040708869046) [Z5 Z12]
+ (0.15337968243314165) [Z2 Z11]
+ (0.15337968243314165) [Z3 Z10]
+ (0.15435748657223636) [Z12 Z13]
+ (0.1556901067175246) [Z2 Z13]
+ (0.1556901067175246) [Z3 Z12]
+ (0.15582269051553116) [Z8 Z13]
+ (0.15582269051553116) [Z9 Z12]
+ (0.1567639617643099) [Z4 Z9]
+ (0.1567639617643099) [Z5 Z8]
+ (0.15755314797985664) [Z4 Z5]
+ (0.16079764534838564) [Z2 Z5]
+ (0.16079764534838564) [Z3 Z4]
+ (0.16756653265461288) [Z6 Z8]
+ (0.16756653265461288) [Z7 Z9]
+ (0.16853486561579947) [Z2 Z7]
+ (0.16853486561579947) [Z3 Z6]
+ (0.18143991440303903) [Z6 Z9]
+ (0.18143991440303903) [Z7 Z8]
+ (0.18189085790751355) [Z2 Z3]
+ (0.1929972393536427) [Z0 Z10]
+ (0.1929972393536427) [Z1 Z11]
+ (0.19392534613270251) [Z6 Z7]
+ (0.19661770890342156) [Z0 Z4]
+ (0.19661770890342156) [Z1 Z5]
+ (0.1993635453736084) [Z0 Z5]
+ (0.1993635453736084) [Z1 Z4]
+ (0.20072866460441802) [Z0 Z11]
+ (0.20072866460441802) [Z1 Z10]
+ (0.21102659849791533) [Z0 Z12]
+ (0.21102659849791533) [Z1 Z13]
+ (0.21631037498631828) [Z0 Z13]
+ (0.21631037498631828) [Z1 Z12]
+ (0.23671080783830425) [Z0 Z2]
+ (0.23671080783830425) [Z1 Z3]
+ (0.2416466393601725) [Z0 Z6]
+ (0.2416466393601725) [Z1 Z7]
+ (0.2485348337131431) [Z0 Z7]
+ (0.2485348337131431) [Z1 Z6]
+ (0.2512944567459169) [Z0 Z3]
+ (0.2512944567459169) [Z1 Z2]
+ (0.272325183066057) [Z0 Z8]
+ (0.272325183066057) [Z1 Z9]
+ (0.2788345442672343) [Z0 Z9]
+ (0.2788345442672343) [Z1 Z8]
+ (1.186176373486051) [Z0 Z1]
+ (-1.2260484988321186e-05) [Y4 Z5 Y6]
+ (-1.2260484988321186e-05) [X4 Z5 X6]
+ (-1.2260484988321182e-05) [Y5 Z6 Y7]
+ (-1.2260484988321182e-05) [X5 Z6 X7]
+ (-1.0722312158032718e-05) [Y11 Z12 Y13]
+ (-1.0722312158032718e-05) [X11 Z12 X13]
+ (-1.0722312158032714e-05) [Y10 Z11 Y12]
+ (-1.0722312158032714e-05) [X10 Z11 X12]
+ (-3.887051673066151e-06) [Y3 Z4 Y5]
+ (-3.887051673066151e-06) [X3 Z4 X5]
+ (-3.887051673066149e-06) [Y2 Z3 Y4]
+ (-3.887051673066149e-06) [X2 Z3 X4]
+ (0.12507032579771812) [Y0 Z1 Y2]
+ (0.12507032579771812) [X0 Z1 X2]
+ (0.12507032579771815) [Y1 Z2 Y3]
+ (0.12507032579771815) [X1 Z2 X3]
+ (-0.03831467029480389) [Y4 Y5 X12 X13]
+ (-0.03831467029480389) [X4 X5 Y12 Y13]
+ (-0.036194123559042564) [Y2 Y3 X8 X9]
+ (-0.036194123559042564) [X2 X3 Y8 Y9]
+ (-0.035839567953353496) [Y2 Y3 X4 X5]
+ (-0.035839567953353496) [X2 X3 Y4 Y5]
+ (-0.0311438179889671) [Y2 Y3 X6 X7]
+ (-0.0311438179889671) [X2 X3 Y6 Y7]
+ (-0.028685183716105952) [Y10 Y11 X12 X13]
+ (-0.028685183716105952) [X10 X11 Y12 Y13]
+ (-0.025996177598021274) [Y3 Z4 Z5 Y7]
+ (-0.025996177598021274) [X3 Z4 Z5 X7]
+ (-0.025384657508457455) [Y2 Y3 X10 X11]
+ (-0.025384657508457455) [X2 X3 Y10 Y11]
+ (-0.019028242443847338) [Y3 Y4 X11 X12]
+ (-0.019028242443847338) [X3 X4 Y11 Y12]
+ (-0.017825140995786432) [Y6 Y7 X10 X11]
+ (-0.017825140995786432) [X6 X7 Y10 Y11]
+ (-0.01768006795248153) [Y4 Y5 X10 X11]
+ (-0.01768006795248153) [X4 X5 Y10 Y11]
+ (-0.01736611899465142) [Y6 Y7 X12 X13]
+ (-0.01736611899465142) [X6 X7 Y12 Y13]
+ (-0.015577208063976455) [Y2 Y3 X12 X13]
+ (-0.015577208063976455) [X2 X3 Y12 Y13]
+ (-0.014583648907612625) [Y0 Y1 X2 X3]
+ (-0.014583648907612625) [X0 X1 Y2 Y3]
+ (-0.013873381748426143) [Y6 Y7 X8 X9]
+ (-0.013873381748426143) [X6 X7 Y8 Y9]
+ (-0.011982389010247953) [Y4 Y5 X6 X7]
+ (-0.011982389010247953) [X4 X5 Y6 Y7]
+ (-0.011285190200840893) [Y5 X6 X11 Y12]
+ (-0.011285190200840893) [X5 Y6 Y11 X12]
+ (-0.009560705729135971) [Y8 Y9 X10 X11]
+ (-0.009560705729135971) [X8 X9 Y10 Y11]
+ (-0.008125251921381048) [Y1 X2 X8 Y9]
+ (-0.008125251921381048) [Y1 Y2 Y8 Y9]
+ (-0.008125251921381048) [X1 X2 X8 X9]
+ (-0.008125251921381048) [X1 Y2 Y8 X9]
+ (-0.007731425250775304) [Y0 Y1 X10 X11]
+ (-0.007731425250775304) [X0 X1 Y10 Y11]
+ (-0.007156934919856943) [Y4 Y5 X8 X9]
+ (-0.007156934919856943) [X4 X5 Y8 Y9]
+ (-0.00688819435297061) [Y0 Y1 X6 X7]
+ (-0.00688819435297061) [X0 X1 Y6 Y7]
+ (-0.006509361201177252) [Y0 Y1 X8 X9]
+ (-0.006509361201177252) [X0 X1 Y8 Y9]
+ (-0.006087822480561859) [Y8 Y9 X12 X13]
+ (-0.006087822480561859) [X8 X9 Y12 Y13]
+ (-0.0052837764884029696) [Y0 Y1 X12 X13]
+ (-0.0052837764884029696) [X0 X1 Y12 Y13]
+ (-0.005143391768825097) [Y3 X4 X5 Y6]
+ (-0.005143391768825097) [X3 Y4 Y5 X6]
+ (-0.004684903388155214) [Y1 X2 X6 Y7]
+ (-0.004684903388155214) [Y1 Y2 Y6 Y7]
+ (-0.004684903388155214) [X1 X2 X6 X7]
+ (-0.004684903388155214) [X1 Y2 Y6 X7]
+ (-0.004575007626639206) [Y1 X2 X12 Y13]
+ (-0.004575007626639206) [Y1 Y2 Y12 Y13]
+ (-0.004575007626639206) [X1 X2 X12 X13]
+ (-0.004575007626639206) [X1 Y2 Y12 X13]
+ (-0.004424855449441865) [Y1 X2 X4 Y5]
+ (-0.004424855449441865) [Y1 Y2 Y4 Y5]
+ (-0.004424855449441865) [X1 X2 X4 X5]
+ (-0.004424855449441865) [X1 Y2 Y4 X5]
+ (-0.003479511890334304) [Y2 Z3 Z5 Y6]
+ (-0.003479511890334304) [X2 Z3 Z5 X6]
+ (-0.003479511890334304) [Y3 Z4 Z6 Y7]
+ (-0.003479511890334304) [X3 Z4 Z6 X7]
+ (-0.0027458364701868163) [Y0 Y1 X4 X5]
+ (-0.0027458364701868163) [X0 X1 Y4 Y5]
+ (-0.0017992194936630366) [Y1 X2 X10 Y11]
+ (-0.0017992194936630366) [Y1 Y2 Y10 Y11]
+ (-0.0017992194936630366) [X1 X2 X10 X11]
+ (-0.0017992194936630366) [X1 Y2 Y10 X11]
+ (-0.0002921986261110867) [Y7 Y8 X9 X10]
+ (-0.0002921986261110867) [X7 X8 Y9 Y10]
+ (-8.194261372433133e-06) [Z10 Y11 Z12 Y13]
+ (-8.194261372433133e-06) [Z10 X11 Z12 X13]
+ (-7.801707500728942e-06) [Y2 Z3 Y4 Z11]
+ (-7.801707500728942e-06) [X2 Z3 X4 Z11]
+ (-7.801707500728942e-06) [Y3 Z4 Y5 Z10]
+ (-7.801707500728942e-06) [X3 Z4 X5 Z10]
+ (-4.64305106864119e-06) [Y3 X4 X10 Y11]
+ (-4.64305106864119e-06) [Y3 Y4 Y10 Y11]
+ (-4.64305106864119e-06) [X3 X4 X10 X11]
+ (-4.64305106864119e-06) [X3 Y4 Y10 X11]
+ (-4.588855155705379e-06) [Y4 Z5 Y6 Z13]
+ (-4.588855155705379e-06) [X4 Z5 X6 Z13]
+ (-4.588855155705379e-06) [Y5 Z6 Y7 Z12]
+ (-4.588855155705379e-06) [X5 Z6 X7 Z12]
+ (-4.556569218296557e-06) [Y5 X6 X12 Y13]
+ (-4.556569218296557e-06) [Y5 Y6 Y12 Y13]
+ (-4.556569218296557e-06) [X5 X6 X12 X13]
+ (-4.556569218296557e-06) [X5 Y6 Y12 X13]
+ (-3.6945132947364453e-06) [Y4 X5 X11 Y12]
+ (-3.6945132947364453e-06) [Y4 Y5 Y11 Y12]
+ (-3.6945132947364453e-06) [X4 X5 X11 X12]
+ (-3.6945132947364453e-06) [X4 Y5 Y11 X12]
+ (-3.3440815563352615e-06) [Z0 Y5 Z6 Y7]
+ (-3.3440815563352615e-06) [Z0 X5 Z6 X7]
+ (-3.3440815563352615e-06) [Z1 Y4 Z5 Y6]
+ (-3.3440815563352615e-06) [Z1 X4 Z5 X6]
+ (-3.1586564320877504e-06) [Y2 Z3 Y4 Z10]
+ (-3.1586564320877504e-06) [X2 Z3 X4 Z10]
+ (-3.1586564320877504e-06) [Y3 Z4 Y5 Z11]
+ (-3.1586564320877504e-06) [X3 Z4 X5 Z11]
+ (-3.0993492434579515e-06) [Z0 Y4 Z5 Y6]
+ (-3.0993492434579515e-06) [Z0 X4 Z5 X6]
+ (-3.0993492434579515e-06) [Z1 Y5 Z6 Y7]
+ (-3.0993492434579515e-06) [Z1 X5 Z6 X7]
+ (-2.8909678817448577e-06) [Z6 Y11 Z12 Y13]
+ (-2.8909678817448577e-06) [Z6 X11 Z12 X13]
+ (-2.8909678817448577e-06) [Z7 Y10 Z11 Y12]
+ (-2.8909678817448577e-06) [Z7 X10 Z11 X12]
+ (-2.1776646051607878e-06) [Z0 Y10 Z11 Y12]
+ (-2.1776646051607878e-06) [Z0 X10 Z11 X12]
+ (-2.1776646051607878e-06) [Z1 Y11 Z12 Y13]
+ (-2.1776646051607878e-06) [Z1 X11 Z12 X13]
+ (-1.8818501831051464e-06) [Y4 Z5 Y6 Z9]
+ (-1.8818501831051464e-06) [X4 Z5 X6 Z9]
+ (-1.8818501831051464e-06) [Y5 Z6 Y7 Z8]
+ (-1.8818501831051464e-06) [X5 Z6 X7 Z8]
+ (-1.8551201216653915e-06) [Z6 Y10 Z11 Y12]
+ (-1.8551201216653915e-06) [Z6 X10 Z11 X12]
+ (-1.8551201216653915e-06) [Z7 Y11 Z12 Y13]
+ (-1.8551201216653915e-06) [Z7 X11 Z12 X13]
+ (-1.854060857852828e-06) [Y4 Z5 Y6 Z7]
+ (-1.854060857852828e-06) [X4 Z5 X6 Z7]
+ (-1.8163031699910057e-06) [Z4 Y11 Z12 Y13]
+ (-1.8163031699910057e-06) [Z4 X11 Z12 X13]
+ (-1.8163031699910057e-06) [Z5 Y10 Z11 Y12]
+ (-1.8163031699910057e-06) [Z5 X10 Z11 X12]
+ (-1.6923978285710195e-06) [Y4 Z5 Y6 Z10]
+ (-1.6923978285710195e-06) [X4 Z5 X6 Z10]
+ (-1.6923978285710195e-06) [Y5 Z6 Y7 Z11]
+ (-1.6923978285710195e-06) [X5 Z6 X7 Z11]
+ (-1.6148794140089826e-06) [Z0 Y11 Z12 Y13]
+ (-1.6148794140089826e-06) [Z0 X11 Z12 X13]
+ (-1.6148794140089826e-06) [Z1 Y10 Z11 Y12]
+ (-1.6148794140089826e-06) [Z1 X10 Z11 X12]
+ (-1.5973171979144427e-06) [Z8 Y10 Z11 Y12]
+ (-1.5973171979144427e-06) [Z8 X10 Z11 X12]
+ (-1.5973171979144427e-06) [Z9 Y11 Z12 Y13]
+ (-1.5973171979144427e-06) [Z9 X11 Z12 X13]
+ (-1.4548424489910612e-06) [Y3 X4 X6 Y7]
+ (-1.4548424489910612e-06) [Y3 Y4 Y6 Y7]
+ (-1.4548424489910612e-06) [X3 X4 X6 X7]
+ (-1.4548424489910612e-06) [X3 Y4 Y6 X7]
+ (-1.3980449080232052e-06) [Y4 Z5 Y6 Z8]
+ (-1.3980449080232052e-06) [X4 Z5 X6 Z8]
+ (-1.3980449080232052e-06) [Y5 Z6 Y7 Z9]
+ (-1.3980449080232052e-06) [X5 Z6 X7 Z9]
+ (-1.195489009863085e-06) [Y2 Z3 Y4 Z7]
+ (-1.195489009863085e-06) [X2 Z3 X4 Z7]
+ (-1.195489009863085e-06) [Y3 Z4 Y5 Z6]
+ (-1.195489009863085e-06) [X3 Z4 X5 Z6]
+ (-1.1908508082775722e-06) [Z0 Y3 Z4 Y5]
+ (-1.1908508082775722e-06) [Z0 X3 Z4 X5]
+ (-1.1908508082775722e-06) [Z1 Y2 Z3 Y4]
+ (-1.1908508082775722e-06) [Z1 X2 Z3 X4]
+ (-1.1708301369641627e-06) [Z2 Y5 Z6 Y7]
+ (-1.1708301369641627e-06) [Z2 X5 Z6 X7]
+ (-1.1708301369641627e-06) [Z3 Y4 Z5 Y6]
+ (-1.1708301369641627e-06) [Z3 X4 Z5 X6]
+ (-1.0632283423926082e-06) [Z2 Y10 Z11 Y12]
+ (-1.0632283423926082e-06) [Z2 X10 Z11 X12]
+ (-1.0632283423926082e-06) [Z3 Y11 Z12 Y13]
+ (-1.0632283423926082e-06) [Z3 X11 Z12 X13]
+ (-1.0358477600794662e-06) [Y6 X7 X11 Y12]
+ (-1.0358477600794662e-06) [Y6 Y7 Y11 Y12]
+ (-1.0358477600794662e-06) [X6 X7 X11 X12]
+ (-1.0358477600794662e-06) [X6 Y7 Y11 X12]
+ (-9.509249750735775e-07) [Z2 Y4 Z5 Y6]
+ (-9.509249750735775e-07) [Z2 X4 Z5 X6]
+ (-9.509249750735775e-07) [Z3 Y5 Z6 Y7]
+ (-9.509249750735775e-07) [Z3 X5 Z6 X7]
+ (-9.344557777299934e-07) [Z8 Y11 Z12 Y13]
+ (-9.344557777299934e-07) [Z8 X11 Z12 X13]
+ (-9.344557777299934e-07) [Z9 Y10 Z11 Y12]
+ (-9.344557777299934e-07) [Z9 X10 Z11 X12]
+ (-8.337746753660904e-07) [Z0 Y2 Z3 Y4]
+ (-8.337746753660904e-07) [Z0 X2 Z3 X4]
+ (-8.337746753660904e-07) [Z1 Y3 Z4 Y5]
+ (-8.337746753660904e-07) [Z1 X3 Z4 X5]
+ (-7.956895372462184e-07) [Y3 X4 X8 Y9]
+ (-7.956895372462184e-07) [Y3 Y4 Y8 Y9]
+ (-7.956895372462184e-07) [X3 X4 X8 X9]
+ (-7.956895372462184e-07) [X3 Y4 Y8 X9]
+ (-7.764994120102223e-07) [Y2 Z3 Y4 Z5]
+ (-7.764994120102223e-07) [X2 Z3 X4 Z5]
+ (-5.929765814229193e-07) [Z4 Y5 Z6 Y7]
+ (-5.929765814229193e-07) [Z4 X5 Z6 X7]
+ (-5.770052994538309e-07) [Y2 Z3 Y4 Z9]
+ (-5.770052994538309e-07) [X2 Z3 X4 Z9]
+ (-5.770052994538309e-07) [Y3 Z4 Y5 Z8]
+ (-5.770052994538309e-07) [X3 Z4 X5 Z8]
+ (-5.471647744841628e-07) [Y1 Y2 X11 X12]
+ (-5.471647744841628e-07) [X1 X2 Y11 Y12]
+ (-4.838052750819412e-07) [Y5 X6 X8 Y9]
+ (-4.838052750819412e-07) [Y5 Y6 Y8 Y9]
+ (-4.838052750819412e-07) [X5 X6 X8 X9]
+ (-4.838052750819412e-07) [X5 Y6 Y8 X9]
+ (-3.5707613291148185e-07) [Y0 X1 X3 Y4]
+ (-3.5707613291148185e-07) [Y0 Y1 Y3 Y4]
+ (-3.5707613291148185e-07) [X0 X1 X3 X4]
+ (-3.5707613291148185e-07) [X0 Y1 Y3 X4]
+ (-2.447323128773102e-07) [Y0 X1 X5 Y6]
+ (-2.447323128773102e-07) [Y0 Y1 Y5 Y6]
+ (-2.447323128773102e-07) [X0 X1 X5 X6]
+ (-2.447323128773102e-07) [X0 Y1 Y5 X6]
+ (-2.199051618905852e-07) [Y2 X3 X5 Y6]
+ (-2.199051618905852e-07) [Y2 Y3 Y5 Y6]
+ (-2.199051618905852e-07) [X2 X3 X5 X6]
+ (-2.199051618905852e-07) [X2 Y3 Y5 X6]
+ (-1.9332412770761808e-07) [Y1 X2 X3 Y4]
+ (-1.9332412770761808e-07) [X1 Y2 Y3 X4]
+ (-1.2919694863364175e-07) [Y1 Z2 Z3 Y5]
+ (-1.2919694863364175e-07) [X1 Z2 Z3 X5]
+ (1.7379332624688032e-07) [Y0 Z1 Z3 Y4]
+ (1.7379332624688032e-07) [X0 Z1 Z3 X4]
+ (1.7379332624688032e-07) [Y1 Z2 Z4 Y5]
+ (1.7379332624688032e-07) [X1 Z2 Z4 X5]
+ (1.9332412770761808e-07) [Y1 Y2 X3 X4]
+ (1.9332412770761808e-07) [X1 X2 Y3 Y4]
+ (2.186842377923874e-07) [Y2 Z3 Y4 Z8]
+ (2.186842377923874e-07) [X2 Z3 X4 Z8]
+ (2.186842377923874e-07) [Y3 Z4 Y5 Z9]
+ (2.186842377923874e-07) [X3 Z4 X5 Z9]
+ (2.593534391279762e-07) [Y2 Z3 Y4 Z6]
+ (2.593534391279762e-07) [X2 Z3 X4 Z6]
+ (2.593534391279762e-07) [Y3 Z4 Y5 Z7]
+ (2.593534391279762e-07) [X3 Z4 X5 Z7]
+ (3.6060718680076775e-07) [Y0 Z1 Z2 Y4]
+ (3.6060718680076775e-07) [X0 Z1 Z2 X4]
+ (3.6060718680076775e-07) [Y1 Z3 Z4 Y5]
+ (3.6060718680076775e-07) [X1 Z3 Z4 X5]
+ (5.471647744841628e-07) [Y1 X2 X11 Y12]
+ (5.471647744841628e-07) [X1 Y2 Y11 X12]
+ (5.627851911518052e-07) [Y0 X1 X11 Y12]
+ (5.627851911518052e-07) [Y0 Y1 Y11 Y12]
+ (5.627851911518052e-07) [X0 X1 X11 X12]
+ (5.627851911518052e-07) [X0 Y1 Y11 X12]
+ (6.628614201844492e-07) [Y8 X9 X11 Y12]
+ (6.628614201844492e-07) [Y8 Y9 Y11 Y12]
+ (6.628614201844492e-07) [X8 X9 X11 X12]
+ (6.628614201844492e-07) [X8 Y9 Y11 X12]
+ (1.109440759195323e-06) [Z2 Y11 Z12 Y13]
+ (1.109440759195323e-06) [Z2 X11 Z12 X13]
+ (1.109440759195323e-06) [Z3 Y10 Z11 Y12]
+ (1.109440759195323e-06) [Z3 X10 Z11 X12]
+ (1.602116740745163e-06) [Z2 Y3 Z4 Y5]
+ (1.602116740745163e-06) [Z2 X3 Z4 X5]
+ (1.8782101247454398e-06) [Z4 Y10 Z11 Y12]
+ (1.8782101247454398e-06) [Z4 X10 Z11 X12]
+ (1.8782101247454398e-06) [Z5 Y11 Z12 Y13]
+ (1.8782101247454398e-06) [Z5 X11 Z12 X13]
+ (2.172669101587931e-06) [Y2 X3 X11 Y12]
+ (2.172669101587931e-06) [Y2 Y3 Y11 Y12]
+ (2.172669101587931e-06) [X2 X3 X11 X12]
+ (2.172669101587931e-06) [X2 Y3 Y11 X12]
+ (3.117447946120398e-06) [Y0 Z2 Z3 Y4]
+ (3.117447946120398e-06) [X0 Z2 Z3 X4]
+ (3.5390541845769056e-06) [Y2 Z3 Y4 Z12]
+ (3.5390541845769056e-06) [X2 Z3 X4 Z12]
+ (3.5390541845769056e-06) [Y3 Z4 Y5 Z13]
+ (3.5390541845769056e-06) [X3 Z4 X5 Z13]
+ (4.281913885024184e-06) [Y4 Z5 Y6 Z11]
+ (4.281913885024184e-06) [X4 Z5 X6 Z11]
+ (4.281913885024184e-06) [Y5 Z6 Y7 Z10]
+ (4.281913885024184e-06) [X5 Z6 X7 Z10]
+ (5.275883122326555e-06) [Y3 X4 X12 Y13]
+ (5.275883122326555e-06) [Y3 Y4 Y12 Y13]
+ (5.275883122326555e-06) [X3 X4 X12 X13]
+ (5.275883122326555e-06) [X3 Y4 Y12 X13]
+ (5.974311713595205e-06) [Y5 X6 X10 Y11]
+ (5.974311713595205e-06) [Y5 Y6 Y10 Y11]
+ (5.974311713595205e-06) [X5 X6 X10 X11]
+ (5.974311713595205e-06) [X5 Y6 Y10 X11]
+ (7.954413176426263e-06) [Y10 Z11 Y12 Z13]
+ (7.954413176426263e-06) [X10 Z11 X12 Z13]
+ (8.814937306903461e-06) [Y2 Z3 Y4 Z13]
+ (8.814937306903461e-06) [X2 Z3 X4 Z13]
+ (8.814937306903461e-06) [Y3 Z4 Y5 Z12]
+ (8.814937306903461e-06) [X3 Z4 X5 Z12]
+ (0.0002921986261110867) [Y7 X8 X9 Y10]
+ (0.0002921986261110867) [X7 Y8 Y9 X10]
+ (0.0004956762314917058) [Y2 Z4 Z5 Y6]
+ (0.0004956762314917058) [X2 Z4 Z5 X6]
+ (0.0011059037691895823) [Y0 Z1 Y2 Z5]
+ (0.0011059037691895823) [X0 Z1 X2 Z5]
+ (0.0011059037691895823) [Y1 Z2 Y3 Z4]
+ (0.0011059037691895823) [X1 Z2 X3 Z4]
+ (0.001663879878490794) [Y2 Z3 Z4 Y6]
+ (0.001663879878490794) [X2 Z3 Z4 X6]
+ (0.001663879878490794) [Y3 Z5 Z6 Y7]
+ (0.001663879878490794) [X3 Z5 Z6 X7]
+ (0.00175607070184114) [Y0 Z1 Y2 Z11]
+ (0.00175607070184114) [X0 Z1 X2 Z11]
+ (0.00175607070184114) [Y1 Z2 Y3 Z10]
+ (0.00175607070184114) [X1 Z2 X3 Z10]
+ (0.0023262306231579726) [Y0 Z1 Y2 Z13]
+ (0.0023262306231579726) [X0 Z1 X2 Z13]
+ (0.0023262306231579726) [Y1 Z2 Y3 Z12]
+ (0.0023262306231579726) [X1 Z2 X3 Z12]
+ (0.0027458364701868163) [Y0 X1 X4 Y5]
+ (0.0027458364701868163) [X0 Y1 Y4 X5]
+ (0.0029297686747509206) [Y0 Z1 Y2 Z9]
+ (0.0029297686747509206) [X0 Z1 X2 Z9]
+ (0.0029297686747509206) [Y1 Z2 Y3 Z8]
+ (0.0029297686747509206) [X1 Z2 X3 Z8]
+ (0.0032769719312315164) [Y0 Z1 Y2 Z3]
+ (0.0032769719312315164) [X0 Z1 X2 Z3]
+ (0.0033476175306660846) [Y0 Z1 Y2 Z7]
+ (0.0033476175306660846) [X0 Z1 X2 Z7]
+ (0.0033476175306660846) [Y1 Z2 Y3 Z6]
+ (0.0033476175306660846) [X1 Z2 X3 Z6]
+ (0.0035552901955041762) [Y0 Z1 Y2 Z10]
+ (0.0035552901955041762) [X0 Z1 X2 Z10]
+ (0.0035552901955041762) [Y1 Z2 Y3 Z11]
+ (0.0035552901955041762) [X1 Z2 X3 Z11]
+ (0.005143391768825097) [Y3 Y4 X5 X6]
+ (0.005143391768825097) [X3 X4 Y5 Y6]
+ (0.0052837764884029696) [Y0 X1 X12 Y13]
+ (0.0052837764884029696) [X0 Y1 Y12 X13]
+ (0.005530759218631447) [Y0 Z1 Y2 Z4]
+ (0.005530759218631447) [X0 Z1 X2 Z4]
+ (0.005530759218631447) [Y1 Z2 Y3 Z5]
+ (0.005530759218631447) [X1 Z2 X3 Z5]
+ (0.006087822480561859) [Y8 X9 X12 Y13]
+ (0.006087822480561859) [X8 Y9 Y12 X13]
+ (0.006509361201177252) [Y0 X1 X8 Y9]
+ (0.006509361201177252) [X0 Y1 Y8 X9]
+ (0.00688819435297061) [Y0 X1 X6 Y7]
+ (0.00688819435297061) [X0 Y1 Y6 X7]
+ (0.006901238249797179) [Y0 Z1 Y2 Z12]
+ (0.006901238249797179) [X0 Z1 X2 Z12]
+ (0.006901238249797179) [Y1 Z2 Y3 Z13]
+ (0.006901238249797179) [X1 Z2 X3 Z13]
+ (0.007156934919856943) [Y4 X5 X8 Y9]
+ (0.007156934919856943) [X4 Y5 Y8 X9]
+ (0.007731425250775304) [Y0 X1 X10 Y11]
+ (0.007731425250775304) [X0 Y1 Y10 X11]
+ (0.008032520918821298) [Y0 Z1 Y2 Z6]
+ (0.008032520918821298) [X0 Z1 X2 Z6]
+ (0.008032520918821298) [Y1 Z2 Y3 Z7]
+ (0.008032520918821298) [X1 Z2 X3 Z7]
+ (0.009560705729135971) [Y8 X9 X10 Y11]
+ (0.009560705729135971) [X8 Y9 Y10 X11]
+ (0.011055020596131969) [Y0 Z1 Y2 Z8]
+ (0.011055020596131969) [X0 Z1 X2 Z8]
+ (0.011055020596131969) [Y1 Z2 Y3 Z9]
+ (0.011055020596131969) [X1 Z2 X3 Z9]
+ (0.011285190200840893) [Y5 Y6 X11 X12]
+ (0.011285190200840893) [X5 X6 Y11 Y12]
+ (0.011307274008848182) [Y7 Z8 Z9 Y11]
+ (0.011307274008848182) [X7 Z8 Z9 X11]
+ (0.011982389010247953) [Y4 X5 X6 Y7]
+ (0.011982389010247953) [X4 Y5 Y6 X7]
+ (0.013873381748426143) [Y6 X7 X8 Y9]
+ (0.013873381748426143) [X6 Y7 Y8 X9]
+ (0.014583648907612625) [Y0 X1 X2 Y3]
+ (0.014583648907612625) [X0 Y1 Y2 X3]
+ (0.015577208063976455) [Y2 X3 X12 Y13]
+ (0.015577208063976455) [X2 Y3 Y12 X13]
+ (0.01736611899465142) [Y6 X7 X12 Y13]
+ (0.01736611899465142) [X6 Y7 Y12 X13]
+ (0.01768006795248153) [Y4 X5 X10 Y11]
+ (0.01768006795248153) [X4 Y5 Y10 X11]
+ (0.017825140995786432) [Y6 X7 X10 Y11]
+ (0.017825140995786432) [X6 Y7 Y10 X11]
+ (0.019028242443847338) [Y3 X4 X11 Y12]
+ (0.019028242443847338) [X3 Y4 Y11 X12]
+ (0.025384657508457455) [Y2 X3 X10 Y11]
+ (0.025384657508457455) [X2 Y3 Y10 X11]
+ (0.028685183716105952) [Y10 X11 X12 Y13]
+ (0.028685183716105952) [X10 Y11 Y12 X13]
+ (0.029812424517345705) [Y6 Z7 Z8 Y10]
+ (0.029812424517345705) [X6 Z7 Z8 X10]
+ (0.029812424517345705) [Y7 Z9 Z10 Y11]
+ (0.029812424517345705) [X7 Z9 Z10 X11]
+ (0.030104623143456792) [Y6 Z7 Z9 Y10]
+ (0.030104623143456792) [X6 Z7 Z9 X10]
+ (0.030104623143456792) [Y7 Z8 Z10 Y11]
+ (0.030104623143456792) [X7 Z8 Z10 X11]
+ (0.030787505389143942) [Y6 Z8 Z9 Y10]
+ (0.030787505389143942) [X6 Z8 Z9 X10]
+ (0.0311438179889671) [Y2 X3 X6 Y7]
+ (0.0311438179889671) [X2 Y3 Y6 X7]
+ (0.035839567953353496) [Y2 X3 X4 Y5]
+ (0.035839567953353496) [X2 Y3 Y4 X5]
+ (0.036194123559042564) [Y2 X3 X8 Y9]
+ (0.036194123559042564) [X2 Y3 Y8 X9]
+ (0.03831467029480389) [Y4 X5 X12 Y13]
+ (0.03831467029480389) [X4 Y5 Y12 X13]
+ (0.10433064780651378) [Z0 Y1 Z2 Y3]
+ (0.10433064780651378) [Z0 X1 Z2 X3]
+ (-0.12133276911042393) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042393) [X2 Z3 Z4 Z5 X6]
+ (-0.1213327691104239) [Y3 Z4 Z5 Z6 Y7]
+ (-0.1213327691104239) [X3 Z4 Z5 Z6 X7]
+ (3.202076880931096e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.202076880931096e-06) [X0 Z1 Z2 Z3 X4]
+ (3.2020768809310965e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.2020768809310965e-06) [X1 Z2 Z3 Z4 X5]
+ (0.2284810656491876) [Y7 Z8 Z9 Z10 Y11]
+ (0.2284810656491876) [X7 Z8 Z9 Z10 X11]
+ (0.22848106564918766) [Y6 Z7 Z8 Z9 Y10]
+ (0.22848106564918766) [X6 Z7 Z8 Z9 X10]
+ (-0.032767657823290545) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.032767657823290545) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.032767657823290545) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.032767657823290545) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273176) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.027115036845273176) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273176) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.027115036845273176) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.025996177598021274) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.025996177598021274) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.017561202409646176) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.017561202409646176) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.017561202409646176) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.017561202409646176) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.014564531231173017) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564531231173017) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564531231173017) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564531231173017) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.012215040997613924) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.012215040997613924) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.012215040997613924) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.012215040997613924) [X4 Z5 X6 X11 Z12 X13]
+ (-0.012215040997613924) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.012215040997613924) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.012215040997613924) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.012215040997613924) [X5 Z6 X7 X10 Z11 X12]
+ (-0.01175601341981928) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.01175601341981928) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.01175601341981928) [X3 Z4 Z5 X6 X8 X9]
+ (-0.01175601341981928) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.008764827575688812) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.008764827575688812) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.008764827575688812) [X2 Z3 Z4 X5 X11 X12]
+ (-0.008764827575688812) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.008764827575688812) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.008764827575688812) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.008764827575688812) [X3 X4 X10 Z11 Z12 X13]
+ (-0.008764827575688812) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.008125251921381048) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125251921381048) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.00730675992883296) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.00730675992883296) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.00730675992883296) [X4 X5 X7 Z8 Z9 X10]
+ (-0.00730675992883296) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005805188989826899) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.005805188989826899) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.005805188989826899) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.005805188989826899) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.005652620978017366) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.005652620978017366) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.005652620978017366) [X0 X1 X3 Z4 Z5 X6]
+ (-0.005652620978017366) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.005143391768825098) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.005143391768825098) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.005143391768825098) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.005143391768825098) [X2 Z3 X4 X5 Z6 X7]
+ (-0.0046849033881552135) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.0046849033881552135) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668620318776319) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668620318776319) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.004424855449441865) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424855449441865) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.004158797381840039) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.004158797381840039) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.004158797381840039) [X3 Z4 Z5 X6 X12 X13]
+ (-0.004158797381840039) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.003493790359890142) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.003493790359890142) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.003493790359890142) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.003493790359890142) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.002779026799025559) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.002779026799025559) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.002293956611352466) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.002293956611352466) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.0017992194936630366) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.0017992194936630366) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278753941369516) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.0017278753941369516) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298507967730315) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298507967730315) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298507967730315) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298507967730315) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533856254125557) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533856254125557) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008145313270956898) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008145313270956898) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008145313270956898) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008145313270956898) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.73503688059193e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.73503688059193e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.73503688059193e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.73503688059193e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.77481786495197e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.77481786495197e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.77481786495197e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.77481786495197e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.518362215958073e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.518362215958073e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.518362215958073e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.518362215958073e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.4443446762389225e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.4443446762389225e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.4443446762389225e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.4443446762389225e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.524373848762999e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.524373848762999e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.524373848762999e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.524373848762999e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.290028433535531e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.290028433535531e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.290028433535531e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.290028433535531e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.974311713595205e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.974311713595205e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.275883122326555e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.275883122326555e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.643051068641192e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.643051068641192e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.556569218296557e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.556569218296557e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.253224225648817e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.253224225648817e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.769659452209393e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.769659452209393e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.694513294736446e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.694513294736446e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.610297130806459e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.610297130806459e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.610297130806459e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.610297130806459e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.313145500163356e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.313145500163356e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.2774831957521467e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.2774831957521467e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.2774831957521467e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.2774831957521467e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.2112283485996423e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.2112283485996423e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.2112283485996423e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.2112283485996423e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.151346311225156e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.151346311225156e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.0882507114720144e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.0882507114720144e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.172669101587931e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.172669101587931e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.454842448991061e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.454842448991061e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.3304731887130484e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.3304731887130484e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2283337824225419e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2283337824225419e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.0358477600794662e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.0358477600794662e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956895372462184e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956895372462184e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733197742631106e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733197742631106e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733197742631106e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733197742631106e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.628614201844492e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.628614201844492e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.556281914809096e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.556281914809096e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.556281914809096e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.556281914809096e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.418291574775133e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.418291574775133e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.418291574775133e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.418291574775133e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.927453083128119e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927453083128119e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927453083128119e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927453083128119e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627851911518052e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627851911518052e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660624810526e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660624810526e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660624810526e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660624810526e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660624810526e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660624810526e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660624810526e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660624810526e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.838052750819412e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.838052750819412e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.570761329114819e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.570761329114819e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.32813935054313e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.32813935054313e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.086826565037493e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.086826565037493e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.086826565037493e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.086826565037493e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.447323128773102e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.447323128773102e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.371328947871858e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.371328947871858e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.371328947871858e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.371328947871858e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.199051618905852e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.199051618905852e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.9332412770761808e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.9332412770761808e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.9332412770761808e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.9332412770761808e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.839420915428128e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.839420915428128e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.839420915428128e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.839420915428128e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.5510539175941372e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.5510539175941372e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.5510539175941372e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.5510539175941372e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.3807781480622212e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.3807781480622212e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.3807781480622212e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.3807781480622212e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.3807781480622212e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.3807781480622212e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.3807781480622212e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.3807781480622212e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.3807781480622212e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.3807781480622212e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.3807781480622212e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.3807781480622212e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.2919694863364175e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.2919694863364175e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.1076325599660381e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.1076325599660381e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.1076325599660381e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.1076325599660381e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.1076325599660381e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.1076325599660381e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.1076325599660381e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.1076325599660381e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.057446595029861e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057446595029861e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057446595029861e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057446595029861e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649310135828356e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649310135828356e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649310135828356e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649310135828356e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.839420915428128e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.839420915428128e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.839420915428128e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.839420915428128e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.199051618905852e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.199051618905852e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.447323128773102e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.447323128773102e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.2362599614546936e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.2362599614546936e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.2362599614546936e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.2362599614546936e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.32813935054313e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.32813935054313e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.570761329114819e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.570761329114819e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.838052750819412e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.838052750819412e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.627851911518052e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627851911518052e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.628614201844492e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.628614201844492e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956895372462184e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956895372462184e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.306536652166998e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.306536652166998e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.306536652166998e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.306536652166998e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.0358477600794662e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.0358477600794662e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2283337824225419e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2283337824225419e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.239336321720449e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.239336321720449e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.239336321720449e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.239336321720449e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.3304731887130484e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.3304731887130484e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.454842448991061e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.454842448991061e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.172669101587931e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.172669101587931e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.0882507114720144e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.0882507114720144e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.117447946120398e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.117447946120398e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.151346311225156e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.151346311225156e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.313145500163356e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.313145500163356e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.3343312894240017e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.3343312894240017e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.694513294736446e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.694513294736446e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.1839325594363625e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.1839325594363625e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.556569218296557e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.556569218296557e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.643051068641192e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.643051068641192e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.275883122326555e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.275883122326555e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.974311713595205e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.974311713595205e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.0002921986261110867) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.0002921986261110867) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.0002921986261110867) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.0002921986261110867) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004956762314917058) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004956762314917058) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650070219498975) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650070219498975) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650070219498975) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650070219498975) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533856254125557) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533856254125557) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.001609531381721383) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.001609531381721383) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.001609531381721383) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.001609531381721383) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0016676041811440785) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.0016676041811440785) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.0016676041811440785) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.0016676041811440785) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.0017278753941369516) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.0017278753941369516) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.0017992194936630366) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.0017992194936630366) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.002293956611352466) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.002293956611352466) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.0024629170071339386) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.0024629170071339386) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.0024629170071339386) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.0024629170071339386) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.003961560792496544) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.003961560792496544) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.003961560792496544) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.003961560792496544) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.004424855449441865) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424855449441865) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004668620318776319) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668620318776319) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.0046849033881552135) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.0046849033881552135) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005324835234221682) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.005324835234221682) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.005324835234221682) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.005324835234221682) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.005368659358109488) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368659358109488) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368659358109488) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368659358109488) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.007960880725921559) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.007960880725921559) [X4 Z5 X6 X10 Z11 X12]
+ (0.007960880725921559) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.007960880725921559) [X5 Z6 X7 X11 Z12 X13]
+ (0.008125251921381048) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125251921381048) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.00889073152269459) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.00889073152269459) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.00889073152269459) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.00889073152269459) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.010263414868158528) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.010263414868158528) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.010263414868158528) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.010263414868158528) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.010540425907671477) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.010540425907671477) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.010540425907671477) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.010540425907671477) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010960074940542552) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010960074940542552) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010960074940542552) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010960074940542552) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307274008848182) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307274008848182) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.014411099430130934) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411099430130934) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411099430130934) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411099430130934) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.015225630757226624) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.015225630757226624) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.015225630757226624) [X3 Z4 Z5 X6 X10 X11]
+ (0.015225630757226624) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.01558825010238021) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.01558825010238021) [X2 Z3 X4 X10 Z11 X12]
+ (0.01558825010238021) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.01558825010238021) [X3 Z4 X5 X11 Z12 X13]
+ (0.018266834869375508) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.018266834869375508) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.018266834869375508) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.018266834869375508) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.019020423173039935) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.019020423173039935) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.019020423173039935) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.019020423173039935) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.020175921723535484) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.020175921723535484) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.020175921723535484) [X4 Z5 Z6 X7 X11 X12]
+ (0.020175921723535484) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.020175921723535484) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.020175921723535484) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.020175921723535484) [X5 X6 X10 Z11 Z12 X13]
+ (0.020175921723535484) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.024353077678069022) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.024353077678069022) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.024353077678069022) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.024353077678069022) [X2 Z3 X4 X11 Z12 X13]
+ (0.024353077678069022) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.024353077678069022) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.024353077678069022) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.024353077678069022) [X3 Z4 X5 X10 Z11 X12]
+ (0.024389082531149422) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.024389082531149422) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.024389082531149422) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.024389082531149422) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.02510495713884449) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.02510495713884449) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.02510495713884449) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.02510495713884449) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.030787505389143942) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.030787505389143942) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.045879470781298295) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.045879470781298295) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.05600733087780757) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.05600733087780757) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.05600733087780757) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.05600733087780757) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661349) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.05608468124661349) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661349) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.05608468124661349) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.631277928708899e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631277928708899e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.631277928708899e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631277928708899e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.5950860071321926e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.5950860071321926e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.5950860071321915e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.5950860071321915e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.04274327701378428) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.04274327701378428) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (0.042743277013784296) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.042743277013784296) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.0476426121763831) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.0476426121763831) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.0476426121763831) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.0476426121763831) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.04171881383982176) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.04171881383982176) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.04171881383982176) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.04171881383982176) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.03956441632289344) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.03956441632289344) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.03956441632289344) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03956441632289344) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03935916802205318) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.03935916802205318) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.03935916802205318) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.03935916802205318) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.03931805194719761) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03931805194719761) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.03931805194719761) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03931805194719761) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.035608378988312595) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.035608378988312595) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.029903789512624915) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.029903789512624915) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.029903789512624915) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.029903789512624915) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.028730779551905544) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.028730779551905544) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.028730779551905544) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.028730779551905544) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.025637238296026862) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.025637238296026862) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.025637238296026862) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.025637238296026862) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.024755463292891033) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.024755463292891033) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.024755463292891033) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.024755463292891033) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.024282117354693034) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.024282117354693034) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.023145130929529044) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.023145130929529044) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.022528440196012994) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.022528440196012994) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.021433810721601013) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.021433810721601013) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.021433810721601013) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.021433810721601013) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.01925750509525161) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.01925750509525161) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019028242443847338) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.019028242443847338) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.01888903030494293) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.01888903030494293) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.01888903030494293) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.01888903030494293) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.016024603689179545) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.016024603689179545) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.015225630757226624) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225630757226624) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.014603704729162142) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.014603704729162142) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564531231173017) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564531231173017) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.011756013419819277) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.011756013419819277) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.011285190200840893) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.011285190200840893) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.009841749246962614) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009841749246962614) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.009612634606847312) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612634606847312) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612634606847312) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612634606847312) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.008469978791023902) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.008469978791023902) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.007306759928832958) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.007306759928832958) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005923798336561344) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.005923798336561344) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.005652620978017366) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.005652620978017366) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368659358109488) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368659358109488) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.004158797381840039) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.004158797381840039) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0033566705638329065) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.0033566705638329065) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.0033566705638329065) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.0033566705638329065) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.003267513854423567) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.003267513854423567) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.003267513854423567) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.003267513854423567) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.0027790267990255592) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.0027790267990255592) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.0026860409778066054) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.0026860409778066054) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.0026860409778066054) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.0026860409778066054) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.002293956611352466) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.002293956611352466) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.002293956611352466) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.002293956611352466) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.000958165583669654) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.000958165583669654) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.000958165583669654) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.000958165583669654) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.000958165583669654) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.000958165583669654) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.000958165583669654) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.000958165583669654) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.00024636437569582824) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00024636437569582824) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0001384017730354882) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.0001384017730354882) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.0001384017730354882) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.0001384017730354882) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.73503688059193e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.73503688059193e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585306400825e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585306400825e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.6103585306400825e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.6103585306400825e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.531680879595225e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.531680879595225e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.531680879595225e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.531680879595225e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.806102775477574e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.806102775477574e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.806102775477574e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.806102775477574e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.089799467860794e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.089799467860794e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.089799467860794e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.089799467860794e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.6522096699053095e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.6522096699053095e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.6522096699053095e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.6522096699053095e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851834370535e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.481851834370535e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851834370535e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.481851834370535e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.071480736514289e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.071480736514289e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.071480736514289e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.071480736514289e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.734622038963284e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.734622038963284e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.734622038963284e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.734622038963284e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.728843147424739e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.728843147424739e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.728843147424739e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.728843147424739e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.253224225648817e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.253224225648817e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.769659452209393e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.769659452209393e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.5443954294410286e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.5443954294410286e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.5443954294410286e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.5443954294410286e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.5443954294410286e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.5443954294410286e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.5443954294410286e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.5443954294410286e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.360956320436055e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.360956320436055e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.360956320436055e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.360956320436055e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.103215604816008e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.103215604816008e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.103215604816008e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.103215604816008e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.011122098355401e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.011122098355401e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.011122098355401e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.011122098355401e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.942946836829545e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.942946836829545e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.942946836829545e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.942946836829545e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174773183052e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.6541174773183052e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174773183052e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.6541174773183052e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.5224930676360523e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224930676360523e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224930676360523e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224930676360523e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224930676360523e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930676360523e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930676360523e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224930676360523e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.2283337824225419e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337824225419e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.2283337824225419e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337824225419e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.988770288344418e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988770288344418e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988770288344418e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988770288344418e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.867765104485731e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104485731e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104485731e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.867765104485731e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.18999097559791e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.18999097559791e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246207120997e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246207120997e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.471647744841628e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.471647744841628e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.561447179690146e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.561447179690146e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.561447179690146e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.561447179690146e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.5233896780759837e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.5233896780759837e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.4273231086542723e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.4273231086542723e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.4273231086542723e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.4273231086542723e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.32813935054313e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.32813935054313e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.32813935054313e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.32813935054313e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.086826565037494e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.086826565037494e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.8882935951123984e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935951123984e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935951123984e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.8882935951123984e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.371328947871858e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.371328947871858e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.839420915428128e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.839420915428128e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.057446595029861e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057446595029861e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.537178096673486e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.537178096673486e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.537178096673486e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.537178096673486e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.057446595029861e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057446595029861e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.209350646060724e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350646060724e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350646060724e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.209350646060724e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783553477481e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783553477481e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783553477481e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783553477481e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.839420915428128e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.839420915428128e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.371328947871858e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.371328947871858e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.086826565037494e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.086826565037494e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.5233896780759837e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.5233896780759837e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.471647744841628e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.471647744841628e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246207120997e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246207120997e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.18999097559791e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.18999097559791e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.3304731887130484e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.3304731887130484e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.3304731887130484e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.3304731887130484e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288532435891045e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288532435891045e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288532435891045e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288532435891045e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.689348951542492e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.689348951542492e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.689348951542492e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.689348951542492e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.745518400606587e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.745518400606587e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.745518400606587e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.745518400606587e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.745518400606587e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.745518400606587e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.745518400606587e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.745518400606587e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.2118420191785443e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420191785443e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.2118420191785443e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420191785443e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.2118420191785443e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420191785443e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.2118420191785443e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420191785443e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.313145500163356e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.313145500163356e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.313145500163356e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.313145500163356e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.3343312894240017e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.3343312894240017e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.1839325594363625e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.1839325594363625e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.73503688059193e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.73503688059193e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00024636437569582824) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00024636437569582824) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00044585351288407686) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.00044585351288407686) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.00044585351288407686) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.00044585351288407686) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940221543005145) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005145) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005145) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005145) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005145) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005145) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940221543005145) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005145) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533856254125559) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533856254125559) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533856254125559) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533856254125559) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0010435246534907646) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435246534907646) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435246534907646) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435246534907646) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.001280306097349678) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.001280306097349678) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.001280306097349678) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.001280306097349678) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.0013038004788127019) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.0013038004788127019) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.0013038004788127019) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.0013038004788127019) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.002261966062482356) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.002261966062482356) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.002261966062482356) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.002261966062482356) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.002261966062482356) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.002261966062482356) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.002261966062482356) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.002261966062482356) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.003989841456619308) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.003989841456619308) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.003989841456619308) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.003989841456619308) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.004158797381840039) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.004158797381840039) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.004311038507914332) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.004311038507914332) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.004311038507914332) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.004311038507914332) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.0046369766611825845) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.0046369766611825845) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.0046369766611825845) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.0046369766611825845) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.005241535382803892) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.005241535382803892) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.005241535382803892) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.005241535382803892) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.005262642473076834) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262642473076834) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262642473076834) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262642473076834) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368659358109488) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368659358109488) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.00537993715583938) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.00537993715583938) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.00537993715583938) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.00537993715583938) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.005652620978017366) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.005652620978017366) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.00570849598596091) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.00570849598596091) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.00570849598596091) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.00570849598596091) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.005923798336561344) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.005923798336561344) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.007306759928832958) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.007306759928832958) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469978791023902) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.008469978791023902) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.009841749246962614) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.009841749246962614) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.011285190200840893) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.011285190200840893) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.011756013419819277) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.011756013419819277) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.014564531231173017) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564531231173017) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603704729162142) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.014603704729162142) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.015225630757226624) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225630757226624) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.016024603689179545) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.016024603689179545) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.019028242443847338) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.019028242443847338) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.01925750509525161) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.01925750509525161) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.045879470781298295) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.045879470781298295) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.36937089366156156) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.36937089366156156) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.36937089366156156) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.36937089366156156) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.28164257767023076) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.28164257767023076) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.28164257767023065) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.28164257767023065) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.09065144207036484) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.09065144207036484) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.09065144207036484) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.09065144207036484) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863628) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684737589863628) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863628) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684737589863628) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0763502195063503) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0763502195063503) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0763502195063503) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0763502195063503) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214045) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.06752385099214045) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214045) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.06752385099214045) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.035608378988312595) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.035608378988312595) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0349033433736617) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0349033433736617) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0349033433736617) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0349033433736617) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883829964) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.024591860883829964) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883829964) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.024591860883829964) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.024282117354693038) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.024282117354693038) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.023145130929529047) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.023145130929529047) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.022528440196012994) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.022528440196012994) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0195380503113148) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.0195380503113148) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.0195380503113148) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.0195380503113148) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.017091553155898893) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.017091553155898893) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.017091553155898893) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.017091553155898893) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.016024603689179545) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.016024603689179545) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.016024603689179545) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.016024603689179545) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.010311482489831734) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831734) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831734) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311482489831734) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.009841749246962614) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962614) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.009841749246962614) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962614) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.00882636851420985) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.00882636851420985) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.00882636851420985) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.00882636851420985) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.00854199662545486) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00854199662545486) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00854199662545486) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00854199662545486) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.00854199662545486) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00854199662545486) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00854199662545486) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00854199662545486) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008469978791023902) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023902) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.008469978791023902) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023902) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.004668620318776319) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668620318776319) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.003876470899336966) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.003876470899336966) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.003804066171728549) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728549) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728549) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003804066171728549) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178878) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178878) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0033566705638329065) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.0033566705638329065) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.003267513854423567) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.003267513854423567) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.002141361223101666) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.002141361223101666) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.0017278753941369516) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.0017278753941369516) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.0016407548553123803) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0016407548553123803) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214169618) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0014528843214169618) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214169618) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0014528843214169618) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870896771024529) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007870896771024529) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192743499487773) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192743499487773) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.0001940085702975598) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0001940085702975598) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0001384017730354882) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.0001384017730354882) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.141625221161903e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.141625221161903e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.141625221161903e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.141625221161903e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.071480736514289e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.071480736514289e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.151346311225156e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.151346311225156e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0882507114720144e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.0882507114720144e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.9885117062359077e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.9885117062359077e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8742990714699816e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.8742990714699816e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.360956320436055e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.360956320436055e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.3002946562171692e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3002946562171692e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.1468376508002032e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.1468376508002032e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.1468376508002032e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.1468376508002032e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.35233210349929e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.35233210349929e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.35233210349929e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.35233210349929e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091637199377769e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637199377769e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637199377769e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637199377769e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637199377769e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637199377769e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637199377769e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091637199377769e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.074305986287118e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.074305986287118e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.074305986287118e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.074305986287118e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128986700249e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128986700249e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128986700249e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128986700249e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.86776510448573e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.86776510448573e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560692465222565e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692465222565e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692465222565e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692465222565e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692465222565e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692465222565e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692465222565e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560692465222565e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.997018422313706e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.997018422313706e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.997018422313706e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.997018422313706e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.997018422313706e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.997018422313706e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.997018422313706e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.997018422313706e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.5682475213017845e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.5682475213017845e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.5682475213017845e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.5682475213017845e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393086242634e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393086242634e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393086242634e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393086242634e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393086242634e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393086242634e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.3767393086242634e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393086242634e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.8882935951123984e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.8882935951123984e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.686381546690907e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.686381546690907e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035783553477481e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7035783553477481e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.209350646060724e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.209350646060724e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244829652e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773244829652e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244829652e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773244829652e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244829652e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773244829652e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773244829652e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773244829652e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253796460882e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253796460882e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9742253796460882e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9742253796460882e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.047471655625974e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.047471655625974e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.047471655625974e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.047471655625974e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.209350646060724e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.209350646060724e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0717282184992604e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0717282184992604e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0717282184992604e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0717282184992604e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.2004287493932137e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.2004287493932137e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.2004287493932137e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.2004287493932137e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.7035783553477481e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7035783553477481e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.3120943052599916e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.3120943052599916e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.3120943052599916e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.3120943052599916e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.686381546690907e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.686381546690907e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.8882935951123984e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.8882935951123984e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.0922506162970806e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506162970806e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.0922506162970806e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506162970806e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.0922506162970806e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506162970806e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.0922506162970806e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506162970806e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.444597854181886e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.444597854181886e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.444597854181886e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.444597854181886e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.684915095385348e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.684915095385348e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.684915095385348e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.684915095385348e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246974425620257e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246974425620257e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246974425620257e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246974425620257e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246974425620257e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246974425620257e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246974425620257e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246974425620257e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.86776510448573e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.86776510448573e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3002946562171692e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3002946562171692e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.360956320436055e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.360956320436055e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.8742990714699816e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.8742990714699816e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.883676576160043e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.883676576160043e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.947356011841944e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.947356011841944e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.947356011841944e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.947356011841944e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117062359077e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9885117062359077e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.0882507114720144e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.0882507114720144e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.151346311225156e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.151346311225156e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.846201671403347e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846201671403347e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846201671403347e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846201671403347e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.071480736514289e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.071480736514289e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.105526722147387e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.105526722147387e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.105526722147387e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.105526722147387e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146496327620516e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146496327620516e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146496327620516e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146496327620516e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159350502020254e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159350502020254e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159350502020254e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159350502020254e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427988656689345e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427988656689345e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427988656689345e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427988656689345e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935867718077852e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935867718077852e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935867718077852e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935867718077852e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348313339e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.253273348313339e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979825793617369e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979825793617369e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979825793617369e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979825793617369e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.205548411219048e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.205548411219048e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.205548411219048e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.205548411219048e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0001384017730354882) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.0001384017730354882) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.00018787053389541827) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.00018787053389541827) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.00018787053389541827) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00018787053389541827) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0001940085702975598) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.0001940085702975598) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00024636437569582824) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569582824) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00024636437569582824) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569582824) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192743499487773) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192743499487773) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156734248908336) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156734248908336) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156734248908336) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156734248908336) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024529) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007870896771024529) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.001532483523072967) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.001532483523072967) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.001532483523072967) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.001532483523072967) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.0016407548553123803) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0016407548553123803) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0017278753941369516) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.0017278753941369516) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.002446497155415908) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.002446497155415908) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.002446497155415908) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.002446497155415908) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.003267513854423567) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.003267513854423567) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.0033566705638329065) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.0033566705638329065) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.0034841573002178878) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0034841573002178878) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.003876470899336966) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.003876470899336966) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.004668620318776319) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668620318776319) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278073) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.004767272188278073) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.004767272188278073) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278073) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.005286546538226852) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.005286546538226852) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.005286546538226852) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.005286546538226852) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.005408954422409932) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.005408954422409932) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.005408954422409932) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.005408954422409932) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.005923798336561344) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561344) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.005923798336561344) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561344) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.010715508469796754) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010715508469796754) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010715508469796754) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010715508469796754) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.010757563953908941) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010757563953908941) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010757563953908941) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010757563953908941) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.014603704729162142) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.014603704729162142) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.014603704729162142) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.014603704729162142) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.0192995605793638) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0192995605793638) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0192995605793638) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0192995605793638) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0192995605793638) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0192995605793638) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0192995605793638) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0192995605793638) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0585919887338619) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0585919887338619) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.775950527466138e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.775950527466138e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.77595052746614e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.77595052746614e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.07165035181002545) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07165035181002545) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0716503518100255) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0716503518100255) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.01925750509525161) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.01925750509525161) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010311482489831734) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311482489831734) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00882636851420985) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.00882636851420985) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.007597464029770589) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.007597464029770589) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.007597464029770589) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.007597464029770589) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311889) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311889) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311889) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311889) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311889) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311889) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311889) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311889) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676589) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005348051582676589) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005348051582676589) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676589) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.003804066171728549) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003804066171728549) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219486) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.0029841661681219486) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.0029841661681219486) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.0029841661681219486) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.002446497155415908) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.002446497155415908) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.002249412447094001) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.002249412447094001) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.002249412447094001) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002249412447094001) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.002141361223101666) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.002141361223101666) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.0018638942824587006) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587006) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587006) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587006) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587006) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587006) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0018638942824587006) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587006) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0016407548553123803) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553123803) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0016407548553123803) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553123803) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.001222337808153841) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.001222337808153841) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.001222337808153841) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.001222337808153841) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.001222337808153841) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.001222337808153841) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.001222337808153841) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.001222337808153841) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.001028329237856281) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.001028329237856281) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.001028329237856281) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.001028329237856281) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1463061453239628e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1463061453239628e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.8742990714699816e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990714699816e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.8742990714699816e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990714699816e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3002946562171692e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3002946562171692e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3002946562171692e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3002946562171692e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.0444941297873573e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.0444941297873573e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.0444941297873573e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.0444941297873573e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.956079229853998e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.956079229853998e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.956079229853998e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.956079229853998e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105515036992181e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105515036992181e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105515036992181e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105515036992181e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661347213031315e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661347213031315e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661347213031315e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661347213031315e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341413905232e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341413905232e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.18999097559791e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.18999097559791e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876621658022974e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876621658022974e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876621658022974e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876621658022974e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246207120997e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246207120997e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.5233896780759837e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.5233896780759837e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0767325318296864e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.0767325318296864e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0767325318296864e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.0767325318296864e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.013471458980756e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.013471458980756e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.90459988396834e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.90459988396834e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.90459988396834e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.90459988396834e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.666731754327264e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.666731754327264e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.666731754327264e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.666731754327264e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641928618165e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.8505641928618165e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.656930932096111e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.656930932096111e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.656930932096111e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.656930932096111e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.8505641928618165e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.8505641928618165e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.686381546690907e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.686381546690907e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.686381546690907e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.686381546690907e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458980756e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.013471458980756e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.5233896780759837e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.5233896780759837e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.6704023910768674e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.6704023910768674e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.6704023910768674e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.6704023910768674e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246207120997e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246207120997e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.18999097559791e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.18999097559791e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341413905232e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341413905232e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949476488018803e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949476488018803e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.79249395774186e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.79249395774186e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.79249395774186e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.79249395774186e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.8836765761600424e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.8836765761600424e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.9885117062359077e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117062359077e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117062359077e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117062359077e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348313339e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253273348313339e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4017109735638455e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4017109735638455e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4017109735638455e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4017109735638455e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603693380317e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.5809603693380317e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603693380317e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.5809603693380317e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192743499487773) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487773) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192743499487773) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487773) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.0007870896771024528) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024528) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024528) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024528) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.001172634831644189) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.001172634831644189) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.001172634831644189) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.001172634831644189) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012366478019244754) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.0012366478019244754) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.0012366478019244754) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.0012366478019244754) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.00220096406950047) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.00220096406950047) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.00220096406950047) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00220096406950047) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00239497263979803) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00239497263979803) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00239497263979803) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00239497263979803) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00239497263979803) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.00239497263979803) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.00239497263979803) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.00239497263979803) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002446497155415908) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.002446497155415908) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.003804066171728549) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.003804066171728549) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.003876470899336966) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.003876470899336966) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.003876470899336966) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.003876470899336966) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.004220813970046424) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.004220813970046424) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.004220813970046424) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.004220813970046424) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.00882636851420985) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00882636851420985) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.010311482489831734) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311482489831734) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01925750509525161) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01925750509525161) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0585919887338619) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0585919887338619) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3987009014843073e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.3987009014843073e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.398700901484307e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.398700901484307e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178878) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178878) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219486) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.0029841661681219486) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.0001940085702975598) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0001940085702975598) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453239628e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453239628e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.79249395774186e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.79249395774186e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540341413905232e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413905232e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540341413905232e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413905232e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.8505641928618165e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641928618165e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641928618165e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641928618165e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458980756e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471458980756e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458980756e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471458980756e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949476488018803e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949476488018803e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.79249395774186e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.79249395774186e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0001940085702975598) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0001940085702975598) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0029841661681219486) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.0029841661681219486) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.0034841573002178878) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0034841573002178878) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
  (-73.13873231352532) [I0]
+ (-0.18066792656583536) [Z6]
+ (-0.18066792656583536) [Z7]
+ (-0.15961432501810052) [Z4]
+ (-0.1596143250181005) [Z5]
+ (0.17419956155055494) [Z3]
+ (0.1741995615505551) [Z2]
+ (0.22757269005453304) [Z1]
+ (0.22757269005453312) [Z0]
+ (-8.194261372468334e-06) [Y4 Y6]
+ (-8.194261372468334e-06) [X4 X6]
+ (7.954413176363581e-06) [Y5 Y7]
+ (7.954413176363581e-06) [X5 X7]
+ (0.11270386920332226) [Z4 Z6]
+ (0.11270386920332226) [Z5 Z7]
+ (0.11952438964682692) [Z0 Z4]
+ (0.11952438964682692) [Z1 Z5]
+ (0.13401715261963723) [Z0 Z6]
+ (0.13401715261963723) [Z1 Z7]
+ (0.1373495306426134) [Z0 Z5]
+ (0.1373495306426134) [Z1 Z4]
+ (0.13766872645852582) [Z2 Z4]
+ (0.13766872645852582) [Z3 Z5]
+ (0.14138905291942816) [Z4 Z7]
+ (0.14138905291942816) [Z5 Z6]
+ (0.1472294321876618) [Z2 Z5]
+ (0.1472294321876618) [Z3 Z4]
+ (0.149263551473889) [Z4 Z5]
+ (0.14973486803496933) [Z2 Z6]
+ (0.14973486803496933) [Z3 Z7]
+ (0.15138327161428863) [Z0 Z7]
+ (0.15138327161428863) [Z1 Z6]
+ (0.15435748657223636) [Z6 Z7]
+ (0.1558226905155312) [Z2 Z7]
+ (0.1558226905155312) [Z3 Z6]
+ (0.16756653265461285) [Z0 Z2]
+ (0.16756653265461285) [Z1 Z3]
+ (0.18143991440303897) [Z0 Z3]
+ (0.18143991440303897) [Z1 Z2]
+ (0.19392534613270235) [Z0 Z1]
+ (-7.037887510081561e-06) [Y4 Z5 Y6]
+ (-7.037887510081561e-06) [X4 Z5 X6]
+ (-7.037887510081561e-06) [Y5 Z6 Y7]
+ (-7.037887510081561e-06) [X5 Z6 X7]
+ (-0.028685183716105907) [Y4 Y5 X6 X7]
+ (-0.028685183716105907) [X4 X5 Y6 Y7]
+ (-0.01782514099578651) [Y0 Y1 X4 X5]
+ (-0.01782514099578651) [X0 X1 Y4 Y5]
+ (-0.017366118994651427) [Y0 Y1 X6 X7]
+ (-0.017366118994651427) [X0 X1 Y6 Y7]
+ (-0.013873381748426105) [Y0 Y1 X2 X3]
+ (-0.013873381748426105) [X0 X1 Y2 Y3]
+ (-0.009560705729135963) [Y2 Y3 X4 X5]
+ (-0.009560705729135963) [X2 X3 Y4 Y5]
+ (-0.006087822480561857) [Y2 Y3 X6 X7]
+ (-0.006087822480561857) [X2 X3 Y6 Y7]
+ (-0.00029219862611105685) [Y1 Y2 X3 X4]
+ (-0.00029219862611105685) [X1 X2 Y3 Y4]
+ (-8.194261372468334e-06) [Z4 Y5 Z6 Y7]
+ (-8.194261372468334e-06) [Z4 X5 Z6 X7]
+ (-2.890967881738732e-06) [Z0 Y5 Z6 Y7]
+ (-2.890967881738732e-06) [Z0 X5 Z6 X7]
+ (-2.890967881738732e-06) [Z1 Y4 Z5 Y6]
+ (-2.890967881738732e-06) [Z1 X4 Z5 X6]
+ (-1.8551201215818162e-06) [Z0 Y4 Z5 Y6]
+ (-1.8551201215818162e-06) [Z0 X4 Z5 X6]
+ (-1.8551201215818162e-06) [Z1 Y5 Z6 Y7]
+ (-1.8551201215818162e-06) [Z1 X5 Z6 X7]
+ (-1.5973171978656977e-06) [Z2 Y4 Z5 Y6]
+ (-1.5973171978656977e-06) [Z2 X4 Z5 X6]
+ (-1.5973171978656977e-06) [Z3 Y5 Z6 Y7]
+ (-1.5973171978656977e-06) [Z3 X5 Z6 X7]
+ (-1.035847760156916e-06) [Y0 X1 X5 Y6]
+ (-1.035847760156916e-06) [Y0 Y1 Y5 Y6]
+ (-1.035847760156916e-06) [X0 X1 X5 X6]
+ (-1.035847760156916e-06) [X0 Y1 Y5 X6]
+ (-9.344557776761096e-07) [Z2 Y5 Z6 Y7]
+ (-9.344557776761096e-07) [Z2 X5 Z6 X7]
+ (-9.344557776761096e-07) [Z3 Y4 Z5 Y6]
+ (-9.344557776761096e-07) [Z3 X4 Z5 X6]
+ (6.628614201895878e-07) [Y2 X3 X5 Y6]
+ (6.628614201895878e-07) [Y2 Y3 Y5 Y6]
+ (6.628614201895878e-07) [X2 X3 X5 X6]
+ (6.628614201895878e-07) [X2 Y3 Y5 X6]
+ (7.954413176363581e-06) [Y4 Z5 Y6 Z7]
+ (7.954413176363581e-06) [X4 Z5 X6 Z7]
+ (0.00029219862611105685) [Y1 X2 X3 Y4]
+ (0.00029219862611105685) [X1 Y2 Y3 X4]
+ (0.006087822480561857) [Y2 X3 X6 Y7]
+ (0.006087822480561857) [X2 Y3 Y6 X7]
+ (0.009560705729135963) [Y2 X3 X4 Y5]
+ (0.009560705729135963) [X2 Y3 Y4 X5]
+ (0.011307274008848237) [Y1 Z2 Z3 Y5]
+ (0.011307274008848237) [X1 Z2 Z3 X5]
+ (0.013873381748426105) [Y0 X1 X2 Y3]
+ (0.013873381748426105) [X0 Y1 Y2 X3]
+ (0.017366118994651427) [Y0 X1 X6 Y7]
+ (0.017366118994651427) [X0 Y1 Y6 X7]
+ (0.01782514099578651) [Y0 X1 X4 Y5]
+ (0.01782514099578651) [X0 Y1 Y4 X5]
+ (0.028685183716105907) [Y4 X5 X6 Y7]
+ (0.028685183716105907) [X4 Y5 Y6 X7]
+ (0.029812424517345823) [Y0 Z1 Z2 Y4]
+ (0.029812424517345823) [X0 Z1 Z2 X4]
+ (0.029812424517345823) [Y1 Z3 Z4 Y5]
+ (0.029812424517345823) [X1 Z3 Z4 X5]
+ (0.03010462314345688) [Y0 Z1 Z3 Y4]
+ (0.03010462314345688) [X0 Z1 Z3 X4]
+ (0.03010462314345688) [Y1 Z2 Z4 Y5]
+ (0.03010462314345688) [X1 Z2 Z4 X5]
+ (0.030787505389143977) [Y0 Z2 Z3 Y4]
+ (0.030787505389143977) [X0 Z2 Z3 X4]
+ (0.04375263801066028) [Y0 Z1 Z2 Z3 Y4]
+ (0.04375263801066028) [X0 Z1 Z2 Z3 X4]
+ (0.04375263801066028) [Y1 Z2 Z3 Z4 Y5]
+ (0.04375263801066028) [X1 Z2 Z3 Z4 X5]
+ (-0.014564531231173017) [Y1 Z2 Z3 X4 X6 Y7]
+ (-0.014564531231173017) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-0.014564531231173017) [X1 Z2 Z3 X4 X6 X7]
+ (-0.014564531231173017) [X1 Z2 Z3 Y4 Y6 X7]
+ (-6.524373848741596e-06) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (-6.524373848741596e-06) [X0 Z1 Z2 Z3 Z4 X6]
+ (-6.524373848741596e-06) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (-6.524373848741596e-06) [X1 Z2 Z3 Z5 Z6 X7]
+ (-3.769659452054297e-06) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (-3.769659452054297e-06) [X0 Z2 Z3 Z4 Z5 X6]
+ (-3.6102971306665382e-06) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (-3.6102971306665382e-06) [X0 Z1 Z3 Z4 Z5 X6]
+ (-3.6102971306665382e-06) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (-3.6102971306665382e-06) [X1 Z2 Z4 Z5 Z6 X7]
+ (-3.3131455002475043e-06) [Y1 Z2 Z3 Y4 X5 X6]
+ (-3.3131455002475043e-06) [X1 Z2 Z3 X4 Y5 Y6]
+ (-3.2774831955990853e-06) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (-3.2774831955990853e-06) [X0 Z1 Z2 Z4 Z5 X6]
+ (-3.2774831955990853e-06) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (-3.2774831955990853e-06) [X1 Z3 Z4 Z5 Z6 X7]
+ (-3.211228348494092e-06) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (-3.211228348494092e-06) [X0 Z1 Z2 Z3 Z5 X6]
+ (-3.211228348494092e-06) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (-3.211228348494092e-06) [X1 Z2 Z3 Z4 Z6 X7]
+ (-1.035847760156916e-06) [Y0 Y1 X4 Z5 Z6 X7]
+ (-1.035847760156916e-06) [X0 X1 Y4 Z5 Z6 Y7]
+ (-6.628614201895878e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (-6.628614201895878e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (-3.328139350674533e-07) [Y1 X2 X3 Z4 Z5 Y6]
+ (-3.328139350674533e-07) [X1 Y2 Y3 Z4 Z5 X6]
+ (3.328139350674533e-07) [Y1 Y2 X3 Z4 Z5 X6]
+ (3.328139350674533e-07) [X1 X2 Y3 Z4 Z5 Y6]
+ (6.628614201895878e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (6.628614201895878e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (1.035847760156916e-06) [Y0 X1 X4 Z5 Z6 Y7]
+ (1.035847760156916e-06) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.3131455002475043e-06) [Y1 Z2 Z3 X4 X5 Y6]
+ (3.3131455002475043e-06) [X1 Z2 Z3 Y4 Y5 X6]
+ (4.183932559479295e-06) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (4.183932559479295e-06) [X1 Z2 Z3 Z4 Z5 X7]
+ (0.0002921986261110569) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (0.0002921986261110569) [Y0 Z1 Y2 X3 Z4 X5]
+ (0.0002921986261110569) [X0 Z1 X2 Y3 Z4 Y5]
+ (0.0002921986261110569) [X0 Z1 X2 X3 Z4 X5]
+ (0.010540425907671545) [Y0 Z1 Z2 Z3 Y4 Z7]
+ (0.010540425907671545) [X0 Z1 Z2 Z3 X4 Z7]
+ (0.010540425907671545) [Y1 Z2 Z3 Z4 Y5 Z6]
+ (0.010540425907671545) [X1 Z2 Z3 Z4 X5 Z6]
+ (0.011307274008848239) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (0.011307274008848239) [X0 Z1 Z2 Z3 X4 Z5]
+ (0.025104957138844565) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (0.025104957138844565) [X0 Z1 Z2 Z3 X4 Z6]
+ (0.025104957138844565) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (0.025104957138844565) [X1 Z2 Z3 Z4 X5 Z7]
+ (0.030787505389143977) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (0.030787505389143977) [Z0 X1 Z2 Z3 Z4 X5]
+ (-5.1053965498135814e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-5.1053965498135814e-06) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-5.105396549813578e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (-5.105396549813578e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (-0.014564531231173019) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-0.014564531231173019) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-3.769659452054297e-06) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-3.769659452054297e-06) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-3.328139350674533e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-3.328139350674533e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-3.328139350674533e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-3.328139350674533e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (3.3131455002475043e-06) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (3.3131455002475043e-06) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (3.3131455002475043e-06) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (3.3131455002475043e-06) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (4.183932559479295e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (4.183932559479295e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (0.014564531231173019) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (0.014564531231173019) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
 </code>
 </pre>
 </details>

---

## 5. tutorial_qnn_module_tf.html <a name="demo4"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_qnn_module_tf.html):

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

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_qnn_module_tf.html):

```
30/30 - 11s - loss: 0.4997 - accuracy: 0.5000 - val_loss: 0.5081 - val_accuracy: 0.4400
30/30 - 11s - loss: 0.4673 - accuracy: 0.6200 - val_loss: 0.4488 - val_accuracy: 0.6200
30/30 - 10s - loss: 0.3230 - accuracy: 0.8267 - val_loss: 0.2562 - val_accuracy: 0.8400
30/30 - 10s - loss: 0.2124 - accuracy: 0.8867 - val_loss: 0.1997 - val_accuracy: 0.8400
30/30 - 10s - loss: 0.1800 - accuracy: 0.8933 - val_loss: 0.1841 - val_accuracy: 0.8400
30/30 - 10s - loss: 0.1593 - accuracy: 0.8667 - val_loss: 0.2177 - val_accuracy: 0.8400
30/30 - 21s - loss: 0.5189 - accuracy: 0.4000 - val_loss: 0.4945 - val_accuracy: 0.5400
30/30 - 21s - loss: 0.4822 - accuracy: 0.6200 - val_loss: 0.4412 - val_accuracy: 0.7200
30/30 - 20s - loss: 0.3850 - accuracy: 0.7133 - val_loss: 0.2898 - val_accuracy: 0.7800
30/30 - 21s - loss: 0.2720 - accuracy: 0.7867 - val_loss: 0.2185 - val_accuracy: 0.8200
30/30 - 20s - loss: 0.2056 - accuracy: 0.8400 - val_loss: 0.1893 - val_accuracy: 0.8400
30/30 - 21s - loss: 0.1753 - accuracy: 0.8400 - val_loss: 0.1856 - val_accuracy: 0.8400
```

---

## 6. tutorial_jax_transformations.html <a name="demo5"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_jax_transformations.html):

```
No jit time: 0.0080 seconds
First run time: 0.0657 seconds
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_jax_transformations.html):

```
No jit time: 0.0093 seconds
First run time: 0.0760 seconds
```

---

## 7. tutorial_adaptive_circuits.html <a name="demo6"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_adaptive_circuits.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Excitation : [0, 1, 2, 9], Gradient: 0.03426451170168727
Excitation : [0, 1, 3, 8], Gradient: -0.008566127925421876
Excitation : [0, 2], Gradient: -0.013361843799981184
Excitation : [0, 8], Gradient: 0.008127419311874395
Excitation : [1, 3], Gradient: 9.609881567210692e-06
Excitation : [1, 5], Gradient: -0.004875127086708462
Excitation : [1, 7], Gradient: -0.004875127086708459
Excitation : [1, 9], Gradient: -0.00750974882210989
n = 0,  E = -7.85513767 H, t = 1.84 s
n = 1,  E = -7.85585993 H, t = 1.83 s
n = 2,  E = -7.85642249 H, t = 1.83 s
n = 3,  E = -7.85686535 H, t = 1.83 s
n = 4,  E = -7.85721832 H, t = 1.82 s
n = 5,  E = -7.85750361 H, t = 1.82 s
n = 6,  E = -7.85773773 H, t = 1.81 s
n = 7,  E = -7.85793296 H, t = 1.81 s
n = 8,  E = -7.85809846 H, t = 1.80 s
n = 9,  E = -7.85824102 H, t = 2.10 s
n = 10,  E = -7.85836572 H, t = 1.57 s
n = 11,  E = -7.85847636 H, t = 2.05 s
n = 12,  E = -7.85857579 H, t = 1.83 s
n = 13,  E = -7.85866614 H, t = 1.82 s
n = 14,  E = -7.85874902 H, t = 1.83 s
n = 15,  E = -7.85882566 H, t = 1.82 s
n = 16,  E = -7.85889701 H, t = 1.82 s
n = 17,  E = -7.85896378 H, t = 1.81 s
n = 18,  E = -7.85902654 H, t = 1.80 s
n = 19,  E = -7.85908573 H, t = 1.80 s
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

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_adaptive_circuits.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Excitation : [0, 1, 2, 9], Gradient: 0.03426451170168039
Excitation : [0, 1, 3, 8], Gradient: -0.008566127925420207
Excitation : [0, 2], Gradient: -0.013361843799982757
Excitation : [0, 8], Gradient: 0.008127419311872944
Excitation : [1, 3], Gradient: 9.609881567030308e-06
Excitation : [1, 5], Gradient: -0.00487512708670836
Excitation : [1, 7], Gradient: -0.004875127086708365
Excitation : [1, 9], Gradient: -0.007509748822108429
n = 0,  E = -7.85513767 H, t = 2.08 s
n = 1,  E = -7.85585993 H, t = 2.04 s
n = 2,  E = -7.85642249 H, t = 2.02 s
n = 3,  E = -7.85686535 H, t = 2.01 s
n = 4,  E = -7.85721832 H, t = 2.02 s
n = 5,  E = -7.85750361 H, t = 2.02 s
n = 6,  E = -7.85773773 H, t = 2.03 s
n = 7,  E = -7.85793296 H, t = 2.01 s
n = 8,  E = -7.85809846 H, t = 2.00 s
n = 9,  E = -7.85824102 H, t = 2.29 s
n = 10,  E = -7.85836572 H, t = 1.81 s
n = 11,  E = -7.85847636 H, t = 2.27 s
n = 12,  E = -7.85857579 H, t = 2.05 s
n = 13,  E = -7.85866614 H, t = 2.04 s
n = 14,  E = -7.85874902 H, t = 2.03 s
n = 15,  E = -7.85882566 H, t = 2.02 s
n = 16,  E = -7.85889701 H, t = 2.03 s
n = 17,  E = -7.85896378 H, t = 2.04 s
n = 18,  E = -7.85902654 H, t = 2.04 s
n = 19,  E = -7.85908573 H, t = 2.02 s
n = 0,  E = -7.86266587 H, t = 0.09 s
n = 1,  E = -7.86373056 H, t = 0.10 s
n = 2,  E = -7.86443636 H, t = 0.10 s
n = 3,  E = -7.86490587 H, t = 0.10 s
n = 4,  E = -7.86521992 H, t = 0.09 s
n = 5,  E = -7.86543166 H, t = 0.10 s
n = 6,  E = -7.86557597 H, t = 0.10 s
n = 7,  E = -7.86567575 H, t = 0.09 s
n = 8,  E = -7.86574604 H, t = 0.09 s
n = 9,  E = -7.86579669 H, t = 0.09 s
n = 10,  E = -7.86583418 H, t = 0.09 s
n = 11,  E = -7.86586277 H, t = 0.10 s
n = 12,  E = -7.86588528 H, t = 0.09 s
n = 13,  E = -7.86590357 H, t = 0.10 s
n = 14,  E = -7.86591886 H, t = 0.10 s
n = 15,  E = -7.86593199 H, t = 0.09 s
n = 16,  E = -7.86594350 H, t = 0.10 s
n = 17,  E = -7.86595377 H, t = 0.09 s
n = 18,  E = -7.86596307 H, t = 0.09 s
n = 19,  E = -7.86597156 H, t = 0.09 s
 </code>
 </pre>
 </details>

---

## 8. tutorial_quanvolution.html <a name="demo7"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quanvolution.html):

```
   16384/11490434 [..............................] - ETA: 1s
 2424832/11490434 [=====>........................] - ETA: 0s
 8724480/11490434 [=====================>........] - ETA: 0s
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_quanvolution.html):

```
   16384/11490434 [..............................] - ETA: 0s
 3145728/11490434 [=======>......................] - ETA: 0s
 4202496/11490434 [=========>....................] - ETA: 0s
```

---

## 9. tutorial_backprop.html <a name="demo8"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_backprop.html):

```
Forward pass (best of 3): 0.019183498099982897 sec per loop
Gradient computation (best of 3): 5.3253430618000035 sec per loop
6.906059315993843
Forward pass (best of 3): 0.05299311939998006 sec per loop
Backward pass (best of 3): 0.1035131679999722 sec per loop
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_backprop.html):

```
Forward pass (best of 3): 0.021551763199931885 sec per loop
Gradient computation (best of 3): 5.997554504699929 sec per loop
7.758634751975479
Forward pass (best of 3): 0.058801325600052225 sec per loop
Backward pass (best of 3): 0.11322735510002531 sec per loop
```

---

