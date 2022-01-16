Last update: 2022-01-15  23:39:08 (All times shown in Eastern time)
# List of differences in demonstration outputs

# Table of contents

1. [tutorial_qubit_rotation.html](#demo0)
2. [tutorial_quantum_transfer_learning.html](#demo1)
3. [tutorial_adaptive_circuits.html](#demo2)
4. [tutorial_doubly_stochastic.html](#demo3)
5. [tutorial_qgrnn.html](#demo4)
6. [tutorial_quanvolution.html](#demo5)
7. [tutorial_error_mitigation.html](#demo6)
8. [tutorial_expressivity_fourier_series.html](#demo7)
9. [tutorial_kernel_based_training.html](#demo8)
10. [tutorial_rosalin.html](#demo9)
11. [tutorial_backprop.html](#demo10)
12. [tutorial_barren_plateaus.html](#demo11)
13. [tutorial_quantum_chemistry.html](#demo12)
14. [tutorial_falqon.html](#demo13)
15. [tutorial_general_parshift.html](#demo14)
16. [tutorial_qnn_module_tf.html](#demo15)
17. [tutorial_vqt.html](#demo16)
18. [tutorial_measurement_optimize.html](#demo17)
19. [tutorial_jax_transformations.html](#demo18)


Number of demos different/all demos: 19/55

## 1. tutorial_qubit_rotation.html <a name="demo0"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_qubit_rotation.html):

```
0.8515405859048366
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_qubit_rotation.html):

```
0.8515405859048367
```

---

## 2. tutorial_quantum_transfer_learning.html <a name="demo1"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
  9%|9         | 4.20M/44.7M [00:00<00:00, 44.1MB/s]
 24%|##4       | 10.7M/44.7M [00:00<00:00, 58.4MB/s]
 36%|###6      | 16.3M/44.7M [00:00<00:00, 55.3MB/s]
 48%|####8     | 21.6M/44.7M [00:00<00:00, 51.9MB/s]
 60%|######    | 26.9M/44.7M [00:00<00:00, 53.2MB/s]
 72%|#######1  | 32.0M/44.7M [00:00<00:00, 51.4MB/s]
 87%|########7 | 39.0M/44.7M [00:00<00:00, 57.9MB/s]
100%|#########9| 44.5M/44.7M [00:00<00:00, 46.8MB/s]
100%|##########| 44.7M/44.7M [00:00<00:00, 50.8MB/s]
Training started:
Phase: train Epoch: 1/3 Iter: 1/62 Batch time: 0.3879
Phase: train Epoch: 1/3 Iter: 2/62 Batch time: 0.3715
Phase: train Epoch: 1/3 Iter: 3/62 Batch time: 0.3693
Phase: train Epoch: 1/3 Iter: 4/62 Batch time: 0.3690
Phase: train Epoch: 1/3 Iter: 5/62 Batch time: 0.3676
Phase: train Epoch: 1/3 Iter: 6/62 Batch time: 0.3702
Phase: train Epoch: 1/3 Iter: 7/62 Batch time: 0.3766
Phase: train Epoch: 1/3 Iter: 8/62 Batch time: 0.3768
Phase: train Epoch: 1/3 Iter: 9/62 Batch time: 0.3765
Phase: train Epoch: 1/3 Iter: 10/62 Batch time: 0.3765
Phase: train Epoch: 1/3 Iter: 11/62 Batch time: 0.3749
Phase: train Epoch: 1/3 Iter: 12/62 Batch time: 0.3837
Phase: train Epoch: 1/3 Iter: 13/62 Batch time: 0.3807
Phase: train Epoch: 1/3 Iter: 14/62 Batch time: 0.3711
Phase: train Epoch: 1/3 Iter: 15/62 Batch time: 0.3687
Phase: train Epoch: 1/3 Iter: 16/62 Batch time: 0.3881
Phase: train Epoch: 1/3 Iter: 17/62 Batch time: 0.3838
Phase: train Epoch: 1/3 Iter: 18/62 Batch time: 0.3689
Phase: train Epoch: 1/3 Iter: 19/62 Batch time: 0.3691
Phase: train Epoch: 1/3 Iter: 20/62 Batch time: 0.3745
Phase: train Epoch: 1/3 Iter: 21/62 Batch time: 0.3780
Phase: train Epoch: 1/3 Iter: 22/62 Batch time: 0.3685
Phase: train Epoch: 1/3 Iter: 23/62 Batch time: 0.3820
Phase: train Epoch: 1/3 Iter: 24/62 Batch time: 0.3778
Phase: train Epoch: 1/3 Iter: 25/62 Batch time: 0.3786
Phase: train Epoch: 1/3 Iter: 26/62 Batch time: 0.3812
Phase: train Epoch: 1/3 Iter: 27/62 Batch time: 0.3734
Phase: train Epoch: 1/3 Iter: 28/62 Batch time: 0.3781
Phase: train Epoch: 1/3 Iter: 29/62 Batch time: 0.3836
Phase: train Epoch: 1/3 Iter: 30/62 Batch time: 0.3688
Phase: train Epoch: 1/3 Iter: 31/62 Batch time: 0.3752
Phase: train Epoch: 1/3 Iter: 32/62 Batch time: 0.3787
Phase: train Epoch: 1/3 Iter: 33/62 Batch time: 0.3743
Phase: train Epoch: 1/3 Iter: 34/62 Batch time: 0.3754
Phase: train Epoch: 1/3 Iter: 35/62 Batch time: 0.3736
Phase: train Epoch: 1/3 Iter: 36/62 Batch time: 0.3696
Phase: train Epoch: 1/3 Iter: 37/62 Batch time: 0.3720
Phase: train Epoch: 1/3 Iter: 38/62 Batch time: 0.3847
Phase: train Epoch: 1/3 Iter: 39/62 Batch time: 0.3942
Phase: train Epoch: 1/3 Iter: 40/62 Batch time: 0.3808
Phase: train Epoch: 1/3 Iter: 41/62 Batch time: 0.3697
Phase: train Epoch: 1/3 Iter: 42/62 Batch time: 0.3834
Phase: train Epoch: 1/3 Iter: 43/62 Batch time: 0.3723
Phase: train Epoch: 1/3 Iter: 44/62 Batch time: 0.3698
Phase: train Epoch: 1/3 Iter: 45/62 Batch time: 0.3714
Phase: train Epoch: 1/3 Iter: 46/62 Batch time: 0.3744
Phase: train Epoch: 1/3 Iter: 47/62 Batch time: 0.3717
Phase: train Epoch: 1/3 Iter: 48/62 Batch time: 0.3871
Phase: train Epoch: 1/3 Iter: 49/62 Batch time: 0.3710
Phase: train Epoch: 1/3 Iter: 50/62 Batch time: 0.3742
Phase: train Epoch: 1/3 Iter: 51/62 Batch time: 0.3765
Phase: train Epoch: 1/3 Iter: 52/62 Batch time: 0.3718
Phase: train Epoch: 1/3 Iter: 53/62 Batch time: 0.3779
Phase: train Epoch: 1/3 Iter: 54/62 Batch time: 0.3821
Phase: train Epoch: 1/3 Iter: 55/62 Batch time: 0.3817
Phase: train Epoch: 1/3 Iter: 56/62 Batch time: 0.3859
Phase: train Epoch: 1/3 Iter: 57/62 Batch time: 0.3959
Phase: train Epoch: 1/3 Iter: 58/62 Batch time: 0.3780
Phase: train Epoch: 1/3 Iter: 59/62 Batch time: 0.4133
Phase: train Epoch: 1/3 Iter: 60/62 Batch time: 0.4300
Phase: train Epoch: 1/3 Iter: 61/62 Batch time: 0.4107
Phase: train Epoch: 1/3 Loss: 0.6993 Acc: 0.5246
Phase: validation Epoch: 1/3 Iter: 1/39 Batch time: 0.3251
Phase: validation Epoch: 1/3 Iter: 2/39 Batch time: 0.3239
Phase: validation Epoch: 1/3 Iter: 3/39 Batch time: 0.3125
Phase: validation Epoch: 1/3 Iter: 4/39 Batch time: 0.3103
Phase: validation Epoch: 1/3 Iter: 5/39 Batch time: 0.3055
Phase: validation Epoch: 1/3 Iter: 6/39 Batch time: 0.2956
Phase: validation Epoch: 1/3 Iter: 7/39 Batch time: 0.3024
Phase: validation Epoch: 1/3 Iter: 8/39 Batch time: 0.2970
Phase: validation Epoch: 1/3 Iter: 9/39 Batch time: 0.3070
Phase: validation Epoch: 1/3 Iter: 10/39 Batch time: 0.3039
Phase: validation Epoch: 1/3 Iter: 11/39 Batch time: 0.2963
Phase: validation Epoch: 1/3 Iter: 12/39 Batch time: 0.2956
Phase: validation Epoch: 1/3 Iter: 13/39 Batch time: 0.3109
Phase: validation Epoch: 1/3 Iter: 14/39 Batch time: 0.3121
Phase: validation Epoch: 1/3 Iter: 15/39 Batch time: 0.3049
Phase: validation Epoch: 1/3 Iter: 16/39 Batch time: 0.3022
Phase: validation Epoch: 1/3 Iter: 17/39 Batch time: 0.3122
Phase: validation Epoch: 1/3 Iter: 18/39 Batch time: 0.3027
Phase: validation Epoch: 1/3 Iter: 19/39 Batch time: 0.3038
Phase: validation Epoch: 1/3 Iter: 20/39 Batch time: 0.3004
Phase: validation Epoch: 1/3 Iter: 21/39 Batch time: 0.3041
Phase: validation Epoch: 1/3 Iter: 22/39 Batch time: 0.3043
Phase: validation Epoch: 1/3 Iter: 23/39 Batch time: 0.2974
Phase: validation Epoch: 1/3 Iter: 24/39 Batch time: 0.3103
Phase: validation Epoch: 1/3 Iter: 25/39 Batch time: 0.3116
Phase: validation Epoch: 1/3 Iter: 26/39 Batch time: 0.3023
Phase: validation Epoch: 1/3 Iter: 27/39 Batch time: 0.3063
Phase: validation Epoch: 1/3 Iter: 28/39 Batch time: 0.3066
Phase: validation Epoch: 1/3 Iter: 29/39 Batch time: 0.3088
Phase: validation Epoch: 1/3 Iter: 30/39 Batch time: 0.3000
Phase: validation Epoch: 1/3 Iter: 31/39 Batch time: 0.2987
Phase: validation Epoch: 1/3 Iter: 32/39 Batch time: 0.2992
Phase: validation Epoch: 1/3 Iter: 33/39 Batch time: 0.3058
Phase: validation Epoch: 1/3 Iter: 34/39 Batch time: 0.3037
Phase: validation Epoch: 1/3 Iter: 35/39 Batch time: 0.3084
Phase: validation Epoch: 1/3 Iter: 36/39 Batch time: 0.3064
Phase: validation Epoch: 1/3 Iter: 37/39 Batch time: 0.2963
Phase: validation Epoch: 1/3 Iter: 38/39 Batch time: 0.2955
Phase: validation Epoch: 1/3 Iter: 39/39 Batch time: 0.0917
Phase: validation   Epoch: 1/3 Loss: 0.6432 Acc: 0.6536
Phase: train Epoch: 2/3 Iter: 1/62 Batch time: 0.3642
Phase: train Epoch: 2/3 Iter: 2/62 Batch time: 0.3740
Phase: train Epoch: 2/3 Iter: 3/62 Batch time: 0.3807
Phase: train Epoch: 2/3 Iter: 4/62 Batch time: 0.3770
Phase: train Epoch: 2/3 Iter: 5/62 Batch time: 0.3802
Phase: train Epoch: 2/3 Iter: 6/62 Batch time: 0.3729
Phase: train Epoch: 2/3 Iter: 7/62 Batch time: 0.3850
Phase: train Epoch: 2/3 Iter: 8/62 Batch time: 0.3942
Phase: train Epoch: 2/3 Iter: 9/62 Batch time: 0.3760
Phase: train Epoch: 2/3 Iter: 10/62 Batch time: 0.3678
Phase: train Epoch: 2/3 Iter: 11/62 Batch time: 0.3703
Phase: train Epoch: 2/3 Iter: 12/62 Batch time: 0.3788
Phase: train Epoch: 2/3 Iter: 13/62 Batch time: 0.3727
Phase: train Epoch: 2/3 Iter: 14/62 Batch time: 0.3771
Phase: train Epoch: 2/3 Iter: 15/62 Batch time: 0.3756
Phase: train Epoch: 2/3 Iter: 16/62 Batch time: 0.3787
Phase: train Epoch: 2/3 Iter: 17/62 Batch time: 0.3804
Phase: train Epoch: 2/3 Iter: 18/62 Batch time: 0.3808
Phase: train Epoch: 2/3 Iter: 19/62 Batch time: 0.3732
Phase: train Epoch: 2/3 Iter: 20/62 Batch time: 0.3828
Phase: train Epoch: 2/3 Iter: 21/62 Batch time: 0.3816
Phase: train Epoch: 2/3 Iter: 22/62 Batch time: 0.3691
Phase: train Epoch: 2/3 Iter: 23/62 Batch time: 0.3698
Phase: train Epoch: 2/3 Iter: 24/62 Batch time: 0.3680
Phase: train Epoch: 2/3 Iter: 25/62 Batch time: 0.3651
Phase: train Epoch: 2/3 Iter: 26/62 Batch time: 0.3744
Phase: train Epoch: 2/3 Iter: 27/62 Batch time: 0.3812
Phase: train Epoch: 2/3 Iter: 28/62 Batch time: 0.3911
Phase: train Epoch: 2/3 Iter: 29/62 Batch time: 0.3850
Phase: train Epoch: 2/3 Iter: 30/62 Batch time: 0.3823
Phase: train Epoch: 2/3 Iter: 31/62 Batch time: 0.3725
Phase: train Epoch: 2/3 Iter: 32/62 Batch time: 0.3700
Phase: train Epoch: 2/3 Iter: 33/62 Batch time: 0.3808
Phase: train Epoch: 2/3 Iter: 34/62 Batch time: 0.3826
Phase: train Epoch: 2/3 Iter: 35/62 Batch time: 0.3781
Phase: train Epoch: 2/3 Iter: 36/62 Batch time: 0.3752
Phase: train Epoch: 2/3 Iter: 37/62 Batch time: 0.3794
Phase: train Epoch: 2/3 Iter: 38/62 Batch time: 0.3808
Phase: train Epoch: 2/3 Iter: 39/62 Batch time: 0.3719
Phase: train Epoch: 2/3 Iter: 40/62 Batch time: 0.3833
Phase: train Epoch: 2/3 Iter: 41/62 Batch time: 0.3987
Phase: train Epoch: 2/3 Iter: 42/62 Batch time: 0.3835
Phase: train Epoch: 2/3 Iter: 43/62 Batch time: 0.3729
Phase: train Epoch: 2/3 Iter: 44/62 Batch time: 0.3923
Phase: train Epoch: 2/3 Iter: 45/62 Batch time: 0.3720
Phase: train Epoch: 2/3 Iter: 46/62 Batch time: 0.3912
Phase: train Epoch: 2/3 Iter: 47/62 Batch time: 0.3827
Phase: train Epoch: 2/3 Iter: 48/62 Batch time: 0.3801
Phase: train Epoch: 2/3 Iter: 49/62 Batch time: 0.3915
Phase: train Epoch: 2/3 Iter: 50/62 Batch time: 0.3759
Phase: train Epoch: 2/3 Iter: 51/62 Batch time: 0.3956
Phase: train Epoch: 2/3 Iter: 52/62 Batch time: 0.3770
Phase: train Epoch: 2/3 Iter: 53/62 Batch time: 0.3894
Phase: train Epoch: 2/3 Iter: 54/62 Batch time: 0.3833
Phase: train Epoch: 2/3 Iter: 55/62 Batch time: 0.3812
Phase: train Epoch: 2/3 Iter: 56/62 Batch time: 0.3769
Phase: train Epoch: 2/3 Iter: 57/62 Batch time: 0.3947
Phase: train Epoch: 2/3 Iter: 58/62 Batch time: 0.3995
Phase: train Epoch: 2/3 Iter: 59/62 Batch time: 0.3883
Phase: train Epoch: 2/3 Iter: 60/62 Batch time: 0.3861
Phase: train Epoch: 2/3 Iter: 61/62 Batch time: 0.3853
Phase: train Epoch: 2/3 Loss: 0.6141 Acc: 0.7049
Phase: validation Epoch: 2/3 Iter: 1/39 Batch time: 0.3101
Phase: validation Epoch: 2/3 Iter: 2/39 Batch time: 0.3078
Phase: validation Epoch: 2/3 Iter: 3/39 Batch time: 0.3082
Phase: validation Epoch: 2/3 Iter: 4/39 Batch time: 0.3200
Phase: validation Epoch: 2/3 Iter: 5/39 Batch time: 0.3079
Phase: validation Epoch: 2/3 Iter: 6/39 Batch time: 0.3080
Phase: validation Epoch: 2/3 Iter: 7/39 Batch time: 0.3011
Phase: validation Epoch: 2/3 Iter: 8/39 Batch time: 0.3080
Phase: validation Epoch: 2/3 Iter: 9/39 Batch time: 0.3164
Phase: validation Epoch: 2/3 Iter: 10/39 Batch time: 0.3082
Phase: validation Epoch: 2/3 Iter: 11/39 Batch time: 0.3053
Phase: validation Epoch: 2/3 Iter: 12/39 Batch time: 0.3054
Phase: validation Epoch: 2/3 Iter: 13/39 Batch time: 0.3036
Phase: validation Epoch: 2/3 Iter: 14/39 Batch time: 0.3018
Phase: validation Epoch: 2/3 Iter: 15/39 Batch time: 0.3042
Phase: validation Epoch: 2/3 Iter: 16/39 Batch time: 0.3023
Phase: validation Epoch: 2/3 Iter: 17/39 Batch time: 0.3052
Phase: validation Epoch: 2/3 Iter: 18/39 Batch time: 0.3026
Phase: validation Epoch: 2/3 Iter: 19/39 Batch time: 0.2932
Phase: validation Epoch: 2/3 Iter: 20/39 Batch time: 0.3052
Phase: validation Epoch: 2/3 Iter: 21/39 Batch time: 0.2998
Phase: validation Epoch: 2/3 Iter: 22/39 Batch time: 0.3003
Phase: validation Epoch: 2/3 Iter: 23/39 Batch time: 0.3022
Phase: validation Epoch: 2/3 Iter: 24/39 Batch time: 0.3054
Phase: validation Epoch: 2/3 Iter: 25/39 Batch time: 0.2985
Phase: validation Epoch: 2/3 Iter: 26/39 Batch time: 0.3010
Phase: validation Epoch: 2/3 Iter: 27/39 Batch time: 0.3057
Phase: validation Epoch: 2/3 Iter: 28/39 Batch time: 0.3084
Phase: validation Epoch: 2/3 Iter: 29/39 Batch time: 0.3014
Phase: validation Epoch: 2/3 Iter: 30/39 Batch time: 0.3043
Phase: validation Epoch: 2/3 Iter: 31/39 Batch time: 0.3069
Phase: validation Epoch: 2/3 Iter: 32/39 Batch time: 0.3103
Phase: validation Epoch: 2/3 Iter: 33/39 Batch time: 0.3030
Phase: validation Epoch: 2/3 Iter: 34/39 Batch time: 0.3104
Phase: validation Epoch: 2/3 Iter: 35/39 Batch time: 0.3103
Phase: validation Epoch: 2/3 Iter: 36/39 Batch time: 0.3026
Phase: validation Epoch: 2/3 Iter: 37/39 Batch time: 0.3127
Phase: validation Epoch: 2/3 Iter: 38/39 Batch time: 0.3089
Phase: validation Epoch: 2/3 Iter: 39/39 Batch time: 0.0871
Phase: validation   Epoch: 2/3 Loss: 0.5392 Acc: 0.8235
Phase: train Epoch: 3/3 Iter: 1/62 Batch time: 0.3719
Phase: train Epoch: 3/3 Iter: 2/62 Batch time: 0.3879
Phase: train Epoch: 3/3 Iter: 3/62 Batch time: 0.3726
Phase: train Epoch: 3/3 Iter: 4/62 Batch time: 0.3799
Phase: train Epoch: 3/3 Iter: 5/62 Batch time: 0.3786
Phase: train Epoch: 3/3 Iter: 6/62 Batch time: 0.3761
Phase: train Epoch: 3/3 Iter: 7/62 Batch time: 0.3990
Phase: train Epoch: 3/3 Iter: 8/62 Batch time: 0.3728
Phase: train Epoch: 3/3 Iter: 9/62 Batch time: 0.3719
Phase: train Epoch: 3/3 Iter: 10/62 Batch time: 0.3861
Phase: train Epoch: 3/3 Iter: 11/62 Batch time: 0.3811
Phase: train Epoch: 3/3 Iter: 12/62 Batch time: 0.3905
Phase: train Epoch: 3/3 Iter: 13/62 Batch time: 0.3814
Phase: train Epoch: 3/3 Iter: 14/62 Batch time: 0.3735
Phase: train Epoch: 3/3 Iter: 15/62 Batch time: 0.3723
Phase: train Epoch: 3/3 Iter: 16/62 Batch time: 0.3713
Phase: train Epoch: 3/3 Iter: 17/62 Batch time: 0.3793
Phase: train Epoch: 3/3 Iter: 18/62 Batch time: 0.3834
Phase: train Epoch: 3/3 Iter: 19/62 Batch time: 0.4126
Phase: train Epoch: 3/3 Iter: 20/62 Batch time: 0.4252
Phase: train Epoch: 3/3 Iter: 21/62 Batch time: 0.3942
Phase: train Epoch: 3/3 Iter: 22/62 Batch time: 0.3898
Phase: train Epoch: 3/3 Iter: 23/62 Batch time: 0.3784
Phase: train Epoch: 3/3 Iter: 24/62 Batch time: 0.3834
Phase: train Epoch: 3/3 Iter: 25/62 Batch time: 0.3798
Phase: train Epoch: 3/3 Iter: 26/62 Batch time: 0.4096
Phase: train Epoch: 3/3 Iter: 27/62 Batch time: 0.3766
Phase: train Epoch: 3/3 Iter: 28/62 Batch time: 0.3878
Phase: train Epoch: 3/3 Iter: 29/62 Batch time: 0.3668
Phase: train Epoch: 3/3 Iter: 30/62 Batch time: 0.3766
Phase: train Epoch: 3/3 Iter: 31/62 Batch time: 0.3807
Phase: train Epoch: 3/3 Iter: 32/62 Batch time: 0.3735
Phase: train Epoch: 3/3 Iter: 33/62 Batch time: 0.4120
Phase: train Epoch: 3/3 Iter: 34/62 Batch time: 0.3821
Phase: train Epoch: 3/3 Iter: 35/62 Batch time: 0.3825
Phase: train Epoch: 3/3 Iter: 36/62 Batch time: 0.3785
Phase: train Epoch: 3/3 Iter: 37/62 Batch time: 0.3726
Phase: train Epoch: 3/3 Iter: 38/62 Batch time: 0.3868
Phase: train Epoch: 3/3 Iter: 39/62 Batch time: 0.3742
Phase: train Epoch: 3/3 Iter: 40/62 Batch time: 0.3772
Phase: train Epoch: 3/3 Iter: 41/62 Batch time: 0.3767
Phase: train Epoch: 3/3 Iter: 42/62 Batch time: 0.3778
Phase: train Epoch: 3/3 Iter: 43/62 Batch time: 0.3743
Phase: train Epoch: 3/3 Iter: 44/62 Batch time: 0.3732
Phase: train Epoch: 3/3 Iter: 45/62 Batch time: 0.3885
Phase: train Epoch: 3/3 Iter: 46/62 Batch time: 0.3820
Phase: train Epoch: 3/3 Iter: 47/62 Batch time: 0.3762
Phase: train Epoch: 3/3 Iter: 48/62 Batch time: 0.3795
Phase: train Epoch: 3/3 Iter: 49/62 Batch time: 0.3709
Phase: train Epoch: 3/3 Iter: 50/62 Batch time: 0.3733
Phase: train Epoch: 3/3 Iter: 51/62 Batch time: 0.3732
Phase: train Epoch: 3/3 Iter: 52/62 Batch time: 0.3778
Phase: train Epoch: 3/3 Iter: 53/62 Batch time: 0.3709
Phase: train Epoch: 3/3 Iter: 54/62 Batch time: 0.3714
Phase: train Epoch: 3/3 Iter: 55/62 Batch time: 0.3804
Phase: train Epoch: 3/3 Iter: 56/62 Batch time: 0.3845
Phase: train Epoch: 3/3 Iter: 57/62 Batch time: 0.3756
Phase: train Epoch: 3/3 Iter: 58/62 Batch time: 0.3717
Phase: train Epoch: 3/3 Iter: 59/62 Batch time: 0.3765
Phase: train Epoch: 3/3 Iter: 60/62 Batch time: 0.3698
Phase: train Epoch: 3/3 Iter: 61/62 Batch time: 0.3900
Phase: train Epoch: 3/3 Loss: 0.5652 Acc: 0.7336
Phase: validation Epoch: 3/3 Iter: 1/39 Batch time: 0.3121
Phase: validation Epoch: 3/3 Iter: 2/39 Batch time: 0.2990
Phase: validation Epoch: 3/3 Iter: 3/39 Batch time: 0.2942
Phase: validation Epoch: 3/3 Iter: 4/39 Batch time: 0.3059
Phase: validation Epoch: 3/3 Iter: 5/39 Batch time: 0.3024
Phase: validation Epoch: 3/3 Iter: 6/39 Batch time: 0.2967
Phase: validation Epoch: 3/3 Iter: 7/39 Batch time: 0.2991
Phase: validation Epoch: 3/3 Iter: 8/39 Batch time: 0.2930
Phase: validation Epoch: 3/3 Iter: 9/39 Batch time: 0.3020
Phase: validation Epoch: 3/3 Iter: 10/39 Batch time: 0.3038
Phase: validation Epoch: 3/3 Iter: 11/39 Batch time: 0.2949
Phase: validation Epoch: 3/3 Iter: 12/39 Batch time: 0.2987
Phase: validation Epoch: 3/3 Iter: 13/39 Batch time: 0.3003
Phase: validation Epoch: 3/3 Iter: 14/39 Batch time: 0.3050
Phase: validation Epoch: 3/3 Iter: 15/39 Batch time: 0.3039
Phase: validation Epoch: 3/3 Iter: 16/39 Batch time: 0.3093
Phase: validation Epoch: 3/3 Iter: 17/39 Batch time: 0.3085
Phase: validation Epoch: 3/3 Iter: 18/39 Batch time: 0.3055
Phase: validation Epoch: 3/3 Iter: 19/39 Batch time: 0.2994
Phase: validation Epoch: 3/3 Iter: 20/39 Batch time: 0.3026
Phase: validation Epoch: 3/3 Iter: 21/39 Batch time: 0.3014
Phase: validation Epoch: 3/3 Iter: 22/39 Batch time: 0.3138
Phase: validation Epoch: 3/3 Iter: 23/39 Batch time: 0.3066
Phase: validation Epoch: 3/3 Iter: 24/39 Batch time: 0.3003
Phase: validation Epoch: 3/3 Iter: 25/39 Batch time: 0.3046
Phase: validation Epoch: 3/3 Iter: 26/39 Batch time: 0.3015
Phase: validation Epoch: 3/3 Iter: 27/39 Batch time: 0.3020
Phase: validation Epoch: 3/3 Iter: 28/39 Batch time: 0.2990
Phase: validation Epoch: 3/3 Iter: 29/39 Batch time: 0.3056
Phase: validation Epoch: 3/3 Iter: 30/39 Batch time: 0.3035
Phase: validation Epoch: 3/3 Iter: 31/39 Batch time: 0.3083
Phase: validation Epoch: 3/3 Iter: 32/39 Batch time: 0.2978
Phase: validation Epoch: 3/3 Iter: 33/39 Batch time: 0.3040
Phase: validation Epoch: 3/3 Iter: 34/39 Batch time: 0.3286
Phase: validation Epoch: 3/3 Iter: 35/39 Batch time: 0.3083
Phase: validation Epoch: 3/3 Iter: 36/39 Batch time: 0.2968
Phase: validation Epoch: 3/3 Iter: 37/39 Batch time: 0.3046
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
 22%|##2       | 9.98M/44.7M [00:00<00:00, 102MB/s]
 44%|####4     | 19.7M/44.7M [00:00<00:00, 98.9MB/s]
 77%|#######7  | 34.6M/44.7M [00:00<00:00, 124MB/s]
100%|##########| 44.7M/44.7M [00:00<00:00, 127MB/s]
Training started:
Phase: train Epoch: 1/3 Iter: 1/62 Batch time: 0.3715
Phase: train Epoch: 1/3 Iter: 2/62 Batch time: 0.3696
Phase: train Epoch: 1/3 Iter: 3/62 Batch time: 0.3774
Phase: train Epoch: 1/3 Iter: 4/62 Batch time: 0.3831
Phase: train Epoch: 1/3 Iter: 5/62 Batch time: 0.3657
Phase: train Epoch: 1/3 Iter: 6/62 Batch time: 0.3741
Phase: train Epoch: 1/3 Iter: 7/62 Batch time: 0.3757
Phase: train Epoch: 1/3 Iter: 8/62 Batch time: 0.3702
Phase: train Epoch: 1/3 Iter: 9/62 Batch time: 0.3999
Phase: train Epoch: 1/3 Iter: 10/62 Batch time: 0.3647
Phase: train Epoch: 1/3 Iter: 11/62 Batch time: 0.4034
Phase: train Epoch: 1/3 Iter: 12/62 Batch time: 0.3818
Phase: train Epoch: 1/3 Iter: 13/62 Batch time: 0.3685
Phase: train Epoch: 1/3 Iter: 14/62 Batch time: 0.3760
Phase: train Epoch: 1/3 Iter: 15/62 Batch time: 0.3715
Phase: train Epoch: 1/3 Iter: 16/62 Batch time: 0.3742
Phase: train Epoch: 1/3 Iter: 17/62 Batch time: 0.3816
Phase: train Epoch: 1/3 Iter: 18/62 Batch time: 0.3857
Phase: train Epoch: 1/3 Iter: 19/62 Batch time: 0.3758
Phase: train Epoch: 1/3 Iter: 20/62 Batch time: 0.3920
Phase: train Epoch: 1/3 Iter: 21/62 Batch time: 0.3872
Phase: train Epoch: 1/3 Iter: 22/62 Batch time: 0.3872
Phase: train Epoch: 1/3 Iter: 23/62 Batch time: 0.3903
Phase: train Epoch: 1/3 Iter: 24/62 Batch time: 0.3800
Phase: train Epoch: 1/3 Iter: 25/62 Batch time: 0.3721
Phase: train Epoch: 1/3 Iter: 26/62 Batch time: 0.3688
Phase: train Epoch: 1/3 Iter: 27/62 Batch time: 0.3606
Phase: train Epoch: 1/3 Iter: 28/62 Batch time: 0.3720
Phase: train Epoch: 1/3 Iter: 29/62 Batch time: 0.3754
Phase: train Epoch: 1/3 Iter: 30/62 Batch time: 0.3760
Phase: train Epoch: 1/3 Iter: 31/62 Batch time: 0.3750
Phase: train Epoch: 1/3 Iter: 32/62 Batch time: 0.3770
Phase: train Epoch: 1/3 Iter: 33/62 Batch time: 0.3752
Phase: train Epoch: 1/3 Iter: 34/62 Batch time: 0.3742
Phase: train Epoch: 1/3 Iter: 35/62 Batch time: 0.3796
Phase: train Epoch: 1/3 Iter: 36/62 Batch time: 0.3678
Phase: train Epoch: 1/3 Iter: 37/62 Batch time: 0.3710
Phase: train Epoch: 1/3 Iter: 38/62 Batch time: 0.3710
Phase: train Epoch: 1/3 Iter: 39/62 Batch time: 0.3681
Phase: train Epoch: 1/3 Iter: 40/62 Batch time: 0.3930
Phase: train Epoch: 1/3 Iter: 41/62 Batch time: 0.3678
Phase: train Epoch: 1/3 Iter: 42/62 Batch time: 0.3618
Phase: train Epoch: 1/3 Iter: 43/62 Batch time: 0.3733
Phase: train Epoch: 1/3 Iter: 44/62 Batch time: 0.3606
Phase: train Epoch: 1/3 Iter: 45/62 Batch time: 0.3702
Phase: train Epoch: 1/3 Iter: 46/62 Batch time: 0.3727
Phase: train Epoch: 1/3 Iter: 47/62 Batch time: 0.3752
Phase: train Epoch: 1/3 Iter: 48/62 Batch time: 0.3675
Phase: train Epoch: 1/3 Iter: 49/62 Batch time: 0.3676
Phase: train Epoch: 1/3 Iter: 50/62 Batch time: 0.3777
Phase: train Epoch: 1/3 Iter: 51/62 Batch time: 0.3536
Phase: train Epoch: 1/3 Iter: 52/62 Batch time: 0.3802
Phase: train Epoch: 1/3 Iter: 53/62 Batch time: 0.3593
Phase: train Epoch: 1/3 Iter: 54/62 Batch time: 0.3738
Phase: train Epoch: 1/3 Iter: 55/62 Batch time: 0.3644
Phase: train Epoch: 1/3 Iter: 56/62 Batch time: 0.3634
Phase: train Epoch: 1/3 Iter: 57/62 Batch time: 0.3778
Phase: train Epoch: 1/3 Iter: 58/62 Batch time: 0.3505
Phase: train Epoch: 1/3 Iter: 59/62 Batch time: 0.3733
Phase: train Epoch: 1/3 Iter: 60/62 Batch time: 0.3687
Phase: train Epoch: 1/3 Iter: 61/62 Batch time: 0.3659
Phase: train Epoch: 1/3 Loss: 0.6993 Acc: 0.5246
Phase: validation Epoch: 1/3 Iter: 1/39 Batch time: 0.2982
Phase: validation Epoch: 1/3 Iter: 2/39 Batch time: 0.2885
Phase: validation Epoch: 1/3 Iter: 3/39 Batch time: 0.2813
Phase: validation Epoch: 1/3 Iter: 4/39 Batch time: 0.2930
Phase: validation Epoch: 1/3 Iter: 5/39 Batch time: 0.2990
Phase: validation Epoch: 1/3 Iter: 6/39 Batch time: 0.2907
Phase: validation Epoch: 1/3 Iter: 7/39 Batch time: 0.2898
Phase: validation Epoch: 1/3 Iter: 8/39 Batch time: 0.2940
Phase: validation Epoch: 1/3 Iter: 9/39 Batch time: 0.2987
Phase: validation Epoch: 1/3 Iter: 10/39 Batch time: 0.3186
Phase: validation Epoch: 1/3 Iter: 11/39 Batch time: 0.3075
Phase: validation Epoch: 1/3 Iter: 12/39 Batch time: 0.2997
Phase: validation Epoch: 1/3 Iter: 13/39 Batch time: 0.3004
Phase: validation Epoch: 1/3 Iter: 14/39 Batch time: 0.3121
Phase: validation Epoch: 1/3 Iter: 15/39 Batch time: 0.2962
Phase: validation Epoch: 1/3 Iter: 16/39 Batch time: 0.3188
Phase: validation Epoch: 1/3 Iter: 17/39 Batch time: 0.3055
Phase: validation Epoch: 1/3 Iter: 18/39 Batch time: 0.3026
Phase: validation Epoch: 1/3 Iter: 19/39 Batch time: 0.3060
Phase: validation Epoch: 1/3 Iter: 20/39 Batch time: 0.3108
Phase: validation Epoch: 1/3 Iter: 21/39 Batch time: 0.3013
Phase: validation Epoch: 1/3 Iter: 22/39 Batch time: 0.3007
Phase: validation Epoch: 1/3 Iter: 23/39 Batch time: 0.2985
Phase: validation Epoch: 1/3 Iter: 24/39 Batch time: 0.2988
Phase: validation Epoch: 1/3 Iter: 25/39 Batch time: 0.2980
Phase: validation Epoch: 1/3 Iter: 26/39 Batch time: 0.2947
Phase: validation Epoch: 1/3 Iter: 27/39 Batch time: 0.2938
Phase: validation Epoch: 1/3 Iter: 28/39 Batch time: 0.2944
Phase: validation Epoch: 1/3 Iter: 29/39 Batch time: 0.3014
Phase: validation Epoch: 1/3 Iter: 30/39 Batch time: 0.2993
Phase: validation Epoch: 1/3 Iter: 31/39 Batch time: 0.3175
Phase: validation Epoch: 1/3 Iter: 32/39 Batch time: 0.3172
Phase: validation Epoch: 1/3 Iter: 33/39 Batch time: 0.2993
Phase: validation Epoch: 1/3 Iter: 34/39 Batch time: 0.3008
Phase: validation Epoch: 1/3 Iter: 35/39 Batch time: 0.2975
Phase: validation Epoch: 1/3 Iter: 36/39 Batch time: 0.2922
Phase: validation Epoch: 1/3 Iter: 37/39 Batch time: 0.2974
Phase: validation Epoch: 1/3 Iter: 38/39 Batch time: 0.2964
Phase: validation Epoch: 1/3 Iter: 39/39 Batch time: 0.0928
Phase: validation   Epoch: 1/3 Loss: 0.6432 Acc: 0.6536
Phase: train Epoch: 2/3 Iter: 1/62 Batch time: 0.3790
Phase: train Epoch: 2/3 Iter: 2/62 Batch time: 0.3683
Phase: train Epoch: 2/3 Iter: 3/62 Batch time: 0.3746
Phase: train Epoch: 2/3 Iter: 4/62 Batch time: 0.3819
Phase: train Epoch: 2/3 Iter: 5/62 Batch time: 0.3672
Phase: train Epoch: 2/3 Iter: 6/62 Batch time: 0.3711
Phase: train Epoch: 2/3 Iter: 7/62 Batch time: 0.3691
Phase: train Epoch: 2/3 Iter: 8/62 Batch time: 0.3892
Phase: train Epoch: 2/3 Iter: 9/62 Batch time: 0.3759
Phase: train Epoch: 2/3 Iter: 10/62 Batch time: 0.3944
Phase: train Epoch: 2/3 Iter: 11/62 Batch time: 0.3637
Phase: train Epoch: 2/3 Iter: 12/62 Batch time: 0.3703
Phase: train Epoch: 2/3 Iter: 13/62 Batch time: 0.3635
Phase: train Epoch: 2/3 Iter: 14/62 Batch time: 0.4211
Phase: train Epoch: 2/3 Iter: 15/62 Batch time: 0.3805
Phase: train Epoch: 2/3 Iter: 16/62 Batch time: 0.3732
Phase: train Epoch: 2/3 Iter: 17/62 Batch time: 0.3824
Phase: train Epoch: 2/3 Iter: 18/62 Batch time: 0.3759
Phase: train Epoch: 2/3 Iter: 19/62 Batch time: 0.3640
Phase: train Epoch: 2/3 Iter: 20/62 Batch time: 0.3732
Phase: train Epoch: 2/3 Iter: 21/62 Batch time: 0.3609
Phase: train Epoch: 2/3 Iter: 22/62 Batch time: 0.3691
Phase: train Epoch: 2/3 Iter: 23/62 Batch time: 0.3626
Phase: train Epoch: 2/3 Iter: 24/62 Batch time: 0.3636
Phase: train Epoch: 2/3 Iter: 25/62 Batch time: 0.3961
Phase: train Epoch: 2/3 Iter: 26/62 Batch time: 0.3698
Phase: train Epoch: 2/3 Iter: 27/62 Batch time: 0.3704
Phase: train Epoch: 2/3 Iter: 28/62 Batch time: 0.3560
Phase: train Epoch: 2/3 Iter: 29/62 Batch time: 0.3722
Phase: train Epoch: 2/3 Iter: 30/62 Batch time: 0.3632
Phase: train Epoch: 2/3 Iter: 31/62 Batch time: 0.3575
Phase: train Epoch: 2/3 Iter: 32/62 Batch time: 0.3697
Phase: train Epoch: 2/3 Iter: 33/62 Batch time: 0.3643
Phase: train Epoch: 2/3 Iter: 34/62 Batch time: 0.3651
Phase: train Epoch: 2/3 Iter: 35/62 Batch time: 0.3622
Phase: train Epoch: 2/3 Iter: 36/62 Batch time: 0.3620
Phase: train Epoch: 2/3 Iter: 37/62 Batch time: 0.3666
Phase: train Epoch: 2/3 Iter: 38/62 Batch time: 0.3600
Phase: train Epoch: 2/3 Iter: 39/62 Batch time: 0.3610
Phase: train Epoch: 2/3 Iter: 40/62 Batch time: 0.3925
Phase: train Epoch: 2/3 Iter: 41/62 Batch time: 0.3705
Phase: train Epoch: 2/3 Iter: 42/62 Batch time: 0.3815
Phase: train Epoch: 2/3 Iter: 43/62 Batch time: 0.3694
Phase: train Epoch: 2/3 Iter: 44/62 Batch time: 0.3685
Phase: train Epoch: 2/3 Iter: 45/62 Batch time: 0.3783
Phase: train Epoch: 2/3 Iter: 46/62 Batch time: 0.3741
Phase: train Epoch: 2/3 Iter: 47/62 Batch time: 0.3704
Phase: train Epoch: 2/3 Iter: 48/62 Batch time: 0.3715
Phase: train Epoch: 2/3 Iter: 49/62 Batch time: 0.3753
Phase: train Epoch: 2/3 Iter: 50/62 Batch time: 0.3862
Phase: train Epoch: 2/3 Iter: 51/62 Batch time: 0.3884
Phase: train Epoch: 2/3 Iter: 52/62 Batch time: 0.4013
Phase: train Epoch: 2/3 Iter: 53/62 Batch time: 0.3724
Phase: train Epoch: 2/3 Iter: 54/62 Batch time: 0.3847
Phase: train Epoch: 2/3 Iter: 55/62 Batch time: 0.3870
Phase: train Epoch: 2/3 Iter: 56/62 Batch time: 0.3784
Phase: train Epoch: 2/3 Iter: 57/62 Batch time: 0.3785
Phase: train Epoch: 2/3 Iter: 58/62 Batch time: 0.3990
Phase: train Epoch: 2/3 Iter: 59/62 Batch time: 0.3710
Phase: train Epoch: 2/3 Iter: 60/62 Batch time: 0.3661
Phase: train Epoch: 2/3 Iter: 61/62 Batch time: 0.3718
Phase: train Epoch: 2/3 Loss: 0.6141 Acc: 0.7049
Phase: validation Epoch: 2/3 Iter: 1/39 Batch time: 0.2853
Phase: validation Epoch: 2/3 Iter: 2/39 Batch time: 0.3009
Phase: validation Epoch: 2/3 Iter: 3/39 Batch time: 0.3100
Phase: validation Epoch: 2/3 Iter: 4/39 Batch time: 0.2962
Phase: validation Epoch: 2/3 Iter: 5/39 Batch time: 0.3042
Phase: validation Epoch: 2/3 Iter: 6/39 Batch time: 0.3059
Phase: validation Epoch: 2/3 Iter: 7/39 Batch time: 0.3108
Phase: validation Epoch: 2/3 Iter: 8/39 Batch time: 0.3043
Phase: validation Epoch: 2/3 Iter: 9/39 Batch time: 0.3006
Phase: validation Epoch: 2/3 Iter: 10/39 Batch time: 0.2960
Phase: validation Epoch: 2/3 Iter: 11/39 Batch time: 0.3026
Phase: validation Epoch: 2/3 Iter: 12/39 Batch time: 0.2969
Phase: validation Epoch: 2/3 Iter: 13/39 Batch time: 0.2977
Phase: validation Epoch: 2/3 Iter: 14/39 Batch time: 0.3061
Phase: validation Epoch: 2/3 Iter: 15/39 Batch time: 0.3161
Phase: validation Epoch: 2/3 Iter: 16/39 Batch time: 0.3051
Phase: validation Epoch: 2/3 Iter: 17/39 Batch time: 0.3014
Phase: validation Epoch: 2/3 Iter: 18/39 Batch time: 0.3095
Phase: validation Epoch: 2/3 Iter: 19/39 Batch time: 0.3125
Phase: validation Epoch: 2/3 Iter: 20/39 Batch time: 0.3109
Phase: validation Epoch: 2/3 Iter: 21/39 Batch time: 0.3148
Phase: validation Epoch: 2/3 Iter: 22/39 Batch time: 0.2993
Phase: validation Epoch: 2/3 Iter: 23/39 Batch time: 0.3165
Phase: validation Epoch: 2/3 Iter: 24/39 Batch time: 0.3023
Phase: validation Epoch: 2/3 Iter: 25/39 Batch time: 0.2944
Phase: validation Epoch: 2/3 Iter: 26/39 Batch time: 0.3084
Phase: validation Epoch: 2/3 Iter: 27/39 Batch time: 0.3163
Phase: validation Epoch: 2/3 Iter: 28/39 Batch time: 0.3001
Phase: validation Epoch: 2/3 Iter: 29/39 Batch time: 0.2893
Phase: validation Epoch: 2/3 Iter: 30/39 Batch time: 0.2864
Phase: validation Epoch: 2/3 Iter: 31/39 Batch time: 0.3117
Phase: validation Epoch: 2/3 Iter: 32/39 Batch time: 0.2947
Phase: validation Epoch: 2/3 Iter: 33/39 Batch time: 0.3149
Phase: validation Epoch: 2/3 Iter: 34/39 Batch time: 0.2954
Phase: validation Epoch: 2/3 Iter: 35/39 Batch time: 0.3124
Phase: validation Epoch: 2/3 Iter: 36/39 Batch time: 0.2985
Phase: validation Epoch: 2/3 Iter: 37/39 Batch time: 0.2935
Phase: validation Epoch: 2/3 Iter: 38/39 Batch time: 0.3002
Phase: validation Epoch: 2/3 Iter: 39/39 Batch time: 0.0876
Phase: validation   Epoch: 2/3 Loss: 0.5392 Acc: 0.8235
Phase: train Epoch: 3/3 Iter: 1/62 Batch time: 0.3503
Phase: train Epoch: 3/3 Iter: 2/62 Batch time: 0.3792
Phase: train Epoch: 3/3 Iter: 3/62 Batch time: 0.3757
Phase: train Epoch: 3/3 Iter: 4/62 Batch time: 0.3596
Phase: train Epoch: 3/3 Iter: 5/62 Batch time: 0.3625
Phase: train Epoch: 3/3 Iter: 6/62 Batch time: 0.3893
Phase: train Epoch: 3/3 Iter: 7/62 Batch time: 0.3696
Phase: train Epoch: 3/3 Iter: 8/62 Batch time: 0.3721
Phase: train Epoch: 3/3 Iter: 9/62 Batch time: 0.3795
Phase: train Epoch: 3/3 Iter: 10/62 Batch time: 0.3960
Phase: train Epoch: 3/3 Iter: 11/62 Batch time: 0.3758
Phase: train Epoch: 3/3 Iter: 12/62 Batch time: 0.3806
Phase: train Epoch: 3/3 Iter: 13/62 Batch time: 0.3728
Phase: train Epoch: 3/3 Iter: 14/62 Batch time: 0.3735
Phase: train Epoch: 3/3 Iter: 15/62 Batch time: 0.3923
Phase: train Epoch: 3/3 Iter: 16/62 Batch time: 0.3873
Phase: train Epoch: 3/3 Iter: 17/62 Batch time: 0.3838
Phase: train Epoch: 3/3 Iter: 18/62 Batch time: 0.3663
Phase: train Epoch: 3/3 Iter: 19/62 Batch time: 0.3819
Phase: train Epoch: 3/3 Iter: 20/62 Batch time: 0.3822
Phase: train Epoch: 3/3 Iter: 21/62 Batch time: 0.4091
Phase: train Epoch: 3/3 Iter: 22/62 Batch time: 0.3986
Phase: train Epoch: 3/3 Iter: 23/62 Batch time: 0.3862
Phase: train Epoch: 3/3 Iter: 24/62 Batch time: 0.3817
Phase: train Epoch: 3/3 Iter: 25/62 Batch time: 0.3857
Phase: train Epoch: 3/3 Iter: 26/62 Batch time: 0.3861
Phase: train Epoch: 3/3 Iter: 27/62 Batch time: 0.3828
Phase: train Epoch: 3/3 Iter: 28/62 Batch time: 0.3759
Phase: train Epoch: 3/3 Iter: 29/62 Batch time: 0.3948
Phase: train Epoch: 3/3 Iter: 30/62 Batch time: 0.3898
Phase: train Epoch: 3/3 Iter: 31/62 Batch time: 0.3917
Phase: train Epoch: 3/3 Iter: 32/62 Batch time: 0.3748
Phase: train Epoch: 3/3 Iter: 33/62 Batch time: 0.4079
Phase: train Epoch: 3/3 Iter: 34/62 Batch time: 0.3851
Phase: train Epoch: 3/3 Iter: 35/62 Batch time: 0.3722
Phase: train Epoch: 3/3 Iter: 36/62 Batch time: 0.3765
Phase: train Epoch: 3/3 Iter: 37/62 Batch time: 0.3729
Phase: train Epoch: 3/3 Iter: 38/62 Batch time: 0.3700
Phase: train Epoch: 3/3 Iter: 39/62 Batch time: 0.3693
Phase: train Epoch: 3/3 Iter: 40/62 Batch time: 0.3732
Phase: train Epoch: 3/3 Iter: 41/62 Batch time: 0.3601
Phase: train Epoch: 3/3 Iter: 42/62 Batch time: 0.3723
Phase: train Epoch: 3/3 Iter: 43/62 Batch time: 0.3815
Phase: train Epoch: 3/3 Iter: 44/62 Batch time: 0.3828
Phase: train Epoch: 3/3 Iter: 45/62 Batch time: 0.3848
Phase: train Epoch: 3/3 Iter: 46/62 Batch time: 0.3829
Phase: train Epoch: 3/3 Iter: 47/62 Batch time: 0.3880
Phase: train Epoch: 3/3 Iter: 48/62 Batch time: 0.3791
Phase: train Epoch: 3/3 Iter: 49/62 Batch time: 0.3830
Phase: train Epoch: 3/3 Iter: 50/62 Batch time: 0.3634
Phase: train Epoch: 3/3 Iter: 51/62 Batch time: 0.3757
Phase: train Epoch: 3/3 Iter: 52/62 Batch time: 0.3775
Phase: train Epoch: 3/3 Iter: 53/62 Batch time: 0.3858
Phase: train Epoch: 3/3 Iter: 54/62 Batch time: 0.3655
Phase: train Epoch: 3/3 Iter: 55/62 Batch time: 0.3657
Phase: train Epoch: 3/3 Iter: 56/62 Batch time: 0.3824
Phase: train Epoch: 3/3 Iter: 57/62 Batch time: 0.3720
Phase: train Epoch: 3/3 Iter: 58/62 Batch time: 0.3765
Phase: train Epoch: 3/3 Iter: 59/62 Batch time: 0.3677
Phase: train Epoch: 3/3 Iter: 60/62 Batch time: 0.3683
Phase: train Epoch: 3/3 Iter: 61/62 Batch time: 0.3810
Phase: train Epoch: 3/3 Loss: 0.5652 Acc: 0.7336
Phase: validation Epoch: 3/3 Iter: 1/39 Batch time: 0.3078
Phase: validation Epoch: 3/3 Iter: 2/39 Batch time: 0.3041
Phase: validation Epoch: 3/3 Iter: 3/39 Batch time: 0.3157
Phase: validation Epoch: 3/3 Iter: 4/39 Batch time: 0.3102
Phase: validation Epoch: 3/3 Iter: 5/39 Batch time: 0.2970
Phase: validation Epoch: 3/3 Iter: 6/39 Batch time: 0.3022
Phase: validation Epoch: 3/3 Iter: 7/39 Batch time: 0.2983
Phase: validation Epoch: 3/3 Iter: 8/39 Batch time: 0.3509
Phase: validation Epoch: 3/3 Iter: 9/39 Batch time: 0.2964
Phase: validation Epoch: 3/3 Iter: 10/39 Batch time: 0.3261
Phase: validation Epoch: 3/3 Iter: 11/39 Batch time: 0.3176
Phase: validation Epoch: 3/3 Iter: 12/39 Batch time: 0.3080
Phase: validation Epoch: 3/3 Iter: 13/39 Batch time: 0.3124
Phase: validation Epoch: 3/3 Iter: 14/39 Batch time: 0.3080
Phase: validation Epoch: 3/3 Iter: 15/39 Batch time: 0.3087
Phase: validation Epoch: 3/3 Iter: 16/39 Batch time: 0.3033
Phase: validation Epoch: 3/3 Iter: 17/39 Batch time: 0.3221
Phase: validation Epoch: 3/3 Iter: 18/39 Batch time: 0.3276
Phase: validation Epoch: 3/3 Iter: 19/39 Batch time: 0.3016
Phase: validation Epoch: 3/3 Iter: 20/39 Batch time: 0.3061
Phase: validation Epoch: 3/3 Iter: 21/39 Batch time: 0.3127
Phase: validation Epoch: 3/3 Iter: 22/39 Batch time: 0.3062
Phase: validation Epoch: 3/3 Iter: 23/39 Batch time: 0.3081
Phase: validation Epoch: 3/3 Iter: 24/39 Batch time: 0.3139
Phase: validation Epoch: 3/3 Iter: 25/39 Batch time: 0.2993
Phase: validation Epoch: 3/3 Iter: 26/39 Batch time: 0.3216
Phase: validation Epoch: 3/3 Iter: 27/39 Batch time: 0.2886
Phase: validation Epoch: 3/3 Iter: 28/39 Batch time: 0.3016
Phase: validation Epoch: 3/3 Iter: 29/39 Batch time: 0.3013
Phase: validation Epoch: 3/3 Iter: 30/39 Batch time: 0.3008
Phase: validation Epoch: 3/3 Iter: 31/39 Batch time: 0.3090
Phase: validation Epoch: 3/3 Iter: 32/39 Batch time: 0.3019
Phase: validation Epoch: 3/3 Iter: 33/39 Batch time: 0.3009
Phase: validation Epoch: 3/3 Iter: 34/39 Batch time: 0.2945
Phase: validation Epoch: 3/3 Iter: 35/39 Batch time: 0.2922
Phase: validation Epoch: 3/3 Iter: 36/39 Batch time: 0.2901
Phase: validation Epoch: 3/3 Iter: 37/39 Batch time: 0.3035
Phase: validation Epoch: 3/3 Iter: 38/39 Batch time: 0.2913
Phase: validation Epoch: 3/3 Iter: 39/39 Batch time: 0.0872
Phase: validation   Epoch: 3/3 Loss: 0.4484 Acc: 0.8497
Training completed in 1m 54s
Best test loss: 0.4484 | Best test accuracy: 0.8497
 </code>
 </pre>
 </details>

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
Excitation : [0, 1, 2, 9], Gradient: 0.03426451170172444
Excitation : [0, 1, 3, 8], Gradient: -0.008566127925431148
Excitation : [0, 2], Gradient: -0.013361843799971474
Excitation : [0, 8], Gradient: 0.008127419311887625
Excitation : [1, 3], Gradient: 9.60988156765077e-06
Excitation : [1, 5], Gradient: -0.004875127086709089
Excitation : [1, 7], Gradient: -0.004875127086709092
Excitation : [1, 9], Gradient: -0.0075097488221232064
n = 0,  E = -7.85513767 H, t = 2.54 s
n = 1,  E = -7.85585993 H, t = 2.56 s
n = 2,  E = -7.85642249 H, t = 2.54 s
n = 4,  E = -7.85721832 H, t = 2.51 s
n = 5,  E = -7.85750361 H, t = 2.50 s
n = 6,  E = -7.85773773 H, t = 2.54 s
n = 7,  E = -7.85793296 H, t = 2.51 s
n = 8,  E = -7.85809846 H, t = 2.49 s
n = 9,  E = -7.85824102 H, t = 2.80 s
n = 10,  E = -7.85836572 H, t = 2.24 s
n = 11,  E = -7.85847636 H, t = 2.76 s
n = 12,  E = -7.85857579 H, t = 2.52 s
n = 13,  E = -7.85866614 H, t = 2.51 s
n = 14,  E = -7.85874902 H, t = 2.52 s
n = 15,  E = -7.85882566 H, t = 2.53 s
n = 16,  E = -7.85889701 H, t = 2.54 s
n = 17,  E = -7.85896378 H, t = 2.52 s
n = 18,  E = -7.85902654 H, t = 2.48 s
n = 19,  E = -7.85908573 H, t = 2.48 s
n = 0,  E = -7.86266587 H, t = 0.13 s
n = 2,  E = -7.86443636 H, t = 0.13 s
n = 5,  E = -7.86543166 H, t = 0.12 s
n = 7,  E = -7.86567575 H, t = 0.13 s
n = 8,  E = -7.86574604 H, t = 0.15 s
n = 10,  E = -7.86583418 H, t = 0.13 s
n = 11,  E = -7.86586277 H, t = 0.13 s
n = 18,  E = -7.86596307 H, t = 0.13 s
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
Excitation : [0, 1, 2, 9], Gradient: 0.034264511701633214
Excitation : [0, 1, 3, 8], Gradient: -0.008566127925408265
Excitation : [0, 2], Gradient: -0.013361843799994867
Excitation : [0, 8], Gradient: 0.00812741931185717
Excitation : [1, 3], Gradient: 9.60988156628004e-06
Excitation : [1, 5], Gradient: -0.004875127086707588
Excitation : [1, 7], Gradient: -0.004875127086707584
Excitation : [1, 9], Gradient: -0.0075097488220925036
n = 0,  E = -7.85513767 H, t = 2.53 s
n = 1,  E = -7.85585993 H, t = 2.47 s
n = 2,  E = -7.85642249 H, t = 2.50 s
n = 4,  E = -7.85721832 H, t = 2.48 s
n = 5,  E = -7.85750361 H, t = 2.78 s
n = 6,  E = -7.85773773 H, t = 2.53 s
n = 7,  E = -7.85793296 H, t = 2.54 s
n = 8,  E = -7.85809846 H, t = 2.55 s
n = 9,  E = -7.85824102 H, t = 2.50 s
n = 10,  E = -7.85836572 H, t = 2.54 s
n = 11,  E = -7.85847636 H, t = 2.49 s
n = 12,  E = -7.85857579 H, t = 2.50 s
n = 13,  E = -7.85866614 H, t = 2.86 s
n = 14,  E = -7.85874902 H, t = 2.55 s
n = 15,  E = -7.85882566 H, t = 2.57 s
n = 16,  E = -7.85889701 H, t = 2.57 s
n = 17,  E = -7.85896378 H, t = 2.49 s
n = 18,  E = -7.85902654 H, t = 2.52 s
n = 19,  E = -7.85908573 H, t = 2.47 s
n = 0,  E = -7.86266587 H, t = 0.35 s
n = 2,  E = -7.86443636 H, t = 0.12 s
n = 5,  E = -7.86543166 H, t = 0.13 s
n = 7,  E = -7.86567575 H, t = 0.12 s
n = 8,  E = -7.86574604 H, t = 0.12 s
n = 10,  E = -7.86583418 H, t = 0.12 s
n = 11,  E = -7.86586277 H, t = 0.12 s
n = 18,  E = -7.86596307 H, t = 0.12 s
 </code>
 </pre>
 </details>

---

## 4. tutorial_doubly_stochastic.html <a name="demo3"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_doubly_stochastic.html):

```
Stochastic gradient descent (shots=100) min energy =  -4.600655176916144
Adaptive QSGD min energy =  -4.592548741613161
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_doubly_stochastic.html):

```
Stochastic gradient descent (shots=100) min energy =  -4.600655176916145
Adaptive QSGD min energy =  -4.59254874161316
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
Cost at Step 290: -0.9999918144420525
Cost at Step 0: -0.9803638573791904
Weights at Step 295: [ 5.95067478e-01  6.59873699e-06  1.33530656e+00  1.78268285e+00
Cost at Step 155: -0.9999873092858327
Cost at Step 5: -0.9974589524428031
Cost at Step 160: -0.9999872637771745
0.56                |  0.5988034096092983
1.24                |  1.3483865512005195
1.67                |   1.786207064845566
Cost at Step 15: -0.998187153312274
Cost at Step 165: -0.9999874624869765
-0.79               | -0.8425475506159125
-1.44               | -1.4067983643944189
Cost at Step 20: -0.9995130692146865
Cost at Step 170: -0.9999886747321688
-1.43               | -1.3529638627174025
1.18                |  1.0349129419831027
-0.93               | -1.0635874966599685
Non-Existing Edge Parameters: [-0.0012651471928343844, -0.0036534472423337514]
Cost at Step 180: -0.9999914836764833
Cost at Step 35: -0.9997858632135158
Cost at Step 185: -0.9999877384049591
Cost at Step 190: -0.9999943811280623
Cost at Step 45: -0.9998796449154709
Cost at Step 195: -0.9999899532834509
Cost at Step 50: -0.9999381279674818
Cost at Step 215: -0.9999886167575431
Cost at Step 220: -0.9999875727826187
Cost at Step 85: -0.9999892005586407
Cost at Step 230: -0.9999922217151833
Cost at Step 235: -0.9999912692867915
Cost at Step 90: -0.9999860418671424
Cost at Step 95: -0.9999837519650678
Cost at Step 240: -0.999989935769989
Cost at Step 100: -0.9999862374755372
Cost at Step 105: -0.9999860417314501
Cost at Step 110: -0.9999889448376539
Cost at Step 255: -0.9999892613809337
Cost at Step 115: -0.9999907527375037
Cost at Step 275: -0.9999913148930893
Cost at Step 135: -0.9999893671898543
Cost at Step 280: -0.9999912773773975
Cost at Step 285: -0.9999871922879656
Cost at Step 145: -0.9999882781659929
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
Cost at Step 290: -0.9999918144420527
Cost at Step 0: -0.9803638573791903
Weights at Step 295: [ 5.95067478e-01  6.59873700e-06  1.33530656e+00  1.78268285e+00
Cost at Step 155: -0.9999873092858328
Cost at Step 5: -0.9974589524428032
Cost at Step 160: -0.9999872637771742
0.56                |  0.5988034096092884
1.24                |  1.3483865512005244
1.67                |   1.786207064845588
Cost at Step 15: -0.9981871533122741
Cost at Step 165: -0.9999874624869768
-0.79               | -0.8425475506159187
-1.44               | -1.4067983643944084
Cost at Step 20: -0.9995130692146866
Cost at Step 170: -0.9999886747321687
-1.43               | -1.3529638627173852
1.18                |    1.03491294198308
-0.93               |   -1.06358749665997
Non-Existing Edge Parameters: [-0.0012651471928231486, -0.003653447242328027]
Cost at Step 180: -0.9999914836764835
Cost at Step 35: -0.9997858632135159
Cost at Step 185: -0.999987738404959
Cost at Step 190: -0.9999943811280624
Cost at Step 45: -0.9998796449154708
Cost at Step 195: -0.9999899532834505
Cost at Step 50: -0.9999381279674817
Cost at Step 215: -0.999988616757543
Cost at Step 220: -0.9999875727826191
Cost at Step 85: -0.9999892005586408
Cost at Step 230: -0.9999922217151831
Cost at Step 235: -0.9999912692867913
Cost at Step 90: -0.9999860418671421
Cost at Step 95: -0.9999837519650677
Cost at Step 240: -0.9999899357699888
Cost at Step 100: -0.9999862374755373
Cost at Step 105: -0.9999860417314499
Cost at Step 110: -0.9999889448376537
Cost at Step 255: -0.9999892613809335
Cost at Step 115: -0.9999907527375036
Cost at Step 275: -0.9999913148930895
Cost at Step 135: -0.9999893671898544
Cost at Step 280: -0.9999912773773976
Cost at Step 285: -0.9999871922879655
Cost at Step 145: -0.9999882781659927
 </code>
 </pre>
 </details>

---

## 6. tutorial_quanvolution.html <a name="demo5"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quanvolution.html):

```
 1261568/11490434 [==>...........................] - ETA: 0s
10756096/11490434 [===========================>..] - ETA: 0s
13/13 - 1s - loss: 2.3619 - accuracy: 0.1400 - val_loss: 2.0567 - val_accuracy: 0.3667
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_quanvolution.html):

```
 1835008/11490434 [===>..........................] - ETA: 0s
 4202496/11490434 [=========>....................] - ETA: 0s
13/13 - 0s - loss: 2.3619 - accuracy: 0.1400 - val_loss: 2.0567 - val_accuracy: 0.3667
```

---

## 7. tutorial_error_mitigation.html <a name="demo6"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_error_mitigation.html):

```
2: ──RY(4.05)──╭C─────────RY(3.32)────────────────────────╰Z──────────RY(1.07)──RY(-1.07)──╰Z──RY(-3.32)──╭C──RY(-4.05)──┤
3: ──RY(3.51)──╰Z─────────RY(3.66)───RY(-3.66)────────────────────────────────────────────────────────────╰Z──RY(-3.51)──┤
0: ──RY(4.56)──╭C──RY(5.93)───RY(-5.93)───────────────────────────────────────────────────────────╭C──RY(-4.56)──┤
1: ──RY(3.6)───╰Z──RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)──RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z──RY(-3.6)───┤
2: ──RY(4.05)──╭C──RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)───────────────────────╰Z──RY(-3.32)──╭C──RY(-4.05)──┤
3: ──RY(3.51)──╰Z──RY(3.66)───RY(-3.66)───────────────────────────────────────────────────────────╰Z──RY(-3.51)──┤
0.9701559539718818
1: ──RY(3.6)───╰Z──RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)──RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z──RY(-3.6)───┤
2: ──RY(4.05)──╭C──RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)───────────────────────╰Z──RY(-3.32)──╭C──RY(-4.05)──┤
0: ──RY(4.56)───────────────────────╭C──────────RY(5.93)───RY(-5.93)──────────────────────────────────────╭C──RY(-4.56)──┤
1: ──RY(3.6)────RY(-3.6)──RY(3.6)───╰Z──────────RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z──RY(-3.6)───┤
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_error_mitigation.html):

```
2: ──RY(4.05)────────────────────────╭C──────────RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)──╰Z──RY(-3.32)──╭C──RY(-4.05)──┤
3: ──RY(3.51)───RY(-3.51)──RY(3.51)──╰Z──────────RY(3.66)───RY(-3.66)──────────────────────────────────────╰Z──RY(-3.51)──┤
0: ──RY(4.56)──╭C──RY(5.93)───RY(-5.93)──────────────────────────────────────╭C───RY(-4.56)─────────────────┤
1: ──RY(3.6)───╰Z──RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z───RY(-3.6)──────────────────┤
2: ──RY(4.05)──╭C──RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)──╰Z──RY(-3.32)──╭C──╭C──────────╭C──RY(-4.05)──┤
3: ──RY(3.51)──╰Z──RY(3.66)───RY(-3.66)──────────────────────────────────────╰Z──╰Z──────────╰Z──RY(-3.51)──┤
0.9594863182014692
1: ──RY(3.6)───╰Z──RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)───────────────────────╭C──RY(-5.9)───╰Z──RY(-3.6)───┤
2: ──RY(4.05)──╭C──RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)──RY(1.07)──RY(-1.07)──╰Z──RY(-3.32)──╭C──RY(-4.05)──┤
0: ──RY(4.56)──╭C──────────RY(5.93)───RY(-5.93)────────────────────────────────────────────────────────────╭C──RY(-4.56)──┤
1: ──RY(3.6)───╰Z──────────RY(5.9)─────────────────────────╭C──────────RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z──RY(-3.6)───┤
```

---

## 8. tutorial_expressivity_fourier_series.html <a name="demo7"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_expressivity_fourier_series.html):

```
Cost at step  10: 0.03212041720004563
Cost at step  20: 0.013853561883024695
Cost at step  30: 0.004049396436389428
Cost at step  40: 0.0005624933894468379
Cost at step  50: 8.145777333271188e-05
Cost at step  10: 0.01716644944531855
Cost at step  20: 0.005497199314426253
Cost at step  30: 0.004784402394898178
Cost at step  40: 0.004015481434555349
Cost at step  50: 0.001399810298978412
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_expressivity_fourier_series.html):

```
Cost at step  10: 0.03212041720004567
Cost at step  20: 0.01385356188302468
Cost at step  30: 0.004049396436389442
Cost at step  40: 0.0005624933894468399
Cost at step  50: 8.145777333271303e-05
Cost at step  10: 0.017166449445319226
Cost at step  20: 0.005497199314426231
Cost at step  30: 0.004784402394898537
Cost at step  40: 0.00401548143455807
Cost at step  50: 0.0013998102989809839
```

---

## 9. tutorial_kernel_based_training.html <a name="demo8"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_kernel_based_training.html):

```
step 0 , loss 1.2128428849025232
step 10 , loss 0.8582750956106431
step 20 , loss 0.43849890579633233
step 30 , loss 0.6458829274590642
step 40 , loss 0.5540116701446127
step 50 , loss 0.4132239145818268
step 70 , loss 0.469419342316038
step 80 , loss 0.4858145744021141
step 90 , loss 0.4196234621534023
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_kernel_based_training.html):

```
step 0 , loss 1.2128428849025235
step 10 , loss 0.8582750956106429
step 20 , loss 0.4384989057963324
step 30 , loss 0.6458829274590638
step 40 , loss 0.5540116701446128
step 50 , loss 0.4132239145818267
step 70 , loss 0.4694193423160379
step 80 , loss 0.48581457440211384
step 90 , loss 0.41962346215340246
```

---

## 10. tutorial_rosalin.html <a name="demo9"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_rosalin.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Step 0: cost = -4.820380999693457, shots_used = 240
Step 1: cost = -4.937944875992972, shots_used = 336
Step 4: cost = -5.740983235298872, shots_used = 768
Step 5: cost = -5.868499507174453, shots_used = 960
Step 6: cost = -5.5725976187549255, shots_used = 1200
Step 7: cost = -6.03851112713168, shots_used = 1512
Step 8: cost = -7.187839850746338, shots_used = 1944
Step 9: cost = -7.244742040043006, shots_used = 2472
Step 10: cost = -6.955119947427081, shots_used = 3144
Step 11: cost = -7.324280331788419, shots_used = 4176
Step 15: cost = -7.095276433317586, shots_used = 9048
Step 17: cost = -7.738585612241667, shots_used = 12624
Step 19: cost = -7.802640154411867, shots_used = 18048
Step 25: cost = -7.790635224170293, shots_used = 42024
Step 26: cost = -7.887177094048005, shots_used = 48096
Step 30: cost = -7.884434864101767, shots_used = 80832
Step 32: cost = -7.854723497132757, shots_used = 100680
Step 33: cost = -7.860436984930214, shots_used = 111984
Step 37: cost = -7.875581235645932, shots_used = 166152
Step 38: cost = -7.809686333605921, shots_used = 183240
Step 41: cost = -7.894948220964508, shots_used = 240192
Step 42: cost = -7.897425239891585, shots_used = 262368
Step 43: cost = -7.8799029002955, shots_used = 285024
Step 47: cost = -7.878308602523566, shots_used = 385320
Step 48: cost = -7.899236702757055, shots_used = 416208
Step 52: cost = -7.893708744149611, shots_used = 547176
Step 53: cost = -7.898823452831049, shots_used = 582960
Step 54: cost = -7.8988892291181925, shots_used = 621072
Step 57: cost = -7.8974256745741105, shots_used = 747480
Step 58: cost = -7.89322196381792, shots_used = 794808
Step 0: cost = -2.12150804866895 shots_used = 2400
Step 1: cost = -3.4462874411421884 shots_used = 4800
Step 2: cost = -4.533723704599173 shots_used = 7200
Step 3: cost = -5.360324618255417 shots_used = 9600
Step 4: cost = -6.010958804727693 shots_used = 12000
Step 5: cost = -6.5450082323750856 shots_used = 14400
Step 10: cost = -7.374281342889537 shots_used = 26400
Step 14: cost = -7.171989037816059 shots_used = 36000
Step 15: cost = -7.2153317942728545 shots_used = 38400
Step 19: cost = -7.5113154529689705 shots_used = 48000
Step 20: cost = -7.5608730559384805 shots_used = 50400
Step 23: cost = -7.607043067486307 shots_used = 57600
Step 24: cost = -7.5940769784963384 shots_used = 60000
Step 25: cost = -7.579153179578798 shots_used = 62400
Step 26: cost = -7.572266109391965 shots_used = 64800
Step 27: cost = -7.568439746440856 shots_used = 67200
Step 28: cost = -7.5819359681781515 shots_used = 69600
Step 33: cost = -7.787293318189747 shots_used = 81600
Step 34: cost = -7.820874827421242 shots_used = 84000
Step 35: cost = -7.840729913365395 shots_used = 86400
Step 39: cost = -7.884473822847104 shots_used = 96000
Step 40: cost = -7.887248861002906 shots_used = 98400
Step 43: cost = -7.875074719805529 shots_used = 105600
Step 44: cost = -7.866129786750198 shots_used = 108000
Step 45: cost = -7.850581153251575 shots_used = 110400
Step 46: cost = -7.843337695237989 shots_used = 112800
Step 49: cost = -7.858018368114793 shots_used = 120000
Step 50: cost = -7.858043805938922 shots_used = 122400
Step 51: cost = -7.855559046474577 shots_used = 124800
Step 53: cost = -7.848969631273945 shots_used = 129600
Step 57: cost = -7.876241759094949 shots_used = 139200
Step 61: cost = -7.870289689150821 shots_used = 148800
Step 63: cost = -7.862715331323386 shots_used = 153600
Step 67: cost = -7.867196452121145 shots_used = 163200
Step 69: cost = -7.869618725213595 shots_used = 168000
Step 75: cost = -7.864825877239139 shots_used = 182400
Step 76: cost = -7.863570824947599 shots_used = 184800
Step 77: cost = -7.863497614169011 shots_used = 187200
Step 78: cost = -7.860326845355642 shots_used = 189600
Step 80: cost = -7.85589006991893 shots_used = 194400
Step 81: cost = -7.857406216142363 shots_used = 196800
Step 82: cost = -7.863450868072559 shots_used = 199200
Step 83: cost = -7.870058142679088 shots_used = 201600
Step 87: cost = -7.879224942149158 shots_used = 211200
Step 88: cost = -7.872184015334468 shots_used = 213600
Step 90: cost = -7.860606976666904 shots_used = 218400
Step 91: cost = -7.85981012759474 shots_used = 220800
Step 97: cost = -7.883690480348111 shots_used = 235200
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
Step 0: cost = -4.820380999693455, shots_used = 240
Step 1: cost = -4.937944875992974, shots_used = 336
Step 4: cost = -5.740983235298874, shots_used = 768
Step 5: cost = -5.86849950717445, shots_used = 960
Step 6: cost = -5.572597618754925, shots_used = 1200
Step 7: cost = -6.038511127131681, shots_used = 1512
Step 8: cost = -7.187839850746337, shots_used = 1944
Step 9: cost = -7.244742040043007, shots_used = 2472
Step 10: cost = -6.9551199474270815, shots_used = 3144
Step 11: cost = -7.32428033178842, shots_used = 4176
Step 15: cost = -7.095276433317588, shots_used = 9048
Step 17: cost = -7.738585612241666, shots_used = 12624
Step 19: cost = -7.802640154411866, shots_used = 18048
Step 25: cost = -7.790635224170295, shots_used = 42024
Step 26: cost = -7.887177094048004, shots_used = 48096
Step 30: cost = -7.884434864101768, shots_used = 80832
Step 32: cost = -7.854723497132756, shots_used = 100680
Step 33: cost = -7.860436984930216, shots_used = 111984
Step 37: cost = -7.8755812356459325, shots_used = 166152
Step 38: cost = -7.80968633360592, shots_used = 183240
Step 41: cost = -7.894948220964509, shots_used = 240192
Step 42: cost = -7.897425239891586, shots_used = 262368
Step 43: cost = -7.879902900295497, shots_used = 285024
Step 47: cost = -7.878308602523567, shots_used = 385320
Step 48: cost = -7.899236702757056, shots_used = 416208
Step 52: cost = -7.8937087441496105, shots_used = 547176
Step 53: cost = -7.898823452831048, shots_used = 582960
Step 54: cost = -7.898889229118192, shots_used = 621072
Step 57: cost = -7.897425674574112, shots_used = 747480
Step 58: cost = -7.893221963817921, shots_used = 794808
Step 0: cost = -2.121508048668951 shots_used = 2400
Step 1: cost = -3.4462874411421875 shots_used = 4800
Step 2: cost = -4.533723704599174 shots_used = 7200
Step 3: cost = -5.360324618255415 shots_used = 9600
Step 4: cost = -6.010958804727692 shots_used = 12000
Step 5: cost = -6.545008232375084 shots_used = 14400
Step 10: cost = -7.374281342889539 shots_used = 26400
Step 14: cost = -7.171989037816058 shots_used = 36000
Step 15: cost = -7.215331794272855 shots_used = 38400
Step 19: cost = -7.511315452968972 shots_used = 48000
Step 20: cost = -7.560873055938481 shots_used = 50400
Step 23: cost = -7.607043067486308 shots_used = 57600
Step 24: cost = -7.594076978496339 shots_used = 60000
Step 25: cost = -7.5791531795788 shots_used = 62400
Step 26: cost = -7.572266109391966 shots_used = 64800
Step 27: cost = -7.568439746440855 shots_used = 67200
Step 28: cost = -7.581935968178152 shots_used = 69600
Step 33: cost = -7.787293318189748 shots_used = 81600
Step 34: cost = -7.820874827421244 shots_used = 84000
Step 35: cost = -7.8407299133653945 shots_used = 86400
Step 39: cost = -7.884473822847106 shots_used = 96000
Step 40: cost = -7.887248861002907 shots_used = 98400
Step 43: cost = -7.875074719805527 shots_used = 105600
Step 44: cost = -7.866129786750201 shots_used = 108000
Step 45: cost = -7.850581153251574 shots_used = 110400
Step 46: cost = -7.843337695237988 shots_used = 112800
Step 49: cost = -7.858018368114795 shots_used = 120000
Step 50: cost = -7.858043805938923 shots_used = 122400
Step 51: cost = -7.855559046474576 shots_used = 124800
Step 53: cost = -7.848969631273947 shots_used = 129600
Step 57: cost = -7.87624175909495 shots_used = 139200
Step 61: cost = -7.870289689150822 shots_used = 148800
Step 63: cost = -7.862715331323385 shots_used = 153600
Step 67: cost = -7.867196452121144 shots_used = 163200
Step 69: cost = -7.869618725213597 shots_used = 168000
Step 75: cost = -7.864825877239141 shots_used = 182400
Step 76: cost = -7.8635708249475975 shots_used = 184800
Step 77: cost = -7.863497614169013 shots_used = 187200
Step 78: cost = -7.860326845355643 shots_used = 189600
Step 80: cost = -7.855890069918929 shots_used = 194400
Step 81: cost = -7.857406216142362 shots_used = 196800
Step 82: cost = -7.863450868072558 shots_used = 199200
Step 83: cost = -7.870058142679087 shots_used = 201600
Step 87: cost = -7.879224942149157 shots_used = 211200
Step 88: cost = -7.872184015334469 shots_used = 213600
Step 90: cost = -7.860606976666903 shots_used = 218400
Step 91: cost = -7.859810127594738 shots_used = 220800
Step 97: cost = -7.883690480348113 shots_used = 235200
 </code>
 </pre>
 </details>

---

## 11. tutorial_backprop.html <a name="demo10"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_backprop.html):

```
Expectation value: -0.11971365706871569
-0.0651887722495813
 -7.61067572e-01  8.32667268e-17]
-0.0651887722495813
[[-6.51887722e-02 -2.72891905e-02  0.00000000e+00 -9.33934621e-02
  -7.61067572e-01  8.32667268e-17]]
0.8947771876917632
Forward pass (best of 3): 0.02806505010003093 sec per loop
Gradient computation (best of 3): 8.016071605300022 sec per loop
10.103418036011135
Forward pass (best of 3): 0.0808696019999843 sec per loop
Backward pass (best of 3): 0.15250693839998347 sec per loop
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_backprop.html):

```
Expectation value: -0.11971365706871566
-0.06518877224958124
 -7.61067572e-01  4.16333634e-17]
-0.06518877224958124
[[-6.51887722e-02 -2.72891905e-02 -2.77555756e-17 -9.33934621e-02
  -7.61067572e-01  4.16333634e-17]]
0.8947771876917631
Forward pass (best of 3): 0.028123336899989228 sec per loop
Gradient computation (best of 3): 7.84624705719998 sec per loop
10.124401283996121
Forward pass (best of 3): 0.07785378629996557 sec per loop
Backward pass (best of 3): 0.1426723481999943 sec per loop
```

---

## 12. tutorial_barren_plateaus.html <a name="demo11"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_barren_plateaus.html):

```
Mean of the gradients for 200 random circuits: -0.0010002268976521357
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_barren_plateaus.html):

```
Mean of the gradients for 200 random circuits: -0.0010002268976521344
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
(-46.46390678868899+0j) [] +
(-0.014583648907612667+0j) [X0 X1 Y2 Y3] +
(-3.5707613293562875e-07+0j) [X0 X1 Y2 Z3 Z4 Y5] +
(-0.005652620978017359+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7] +
(-0.008826368514209827+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.7924939576642917e-06+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.5707613293562875e-07+0j) [X0 X1 X3 X4] +
(-0.005652620978017359+0j) [X0 X1 X3 Z4 Z5 X6] +
(-0.008826368514209827+0j) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.7924939576642917e-06+0j) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-2.4473231289095723e-07+0j) [X0 X1 Y4 Z5 Z6 Y7] +
(-7.867765104061585e-07+0j) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0038040661717285355+0j) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.4473231289095723e-07+0j) [X0 X1 X5 X6] +
(-7.867765104061585e-07+0j) [X0 X1 X5 Z6 Z7 Z8 Z9 X10] +
(-0.0038040661717285355+0j) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.006888194352970561+0j) [X0 X1 Y6 Y7] +
(-7.735036880590183e-05+0j) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11] +
(1.70357835560182e-07+0j) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.735036880590183e-05+0j) [X0 X1 X7 Z8 Z9 X10] +
(1.70357835560182e-07+0j) [X0 X1 X7 Z8 Z9 Z10 Z11 X12] +
(-0.007731425250775274+0j) [X0 X1 Y10 Y11] +
(5.62785191139709e-07+0j) [X0 X1 Y10 Z11 Z12 Y13] +
(5.62785191139709e-07+0j) [X0 X1 X11 X12] +
(-0.005283776488402954+0j) [X0 X1 Y12 Y13] +
(0.014583648907612667+0j) [X0 Y1 Y2 X3] +
(3.5707613293562875e-07+0j) [X0 Y1 Y2 Z3 Z4 X5] +
(0.005652620978017359+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7] +
(0.008826368514209827+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-1.7924939576642917e-06+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.5707613293562875e-07+0j) [X0 Y1 Y3 X4] +
(-0.005652620978017359+0j) [X0 Y1 Y3 Z4 Z5 X6] +
(-0.008826368514209827+0j) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.7924939576642917e-06+0j) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(2.4473231289095723e-07+0j) [X0 Y1 Y4 Z5 Z6 X7] +
(7.867765104061585e-07+0j) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0038040661717285355+0j) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.4473231289095723e-07+0j) [X0 Y1 Y5 X6] +
(-7.867765104061585e-07+0j) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.0038040661717285355+0j) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.006888194352970561+0j) [X0 Y1 Y6 X7] +
(7.735036880590183e-05+0j) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11] +
(-1.70357835560182e-07+0j) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.735036880590183e-05+0j) [X0 Y1 Y7 Z8 Z9 X10] +
(1.70357835560182e-07+0j) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12] +
(0.007731425250775274+0j) [X0 Y1 Y10 X11] +
(-5.62785191139709e-07+0j) [X0 Y1 Y10 Z11 Z12 X13] +
(5.62785191139709e-07+0j) [X0 Y1 Y11 X12] +
(0.005283776488402954+0j) [X0 Y1 Y12 X13] +
(0.12507032579772082+0j) [X0 Z1 X2] +
(-1.9332412771863424e-07+0j) [X0 Z1 X2 X3 Z4 X5] +
(-0.002293956611352469+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 X7] +
(-0.0016407548553124263+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(3.0134714588406195e-07+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.9332412771863424e-07+0j) [X0 Z1 X2 Y3 Z4 Y5] +
(-0.002293956611352469+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7] +
(-0.0016407548553124263+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(3.0134714588406195e-07+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.003276971931231708+0j) [X0 Z1 X2 Z3] +
(-1.5510539176372522e-07+0j) [X0 Z1 X2 X4 Z5 X6] +
(-1.146837650695226e-06+0j) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0075974640297706095+0j) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.380778148106839e-07+0j) [X0 Z1 X2 Y4 Z5 Y6] +
(-7.900128985880609e-07+0j) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0053480515826766235+0j) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005530759218631567+0j) [X0 Z1 X2 Z4] +
(-1.380778148106839e-07+0j) [X0 Z1 X2 X5 Z6 X7] +
(-3.3767393083408265e-07+0j) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0018638942824587476+0j) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.380778148106839e-07+0j) [X0 Z1 X2 Y5 Z6 Y7] +
(-3.3767393083408265e-07+0j) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0018638942824587476+0j) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0011059037691897233+0j) [X0 Z1 X2 Z5] +
(0.005708495985960939+0j) [X0 Z1 X2 X6 Z7 Z8 Z9 X10] +
(-8.352332102599317e-07+0j) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
1.97422537919719e-08j [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005262642473076852+0j) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10] +
(-8.074305985494647e-07+0j) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.008032520918821413+0j) [X0 Z1 X2 Z6] +
(0.000594022154300555+0j) [X0 Z1 X2 X7 Z8 Z9 Z10 X11] +
(-8.379773243415852e-08+0j) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.000594022154300555+0j) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11] +
(-8.379773243415852e-08+0j) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.003347617530666225+0j) [X0 Z1 X2 Z7] +
(0.011055020596132122+0j) [X0 Z1 X2 Z8] +
(0.002929768674751105+0j) [X0 Z1 X2 Z9] +
(-6.418291574438277e-07+0j) [X0 Z1 X2 X10 Z11 X12] +
(-6.556281914371733e-07+0j) [X0 Z1 X2 Y10 Z11 Y12] +
(0.0035552901955043146+0j) [X0 Z1 X2 Z10] +
(-1.1076325598976819e-07+0j) [X0 Z1 X2 X11 Z12 X13] +
(-1.1076325598976819e-07+0j) [X0 Z1 X2 Y11 Z12 Y13] +
(0.001756070701841283+0j) [X0 Z1 X2 Z11] +
(0.006901238249797321+0j) [X0 Z1 X2 Z12] +
(0.002326230623158121+0j) [X0 Z1 X2 Z13] +
(-3.5682475210716536e-07+0j) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.002249412447093986+0j) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.0474716554532454e-08+0j) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.00044585351288408846+0j) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10] +
(-1.9742253791434932e-08+0j) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441844+0j) [X0 Z1 Z2 X3 Y4 Y5] +
(-4.5233896775397796e-07+0j) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.003484157300217876+0j) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-8.09163719861143e-07+0j) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10] +
(-0.0057335697473118626+0j) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004684903388155188+0j) [X0 Z1 Z2 X3 Y6 Y7] +
(0.004668620318776296+0j) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11] +
(-7.189990974965863e-07+0j) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005114473831660384+0j) [X0 Z1 Z2 X3 X7 Z8 Z9 X10] +
(-7.560692464444928e-07+0j) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381015+0j) [X0 Z1 Z2 X3 Y8 Y9] +
(-0.0017992194936630318+0j) [X0 Z1 Z2 X3 Y10 Y11] +
(-5.471647744458417e-07+0j) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13] +
(-5.287660624556228e-07+0j) [X0 Z1 Z2 X3 X11 X12] +
(-0.0045750076266392+0j) [X0 Z1 Z2 X3 Y12 Y13] +
(0.004424855449441844+0j) [X0 Z1 Z2 Y3 Y4 X5] +
(4.5233896775397796e-07+0j) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.003484157300217876+0j) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.09163719861143e-07+0j) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.0057335697473118626+0j) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.004684903388155188+0j) [X0 Z1 Z2 Y3 Y6 X7] +
(-0.004668620318776296+0j) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11] +
(7.189990974965863e-07+0j) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005114473831660384+0j) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10] +
(-7.560692464444928e-07+0j) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12] +
(0.008125251921381015+0j) [X0 Z1 Z2 Y3 Y8 X9] +
(0.0017992194936630318+0j) [X0 Z1 Z2 Y3 Y10 X11] +
(5.471647744458417e-07+0j) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13] +
(-5.287660624556228e-07+0j) [X0 Z1 Z2 Y3 Y11 X12] +
(0.0045750076266392+0j) [X0 Z1 Z2 Y3 Y12 X13] +
(3.2020768803042663e-06+0j) [X0 Z1 Z2 Z3 X4] +
(0.0008533856254125463+0j) [X0 Z1 Z2 Z3 X4 X5 Z6 X7] +
(0.0007870896771024439+0j) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0008533856254125463+0j) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7] +
(0.0007870896771024439+0j) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.2919694862345983e-07+0j) [X0 Z1 Z2 Z3 X4 Z5] +
(4.4445978541318083e-07+0j) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10] +
(0.0011726348316441863+0j) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.6849150950897495e-07+0j) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10] +
(0.0022009640695004576+0j) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.8394209155684873e-07+0j) [X0 Z1 Z2 Z3 X4 Z6] +
(4.092250615935313e-07+0j) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11] +
(0.0023949726397980197+0j) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.092250615935313e-07+0j) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11] +
(0.0023949726397980197+0j) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.2362599615884835e-07+0j) [X0 Z1 Z2 Z3 X4 Z8] +
(8.649310135408714e-08+0j) [X0 Z1 Z2 Z3 X4 Z9] +
(0.001303800478812689+0j) [X0 Z1 Z2 Z3 X4 X10 Z11 X12] +
(0.003989841456619299+0j) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12] +
(-6.733197741874332e-07+0j) [X0 Z1 Z2 Z3 X4 Z10] +
(0.0022619660624823438+0j) [X0 Z1 Z2 Z3 X4 X11 Z12 X13] +
(0.0022619660624823438+0j) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13] +
(-5.927453082360505e-07+0j) [X0 Z1 Z2 Z3 X4 Z11] +
(1.2393363217097353e-06+0j) [X0 Z1 Z2 Z3 X4 Z12] +
(9.30653665177415e-07+0j) [X0 Z1 Z2 Z3 X4 Z13] +
(-0.001028329237856271+0j) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.0026860409778066098+0j) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12] +
(-1.8394209155684873e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7] +
(-0.00019400857029756258+0j) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0012223378081538336+0j) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-2.3713289480476123e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9] +
(8.057446595138264e-08+0j) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11] +
(0.0017278753941369555+0j) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13] +
(-0.0009581655836696545+0j) [X0 Z1 Z2 Z3 Z4 X5 X11 X12] +
(-3.086826565323204e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13] +
(1.8394209155684873e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7] +
(0.00019400857029756258+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0012223378081538336+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(2.3713289480476123e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9] +
(-8.057446595138264e-08+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11] +
(-0.0017278753941369555+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13] +
(-0.0009581655836696545+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12] +
(3.086826565323204e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13] +
(0.0427432770137828+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6] +
(0.0005192743499487639+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(-1.8505641929092344e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005192743499487639+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(-1.8505641929092344e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.002779026799025534+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7] +
(0.004636976661182563+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8] +
(0.0012803060973496753+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9] +
(2.3120943051683792e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12] +
(1.0717282182278833e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12] +
(0.00537993715583937+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10] +
(7.2469744251416e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13] +
(7.2469744251416e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13] +
(0.0052415353828038636+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11] +
(0.004311038507914311+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12] +
(0.0010435246534907625+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13] +
(1.2004287493877075e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12] +
(-0.0033566705638328875+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9] +
(-0.0001384017730355058+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11] +
(-6.175246206913717e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(-4.997018421885545e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12] +
(-0.0032675138544235476+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13] +
(0.0033566705638328875+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9] +
(0.0001384017730355058+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11] +
(6.175246206913717e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(-4.997018421885545e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12] +
(0.0032675138544235476+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13] +
(0.0038764708993369403+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(-7.540341413494145e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.0038764708993369403+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(-7.540341413494145e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(0.07165035181002508+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0021413612231015893+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(0.004220813970046467+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(0.0012366478019245446+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(-0.0029841661681219225+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(0.0029841661681219225+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-1.3987009015992287e-05+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(8.949476486818995e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-6.876621658250755e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(-7.661347213093929e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(0.0015324835230730396+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10] +
(-2.904599884521583e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(0.00540895442240998+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10] +
(-1.0444941298015726e-06+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(0.004767272188278112+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10] +
(-8.105515037059819e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(0.005286546538226876+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10] +
(-9.956079229969055e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0016095313817213789+0j) [X0 Z1 Z2 Z3 Z4 X6] +
(-7.141625221155014e-05+0j) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10] +
(-2.666731755033326e-07+0j) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.002462917007133925+0j) [X0 Z1 Z2 Z3 Z5 X6] +
(0.0007156734248908936+0j) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10] +
(-3.076732532136001e-07+0j) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.606071868103237e-07+0j) [X0 Z1 Z2 X4] +
(0.0039615607924965174+0j) [X0 Z1 Z2 Z4 Z5 X6] +
(0.00018787053389551263+0j) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.6569309313210045e-07+0j) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.7379332624862986e-07+0j) [X0 Z1 Z3 X4] +
(0.001667604181144049+0j) [X0 Z1 Z3 Z4 Z5 X6] +
(-0.0014528843214169135+0j) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(4.6704023901616245e-07+0j) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.1043306478065141+0j) [X0 X2] +
(3.117447946268268e-06+0j) [X0 Z2 Z3 X4] +
(0.04587947078129806+0j) [X0 Z2 Z3 Z4 Z5 X6] +
(0.058591988733861775+0j) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-1.1463061452834929e-05+0j) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.014583648907612667+0j) [Y0 X1 X2 Y3] +
(3.5707613293562875e-07+0j) [Y0 X1 X2 Z3 Z4 Y5] +
(0.005652620978017359+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7] +
(0.008826368514209827+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.7924939576642917e-06+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.5707613293562875e-07+0j) [Y0 X1 X3 Y4] +
(-0.005652620978017359+0j) [Y0 X1 X3 Z4 Z5 Y6] +
(-0.008826368514209827+0j) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.7924939576642917e-06+0j) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.4473231289095723e-07+0j) [Y0 X1 X4 Z5 Z6 Y7] +
(7.867765104061585e-07+0j) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.0038040661717285355+0j) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.4473231289095723e-07+0j) [Y0 X1 X5 Y6] +
(-7.867765104061585e-07+0j) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.0038040661717285355+0j) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.006888194352970561+0j) [Y0 X1 X6 Y7] +
(7.735036880590183e-05+0j) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11] +
(-1.70357835560182e-07+0j) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.735036880590183e-05+0j) [Y0 X1 X7 Z8 Z9 Y10] +
(1.70357835560182e-07+0j) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12] +
(0.007731425250775274+0j) [Y0 X1 X10 Y11] +
(-5.62785191139709e-07+0j) [Y0 X1 X10 Z11 Z12 Y13] +
(5.62785191139709e-07+0j) [Y0 X1 X11 Y12] +
(0.005283776488402954+0j) [Y0 X1 X12 Y13] +
(-0.014583648907612667+0j) [Y0 Y1 X2 X3] +
(-3.5707613293562875e-07+0j) [Y0 Y1 X2 Z3 Z4 X5] +
(-0.005652620978017359+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7] +
(-0.008826368514209827+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.7924939576642917e-06+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.5707613293562875e-07+0j) [Y0 Y1 Y3 Y4] +
(-0.005652620978017359+0j) [Y0 Y1 Y3 Z4 Z5 Y6] +
(-0.008826368514209827+0j) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.7924939576642917e-06+0j) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-2.4473231289095723e-07+0j) [Y0 Y1 X4 Z5 Z6 X7] +
(-7.867765104061585e-07+0j) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0038040661717285355+0j) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.4473231289095723e-07+0j) [Y0 Y1 Y5 Y6] +
(-7.867765104061585e-07+0j) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.0038040661717285355+0j) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.006888194352970561+0j) [Y0 Y1 X6 X7] +
(-7.735036880590183e-05+0j) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11] +
(1.70357835560182e-07+0j) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.735036880590183e-05+0j) [Y0 Y1 Y7 Z8 Z9 Y10] +
(1.70357835560182e-07+0j) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.007731425250775274+0j) [Y0 Y1 X10 X11] +
(5.62785191139709e-07+0j) [Y0 Y1 X10 Z11 Z12 X13] +
(5.62785191139709e-07+0j) [Y0 Y1 Y11 Y12] +
(-0.005283776488402954+0j) [Y0 Y1 X12 X13] +
(-3.5682475210716536e-07+0j) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.002249412447093986+0j) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.00044585351288408846+0j) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10] +
(-1.9742253791434932e-08+0j) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.0474716554532454e-08+0j) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.12507032579772082+0j) [Y0 Z1 Y2] +
(-1.9332412771863424e-07+0j) [Y0 Z1 Y2 X3 Z4 X5] +
(-0.002293956611352469+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7] +
(-0.0016407548553124263+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(3.0134714588406195e-07+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.9332412771863424e-07+0j) [Y0 Z1 Y2 Y3 Z4 Y5] +
(-0.002293956611352469+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7] +
(-0.0016407548553124263+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(3.0134714588406195e-07+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.003276971931231708+0j) [Y0 Z1 Y2 Z3] +
(-1.380778148106839e-07+0j) [Y0 Z1 Y2 X4 Z5 X6] +
(-7.900128985880609e-07+0j) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0053480515826766235+0j) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.5510539176372522e-07+0j) [Y0 Z1 Y2 Y4 Z5 Y6] +
(-1.146837650695226e-06+0j) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0075974640297706095+0j) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005530759218631567+0j) [Y0 Z1 Y2 Z4] +
(-1.380778148106839e-07+0j) [Y0 Z1 Y2 X5 Z6 X7] +
(-3.3767393083408265e-07+0j) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0018638942824587476+0j) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.380778148106839e-07+0j) [Y0 Z1 Y2 Y5 Z6 Y7] +
(-3.3767393083408265e-07+0j) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0018638942824587476+0j) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0011059037691897233+0j) [Y0 Z1 Y2 Z5] +
(0.005262642473076852+0j) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10] +
(-8.074305985494647e-07+0j) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.005708495985960939+0j) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10] +
-1.97422537919719e-08j [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-8.352332102599317e-07+0j) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.008032520918821413+0j) [Y0 Z1 Y2 Z6] +
(0.000594022154300555+0j) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11] +
(-8.379773243415852e-08+0j) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.000594022154300555+0j) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11] +
(-8.379773243415852e-08+0j) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.003347617530666225+0j) [Y0 Z1 Y2 Z7] +
(0.011055020596132122+0j) [Y0 Z1 Y2 Z8] +
(0.002929768674751105+0j) [Y0 Z1 Y2 Z9] +
(-6.556281914371733e-07+0j) [Y0 Z1 Y2 X10 Z11 X12] +
(-6.418291574438277e-07+0j) [Y0 Z1 Y2 Y10 Z11 Y12] +
(0.0035552901955043146+0j) [Y0 Z1 Y2 Z10] +
(-1.1076325598976819e-07+0j) [Y0 Z1 Y2 X11 Z12 X13] +
(-1.1076325598976819e-07+0j) [Y0 Z1 Y2 Y11 Z12 Y13] +
(0.001756070701841283+0j) [Y0 Z1 Y2 Z11] +
(0.006901238249797321+0j) [Y0 Z1 Y2 Z12] +
(0.002326230623158121+0j) [Y0 Z1 Y2 Z13] +
(0.004424855449441844+0j) [Y0 Z1 Z2 X3 X4 Y5] +
(4.5233896775397796e-07+0j) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.003484157300217876+0j) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-8.09163719861143e-07+0j) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.0057335697473118626+0j) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.004684903388155188+0j) [Y0 Z1 Z2 X3 X6 Y7] +
(-0.004668620318776296+0j) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11] +
(7.189990974965863e-07+0j) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005114473831660384+0j) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10] +
(-7.560692464444928e-07+0j) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12] +
(0.008125251921381015+0j) [Y0 Z1 Z2 X3 X8 Y9] +
(0.0017992194936630318+0j) [Y0 Z1 Z2 X3 X10 Y11] +
(5.471647744458417e-07+0j) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13] +
(-5.287660624556228e-07+0j) [Y0 Z1 Z2 X3 X11 Y12] +
(0.0045750076266392+0j) [Y0 Z1 Z2 X3 X12 Y13] +
(-0.004424855449441844+0j) [Y0 Z1 Z2 Y3 X4 X5] +
(-4.5233896775397796e-07+0j) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.003484157300217876+0j) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.09163719861143e-07+0j) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.0057335697473118626+0j) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004684903388155188+0j) [Y0 Z1 Z2 Y3 X6 X7] +
(0.004668620318776296+0j) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11] +
(-7.189990974965863e-07+0j) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005114473831660384+0j) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10] +
(-7.560692464444928e-07+0j) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381015+0j) [Y0 Z1 Z2 Y3 X8 X9] +
(-0.0017992194936630318+0j) [Y0 Z1 Z2 Y3 X10 X11] +
(-5.471647744458417e-07+0j) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13] +
(-5.287660624556228e-07+0j) [Y0 Z1 Z2 Y3 Y11 Y12] +
(-0.0045750076266392+0j) [Y0 Z1 Z2 Y3 X12 X13] +
(-0.001028329237856271+0j) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.0026860409778066098+0j) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12] +
(3.2020768803042663e-06+0j) [Y0 Z1 Z2 Z3 Y4] +
(0.0008533856254125463+0j) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7] +
(0.0007870896771024439+0j) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0008533856254125463+0j) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7] +
(0.0007870896771024439+0j) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.2919694862345983e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z5] +
(4.6849150950897495e-07+0j) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10] +
(0.0022009640695004576+0j) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.4445978541318083e-07+0j) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10] +
(0.0011726348316441863+0j) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.8394209155684873e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z6] +
(4.092250615935313e-07+0j) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11] +
(0.0023949726397980197+0j) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.092250615935313e-07+0j) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11] +
(0.0023949726397980197+0j) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.2362599615884835e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z8] +
(8.649310135408714e-08+0j) [Y0 Z1 Z2 Z3 Y4 Z9] +
(0.003989841456619299+0j) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12] +
(0.001303800478812689+0j) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12] +
(-6.733197741874332e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z10] +
(0.0022619660624823438+0j) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13] +
(0.0022619660624823438+0j) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13] +
(-5.927453082360505e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z11] +
(1.2393363217097353e-06+0j) [Y0 Z1 Z2 Z3 Y4 Z12] +
(9.30653665177415e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z13] +
(1.8394209155684873e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7] +
(0.00019400857029756258+0j) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0012223378081538336+0j) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(2.3713289480476123e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9] +
(-8.057446595138264e-08+0j) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11] +
(-0.0017278753941369555+0j) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13] +
(-0.0009581655836696545+0j) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12] +
(3.086826565323204e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13] +
(-1.8394209155684873e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7] +
(-0.00019400857029756258+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0012223378081538336+0j) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-2.3713289480476123e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9] +
(8.057446595138264e-08+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11] +
(0.0017278753941369555+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13] +
(-0.0009581655836696545+0j) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12] +
(-3.086826565323204e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13] +
(1.2004287493877075e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12] +
(0.0427432770137828+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6] +
(0.0005192743499487639+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(-1.8505641929092344e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005192743499487639+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(-1.8505641929092344e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.002779026799025534+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7] +
(0.004636976661182563+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8] +
(0.0012803060973496753+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9] +
(1.0717282182278833e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12] +
(2.3120943051683792e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12] +
(0.00537993715583937+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10] +
(7.2469744251416e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13] +
(7.2469744251416e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13] +
(0.0052415353828038636+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11] +
(0.004311038507914311+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12] +
(0.0010435246534907625+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13] +
(0.0033566705638328875+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9] +
(0.0001384017730355058+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11] +
(6.175246206913717e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(-4.997018421885545e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12] +
(0.0032675138544235476+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13] +
(-0.0033566705638328875+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9] +
(-0.0001384017730355058+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11] +
(-6.175246206913717e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(-4.997018421885545e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12] +
(-0.0032675138544235476+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13] +
(0.0038764708993369403+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(-7.540341413494145e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.0038764708993369403+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(-7.540341413494145e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(0.07165035181002508+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0021413612231015893+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(0.004220813970046467+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(0.0012366478019245446+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(0.0029841661681219225+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-0.0029841661681219225+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-1.3987009015992287e-05+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(8.949476486818995e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-6.876621658250755e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(-7.661347213093929e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(0.0015324835230730396+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10] +
(-2.904599884521583e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(0.00540895442240998+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10] +
(-1.0444941298015726e-06+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(0.004767272188278112+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10] +
(-8.105515037059819e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(0.005286546538226876+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10] +
(-9.956079229969055e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0016095313817213789+0j) [Y0 Z1 Z2 Z3 Z4 Y6] +
(-7.141625221155014e-05+0j) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10] +
(-2.666731755033326e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.002462917007133925+0j) [Y0 Z1 Z2 Z3 Z5 Y6] +
(0.0007156734248908936+0j) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10] +
(-3.076732532136001e-07+0j) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(3.606071868103237e-07+0j) [Y0 Z1 Z2 Y4] +
(0.0039615607924965174+0j) [Y0 Z1 Z2 Z4 Z5 Y6] +
(0.00018787053389551263+0j) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.6569309313210045e-07+0j) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.7379332624862986e-07+0j) [Y0 Z1 Z3 Y4] +
(0.001667604181144049+0j) [Y0 Z1 Z3 Z4 Z5 Y6] +
(-0.0014528843214169135+0j) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(4.6704023901616245e-07+0j) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.1043306478065141+0j) [Y0 Y2] +
(3.117447946268268e-06+0j) [Y0 Z2 Z3 Y4] +
(0.04587947078129806+0j) [Y0 Z2 Z3 Z4 Z5 Y6] +
(0.058591988733861775+0j) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-1.1463061452834929e-05+0j) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(12.41263074211177+0j) [Z0] +
(0.1043306478065141+0j) [Z0 X1 Z2 X3] +
(3.1174479462682685e-06+0j) [Z0 X1 Z2 Z3 Z4 X5] +
(0.04587947078129806+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7] +
(0.058591988733861775+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-1.146306145283493e-05+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.1043306478065141+0j) [Z0 Y1 Z2 Y3] +
(3.1174479462682685e-06+0j) [Z0 Y1 Z2 Z3 Z4 Y5] +
(0.04587947078129806+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7] +
(0.058591988733861775+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.146306145283493e-05+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-8.337746755836362e-07+0j) [Z0 X2 Z3 X4] +
(-0.027115036845273253+0j) [Z0 X2 Z3 Z4 Z5 X6] +
(-0.0675238509921402+0j) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.4017109734899417e-05+0j) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-8.337746755836362e-07+0j) [Z0 Y2 Z3 Y4] +
(-0.027115036845273253+0j) [Z0 Y2 Z3 Z4 Z5 Y6] +
(-0.0675238509921402+0j) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.4017109734899417e-05+0j) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.2367108078383041+0j) [Z0 Z2] +
(-1.1908508085192648e-06+0j) [Z0 X3 Z4 X5] +
(-0.03276765782329061+0j) [Z0 X3 Z4 Z5 Z6 X7] +
(-0.07635021950635001+0j) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.5809603692563706e-05+0j) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.1908508085192648e-06+0j) [Z0 Y3 Z4 Y5] +
(-0.03276765782329061+0j) [Z0 Y3 Z4 Z5 Z6 Y7] +
(-0.07635021950635001+0j) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.5809603692563706e-05+0j) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.25129445674591677+0j) [Z0 Z3] +
(-3.0993492435998888e-06+0j) [Z0 X4 Z5 X6] +
(-1.5316808794758076e-05+0j) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.08684737589863618+0j) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-3.0993492435998888e-06+0j) [Z0 Y4 Z5 Y6] +
(-1.5316808794758076e-05+0j) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.08684737589863618+0j) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.19661770890342142+0j) [Z0 Z4] +
(-3.3440815564908457e-06+0j) [Z0 X5 Z6 X7] +
(-1.610358530516423e-05+0j) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.09065144207036471+0j) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.3440815564908457e-06+0j) [Z0 Y5 Z6 Y7] +
(-1.610358530516423e-05+0j) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.09065144207036471+0j) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.05608468124661368+0j) [Z0 X6 Z7 Z8 Z9 X10] +
(-6.65220966889595e-06+0j) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.05608468124661368+0j) [Z0 Y6 Z7 Z8 Z9 Y10] +
(-6.65220966889595e-06+0j) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.24164663936017197+0j) [Z0 Z6] +
(0.05600733087780777+0j) [Z0 X7 Z8 Z9 Z10 X11] +
(-6.481851833335769e-06+0j) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.05600733087780777+0j) [Z0 Y7 Z8 Z9 Z10 Y11] +
(-6.481851833335769e-06+0j) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.2485348337131425+0j) [Z0 Z7] +
(0.2723251830660569+0j) [Z0 Z8] +
(0.2788345442672341+0j) [Z0 Z9] +
(-2.1776646048062753e-06+0j) [Z0 X10 Z11 X12] +
(-2.1776646048062753e-06+0j) [Z0 Y10 Z11 Y12] +
(0.19299723935364252+0j) [Z0 Z10] +
(-1.6148794136665662e-06+0j) [Z0 X11 Z12 X13] +
(-1.6148794136665662e-06+0j) [Z0 Y11 Z12 Y13] +
(0.2007286646044178+0j) [Z0 Z11] +
(0.2110265984979151+0j) [Z0 Z12] +
(0.21631037498631805+0j) [Z0 Z13] +
(1.9332412771863427e-07+0j) [X1 X2 Y3 Y4] +
(0.002293956611352469+0j) [X1 X2 Y3 Z4 Z5 Y6] +
(0.0016407548553124263+0j) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-3.0134714588406195e-07+0j) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004424855449441844+0j) [X1 X2 X4 X5] +
(-8.09163719861143e-07+0j) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0057335697473118626+0j) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-4.5233896775397796e-07+0j) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.003484157300217876+0j) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.0046849033881551875+0j) [X1 X2 X6 X7] +
(0.005114473831660384+0j) [X1 X2 X6 Z7 Z8 Z9 Z10 X11] +
(-7.560692464444928e-07+0j) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.004668620318776296+0j) [X1 X2 Y7 Z8 Z9 Y10] +
(-7.189990974965863e-07+0j) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381015+0j) [X1 X2 X8 X9] +
(-0.0017992194936630316+0j) [X1 X2 X10 X11] +
(-5.287660624556228e-07+0j) [X1 X2 X10 Z11 Z12 X13] +
(-5.471647744458417e-07+0j) [X1 X2 Y11 Y12] +
(-0.004575007626639199+0j) [X1 X2 X12 X13] +
(-1.9332412771863427e-07+0j) [X1 Y2 Y3 X4] +
(-0.002293956611352469+0j) [X1 Y2 Y3 Z4 Z5 X6] +
(-0.0016407548553124263+0j) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(3.0134714588406195e-07+0j) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441844+0j) [X1 Y2 Y4 X5] +
(-8.09163719861143e-07+0j) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0057335697473118626+0j) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.5233896775397796e-07+0j) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10] +
(0.003484157300217876+0j) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.0046849033881551875+0j) [X1 Y2 Y6 X7] +
(0.005114473831660384+0j) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11] +
(-7.560692464444928e-07+0j) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.004668620318776296+0j) [X1 Y2 Y7 Z8 Z9 X10] +
(7.189990974965863e-07+0j) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381015+0j) [X1 Y2 Y8 X9] +
(-0.0017992194936630316+0j) [X1 Y2 Y10 X11] +
(-5.287660624556228e-07+0j) [X1 Y2 Y10 Z11 Z12 X13] +
(5.471647744458417e-07+0j) [X1 Y2 Y11 X12] +
(-0.004575007626639199+0j) [X1 Y2 Y12 X13] +
(0.12507032579772087+0j) [X1 Z2 X3] +
(-1.380778148106839e-07+0j) [X1 Z2 X3 X4 Z5 X6] +
(-3.3767393083408265e-07+0j) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0018638942824587476+0j) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.380778148106839e-07+0j) [X1 Z2 X3 Y4 Z5 Y6] +
(-3.3767393083408265e-07+0j) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0018638942824587476+0j) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0011059037691897233+0j) [X1 Z2 X3 Z4] +
(-1.5510539176372522e-07+0j) [X1 Z2 X3 X5 Z6 X7] +
(-1.146837650695226e-06+0j) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0075974640297706095+0j) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.380778148106839e-07+0j) [X1 Z2 X3 Y5 Z6 Y7] +
(-7.900128985880609e-07+0j) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0053480515826766235+0j) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005530759218631567+0j) [X1 Z2 X3 Z5] +
(0.000594022154300555+0j) [X1 Z2 X3 X6 Z7 Z8 Z9 X10] +
(-8.379773243415852e-08+0j) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.000594022154300555+0j) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10] +
(-8.379773243415852e-08+0j) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.003347617530666225+0j) [X1 Z2 X3 Z6] +
(0.005708495985960939+0j) [X1 Z2 X3 X7 Z8 Z9 Z10 X11] +
(-8.352332102599317e-07+0j) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
1.97422537919719e-08j [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005262642473076852+0j) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11] +
(-8.074305985494647e-07+0j) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.008032520918821413+0j) [X1 Z2 X3 Z7] +
(0.002929768674751105+0j) [X1 Z2 X3 Z8] +
(0.011055020596132122+0j) [X1 Z2 X3 Z9] +
(-1.1076325598976819e-07+0j) [X1 Z2 X3 X10 Z11 X12] +
(-1.1076325598976819e-07+0j) [X1 Z2 X3 Y10 Z11 Y12] +
(0.001756070701841283+0j) [X1 Z2 X3 Z10] +
(-6.418291574438277e-07+0j) [X1 Z2 X3 X11 Z12 X13] +
(-6.556281914371733e-07+0j) [X1 Z2 X3 Y11 Z12 Y13] +
(0.0035552901955043146+0j) [X1 Z2 X3 Z11] +
(0.002326230623158121+0j) [X1 Z2 X3 Z12] +
(0.006901238249797321+0j) [X1 Z2 X3 Z13] +
(-3.5682475210716536e-07+0j) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.002249412447093986+0j) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.0474716554532454e-08+0j) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.00044585351288408846+0j) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11] +
(-1.9742253791434932e-08+0j) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0008533856254125463+0j) [X1 Z2 Z3 X4 Y5 Y6] +
(-0.0007870896771024438+0j) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10] +
(-1.8394209155684873e-07+0j) [X1 Z2 Z3 X4 X6 X7] +
(-0.0012223378081538336+0j) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.00019400857029756258+0j) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12] +
(-2.3713289480476123e-07+0j) [X1 Z2 Z3 X4 X8 X9] +
(8.057446595138265e-08+0j) [X1 Z2 Z3 X4 X10 X11] +
(-0.0009581655836696545+0j) [X1 Z2 Z3 X4 X10 Z11 Z12 X13] +
(0.0017278753941369555+0j) [X1 Z2 Z3 X4 Y11 Y12] +
(-3.0868265653232043e-07+0j) [X1 Z2 Z3 X4 X12 X13] +
(0.0008533856254125463+0j) [X1 Z2 Z3 Y4 Y5 X6] +
(0.0007870896771024438+0j) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10] +
(-1.8394209155684873e-07+0j) [X1 Z2 Z3 Y4 Y6 X7] +
(-0.0012223378081538336+0j) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.00019400857029756258+0j) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12] +
(-2.3713289480476123e-07+0j) [X1 Z2 Z3 Y4 Y8 X9] +
(8.057446595138265e-08+0j) [X1 Z2 Z3 Y4 Y10 X11] +
(-0.0009581655836696545+0j) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13] +
(-0.0017278753941369555+0j) [X1 Z2 Z3 Y4 Y11 X12] +
(-3.0868265653232043e-07+0j) [X1 Z2 Z3 Y4 Y12 X13] +
(3.2020768803042655e-06+0j) [X1 Z2 Z3 Z4 X5] +
(4.092250615935313e-07+0j) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10] +
(0.0023949726397980197+0j) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.092250615935313e-07+0j) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10] +
(0.0023949726397980197+0j) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.4445978541318083e-07+0j) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11] +
(0.0011726348316441863+0j) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.6849150950897495e-07+0j) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11] +
(0.0022009640695004576+0j) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.8394209155684873e-07+0j) [X1 Z2 Z3 Z4 X5 Z7] +
(8.649310135408714e-08+0j) [X1 Z2 Z3 Z4 X5 Z8] +
(3.2362599615884835e-07+0j) [X1 Z2 Z3 Z4 X5 Z9] +
(0.0022619660624823438+0j) [X1 Z2 Z3 Z4 X5 X10 Z11 X12] +
(0.0022619660624823438+0j) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12] +
(-5.927453082360505e-07+0j) [X1 Z2 Z3 Z4 X5 Z10] +
(0.001303800478812689+0j) [X1 Z2 Z3 Z4 X5 X11 Z12 X13] +
(0.003989841456619299+0j) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13] +
(-6.733197741874332e-07+0j) [X1 Z2 Z3 Z4 X5 Z11] +
(9.30653665177415e-07+0j) [X1 Z2 Z3 Z4 X5 Z12] +
(1.2393363217097353e-06+0j) [X1 Z2 Z3 Z4 X5 Z13] +
(-0.001028329237856271+0j) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0026860409778066098+0j) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13] +
(-0.0005192743499487639+0j) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10] +
(1.8505641929092344e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.003356670563832888+0j) [X1 Z2 Z3 Z4 Z5 X6 X8 X9] +
(-0.0001384017730355058+0j) [X1 Z2 Z3 Z4 Z5 X6 X10 X11] +
(-4.997018421885545e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13] +
(-6.175246206913717e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12] +
(-0.0032675138544235476+0j) [X1 Z2 Z3 Z4 Z5 X6 X12 X13] +
(0.0005192743499487639+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10] +
(-1.8505641929092344e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.003356670563832888+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9] +
(-0.0001384017730355058+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11] +
(-4.997018421885545e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13] +
(6.175246206913717e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12] +
(-0.0032675138544235476+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13] +
(0.0427432770137828+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7] +
(0.0012803060973496753+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8] +
(0.004636976661182563+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9] +
(7.2469744251416e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12] +
(7.2469744251416e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12] +
(0.0052415353828038636+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10] +
(2.3120943051683792e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13] +
(1.0717282182278833e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13] +
(0.00537993715583937+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11] +
(0.0010435246534907625+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12] +
(0.004311038507914311+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13] +
(1.2004287493877075e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13] +
(-0.0038764708993369403+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10] +
(7.540341413494145e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(0.0038764708993369403+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10] +
(-7.540341413494145e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(-0.0029841661681219225+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-0.0029841661681219225+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(0.07165035181002502+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0012366478019245446+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(0.004220813970046467+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(-1.3987009015992283e-05+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(8.949476486818996e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(-7.661347213093929e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(-0.0021413612231015893+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11] +
(-6.876621658250755e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(0.00540895442240998+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11] +
(-1.0444941298015726e-06+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(0.0015324835230730396+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11] +
(-2.904599884521583e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(0.005286546538226876+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11] +
(-9.956079229969055e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.002779026799025534+0j) [X1 Z2 Z3 Z4 Z5 X7] +
(0.004767272188278112+0j) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11] +
(-8.105515037059819e-07+0j) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.002462917007133925+0j) [X1 Z2 Z3 Z4 Z6 X7] +
(0.0007156734248908936+0j) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11] +
(-3.076732532136001e-07+0j) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2919694862345983e-07+0j) [X1 Z2 Z3 X5] +
(0.0016095313817213789+0j) [X1 Z2 Z3 Z5 Z6 X7] +
(-7.141625221155014e-05+0j) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-2.666731755033326e-07+0j) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.7379332624862986e-07+0j) [X1 Z2 Z4 X5] +
(0.001667604181144049+0j) [X1 Z2 Z4 Z5 Z6 X7] +
(-0.0014528843214169135+0j) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(4.6704023901616245e-07+0j) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0032769719312317077+0j) [X1 X3] +
(3.606071868103237e-07+0j) [X1 Z3 Z4 X5] +
(0.0039615607924965174+0j) [X1 Z3 Z4 Z5 Z6 X7] +
(0.00018787053389551263+0j) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.6569309313210045e-07+0j) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.9332412771863427e-07+0j) [Y1 X2 X3 Y4] +
(-0.002293956611352469+0j) [Y1 X2 X3 Z4 Z5 Y6] +
(-0.0016407548553124263+0j) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(3.0134714588406195e-07+0j) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004424855449441844+0j) [Y1 X2 X4 Y5] +
(-8.09163719861143e-07+0j) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0057335697473118626+0j) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(4.5233896775397796e-07+0j) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10] +
(0.003484157300217876+0j) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.0046849033881551875+0j) [Y1 X2 X6 Y7] +
(0.005114473831660384+0j) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11] +
(-7.560692464444928e-07+0j) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.004668620318776296+0j) [Y1 X2 X7 Z8 Z9 Y10] +
(7.189990974965863e-07+0j) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381015+0j) [Y1 X2 X8 Y9] +
(-0.0017992194936630316+0j) [Y1 X2 X10 Y11] +
(-5.287660624556228e-07+0j) [Y1 X2 X10 Z11 Z12 Y13] +
(5.471647744458417e-07+0j) [Y1 X2 X11 Y12] +
(-0.004575007626639199+0j) [Y1 X2 X12 Y13] +
(1.9332412771863427e-07+0j) [Y1 Y2 X3 X4] +
(0.002293956611352469+0j) [Y1 Y2 X3 Z4 Z5 X6] +
(0.0016407548553124263+0j) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-3.0134714588406195e-07+0j) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441844+0j) [Y1 Y2 Y4 Y5] +
(-8.09163719861143e-07+0j) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0057335697473118626+0j) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-4.5233896775397796e-07+0j) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10] +
(-0.003484157300217876+0j) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.0046849033881551875+0j) [Y1 Y2 Y6 Y7] +
(0.005114473831660384+0j) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11] +
(-7.560692464444928e-07+0j) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.004668620318776296+0j) [Y1 Y2 X7 Z8 Z9 X10] +
(-7.189990974965863e-07+0j) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381015+0j) [Y1 Y2 Y8 Y9] +
(-0.0017992194936630316+0j) [Y1 Y2 Y10 Y11] +
(-5.287660624556228e-07+0j) [Y1 Y2 Y10 Z11 Z12 Y13] +
(-5.471647744458417e-07+0j) [Y1 Y2 X11 X12] +
(-0.004575007626639199+0j) [Y1 Y2 Y12 Y13] +
(-3.5682475210716536e-07+0j) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.002249412447093986+0j) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.00044585351288408846+0j) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11] +
(-1.9742253791434932e-08+0j) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.0474716554532454e-08+0j) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.12507032579772087+0j) [Y1 Z2 Y3] +
(-1.380778148106839e-07+0j) [Y1 Z2 Y3 X4 Z5 X6] +
(-3.3767393083408265e-07+0j) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0018638942824587476+0j) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.380778148106839e-07+0j) [Y1 Z2 Y3 Y4 Z5 Y6] +
(-3.3767393083408265e-07+0j) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0018638942824587476+0j) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0011059037691897233+0j) [Y1 Z2 Y3 Z4] +
(-1.380778148106839e-07+0j) [Y1 Z2 Y3 X5 Z6 X7] +
(-7.900128985880609e-07+0j) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0053480515826766235+0j) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.5510539176372522e-07+0j) [Y1 Z2 Y3 Y5 Z6 Y7] +
(-1.146837650695226e-06+0j) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0075974640297706095+0j) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005530759218631567+0j) [Y1 Z2 Y3 Z5] +
(0.000594022154300555+0j) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10] +
(-8.379773243415852e-08+0j) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.000594022154300555+0j) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10] +
(-8.379773243415852e-08+0j) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.003347617530666225+0j) [Y1 Z2 Y3 Z6] +
(0.005262642473076852+0j) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11] +
(-8.074305985494647e-07+0j) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005708495985960939+0j) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11] +
-1.97422537919719e-08j [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.352332102599317e-07+0j) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.008032520918821413+0j) [Y1 Z2 Y3 Z7] +
(0.002929768674751105+0j) [Y1 Z2 Y3 Z8] +
(0.011055020596132122+0j) [Y1 Z2 Y3 Z9] +
(-1.1076325598976819e-07+0j) [Y1 Z2 Y3 X10 Z11 X12] +
(-1.1076325598976819e-07+0j) [Y1 Z2 Y3 Y10 Z11 Y12] +
(0.001756070701841283+0j) [Y1 Z2 Y3 Z10] +
(-6.556281914371733e-07+0j) [Y1 Z2 Y3 X11 Z12 X13] +
(-6.418291574438277e-07+0j) [Y1 Z2 Y3 Y11 Z12 Y13] +
(0.0035552901955043146+0j) [Y1 Z2 Y3 Z11] +
(0.002326230623158121+0j) [Y1 Z2 Y3 Z12] +
(0.006901238249797321+0j) [Y1 Z2 Y3 Z13] +
(0.0008533856254125463+0j) [Y1 Z2 Z3 X4 X5 Y6] +
(0.0007870896771024438+0j) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10] +
(-1.8394209155684873e-07+0j) [Y1 Z2 Z3 X4 X6 Y7] +
(-0.0012223378081538336+0j) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.00019400857029756258+0j) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12] +
(-2.3713289480476123e-07+0j) [Y1 Z2 Z3 X4 X8 Y9] +
(8.057446595138265e-08+0j) [Y1 Z2 Z3 X4 X10 Y11] +
(-0.0009581655836696545+0j) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13] +
(-0.0017278753941369555+0j) [Y1 Z2 Z3 X4 X11 Y12] +
(-3.0868265653232043e-07+0j) [Y1 Z2 Z3 X4 X12 Y13] +
(-0.0008533856254125463+0j) [Y1 Z2 Z3 Y4 X5 X6] +
(-0.0007870896771024438+0j) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10] +
(-1.8394209155684873e-07+0j) [Y1 Z2 Z3 Y4 Y6 Y7] +
(-0.0012223378081538336+0j) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.00019400857029756258+0j) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12] +
(-2.3713289480476123e-07+0j) [Y1 Z2 Z3 Y4 Y8 Y9] +
(8.057446595138265e-08+0j) [Y1 Z2 Z3 Y4 Y10 Y11] +
(-0.0009581655836696545+0j) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13] +
(0.0017278753941369555+0j) [Y1 Z2 Z3 Y4 X11 X12] +
(-3.0868265653232043e-07+0j) [Y1 Z2 Z3 Y4 Y12 Y13] +
(-0.001028329237856271+0j) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0026860409778066098+0j) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13] +
(3.2020768803042655e-06+0j) [Y1 Z2 Z3 Z4 Y5] +
(4.092250615935313e-07+0j) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10] +
(0.0023949726397980197+0j) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.092250615935313e-07+0j) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10] +
(0.0023949726397980197+0j) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.6849150950897495e-07+0j) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11] +
(0.0022009640695004576+0j) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.4445978541318083e-07+0j) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11] +
(0.0011726348316441863+0j) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.8394209155684873e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z7] +
(8.649310135408714e-08+0j) [Y1 Z2 Z3 Z4 Y5 Z8] +
(3.2362599615884835e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z9] +
(0.0022619660624823438+0j) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12] +
(0.0022619660624823438+0j) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12] +
(-5.927453082360505e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z10] +
(0.003989841456619299+0j) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13] +
(0.001303800478812689+0j) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13] +
(-6.733197741874332e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z11] +
(9.30653665177415e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z12] +
(1.2393363217097353e-06+0j) [Y1 Z2 Z3 Z4 Y5 Z13] +
(0.0005192743499487639+0j) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10] +
(-1.8505641929092344e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.003356670563832888+0j) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9] +
(-0.0001384017730355058+0j) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11] +
(-4.997018421885545e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13] +
(6.175246206913717e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12] +
(-0.0032675138544235476+0j) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13] +
(-0.0005192743499487639+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10] +
(1.8505641929092344e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-0.003356670563832888+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9] +
(-0.0001384017730355058+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11] +
(-4.997018421885545e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13] +
(-6.175246206913717e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12] +
(-0.0032675138544235476+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13] +
(1.2004287493877075e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13] +
(0.0427432770137828+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7] +
(0.0012803060973496753+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8] +
(0.004636976661182563+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9] +
(7.2469744251416e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12] +
(7.2469744251416e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12] +
(0.0052415353828038636+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10] +
(1.0717282182278833e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13] +
(2.3120943051683792e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13] +
(0.00537993715583937+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11] +
(0.0010435246534907625+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12] +
(0.004311038507914311+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13] +
(0.0038764708993369403+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10] +
(-7.540341413494145e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-0.0038764708993369403+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10] +
(7.540341413494145e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-0.0029841661681219225+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(-0.0029841661681219225+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(0.07165035181002502+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.0012366478019245446+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(0.004220813970046467+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(-1.3987009015992283e-05+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(8.949476486818996e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(-7.661347213093929e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(-0.0021413612231015893+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11] +
(-6.876621658250755e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(0.00540895442240998+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11] +
(-1.0444941298015726e-06+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(0.0015324835230730396+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11] +
(-2.904599884521583e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(0.005286546538226876+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11] +
(-9.956079229969055e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.002779026799025534+0j) [Y1 Z2 Z3 Z4 Z5 Y7] +
(0.004767272188278112+0j) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11] +
(-8.105515037059819e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.002462917007133925+0j) [Y1 Z2 Z3 Z4 Z6 Y7] +
(0.0007156734248908936+0j) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11] +
(-3.076732532136001e-07+0j) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.2919694862345983e-07+0j) [Y1 Z2 Z3 Y5] +
(0.0016095313817213789+0j) [Y1 Z2 Z3 Z5 Z6 Y7] +
(-7.141625221155014e-05+0j) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-2.666731755033326e-07+0j) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.7379332624862986e-07+0j) [Y1 Z2 Z4 Y5] +
(0.001667604181144049+0j) [Y1 Z2 Z4 Z5 Z6 Y7] +
(-0.0014528843214169135+0j) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(4.6704023901616245e-07+0j) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0032769719312317077+0j) [Y1 Y3] +
(3.606071868103237e-07+0j) [Y1 Z3 Z4 Y5] +
(0.0039615607924965174+0j) [Y1 Z3 Z4 Z5 Z6 Y7] +
(0.00018787053389551263+0j) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.6569309313210045e-07+0j) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(12.41263074211177+0j) [Z1] +
(-1.1908508085192648e-06+0j) [Z1 X2 Z3 X4] +
(-0.03276765782329061+0j) [Z1 X2 Z3 Z4 Z5 X6] +
(-0.07635021950635001+0j) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.5809603692563706e-05+0j) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.1908508085192648e-06+0j) [Z1 Y2 Z3 Y4] +
(-0.03276765782329061+0j) [Z1 Y2 Z3 Z4 Z5 Y6] +
(-0.07635021950635001+0j) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.5809603692563706e-05+0j) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.25129445674591677+0j) [Z1 Z2] +
(-8.337746755836362e-07+0j) [Z1 X3 Z4 X5] +
(-0.027115036845273253+0j) [Z1 X3 Z4 Z5 Z6 X7] +
(-0.0675238509921402+0j) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.4017109734899417e-05+0j) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.337746755836362e-07+0j) [Z1 Y3 Z4 Y5] +
(-0.027115036845273253+0j) [Z1 Y3 Z4 Z5 Z6 Y7] +
(-0.0675238509921402+0j) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.4017109734899417e-05+0j) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.2367108078383041+0j) [Z1 Z3] +
(-3.3440815564908457e-06+0j) [Z1 X4 Z5 X6] +
(-1.610358530516423e-05+0j) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.09065144207036471+0j) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-3.3440815564908457e-06+0j) [Z1 Y4 Z5 Y6] +
(-1.610358530516423e-05+0j) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.09065144207036471+0j) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-3.0993492435998888e-06+0j) [Z1 X5 Z6 X7] +
(-1.5316808794758076e-05+0j) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.08684737589863618+0j) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.0993492435998888e-06+0j) [Z1 Y5 Z6 Y7] +
(-1.5316808794758076e-05+0j) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.08684737589863618+0j) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.19661770890342142+0j) [Z1 Z5] +
(0.05600733087780777+0j) [Z1 X6 Z7 Z8 Z9 X10] +
(-6.481851833335769e-06+0j) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.05600733087780777+0j) [Z1 Y6 Z7 Z8 Z9 Y10] +
(-6.481851833335769e-06+0j) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.2485348337131425+0j) [Z1 Z6] +
(0.05608468124661368+0j) [Z1 X7 Z8 Z9 Z10 X11] +
(-6.65220966889595e-06+0j) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.05608468124661368+0j) [Z1 Y7 Z8 Z9 Z10 Y11] +
(-6.65220966889595e-06+0j) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.24164663936017197+0j) [Z1 Z7] +
(0.2788345442672341+0j) [Z1 Z8] +
(0.2723251830660569+0j) [Z1 Z9] +
(-1.6148794136665662e-06+0j) [Z1 X10 Z11 X12] +
(-1.6148794136665662e-06+0j) [Z1 Y10 Z11 Y12] +
(0.2007286646044178+0j) [Z1 Z10] +
(-2.1776646048062753e-06+0j) [Z1 X11 Z12 X13] +
(-2.1776646048062753e-06+0j) [Z1 Y11 Z12 Y13] +
(0.19299723935364252+0j) [Z1 Z11] +
(0.21631037498631805+0j) [Z1 Z12] +
(0.2110265984979151+0j) [Z1 Z13] +
(-0.03583956795335346+0j) [X2 X3 Y4 Y5] +
(-2.1990516182973136e-07+0j) [X2 X3 Y4 Z5 Z6 Y7] +
(-2.3609563202562002e-06+0j) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.01031148248983178+0j) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.1990516182973138e-07+0j) [X2 X3 X5 X6] +
(-2.3609563202562007e-06+0j) [X2 X3 X5 Z6 Z7 Z8 Z9 X10] +
(-0.010311482489831781+0j) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.031143817988967145+0j) [X2 X3 Y6 Y7] +
(0.0053686593581095485+0j) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11] +
(9.209350651465036e-08+0j) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0053686593581095485+0j) [X2 X3 X7 Z8 Z9 X10] +
(9.209350651465037e-08+0j) [X2 X3 X7 Z8 Z9 Z10 Z11 X12] +
(-0.03619412355904261+0j) [X2 X3 Y8 Y9] +
(-0.025384657508457392+0j) [X2 X3 Y10 Y11] +
(2.1726691014774407e-06+0j) [X2 X3 Y10 Z11 Z12 Y13] +
(2.1726691014774407e-06+0j) [X2 X3 X11 X12] +
(-0.015577208063976462+0j) [X2 X3 Y12 Y13] +
(0.03583956795335346+0j) [X2 Y3 Y4 X5] +
(2.1990516182973136e-07+0j) [X2 Y3 Y4 Z5 Z6 X7] +
(2.3609563202562002e-06+0j) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.01031148248983178+0j) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.1990516182973138e-07+0j) [X2 Y3 Y5 X6] +
(-2.3609563202562007e-06+0j) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.010311482489831781+0j) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.031143817988967145+0j) [X2 Y3 Y6 X7] +
(-0.0053686593581095485+0j) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11] +
(-9.209350651465036e-08+0j) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0053686593581095485+0j) [X2 Y3 Y7 Z8 Z9 X10] +
(9.209350651465037e-08+0j) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12] +
(0.03619412355904261+0j) [X2 Y3 Y8 X9] +
(0.025384657508457392+0j) [X2 Y3 Y10 X11] +
(-2.1726691014774407e-06+0j) [X2 Y3 Y10 Z11 Z12 X13] +
(2.1726691014774407e-06+0j) [X2 Y3 Y11 X12] +
(0.015577208063976462+0j) [X2 Y3 Y12 X13] +
(-3.887051673915365e-06+0j) [X2 Z3 X4] +
(-0.005143391768825145+0j) [X2 Z3 X4 X5 Z6 X7] +
(-0.0098417492469626+0j) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.9885117064238046e-06+0j) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.005143391768825145+0j) [X2 Z3 X4 Y5 Z6 Y7] +
(-0.0098417492469626+0j) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.9885117064238046e-06+0j) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.76499411853114e-07+0j) [X2 Z3 X4 Z5] +
(1.6893489514315483e-06+0j) [X2 Z3 X4 X6 Z7 Z8 Z9 X10] +
(0.010757563953908957+0j) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.537178095534251e-08+0j) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10] +
(4.2055484112180075e-05+0j) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.593534390687222e-07+0j) [X2 Z3 X4 Z6] +
(3.21184201899509e-06+0j) [X2 Z3 X4 X7 Z8 Z9 Z10 X11] +
(0.019299560579363804+0j) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.21184201899509e-06+0j) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11] +
(0.019299560579363804+0j) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.195489010063762e-06+0j) [X2 Z3 X4 Z7] +
(2.1868423771818334e-07+0j) [X2 Z3 X4 Z8] +
(-5.770052995860509e-07+0j) [X2 Z3 X4 Z9] +
(0.015588250102380173+0j) [X2 Z3 X4 X10 Z11 X12] +
(0.005324835234221689+0j) [X2 Z3 X4 Y10 Z11 Y12] +
(-3.1586564319536888e-06+0j) [X2 Z3 X4 Z10] +
(0.024353077678068935+0j) [X2 Z3 X4 X11 Z12 X13] +
(0.024353077678068935+0j) [X2 Z3 X4 Y11 Z12 Y13] +
(-7.801707500270743e-06+0j) [X2 Z3 X4 Z11] +
(3.5390541843935785e-06+0j) [X2 Z3 X4 Z12] +
(8.814937306387542e-06+0j) [X2 Z3 X4 Z13] +
(1.6288532434908726e-06+0j) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10] +
(0.010715508469796776+0j) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.010263414868158483+0j) [X2 Z3 Y4 Y10 Z11 X12] +
(-1.4548424491324842e-06+0j) [X2 Z3 Z4 X5 Y6 Y7] +
(-3.151346311054413e-06+0j) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11] +
(-0.019257505095251627+0j) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.5224930675635414e-06+0j) [X2 Z3 Z4 X5 X7 Z8 Z9 X10] +
(-0.008541996625454849+0j) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-7.956895373042342e-07+0j) [X2 Z3 Z4 X5 Y8 Y9] +
(-4.643051068317055e-06+0j) [X2 Z3 Z4 X5 Y10 Y11] +
(-0.019028242443847248+0j) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13] +
(-0.008764827575688765+0j) [X2 Z3 Z4 X5 X11 X12] +
(5.275883121993964e-06+0j) [X2 Z3 Z4 X5 Y12 Y13] +
(1.4548424491324842e-06+0j) [X2 Z3 Z4 Y5 Y6 X7] +
(3.151346311054413e-06+0j) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11] +
(0.019257505095251627+0j) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.5224930675635414e-06+0j) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10] +
(-0.008541996625454849+0j) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(7.956895373042342e-07+0j) [X2 Z3 Z4 Y5 Y8 X9] +
(4.643051068317055e-06+0j) [X2 Z3 Z4 Y5 Y10 X11] +
(0.019028242443847248+0j) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13] +
(-0.008764827575688765+0j) [X2 Z3 Z4 Y5 Y11 X12] +
(-5.275883121993964e-06+0j) [X2 Z3 Z4 Y5 Y12 X13] +
(-0.12133276911042373+0j) [X2 Z3 Z4 Z5 X6] +
(-0.008469978791023989+0j) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(2.686381543842499e-07+0j) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.008469978791023989+0j) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(2.686381543842499e-07+0j) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.025996177598021204+0j) [X2 Z3 Z4 Z5 X6 Z7] +
(-0.00580518898982697+0j) [X2 Z3 Z4 Z5 X6 Z8] +
(-0.017561202409646235+0j) [X2 Z3 Z4 Z5 X6 Z9] +
(-7.988770289156996e-07+0j) [X2 Z3 Z4 Z5 X6 X10 Z11 X12] +
(-3.4273231088380753e-07+0j) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12] +
(-0.0008145313270957245+0j) [X2 Z3 Z4 Z5 X6 Z10] +
(2.7455184002536466e-06+0j) [X2 Z3 Z4 Z5 X6 X11 Z12 X13] +
(2.7455184002536466e-06+0j) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13] +
(0.014411099430130862+0j) [X2 Z3 Z4 Z5 X6 Z11] +
(0.0006650070219498812+0j) [X2 Z3 Z4 Z5 X6 Z12] +
(-0.003493790359890169+0j) [X2 Z3 Z4 Z5 X6 Z13] +
(-4.5614471803189203e-07+0j) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12] +
(-0.011756013419819265+0j) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9] +
(0.015225630757226587+0j) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11] +
(-3.088250711137455e-06+0j) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(-3.544395429169346e-06+0j) [X2 Z3 Z4 Z5 Z6 X7 X11 X12] +
(-0.0041587973818400506+0j) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13] +
(0.011756013419819265+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9] +
(-0.015225630757226587+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11] +
(3.088250711137455e-06+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(-3.544395429169346e-06+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12] +
(0.0041587973818400506+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13] +
(0.01460370472916211+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(-2.8742990713108648e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.01460370472916211+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(-2.8742990713108648e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(-0.2816425776702285+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-1.3002946562341067e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(-1.3002946562341067e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(-0.02428211735469305+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(-0.019538050311314722+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(-0.01709155315589885+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(0.0024464971554158717+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(-0.0024464971554158717+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(5.7759505271271167e-05+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(2.883676575983934e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(5.1464963273862535e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(3.846201671152147e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(-0.03935916802205309+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10] +
(7.979825793187526e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(-0.02475546329289098+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10] +
(5.105526721876662e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(-0.021433810721600846+0j) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10] +
(5.159350501872071e-06+0j) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(-0.029903789512624842+0j) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10] +
(5.4279886562563205e-06+0j) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0016638798784907708+0j) [X2 Z3 Z4 X6] +
(-0.018889030304942905+0j) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10] +
(2.9473560115420034e-06+0j) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.003479511890334374+0j) [X2 Z3 Z5 X6] +
(-0.028730779551905505+0j) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10] +
(5.935867717965807e-06+0j) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.602116740622521e-06+0j) [X2 X4] +
(0.0004956762314915631+0j) [X2 Z4 Z5 X6] +
(-0.035608378988312525+0j) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(7.253273347923426e-06+0j) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.03583956795335346+0j) [Y2 X3 X4 Y5] +
(2.1990516182973136e-07+0j) [Y2 X3 X4 Z5 Z6 Y7] +
(2.3609563202562002e-06+0j) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.01031148248983178+0j) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.1990516182973138e-07+0j) [Y2 X3 X5 Y6] +
(-2.3609563202562007e-06+0j) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.010311482489831781+0j) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.031143817988967145+0j) [Y2 X3 X6 Y7] +
(-0.0053686593581095485+0j) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11] +
(-9.209350651465036e-08+0j) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0053686593581095485+0j) [Y2 X3 X7 Z8 Z9 Y10] +
(9.209350651465037e-08+0j) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12] +
(0.03619412355904261+0j) [Y2 X3 X8 Y9] +
(0.025384657508457392+0j) [Y2 X3 X10 Y11] +
(-2.1726691014774407e-06+0j) [Y2 X3 X10 Z11 Z12 Y13] +
(2.1726691014774407e-06+0j) [Y2 X3 X11 Y12] +
(0.015577208063976462+0j) [Y2 X3 X12 Y13] +
(-0.03583956795335346+0j) [Y2 Y3 X4 X5] +
(-2.1990516182973136e-07+0j) [Y2 Y3 X4 Z5 Z6 X7] +
(-2.3609563202562002e-06+0j) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.01031148248983178+0j) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.1990516182973138e-07+0j) [Y2 Y3 Y5 Y6] +
(-2.3609563202562007e-06+0j) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.010311482489831781+0j) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.031143817988967145+0j) [Y2 Y3 X6 X7] +
(0.0053686593581095485+0j) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11] +
(9.209350651465036e-08+0j) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0053686593581095485+0j) [Y2 Y3 Y7 Z8 Z9 Y10] +
(9.209350651465037e-08+0j) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.03619412355904261+0j) [Y2 Y3 X8 X9] +
(-0.025384657508457392+0j) [Y2 Y3 X10 X11] +
(2.1726691014774407e-06+0j) [Y2 Y3 X10 Z11 Z12 X13] +
(2.1726691014774407e-06+0j) [Y2 Y3 Y11 Y12] +
(-0.015577208063976462+0j) [Y2 Y3 X12 X13] +
(1.6288532434908726e-06+0j) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10] +
(0.010715508469796776+0j) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.010263414868158483+0j) [Y2 Z3 X4 X10 Z11 Y12] +
(-3.887051673915365e-06+0j) [Y2 Z3 Y4] +
(-0.005143391768825145+0j) [Y2 Z3 Y4 X5 Z6 X7] +
(-0.0098417492469626+0j) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.9885117064238046e-06+0j) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.005143391768825145+0j) [Y2 Z3 Y4 Y5 Z6 Y7] +
(-0.0098417492469626+0j) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.9885117064238046e-06+0j) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.76499411853114e-07+0j) [Y2 Z3 Y4 Z5] +
(4.537178095534251e-08+0j) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10] +
(4.2055484112180075e-05+0j) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.6893489514315483e-06+0j) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10] +
(0.010757563953908957+0j) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.593534390687222e-07+0j) [Y2 Z3 Y4 Z6] +
(3.21184201899509e-06+0j) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11] +
(0.019299560579363804+0j) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.21184201899509e-06+0j) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11] +
(0.019299560579363804+0j) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.195489010063762e-06+0j) [Y2 Z3 Y4 Z7] +
(2.1868423771818334e-07+0j) [Y2 Z3 Y4 Z8] +
(-5.770052995860509e-07+0j) [Y2 Z3 Y4 Z9] +
(0.005324835234221689+0j) [Y2 Z3 Y4 X10 Z11 X12] +
(0.015588250102380173+0j) [Y2 Z3 Y4 Y10 Z11 Y12] +
(-3.1586564319536888e-06+0j) [Y2 Z3 Y4 Z10] +
(0.024353077678068935+0j) [Y2 Z3 Y4 X11 Z12 X13] +
(0.024353077678068935+0j) [Y2 Z3 Y4 Y11 Z12 Y13] +
(-7.801707500270743e-06+0j) [Y2 Z3 Y4 Z11] +
(3.5390541843935785e-06+0j) [Y2 Z3 Y4 Z12] +
(8.814937306387542e-06+0j) [Y2 Z3 Y4 Z13] +
(1.4548424491324842e-06+0j) [Y2 Z3 Z4 X5 X6 Y7] +
(3.151346311054413e-06+0j) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11] +
(0.019257505095251627+0j) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.5224930675635414e-06+0j) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10] +
(-0.008541996625454849+0j) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(7.956895373042342e-07+0j) [Y2 Z3 Z4 X5 X8 Y9] +
(4.643051068317055e-06+0j) [Y2 Z3 Z4 X5 X10 Y11] +
(0.019028242443847248+0j) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13] +
(-0.008764827575688765+0j) [Y2 Z3 Z4 X5 X11 Y12] +
(-5.275883121993964e-06+0j) [Y2 Z3 Z4 X5 X12 Y13] +
(-1.4548424491324842e-06+0j) [Y2 Z3 Z4 Y5 X6 X7] +
(-3.151346311054413e-06+0j) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11] +
(-0.019257505095251627+0j) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.5224930675635414e-06+0j) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10] +
(-0.008541996625454849+0j) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895373042342e-07+0j) [Y2 Z3 Z4 Y5 X8 X9] +
(-4.643051068317055e-06+0j) [Y2 Z3 Z4 Y5 X10 X11] +
(-0.019028242443847248+0j) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13] +
(-0.008764827575688765+0j) [Y2 Z3 Z4 Y5 Y11 Y12] +
(5.275883121993964e-06+0j) [Y2 Z3 Z4 Y5 X12 X13] +
(-4.5614471803189203e-07+0j) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12] +
(-0.12133276911042373+0j) [Y2 Z3 Z4 Z5 Y6] +
(-0.008469978791023989+0j) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(2.686381543842499e-07+0j) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.008469978791023989+0j) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(2.686381543842499e-07+0j) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.025996177598021204+0j) [Y2 Z3 Z4 Z5 Y6 Z7] +
(-0.00580518898982697+0j) [Y2 Z3 Z4 Z5 Y6 Z8] +
(-0.017561202409646235+0j) [Y2 Z3 Z4 Z5 Y6 Z9] +
(-3.4273231088380753e-07+0j) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12] +
(-7.988770289156996e-07+0j) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12] +
(-0.0008145313270957245+0j) [Y2 Z3 Z4 Z5 Y6 Z10] +
(2.7455184002536466e-06+0j) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13] +
(2.7455184002536466e-06+0j) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13] +
(0.014411099430130862+0j) [Y2 Z3 Z4 Z5 Y6 Z11] +
(0.0006650070219498812+0j) [Y2 Z3 Z4 Z5 Y6 Z12] +
(-0.003493790359890169+0j) [Y2 Z3 Z4 Z5 Y6 Z13] +
(0.011756013419819265+0j) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9] +
(-0.015225630757226587+0j) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11] +
(3.088250711137455e-06+0j) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(-3.544395429169346e-06+0j) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12] +
(0.0041587973818400506+0j) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13] +
(-0.011756013419819265+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9] +
(0.015225630757226587+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11] +
(-3.088250711137455e-06+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(-3.544395429169346e-06+0j) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12] +
(-0.0041587973818400506+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13] +
(0.01460370472916211+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(-2.8742990713108648e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.01460370472916211+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(-2.8742990713108648e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(-0.2816425776702285+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-1.3002946562341067e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(-1.3002946562341067e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(-0.02428211735469305+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(-0.019538050311314722+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(-0.01709155315589885+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(-0.0024464971554158717+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(0.0024464971554158717+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(5.7759505271271167e-05+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.883676575983934e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(5.1464963273862535e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(3.846201671152147e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(-0.03935916802205309+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10] +
(7.979825793187526e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(-0.02475546329289098+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10] +
(5.105526721876662e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(-0.021433810721600846+0j) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10] +
(5.159350501872071e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(-0.029903789512624842+0j) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10] +
(5.4279886562563205e-06+0j) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0016638798784907708+0j) [Y2 Z3 Z4 Y6] +
(-0.018889030304942905+0j) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10] +
(2.9473560115420034e-06+0j) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.003479511890334374+0j) [Y2 Z3 Z5 Y6] +
(-0.028730779551905505+0j) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10] +
(5.935867717965807e-06+0j) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.602116740622521e-06+0j) [Y2 Y4] +
(0.0004956762314915631+0j) [Y2 Z4 Z5 Y6] +
(-0.035608378988312525+0j) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(7.253273347923426e-06+0j) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.65389422268317+0j) [Z2] +
(1.602116740622521e-06+0j) [Z2 X3 Z4 X5] +
(0.0004956762314915632+0j) [Z2 X3 Z4 Z5 Z6 X7] +
(-0.03560837898831252+0j) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(7.253273347923426e-06+0j) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.602116740622521e-06+0j) [Z2 Y3 Z4 Y5] +
(0.0004956762314915632+0j) [Z2 Y3 Z4 Z5 Z6 Y7] +
(-0.03560837898831252+0j) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(7.253273347923426e-06+0j) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.18189085790751344+0j) [Z2 Z3] +
(-9.509249751315749e-07+0j) [Z2 X4 Z5 X6] +
(-4.728843147054139e-06+0j) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-9.509249751315749e-07+0j) [Z2 Y4 Z5 Y6] +
(-4.728843147054139e-06+0j) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(0.12495807739503215+0j) [Z2 Z4] +
(-1.1708301369613063e-06+0j) [Z2 X5 Z6 X7] +
(-7.0897994673103395e-06+0j) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-1.1708301369613063e-06+0j) [Z2 Y5 Z6 Y7] +
(-7.0897994673103395e-06+0j) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.1607976453483856+0j) [Z2 Z5] +
(0.019020423173039987+0j) [Z2 X6 Z7 Z8 Z9 X10] +
(-2.1032156044926093e-06+0j) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.019020423173039987+0j) [Z2 Y6 Z7 Z8 Z9 Y10] +
(-2.1032156044926093e-06+0j) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.13739104762683219+0j) [Z2 Z6] +
(0.024389082531149534+0j) [Z2 X7 Z8 Z9 Z10 X11] +
(-2.0111220979779585e-06+0j) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.024389082531149534+0j) [Z2 Y7 Z8 Z9 Z10 Y11] +
(-2.0111220979779585e-06+0j) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.16853486561579933+0j) [Z2 Z7] +
(0.1507140812100829+0j) [Z2 Z8] +
(0.1869082047691255+0j) [Z2 Z9] +
(-1.0632283422530125e-06+0j) [Z2 X10 Z11 X12] +
(-1.0632283422530125e-06+0j) [Z2 Y10 Z11 Y12] +
(0.12799502492468412+0j) [Z2 Z10] +
(1.1094407592244284e-06+0j) [Z2 X11 Z12 X13] +
(1.1094407592244284e-06+0j) [Z2 Y11 Z12 Y13] +
(0.15337968243314148+0j) [Z2 Z11] +
(0.1401128986535481+0j) [Z2 Z12] +
(0.15569010671752453+0j) [Z2 Z13] +
(0.005143391768825145+0j) [X3 X4 Y5 Y6] +
(0.009841749246962602+0j) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10] +
(-2.9885117064238046e-06+0j) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.4548424491324842e-06+0j) [X3 X4 X6 X7] +
(-1.5224930675635414e-06+0j) [X3 X4 X6 Z7 Z8 Z9 Z10 X11] +
(-0.008541996625454849+0j) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.151346311054413e-06+0j) [X3 X4 Y7 Z8 Z9 Y10] +
(-0.019257505095251627+0j) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895373042342e-07+0j) [X3 X4 X8 X9] +
(-4.643051068317055e-06+0j) [X3 X4 X10 X11] +
(-0.008764827575688765+0j) [X3 X4 X10 Z11 Z12 X13] +
(-0.019028242443847248+0j) [X3 X4 Y11 Y12] +
(5.275883121993964e-06+0j) [X3 X4 X12 X13] +
(-0.005143391768825145+0j) [X3 Y4 Y5 X6] +
(-0.009841749246962602+0j) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10] +
(2.9885117064238046e-06+0j) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.4548424491324842e-06+0j) [X3 Y4 Y6 X7] +
(-1.5224930675635414e-06+0j) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11] +
(-0.008541996625454849+0j) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.151346311054413e-06+0j) [X3 Y4 Y7 Z8 Z9 X10] +
(0.019257505095251627+0j) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12] +
(-7.956895373042342e-07+0j) [X3 Y4 Y8 X9] +
(-4.643051068317055e-06+0j) [X3 Y4 Y10 X11] +
(-0.008764827575688765+0j) [X3 Y4 Y10 Z11 Z12 X13] +
(0.019028242443847248+0j) [X3 Y4 Y11 X12] +
(5.275883121993964e-06+0j) [X3 Y4 Y12 X13] +
(-3.887051673915361e-06+0j) [X3 Z4 X5] +
(3.21184201899509e-06+0j) [X3 Z4 X5 X6 Z7 Z8 Z9 X10] +
(0.019299560579363804+0j) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.21184201899509e-06+0j) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10] +
(0.019299560579363804+0j) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.195489010063762e-06+0j) [X3 Z4 X5 Z6] +
(1.6893489514315483e-06+0j) [X3 Z4 X5 X7 Z8 Z9 Z10 X11] +
(0.010757563953908957+0j) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.537178095534251e-08+0j) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11] +
(4.2055484112180075e-05+0j) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.593534390687222e-07+0j) [X3 Z4 X5 Z7] +
(-5.770052995860509e-07+0j) [X3 Z4 X5 Z8] +
(2.1868423771818334e-07+0j) [X3 Z4 X5 Z9] +
(0.024353077678068935+0j) [X3 Z4 X5 X10 Z11 X12] +
(0.024353077678068935+0j) [X3 Z4 X5 Y10 Z11 Y12] +
(-7.801707500270743e-06+0j) [X3 Z4 X5 Z10] +
(0.015588250102380173+0j) [X3 Z4 X5 X11 Z12 X13] +
(0.005324835234221689+0j) [X3 Z4 X5 Y11 Z12 Y13] +
(-3.1586564319536888e-06+0j) [X3 Z4 X5 Z11] +
(8.814937306387542e-06+0j) [X3 Z4 X5 Z12] +
(3.5390541843935785e-06+0j) [X3 Z4 X5 Z13] +
(1.6288532434908726e-06+0j) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11] +
(0.010715508469796776+0j) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.010263414868158483+0j) [X3 Z4 Y5 Y11 Z12 X13] +
(0.008469978791023989+0j) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10] +
(-2.686381543842499e-07+0j) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.011756013419819262+0j) [X3 Z4 Z5 X6 X8 X9] +
(0.015225630757226587+0j) [X3 Z4 Z5 X6 X10 X11] +
(-3.544395429169346e-06+0j) [X3 Z4 Z5 X6 X10 Z11 Z12 X13] +
(-3.088250711137455e-06+0j) [X3 Z4 Z5 X6 Y11 Y12] +
(-0.0041587973818400506+0j) [X3 Z4 Z5 X6 X12 X13] +
(-0.008469978791023989+0j) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10] +
(2.686381543842499e-07+0j) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.011756013419819262+0j) [X3 Z4 Z5 Y6 Y8 X9] +
(0.015225630757226587+0j) [X3 Z4 Z5 Y6 Y10 X11] +
(-3.544395429169346e-06+0j) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13] +
(3.088250711137455e-06+0j) [X3 Z4 Z5 Y6 Y11 X12] +
(-0.0041587973818400506+0j) [X3 Z4 Z5 Y6 Y12 X13] +
(-0.12133276911042364+0j) [X3 Z4 Z5 Z6 X7] +
(-0.017561202409646235+0j) [X3 Z4 Z5 Z6 X7 Z8] +
(-0.00580518898982697+0j) [X3 Z4 Z5 Z6 X7 Z9] +
(2.7455184002536466e-06+0j) [X3 Z4 Z5 Z6 X7 X10 Z11 X12] +
(2.7455184002536466e-06+0j) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12] +
(0.014411099430130862+0j) [X3 Z4 Z5 Z6 X7 Z10] +
(-7.988770289156996e-07+0j) [X3 Z4 Z5 Z6 X7 X11 Z12 X13] +
(-3.4273231088380753e-07+0j) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13] +
(-0.0008145313270957245+0j) [X3 Z4 Z5 Z6 X7 Z11] +
(-0.003493790359890169+0j) [X3 Z4 Z5 Z6 X7 Z12] +
(0.0006650070219498812+0j) [X3 Z4 Z5 Z6 X7 Z13] +
(-4.5614471803189203e-07+0j) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13] +
(-0.014603704729162108+0j) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10] +
(2.8742990713108648e-06+0j) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(0.014603704729162108+0j) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10] +
(-2.8742990713108648e-06+0j) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(1.3002946562341067e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12] +
(0.0024464971554158717+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-1.3002946562341067e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12] +
(0.0024464971554158717+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(-0.2816425776702286+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.01709155315589885+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(-0.019538050311314722+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(5.775950527127117e-05+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(2.883676575983934e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(3.846201671152147e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(-0.024282117354693048+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11] +
(5.1464963273862535e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(-0.02475546329289098+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11] +
(5.105526721876662e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(-0.03935916802205309+0j) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11] +
(7.979825793187526e-06+0j) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(-0.029903789512624842+0j) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11] +
(5.4279886562563205e-06+0j) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.025996177598021204+0j) [X3 Z4 Z5 X7] +
(-0.021433810721600846+0j) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11] +
(5.159350501872071e-06+0j) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.003479511890334374+0j) [X3 Z4 Z6 X7] +
(-0.028730779551905505+0j) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11] +
(5.935867717965807e-06+0j) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.76499411853114e-07+0j) [X3 X5] +
(0.0016638798784907708+0j) [X3 Z5 Z6 X7] +
(-0.018889030304942905+0j) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.9473560115420034e-06+0j) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.005143391768825145+0j) [Y3 X4 X5 Y6] +
(-0.009841749246962602+0j) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10] +
(2.9885117064238046e-06+0j) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.4548424491324842e-06+0j) [Y3 X4 X6 Y7] +
(-1.5224930675635414e-06+0j) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11] +
(-0.008541996625454849+0j) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.151346311054413e-06+0j) [Y3 X4 X7 Z8 Z9 Y10] +
(0.019257505095251627+0j) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895373042342e-07+0j) [Y3 X4 X8 Y9] +
(-4.643051068317055e-06+0j) [Y3 X4 X10 Y11] +
(-0.008764827575688765+0j) [Y3 X4 X10 Z11 Z12 Y13] +
(0.019028242443847248+0j) [Y3 X4 X11 Y12] +
(5.275883121993964e-06+0j) [Y3 X4 X12 Y13] +
(0.005143391768825145+0j) [Y3 Y4 X5 X6] +
(0.009841749246962602+0j) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10] +
(-2.9885117064238046e-06+0j) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.4548424491324842e-06+0j) [Y3 Y4 Y6 Y7] +
(-1.5224930675635414e-06+0j) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11] +
(-0.008541996625454849+0j) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.151346311054413e-06+0j) [Y3 Y4 X7 Z8 Z9 X10] +
(-0.019257505095251627+0j) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12] +
(-7.956895373042342e-07+0j) [Y3 Y4 Y8 Y9] +
(-4.643051068317055e-06+0j) [Y3 Y4 Y10 Y11] +
(-0.008764827575688765+0j) [Y3 Y4 Y10 Z11 Z12 Y13] +
(-0.019028242443847248+0j) [Y3 Y4 X11 X12] +
(5.275883121993964e-06+0j) [Y3 Y4 Y12 Y13] +
(1.6288532434908726e-06+0j) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11] +
(0.010715508469796776+0j) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.010263414868158483+0j) [Y3 Z4 X5 X11 Z12 Y13] +
(-3.887051673915361e-06+0j) [Y3 Z4 Y5] +
(3.21184201899509e-06+0j) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10] +
(0.019299560579363804+0j) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.21184201899509e-06+0j) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10] +
(0.019299560579363804+0j) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.195489010063762e-06+0j) [Y3 Z4 Y5 Z6] +
(4.537178095534251e-08+0j) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11] +
(4.2055484112180075e-05+0j) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.6893489514315483e-06+0j) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11] +
(0.010757563953908957+0j) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.593534390687222e-07+0j) [Y3 Z4 Y5 Z7] +
(-5.770052995860509e-07+0j) [Y3 Z4 Y5 Z8] +
(2.1868423771818334e-07+0j) [Y3 Z4 Y5 Z9] +
(0.024353077678068935+0j) [Y3 Z4 Y5 X10 Z11 X12] +
(0.024353077678068935+0j) [Y3 Z4 Y5 Y10 Z11 Y12] +
(-7.801707500270743e-06+0j) [Y3 Z4 Y5 Z10] +
(0.005324835234221689+0j) [Y3 Z4 Y5 X11 Z12 X13] +
(0.015588250102380173+0j) [Y3 Z4 Y5 Y11 Z12 Y13] +
(-3.1586564319536888e-06+0j) [Y3 Z4 Y5 Z11] +
(8.814937306387542e-06+0j) [Y3 Z4 Y5 Z12] +
(3.5390541843935785e-06+0j) [Y3 Z4 Y5 Z13] +
(-0.008469978791023989+0j) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10] +
(2.686381543842499e-07+0j) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.011756013419819262+0j) [Y3 Z4 Z5 X6 X8 Y9] +
(0.015225630757226587+0j) [Y3 Z4 Z5 X6 X10 Y11] +
(-3.544395429169346e-06+0j) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13] +
(3.088250711137455e-06+0j) [Y3 Z4 Z5 X6 X11 Y12] +
(-0.0041587973818400506+0j) [Y3 Z4 Z5 X6 X12 Y13] +
(0.008469978791023989+0j) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10] +
(-2.686381543842499e-07+0j) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-0.011756013419819262+0j) [Y3 Z4 Z5 Y6 Y8 Y9] +
(0.015225630757226587+0j) [Y3 Z4 Z5 Y6 Y10 Y11] +
(-3.544395429169346e-06+0j) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13] +
(-3.088250711137455e-06+0j) [Y3 Z4 Z5 Y6 X11 X12] +
(-0.0041587973818400506+0j) [Y3 Z4 Z5 Y6 Y12 Y13] +
(-4.5614471803189203e-07+0j) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13] +
(-0.12133276911042364+0j) [Y3 Z4 Z5 Z6 Y7] +
(-0.017561202409646235+0j) [Y3 Z4 Z5 Z6 Y7 Z8] +
(-0.00580518898982697+0j) [Y3 Z4 Z5 Z6 Y7 Z9] +
(2.7455184002536466e-06+0j) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12] +
(2.7455184002536466e-06+0j) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12] +
(0.014411099430130862+0j) [Y3 Z4 Z5 Z6 Y7 Z10] +
(-3.4273231088380753e-07+0j) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13] +
(-7.988770289156996e-07+0j) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13] +
(-0.0008145313270957245+0j) [Y3 Z4 Z5 Z6 Y7 Z11] +
(-0.003493790359890169+0j) [Y3 Z4 Z5 Z6 Y7 Z12] +
(0.0006650070219498812+0j) [Y3 Z4 Z5 Z6 Y7 Z13] +
(0.014603704729162108+0j) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10] +
(-2.8742990713108648e-06+0j) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-0.014603704729162108+0j) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10] +
(2.8742990713108648e-06+0j) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-1.3002946562341067e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12] +
(0.0024464971554158717+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(1.3002946562341067e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12] +
(0.0024464971554158717+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(-0.2816425776702286+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.01709155315589885+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(-0.019538050311314722+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(5.775950527127117e-05+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.883676575983934e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(3.846201671152147e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(-0.024282117354693048+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11] +
(5.1464963273862535e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(-0.02475546329289098+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11] +
(5.105526721876662e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(-0.03935916802205309+0j) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11] +
(7.979825793187526e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(-0.029903789512624842+0j) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11] +
(5.4279886562563205e-06+0j) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.025996177598021204+0j) [Y3 Z4 Z5 Y7] +
(-0.021433810721600846+0j) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11] +
(5.159350501872071e-06+0j) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.003479511890334374+0j) [Y3 Z4 Z6 Y7] +
(-0.028730779551905505+0j) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11] +
(5.935867717965807e-06+0j) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.76499411853114e-07+0j) [Y3 Y5] +
(0.0016638798784907708+0j) [Y3 Z5 Z6 Y7] +
(-0.018889030304942905+0j) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.9473560115420034e-06+0j) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.65389422268317+0j) [Z3] +
(-1.1708301369613063e-06+0j) [Z3 X4 Z5 X6] +
(-7.0897994673103395e-06+0j) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-1.1708301369613063e-06+0j) [Z3 Y4 Z5 Y6] +
(-7.0897994673103395e-06+0j) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(0.1607976453483856+0j) [Z3 Z4] +
(-9.509249751315749e-07+0j) [Z3 X5 Z6 X7] +
(-4.728843147054139e-06+0j) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-9.509249751315749e-07+0j) [Z3 Y5 Z6 Y7] +
(-4.728843147054139e-06+0j) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.12495807739503215+0j) [Z3 Z5] +
(0.024389082531149534+0j) [Z3 X6 Z7 Z8 Z9 X10] +
(-2.0111220979779585e-06+0j) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.024389082531149534+0j) [Z3 Y6 Z7 Z8 Z9 Y10] +
(-2.0111220979779585e-06+0j) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.16853486561579933+0j) [Z3 Z6] +
(0.019020423173039987+0j) [Z3 X7 Z8 Z9 Z10 X11] +
(-2.1032156044926093e-06+0j) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.019020423173039987+0j) [Z3 Y7 Z8 Z9 Z10 Y11] +
(-2.1032156044926093e-06+0j) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.13739104762683219+0j) [Z3 Z7] +
(0.1869082047691255+0j) [Z3 Z8] +
(0.1507140812100829+0j) [Z3 Z9] +
(1.1094407592244284e-06+0j) [Z3 X10 Z11 X12] +
(1.1094407592244284e-06+0j) [Z3 Y10 Z11 Y12] +
(0.15337968243314148+0j) [Z3 Z10] +
(-1.0632283422530125e-06+0j) [Z3 X11 Z12 X13] +
(-1.0632283422530125e-06+0j) [Z3 Y11 Z12 Y13] +
(0.12799502492468412+0j) [Z3 Z11] +
(0.15569010671752453+0j) [Z3 Z12] +
(0.1401128986535481+0j) [Z3 Z13] +
(-0.011982389010247983+0j) [X4 X5 Y6 Y7] +
(-0.007306759928832981+0j) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11] +
(-2.8882935957895404e-07+0j) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0073067599288329805+0j) [X4 X5 X7 Z8 Z9 X10] +
(-2.8882935957895404e-07+0j) [X4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-0.007156934919856945+0j) [X4 X5 Y8 Y9] +
(-0.0176800679524815+0j) [X4 X5 Y10 Y11] +
(-3.694513294290817e-06+0j) [X4 X5 Y10 Z11 Z12 Y13] +
(-3.694513294290817e-06+0j) [X4 X5 X11 X12] +
(-0.038314670294803864+0j) [X4 X5 Y12 Y13] +
(0.011982389010247983+0j) [X4 Y5 Y6 X7] +
(0.007306759928832981+0j) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11] +
(2.8882935957895404e-07+0j) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0073067599288329805+0j) [X4 Y5 Y7 Z8 Z9 X10] +
(-2.8882935957895404e-07+0j) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(0.007156934919856945+0j) [X4 Y5 Y8 X9] +
(0.0176800679524815+0j) [X4 Y5 Y10 X11] +
(3.694513294290817e-06+0j) [X4 Y5 Y10 Z11 Z12 X13] +
(-3.694513294290817e-06+0j) [X4 Y5 Y11 X12] +
(0.038314670294803864+0j) [X4 Y5 Y12 X13] +
(-1.2260484988626412e-05+0j) [X4 Z5 X6] +
(-1.228333782542861e-06+0j) [X4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(0.0002463643756957518+0j) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.228333782542861e-06+0j) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(0.0002463643756957518+0j) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.8540608579247497e-06+0j) [X4 Z5 X6 Z7] +
(-1.3980449080893421e-06+0j) [X4 Z5 X6 Z8] +
(-1.881850183196008e-06+0j) [X4 Z5 X6 Z9] +
(0.007960880725921576+0j) [X4 Z5 X6 X10 Z11 X12] +
(-0.0009298507967730484+0j) [X4 Z5 X6 Y10 Z11 Y12] +
(-1.6923978284614469e-06+0j) [X4 Z5 X6 Z10] +
(-0.01221504099761398+0j) [X4 Z5 X6 X11 Z12 X13] +
(-0.01221504099761398+0j) [X4 Z5 X6 Y11 Z12 Y13] +
(4.281913884822879e-06+0j) [X4 Z5 X6 Z11] +
(-4.588855155497658e-06+0j) [X4 Z5 X6 Z13] +
(0.008890731522694624+0j) [X4 Z5 Y6 Y10 Z11 X12] +
(-4.838052751066658e-07+0j) [X4 Z5 Z6 X7 Y8 Y9] +
(5.974311713284325e-06+0j) [X4 Z5 Z6 X7 Y10 Y11] +
(0.011285190200840931+0j) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(0.020175921723535554+0j) [X4 Z5 Z6 X7 X11 X12] +
(-4.5565692179352414e-06+0j) [X4 Z5 Z6 X7 Y12 Y13] +
(4.838052751066658e-07+0j) [X4 Z5 Z6 Y7 Y8 X9] +
(-5.974311713284325e-06+0j) [X4 Z5 Z6 Y7 Y10 X11] +
(-0.011285190200840931+0j) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(0.020175921723535554+0j) [X4 Z5 Z6 Y7 Y11 X12] +
(4.5565692179352414e-06+0j) [X4 Z5 Z6 Y7 Y12 X13] +
(1.3304731886481093e-06+0j) [X4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(0.005923798336561347+0j) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(1.3304731886481093e-06+0j) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(0.005923798336561347+0j) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(-6.631277928181194e-05+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.016024603689179507+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(-0.016024603689179507+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(3.334331289356844e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(-4.734622038557671e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(-9.806102774864012e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(-5.0714807363063405e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(5.0714807363063405e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-0.3693708936615611+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.02314513092952901+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-0.009612634606847314+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(-0.025637238296026817+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(-8.774817864287861e-06+0j) [X4 Z5 Z6 Z7 Z8 X10] +
(-0.0476426121763831+0j) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(-7.444344675639752e-06+0j) [X4 Z5 Z6 Z7 Z9 X10] +
(-0.04171881383982176+0j) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(-6.2900284329000984e-06+0j) [X4 Z5 Z6 Z8 Z9 X10] +
(-0.03956441632289327+0j) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(-7.51836221544296e-06+0j) [X4 Z5 Z7 Z8 Z9 X10] +
(-0.03931805194719752+0j) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(-5.929765815612109e-07+0j) [X4 X6] +
(-4.25322422550171e-06+0j) [X4 Z6 Z7 Z8 Z9 X10] +
(0.011982389010247983+0j) [Y4 X5 X6 Y7] +
(0.007306759928832981+0j) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11] +
(2.8882935957895404e-07+0j) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0073067599288329805+0j) [Y4 X5 X7 Z8 Z9 Y10] +
(-2.8882935957895404e-07+0j) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(0.007156934919856945+0j) [Y4 X5 X8 Y9] +
(0.0176800679524815+0j) [Y4 X5 X10 Y11] +
(3.694513294290817e-06+0j) [Y4 X5 X10 Z11 Z12 Y13] +
(-3.694513294290817e-06+0j) [Y4 X5 X11 Y12] +
(0.038314670294803864+0j) [Y4 X5 X12 Y13] +
(-0.011982389010247983+0j) [Y4 Y5 X6 X7] +
(-0.007306759928832981+0j) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11] +
(-2.8882935957895404e-07+0j) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0073067599288329805+0j) [Y4 Y5 Y7 Z8 Z9 Y10] +
(-2.8882935957895404e-07+0j) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.007156934919856945+0j) [Y4 Y5 X8 X9] +
(-0.0176800679524815+0j) [Y4 Y5 X10 X11] +
(-3.694513294290817e-06+0j) [Y4 Y5 X10 Z11 Z12 X13] +
(-3.694513294290817e-06+0j) [Y4 Y5 Y11 Y12] +
(-0.038314670294803864+0j) [Y4 Y5 X12 X13] +
(0.008890731522694624+0j) [Y4 Z5 X6 X10 Z11 Y12] +
(-1.2260484988626412e-05+0j) [Y4 Z5 Y6] +
(-1.228333782542861e-06+0j) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(0.0002463643756957518+0j) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.228333782542861e-06+0j) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(0.0002463643756957518+0j) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.8540608579247497e-06+0j) [Y4 Z5 Y6 Z7] +
(-1.3980449080893421e-06+0j) [Y4 Z5 Y6 Z8] +
(-1.881850183196008e-06+0j) [Y4 Z5 Y6 Z9] +
(-0.0009298507967730484+0j) [Y4 Z5 Y6 X10 Z11 X12] +
(0.007960880725921576+0j) [Y4 Z5 Y6 Y10 Z11 Y12] +
(-1.6923978284614469e-06+0j) [Y4 Z5 Y6 Z10] +
(-0.01221504099761398+0j) [Y4 Z5 Y6 X11 Z12 X13] +
(-0.01221504099761398+0j) [Y4 Z5 Y6 Y11 Z12 Y13] +
(4.281913884822879e-06+0j) [Y4 Z5 Y6 Z11] +
(-4.588855155497658e-06+0j) [Y4 Z5 Y6 Z13] +
(4.838052751066658e-07+0j) [Y4 Z5 Z6 X7 X8 Y9] +
(-5.974311713284325e-06+0j) [Y4 Z5 Z6 X7 X10 Y11] +
(-0.011285190200840931+0j) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(0.020175921723535554+0j) [Y4 Z5 Z6 X7 X11 Y12] +
(4.5565692179352414e-06+0j) [Y4 Z5 Z6 X7 X12 Y13] +
(-4.838052751066658e-07+0j) [Y4 Z5 Z6 Y7 X8 X9] +
(5.974311713284325e-06+0j) [Y4 Z5 Z6 Y7 X10 X11] +
(0.011285190200840931+0j) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(0.020175921723535554+0j) [Y4 Z5 Z6 Y7 Y11 Y12] +
(-4.5565692179352414e-06+0j) [Y4 Z5 Z6 Y7 X12 X13] +
(1.3304731886481093e-06+0j) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(0.005923798336561347+0j) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(1.3304731886481093e-06+0j) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(0.005923798336561347+0j) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(-6.631277928181194e-05+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.016024603689179507+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(-0.016024603689179507+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(3.334331289356844e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(-4.734622038557671e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(-9.806102774864012e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(5.0714807363063405e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-5.0714807363063405e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-0.3693708936615611+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.02314513092952901+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-0.009612634606847314+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(-0.025637238296026817+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(-8.774817864287861e-06+0j) [Y4 Z5 Z6 Z7 Z8 Y10] +
(-0.0476426121763831+0j) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(-7.444344675639752e-06+0j) [Y4 Z5 Z6 Z7 Z9 Y10] +
(-0.04171881383982176+0j) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(-6.2900284329000984e-06+0j) [Y4 Z5 Z6 Z8 Z9 Y10] +
(-0.03956441632289327+0j) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(-7.51836221544296e-06+0j) [Y4 Z5 Z7 Z8 Z9 Y10] +
(-0.03931805194719752+0j) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(-5.929765815612109e-07+0j) [Y4 Y6] +
(-4.25322422550171e-06+0j) [Y4 Z6 Z7 Z8 Z9 Y10] +
(-5.929765815612109e-07+0j) [Z4 X5 Z6 X7] +
(-4.25322422550171e-06+0j) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-5.929765815612109e-07+0j) [Z4 Y5 Z6 Y7] +
(-4.25322422550171e-06+0j) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.15755314797985667+0j) [Z4 Z5] +
(0.01826683486937558+0j) [Z4 X6 Z7 Z8 Z9 X10] +
(-1.6541174769226934e-06+0j) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.01826683486937558+0j) [Z4 Y6 Z7 Z8 Z9 Y10] +
(-1.6541174769226934e-06+0j) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.13701191674040752+0j) [Z4 Z6] +
(0.010960074940542604+0j) [Z4 X7 Z8 Z9 Z10 X11] +
(-1.9429468365016473e-06+0j) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.010960074940542604+0j) [Z4 Y7 Z8 Z9 Z10 Y11] +
(-1.9429468365016473e-06+0j) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.14899430575065553+0j) [Z4 Z7] +
(0.149607026844453+0j) [Z4 Z8] +
(0.15676396176430998+0j) [Z4 Z9] +
(1.8782101247102625e-06+0j) [Z4 X10 Z11 X12] +
(1.8782101247102625e-06+0j) [Z4 Y10 Z11 Y12] +
(0.12489990917237609+0j) [Z4 Z10] +
(-1.8163031695805545e-06+0j) [Z4 X11 Z12 X13] +
(-1.8163031695805545e-06+0j) [Z4 Y11 Z12 Y13] +
(0.1425799771248576+0j) [Z4 Z11] +
(0.11383573679388657+0j) [Z4 Z12] +
(0.15215040708869046+0j) [Z4 Z13] +
(1.228333782542861e-06+0j) [X5 X6 Y7 Z8 Z9 Y10] +
(-0.0002463643756957518+0j) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-4.838052751066658e-07+0j) [X5 X6 X8 X9] +
(5.974311713284326e-06+0j) [X5 X6 X10 X11] +
(0.020175921723535554+0j) [X5 X6 X10 Z11 Z12 X13] +
(0.011285190200840931+0j) [X5 X6 Y11 Y12] +
(-4.556569217935241e-06+0j) [X5 X6 X12 X13] +
(-1.228333782542861e-06+0j) [X5 Y6 Y7 Z8 Z9 X10] +
(0.0002463643756957518+0j) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-4.838052751066658e-07+0j) [X5 Y6 Y8 X9] +
(5.974311713284326e-06+0j) [X5 Y6 Y10 X11] +
(0.020175921723535554+0j) [X5 Y6 Y10 Z11 Z12 X13] +
(-0.011285190200840931+0j) [X5 Y6 Y11 X12] +
(-4.556569217935241e-06+0j) [X5 Y6 Y12 X13] +
(-1.2260484988626417e-05+0j) [X5 Z6 X7] +
(-1.881850183196008e-06+0j) [X5 Z6 X7 Z8] +
(-1.3980449080893421e-06+0j) [X5 Z6 X7 Z9] +
(-0.01221504099761398+0j) [X5 Z6 X7 X10 Z11 X12] +
(-0.01221504099761398+0j) [X5 Z6 X7 Y10 Z11 Y12] +
(4.281913884822879e-06+0j) [X5 Z6 X7 Z10] +
(0.007960880725921576+0j) [X5 Z6 X7 X11 Z12 X13] +
(-0.0009298507967730484+0j) [X5 Z6 X7 Y11 Z12 Y13] +
(-1.6923978284614469e-06+0j) [X5 Z6 X7 Z11] +
(-4.588855155497658e-06+0j) [X5 Z6 X7 Z12] +
(0.008890731522694624+0j) [X5 Z6 Y7 Y11 Z12 X13] +
(-1.3304731886481093e-06+0j) [X5 Z6 Z7 X8 Y9 Y10] +
(-0.005923798336561347+0j) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(1.3304731886481093e-06+0j) [X5 Z6 Z7 Y8 Y9 X10] +
(0.005923798336561347+0j) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(0.016024603689179507+0j) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12] +
(-5.071480736306339e-06+0j) [X5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-0.016024603689179507+0j) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12] +
(-5.071480736306339e-06+0j) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(-6.631277928181194e-05+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-9.806102774864012e-06+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(-4.734622038557671e-06+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(-0.3693708936615611+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.02314513092952901+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(-0.025637238296026817+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(3.334331289356844e-06+0j) [X5 Z6 Z7 Z8 Z9 X11] +
(-0.009612634606847314+0j) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(-7.444344675639752e-06+0j) [X5 Z6 Z7 Z8 Z10 X11] +
(-0.04171881383982176+0j) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(-8.774817864287861e-06+0j) [X5 Z6 Z7 Z9 Z10 X11] +
(-0.0476426121763831+0j) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(-7.51836221544296e-06+0j) [X5 Z6 Z8 Z9 Z10 X11] +
(-0.03931805194719752+0j) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.8540608579247494e-06+0j) [X5 X7] +
(-6.2900284329000984e-06+0j) [X5 Z7 Z8 Z9 Z10 X11] +
(-0.03956441632289327+0j) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.228333782542861e-06+0j) [Y5 X6 X7 Z8 Z9 Y10] +
(0.0002463643756957518+0j) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-4.838052751066658e-07+0j) [Y5 X6 X8 Y9] +
(5.974311713284326e-06+0j) [Y5 X6 X10 Y11] +
(0.020175921723535554+0j) [Y5 X6 X10 Z11 Z12 Y13] +
(-0.011285190200840931+0j) [Y5 X6 X11 Y12] +
(-4.556569217935241e-06+0j) [Y5 X6 X12 Y13] +
(1.228333782542861e-06+0j) [Y5 Y6 X7 Z8 Z9 X10] +
(-0.0002463643756957518+0j) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-4.838052751066658e-07+0j) [Y5 Y6 Y8 Y9] +
(5.974311713284326e-06+0j) [Y5 Y6 Y10 Y11] +
(0.020175921723535554+0j) [Y5 Y6 Y10 Z11 Z12 Y13] +
(0.011285190200840931+0j) [Y5 Y6 X11 X12] +
(-4.556569217935241e-06+0j) [Y5 Y6 Y12 Y13] +
(0.008890731522694624+0j) [Y5 Z6 X7 X11 Z12 Y13] +
(-1.2260484988626417e-05+0j) [Y5 Z6 Y7] +
(-1.881850183196008e-06+0j) [Y5 Z6 Y7 Z8] +
(-1.3980449080893421e-06+0j) [Y5 Z6 Y7 Z9] +
(-0.01221504099761398+0j) [Y5 Z6 Y7 X10 Z11 X12] +
(-0.01221504099761398+0j) [Y5 Z6 Y7 Y10 Z11 Y12] +
(4.281913884822879e-06+0j) [Y5 Z6 Y7 Z10] +
(-0.0009298507967730484+0j) [Y5 Z6 Y7 X11 Z12 X13] +
(0.007960880725921576+0j) [Y5 Z6 Y7 Y11 Z12 Y13] +
(-1.6923978284614469e-06+0j) [Y5 Z6 Y7 Z11] +
(-4.588855155497658e-06+0j) [Y5 Z6 Y7 Z12] +
(1.3304731886481093e-06+0j) [Y5 Z6 Z7 X8 X9 Y10] +
(0.005923798336561347+0j) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-1.3304731886481093e-06+0j) [Y5 Z6 Z7 Y8 X9 X10] +
(-0.005923798336561347+0j) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-0.016024603689179507+0j) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12] +
(-5.071480736306339e-06+0j) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(0.016024603689179507+0j) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12] +
(-5.071480736306339e-06+0j) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(-6.631277928181194e-05+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-9.806102774864012e-06+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(-4.734622038557671e-06+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(-0.3693708936615611+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.02314513092952901+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(-0.025637238296026817+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(3.334331289356844e-06+0j) [Y5 Z6 Z7 Z8 Z9 Y11] +
(-0.009612634606847314+0j) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(-7.444344675639752e-06+0j) [Y5 Z6 Z7 Z8 Z10 Y11] +
(-0.04171881383982176+0j) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(-8.774817864287861e-06+0j) [Y5 Z6 Z7 Z9 Z10 Y11] +
(-0.0476426121763831+0j) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(-7.51836221544296e-06+0j) [Y5 Z6 Z8 Z9 Z10 Y11] +
(-0.03931805194719752+0j) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.8540608579247494e-06+0j) [Y5 Y7] +
(-6.2900284329000984e-06+0j) [Y5 Z7 Z8 Z9 Z10 Y11] +
(-0.03956441632289327+0j) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.010960074940542604+0j) [Z5 X6 Z7 Z8 Z9 X10] +
(-1.9429468365016473e-06+0j) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.010960074940542604+0j) [Z5 Y6 Z7 Z8 Z9 Y10] +
(-1.9429468365016473e-06+0j) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.14899430575065553+0j) [Z5 Z6] +
(0.01826683486937558+0j) [Z5 X7 Z8 Z9 Z10 X11] +
(-1.6541174769226934e-06+0j) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.01826683486937558+0j) [Z5 Y7 Z8 Z9 Z10 Y11] +
(-1.6541174769226934e-06+0j) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.13701191674040752+0j) [Z5 Z7] +
(0.15676396176430998+0j) [Z5 Z8] +
(0.149607026844453+0j) [Z5 Z9] +
(-1.8163031695805545e-06+0j) [Z5 X10 Z11 X12] +
(-1.8163031695805545e-06+0j) [Z5 Y10 Z11 Y12] +
(0.1425799771248576+0j) [Z5 Z10] +
(1.8782101247102625e-06+0j) [Z5 X11 Z12 X13] +
(1.8782101247102625e-06+0j) [Z5 Y11 Z12 Y13] +
(0.12489990917237609+0j) [Z5 Z11] +
(0.15215040708869046+0j) [Z5 Z12] +
(0.11383573679388657+0j) [Z5 Z13] +
(-0.013873381748426105+0j) [X6 X7 Y8 Y9] +
(-0.017825140995786543+0j) [X6 X7 Y10 Y11] +
(-1.035847760232373e-06+0j) [X6 X7 Y10 Z11 Z12 Y13] +
(-1.035847760232373e-06+0j) [X6 X7 X11 X12] +
(-0.017366118994651434+0j) [X6 X7 Y12 Y13] +
(0.013873381748426105+0j) [X6 Y7 Y8 X9] +
(0.017825140995786543+0j) [X6 Y7 Y10 X11] +
(1.035847760232373e-06+0j) [X6 Y7 Y10 Z11 Z12 X13] +
(-1.035847760232373e-06+0j) [X6 Y7 Y11 X12] +
(0.017366118994651434+0j) [X6 Y7 Y12 X13] +
(0.0002921986261110502+0j) [X6 Z7 X8 X9 Z10 X11] +
(-3.3281393507656363e-07+0j) [X6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.0002921986261110502+0j) [X6 Z7 X8 Y9 Z10 Y11] +
(-3.3281393507656363e-07+0j) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(0.22848106564918938+0j) [X6 Z7 Z8 Z9 X10] +
(3.313145500155433e-06+0j) [X6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(3.313145500155433e-06+0j) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(0.011307274008848262+0j) [X6 Z7 Z8 Z9 X10 Z11] +
(0.02510495713884457+0j) [X6 Z7 Z8 Z9 X10 Z12] +
(0.010540425907671545+0j) [X6 Z7 Z8 Z9 X10 Z13] +
(-0.014564531231173022+0j) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(0.014564531231173022+0j) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-2.595086006760903e-05+0j) [X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.18393255937118e-06+0j) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-6.52437384833992e-06+0j) [X6 Z7 Z8 Z9 Z10 X12] +
(-3.211228348184486e-06+0j) [X6 Z7 Z8 Z9 Z11 X12] +
(0.029812424517345858+0j) [X6 Z7 Z8 X10] +
(-3.277483195205242e-06+0j) [X6 Z7 Z8 Z10 Z11 X12] +
(0.030104623143456907+0j) [X6 Z7 Z9 X10] +
(-3.6102971302818057e-06+0j) [X6 Z7 Z9 Z10 Z11 X12] +
(0.030787505389143995+0j) [X6 Z8 Z9 X10] +
(-3.7696594516737165e-06+0j) [X6 Z8 Z9 Z10 Z11 X12] +
(0.013873381748426105+0j) [Y6 X7 X8 Y9] +
(0.017825140995786543+0j) [Y6 X7 X10 Y11] +
(1.035847760232373e-06+0j) [Y6 X7 X10 Z11 Z12 Y13] +
(-1.035847760232373e-06+0j) [Y6 X7 X11 Y12] +
(0.017366118994651434+0j) [Y6 X7 X12 Y13] +
(-0.013873381748426105+0j) [Y6 Y7 X8 X9] +
(-0.017825140995786543+0j) [Y6 Y7 X10 X11] +
(-1.035847760232373e-06+0j) [Y6 Y7 X10 Z11 Z12 X13] +
(-1.035847760232373e-06+0j) [Y6 Y7 Y11 Y12] +
(-0.017366118994651434+0j) [Y6 Y7 X12 X13] +
(0.0002921986261110502+0j) [Y6 Z7 Y8 X9 Z10 X11] +
(-3.3281393507656363e-07+0j) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.0002921986261110502+0j) [Y6 Z7 Y8 Y9 Z10 Y11] +
(-3.3281393507656363e-07+0j) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(0.22848106564918938+0j) [Y6 Z7 Z8 Z9 Y10] +
(3.313145500155433e-06+0j) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(3.313145500155433e-06+0j) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(0.011307274008848262+0j) [Y6 Z7 Z8 Z9 Y10 Z11] +
(0.02510495713884457+0j) [Y6 Z7 Z8 Z9 Y10 Z12] +
(0.010540425907671545+0j) [Y6 Z7 Z8 Z9 Y10 Z13] +
(0.014564531231173022+0j) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-0.014564531231173022+0j) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-2.595086006760903e-05+0j) [Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.18393255937118e-06+0j) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-6.52437384833992e-06+0j) [Y6 Z7 Z8 Z9 Z10 Y12] +
(-3.211228348184486e-06+0j) [Y6 Z7 Z8 Z9 Z11 Y12] +
(0.029812424517345858+0j) [Y6 Z7 Z8 Y10] +
(-3.277483195205242e-06+0j) [Y6 Z7 Z8 Z10 Z11 Y12] +
(0.030104623143456907+0j) [Y6 Z7 Z9 Y10] +
(-3.6102971302818057e-06+0j) [Y6 Z7 Z9 Z10 Z11 Y12] +
(0.030787505389143995+0j) [Y6 Z8 Z9 Y10] +
(-3.7696594516737165e-06+0j) [Y6 Z8 Z9 Z10 Z11 Y12] +
(1.3096862988615425+0j) [Z6] +
(0.030787505389143995+0j) [Z6 X7 Z8 Z9 Z10 X11] +
(-3.7696594516737165e-06+0j) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.030787505389143995+0j) [Z6 Y7 Z8 Z9 Z10 Y11] +
(-3.7696594516737165e-06+0j) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.19392534613270201+0j) [Z6 Z7] +
(0.16756653265461274+0j) [Z6 Z8] +
(0.18143991440303886+0j) [Z6 Z9] +
(-1.8551201213792047e-06+0j) [Z6 X10 Z11 X12] +
(-1.8551201213792047e-06+0j) [Z6 Y10 Z11 Y12] +
(0.11952438964682681+0j) [Z6 Z10] +
(-2.890967881611578e-06+0j) [Z6 X11 Z12 X13] +
(-2.890967881611578e-06+0j) [Z6 Y11 Z12 Y13] +
(0.13734953064261335+0j) [Z6 Z11] +
(0.134017152619637+0j) [Z6 Z12] +
(0.15138327161428844+0j) [Z6 Z13] +
(-0.00029219862611105024+0j) [X7 X8 Y9 Y10] +
(3.3281393507656363e-07+0j) [X7 X8 Y9 Z10 Z11 Y12] +
(0.00029219862611105024+0j) [X7 Y8 Y9 X10] +
(-3.3281393507656363e-07+0j) [X7 Y8 Y9 Z10 Z11 X12] +
(-3.313145500155433e-06+0j) [X7 Z8 Z9 X10 Y11 Y12] +
(-0.014564531231173022+0j) [X7 Z8 Z9 X10 X12 X13] +
(3.313145500155433e-06+0j) [X7 Z8 Z9 Y10 Y11 X12] +
(-0.014564531231173022+0j) [X7 Z8 Z9 Y10 Y12 X13] +
(0.2284810656491892+0j) [X7 Z8 Z9 Z10 X11] +
(0.010540425907671545+0j) [X7 Z8 Z9 Z10 X11 Z12] +
(0.02510495713884457+0j) [X7 Z8 Z9 Z10 X11 Z13] +
(-2.595086006760903e-05+0j) [X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.18393255937118e-06+0j) [X7 Z8 Z9 Z10 Z11 X13] +
(-3.211228348184486e-06+0j) [X7 Z8 Z9 Z10 Z12 X13] +
(0.011307274008848262+0j) [X7 Z8 Z9 X11] +
(-6.52437384833992e-06+0j) [X7 Z8 Z9 Z11 Z12 X13] +
(0.030104623143456907+0j) [X7 Z8 Z10 X11] +
(-3.6102971302818057e-06+0j) [X7 Z8 Z10 Z11 Z12 X13] +
(0.029812424517345858+0j) [X7 Z9 Z10 X11] +
(-3.277483195205242e-06+0j) [X7 Z9 Z10 Z11 Z12 X13] +
(0.00029219862611105024+0j) [Y7 X8 X9 Y10] +
(-3.3281393507656363e-07+0j) [Y7 X8 X9 Z10 Z11 Y12] +
(-0.00029219862611105024+0j) [Y7 Y8 X9 X10] +
(3.3281393507656363e-07+0j) [Y7 Y8 X9 Z10 Z11 X12] +
(3.313145500155433e-06+0j) [Y7 Z8 Z9 X10 X11 Y12] +
(-0.014564531231173022+0j) [Y7 Z8 Z9 X10 X12 Y13] +
(-3.313145500155433e-06+0j) [Y7 Z8 Z9 Y10 X11 X12] +
(-0.014564531231173022+0j) [Y7 Z8 Z9 Y10 Y12 Y13] +
(0.2284810656491892+0j) [Y7 Z8 Z9 Z10 Y11] +
(0.010540425907671545+0j) [Y7 Z8 Z9 Z10 Y11 Z12] +
(0.02510495713884457+0j) [Y7 Z8 Z9 Z10 Y11 Z13] +
(-2.595086006760903e-05+0j) [Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(4.18393255937118e-06+0j) [Y7 Z8 Z9 Z10 Z11 Y13] +
(-3.211228348184486e-06+0j) [Y7 Z8 Z9 Z10 Z12 Y13] +
(0.011307274008848262+0j) [Y7 Z8 Z9 Y11] +
(-6.52437384833992e-06+0j) [Y7 Z8 Z9 Z11 Z12 Y13] +
(0.030104623143456907+0j) [Y7 Z8 Z10 Y11] +
(-3.6102971302818057e-06+0j) [Y7 Z8 Z10 Z11 Z12 Y13] +
(0.029812424517345858+0j) [Y7 Z9 Z10 Y11] +
(-3.277483195205242e-06+0j) [Y7 Z9 Z10 Z11 Z12 Y13] +
(1.3096862988615428+0j) [Z7] +
(0.18143991440303886+0j) [Z7 Z8] +
(0.16756653265461274+0j) [Z7 Z9] +
(-2.890967881611578e-06+0j) [Z7 X10 Z11 X12] +
(-2.890967881611578e-06+0j) [Z7 Y10 Z11 Y12] +
(0.13734953064261335+0j) [Z7 Z10] +
(-1.8551201213792047e-06+0j) [Z7 X11 Z12 X13] +
(-1.8551201213792047e-06+0j) [Z7 Y11 Z12 Y13] +
(0.11952438964682681+0j) [Z7 Z11] +
(0.15138327161428844+0j) [Z7 Z12] +
(0.134017152619637+0j) [Z7 Z13] +
(-0.009560705729135961+0j) [X8 X9 Y10 Y11] +
(6.628614201565468e-07+0j) [X8 X9 Y10 Z11 Z12 Y13] +
(6.628614201565468e-07+0j) [X8 X9 X11 X12] +
(-0.00608782248056186+0j) [X8 X9 Y12 Y13] +
(0.009560705729135961+0j) [X8 Y9 Y10 X11] +
(-6.628614201565468e-07+0j) [X8 Y9 Y10 Z11 Z12 X13] +
(6.628614201565468e-07+0j) [X8 Y9 Y11 X12] +
(0.00608782248056186+0j) [X8 Y9 Y12 X13] +
(0.009560705729135961+0j) [Y8 X9 X10 Y11] +
(-6.628614201565468e-07+0j) [Y8 X9 X10 Z11 Z12 Y13] +
(6.628614201565468e-07+0j) [Y8 X9 X11 Y12] +
(0.00608782248056186+0j) [Y8 X9 X12 Y13] +
(-0.009560705729135961+0j) [Y8 Y9 X10 X11] +
(6.628614201565468e-07+0j) [Y8 Y9 X10 Z11 Z12 X13] +
(6.628614201565468e-07+0j) [Y8 Y9 Y11 Y12] +
(-0.00608782248056186+0j) [Y8 Y9 X12 X13] +
(1.3693525634718196+0j) [Z8] +
(0.22003977334376112+0j) [Z8 Z9] +
(-1.5973171976934442e-06+0j) [Z8 X10 Z11 X12] +
(-1.5973171976934442e-06+0j) [Z8 Y10 Z11 Y12] +
(0.13766872645852593+0j) [Z8 Z10] +
(-9.344557775368971e-07+0j) [Z8 X11 Z12 X13] +
(-9.344557775368971e-07+0j) [Z8 Y11 Z12 Y13] +
(0.1472294321876619+0j) [Z8 Z11] +
(0.14973486803496933+0j) [Z8 Z12] +
(0.1558226905155312+0j) [Z8 Z13] +
(1.3693525634718196+0j) [Z9] +
(-9.344557775368971e-07+0j) [Z9 X10 Z11 X12] +
(-9.344557775368971e-07+0j) [Z9 Y10 Z11 Y12] +
(0.1472294321876619+0j) [Z9 Z10] +
(-1.5973171976934442e-06+0j) [Z9 X11 Z12 X13] +
(-1.5973171976934442e-06+0j) [Z9 Y11 Z12 Y13] +
(0.13766872645852593+0j) [Z9 Z11] +
(0.1558226905155312+0j) [Z9 Z12] +
(0.14973486803496933+0j) [Z9 Z13] +
(-0.02868518371610587+0j) [X10 X11 Y12 Y13] +
(0.02868518371610587+0j) [X10 Y11 Y12 X13] +
(-1.0722312156862755e-05+0j) [X10 Z11 X12] +
(7.954413176055897e-06+0j) [X10 Z11 X12 Z13] +
(-8.194261371989891e-06+0j) [X10 X12] +
(0.02868518371610587+0j) [Y10 X11 X12 Y13] +
(-0.02868518371610587+0j) [Y10 Y11 X12 X13] +
(-1.0722312156862755e-05+0j) [Y10 Z11 Y12] +
(7.954413176055897e-06+0j) [Y10 Z11 Y12 Z13] +
(-8.194261371989891e-06+0j) [Y10 Y12] +
(0.7829661725950182+0j) [Z10] +
(-8.194261371989891e-06+0j) [Z10 X11 Z12 X13] +
(-8.194261371989891e-06+0j) [Z10 Y11 Z12 Y13] +
(0.1492635514738891+0j) [Z10 Z11] +
(0.11270386920332219+0j) [Z10 Z12] +
(0.14138905291942808+0j) [Z10 Z13] +
(-1.0722312156862757e-05+0j) [X11 Z12 X13] +
(7.954413176055897e-06+0j) [X11 X13] +
(-1.0722312156862757e-05+0j) [Y11 Z12 Y13] +
(7.954413176055897e-06+0j) [Y11 Y13] +
(0.7829661725950184+0j) [Z11] +
(0.14138905291942808+0j) [Z11 Z12] +
(0.11270386920332219+0j) [Z11 Z13] +
(0.8084581961720474+0j) [Z12] +
(0.15435748657223636+0j) [Z12 Z13] +
(0.8084581961720475+0j) [Z13]
  (-46.463906788688945) [I0]
+ (0.7829661725950192) [Z10]
+ (0.7829661725950192) [Z11]
+ (0.8084581961720478) [Z12]
+ (0.808458196172049) [Z13]
+ (1.203440228914564) [Z4]
+ (1.2034402289145643) [Z5]
+ (1.3096862988615405) [Z6]
+ (1.3096862988615405) [Z7]
+ (1.369352563471818) [Z9]
+ (1.3693525634718189) [Z8]
+ (1.65389422268317) [Z3]
+ (1.6538942226831712) [Z2]
+ (12.41263074211177) [Z0]
+ (12.41263074211177) [Z1]
+ (-8.19426137298764e-06) [Y10 Y12]
+ (-8.19426137298764e-06) [X10 X12]
+ (-1.8540608579209804e-06) [Y5 Y7]
+ (-1.8540608579209804e-06) [X5 X7]
+ (-7.764994118585462e-07) [Y3 Y5]
+ (-7.764994118585462e-07) [X3 X5]
+ (-5.929765815797967e-07) [Y4 Y6]
+ (-5.929765815797967e-07) [X4 X6]
+ (1.602116740510007e-06) [Y2 Y4]
+ (1.602116740510007e-06) [X2 X4]
+ (7.954413176813625e-06) [Y11 Y13]
+ (7.954413176813625e-06) [X11 X13]
+ (0.003276971931231706) [Y1 Y3]
+ (0.003276971931231706) [X1 X3]
+ (0.10433064780651419) [Y0 Y2]
+ (0.10433064780651419) [X0 X2]
+ (0.11270386920332225) [Z10 Z12]
+ (0.11270386920332225) [Z11 Z13]
+ (0.11383573679388659) [Z4 Z12]
+ (0.11383573679388659) [Z5 Z13]
+ (0.11952438964682668) [Z6 Z10]
+ (0.11952438964682668) [Z7 Z11]
+ (0.12489990917237605) [Z4 Z10]
+ (0.12489990917237605) [Z5 Z11]
+ (0.12495807739503216) [Z2 Z4]
+ (0.12495807739503216) [Z3 Z5]
+ (0.12799502492468412) [Z2 Z10]
+ (0.12799502492468412) [Z3 Z11]
+ (0.13401715261963695) [Z6 Z12]
+ (0.13401715261963695) [Z7 Z13]
+ (0.13701191674040747) [Z4 Z6]
+ (0.13701191674040747) [Z5 Z7]
+ (0.13734953064261327) [Z6 Z11]
+ (0.13734953064261327) [Z7 Z10]
+ (0.13739104762683213) [Z2 Z6]
+ (0.13739104762683213) [Z3 Z7]
+ (0.1376687264585258) [Z8 Z10]
+ (0.1376687264585258) [Z9 Z11]
+ (0.1401128986535481) [Z2 Z12]
+ (0.1401128986535481) [Z3 Z13]
+ (0.1413890529194281) [Z10 Z13]
+ (0.1413890529194281) [Z11 Z12]
+ (0.14257997712485754) [Z4 Z11]
+ (0.14257997712485754) [Z5 Z10]
+ (0.1472294321876617) [Z8 Z11]
+ (0.1472294321876617) [Z9 Z10]
+ (0.14899430575065548) [Z4 Z7]
+ (0.14899430575065548) [Z5 Z6]
+ (0.149263551473889) [Z10 Z11]
+ (0.149607026844453) [Z4 Z8]
+ (0.149607026844453) [Z5 Z9]
+ (0.14973486803496927) [Z8 Z12]
+ (0.14973486803496927) [Z9 Z13]
+ (0.1507140812100829) [Z2 Z8]
+ (0.1507140812100829) [Z3 Z9]
+ (0.15138327161428833) [Z6 Z13]
+ (0.15138327161428833) [Z7 Z12]
+ (0.1521504070886905) [Z4 Z13]
+ (0.1521504070886905) [Z5 Z12]
+ (0.15337968243314143) [Z2 Z11]
+ (0.15337968243314143) [Z3 Z10]
+ (0.15435748657223625) [Z12 Z13]
+ (0.15569010671752453) [Z2 Z13]
+ (0.15569010671752453) [Z3 Z12]
+ (0.1558226905155311) [Z8 Z13]
+ (0.1558226905155311) [Z9 Z12]
+ (0.15676396176430996) [Z4 Z9]
+ (0.15676396176430996) [Z5 Z8]
+ (0.1575531479798567) [Z4 Z5]
+ (0.1607976453483856) [Z2 Z5]
+ (0.1607976453483856) [Z3 Z4]
+ (0.16756653265461258) [Z6 Z8]
+ (0.16756653265461258) [Z7 Z9]
+ (0.16853486561579936) [Z2 Z7]
+ (0.16853486561579936) [Z3 Z6]
+ (0.18143991440303858) [Z6 Z9]
+ (0.18143991440303858) [Z7 Z8]
+ (0.18189085790751353) [Z2 Z3]
+ (0.18690820476912556) [Z2 Z9]
+ (0.18690820476912556) [Z3 Z8]
+ (0.19299723935364232) [Z0 Z10]
+ (0.19299723935364232) [Z1 Z11]
+ (0.1939253461327015) [Z6 Z7]
+ (0.19661770890342145) [Z0 Z4]
+ (0.19661770890342145) [Z1 Z5]
+ (0.1993635453736083) [Z0 Z5]
+ (0.1993635453736083) [Z1 Z4]
+ (0.20072866460441757) [Z0 Z11]
+ (0.20072866460441757) [Z1 Z10]
+ (0.211026598497915) [Z0 Z12]
+ (0.211026598497915) [Z1 Z13]
+ (0.21631037498631794) [Z0 Z13]
+ (0.21631037498631794) [Z1 Z12]
+ (0.23671080783830423) [Z0 Z2]
+ (0.23671080783830423) [Z1 Z3]
+ (0.2416466393601716) [Z0 Z6]
+ (0.2416466393601716) [Z1 Z7]
+ (0.2485348337131421) [Z0 Z7]
+ (0.2485348337131421) [Z1 Z6]
+ (0.25129445674591694) [Z0 Z3]
+ (0.25129445674591694) [Z1 Z2]
+ (0.2723251830660567) [Z0 Z8]
+ (0.2723251830660567) [Z1 Z9]
+ (0.27883454426723386) [Z0 Z9]
+ (0.27883454426723386) [Z1 Z8]
+ (1.1861763734860487) [Z0 Z1]
+ (-1.2260484988743093e-05) [Y4 Z5 Y6]
+ (-1.2260484988743093e-05) [X4 Z5 X6]
+ (-1.2260484988743093e-05) [Y5 Z6 Y7]
+ (-1.2260484988743093e-05) [X5 Z6 X7]
+ (-1.0722312157927411e-05) [Y11 Z12 Y13]
+ (-1.0722312157927411e-05) [X11 Z12 X13]
+ (-1.0722312157927408e-05) [Y10 Z11 Y12]
+ (-1.0722312157927408e-05) [X10 Z11 X12]
+ (-3.887051672999319e-06) [Y3 Z4 Y5]
+ (-3.887051672999319e-06) [X3 Z4 X5]
+ (-3.887051672999318e-06) [Y2 Z3 Y4]
+ (-3.887051672999318e-06) [X2 Z3 X4]
+ (0.12507032579772148) [Y0 Z1 Y2]
+ (0.12507032579772148) [X0 Z1 X2]
+ (0.1250703257977215) [Y1 Z2 Y3]
+ (0.1250703257977215) [X1 Z2 X3]
+ (-0.03831467029480389) [Y4 Y5 X12 X13]
+ (-0.03831467029480389) [X4 X5 Y12 Y13]
+ (-0.03619412355904267) [Y2 Y3 X8 X9]
+ (-0.03619412355904267) [X2 X3 Y8 Y9]
+ (-0.03583956795335344) [Y2 Y3 X4 X5]
+ (-0.03583956795335344) [X2 X3 Y4 Y5]
+ (-0.031143817988967235) [Y2 Y3 X6 X7]
+ (-0.031143817988967235) [X2 X3 Y6 Y7]
+ (-0.028685183716105876) [Y10 Y11 X12 X13]
+ (-0.028685183716105876) [X10 X11 Y12 Y13]
+ (-0.025996177598021065) [Y3 Z4 Z5 Y7]
+ (-0.025996177598021065) [X3 Z4 Z5 X7]
+ (-0.025384657508457288) [Y2 Y3 X10 X11]
+ (-0.025384657508457288) [X2 X3 Y10 Y11]
+ (-0.019028242443847203) [Y3 Y4 X11 X12]
+ (-0.019028242443847203) [X3 X4 Y11 Y12]
+ (-0.017825140995786585) [Y6 Y7 X10 X11]
+ (-0.017825140995786585) [X6 X7 Y10 Y11]
+ (-0.01768006795248148) [Y4 Y5 X10 X11]
+ (-0.01768006795248148) [X4 X5 Y10 Y11]
+ (-0.01736611899465139) [Y6 Y7 X12 X13]
+ (-0.01736611899465139) [X6 X7 Y12 Y13]
+ (-0.015577208063976469) [Y2 Y3 X12 X13]
+ (-0.015577208063976469) [X2 X3 Y12 Y13]
+ (-0.014583648907612708) [Y0 Y1 X2 X3]
+ (-0.014583648907612708) [X0 X1 Y2 Y3]
+ (-0.013873381748426034) [Y6 Y7 X8 X9]
+ (-0.013873381748426034) [X6 X7 Y8 Y9]
+ (-0.011982389010247986) [Y4 Y5 X6 X7]
+ (-0.011982389010247986) [X4 X5 Y6 Y7]
+ (-0.011285190200840936) [Y5 X6 X11 Y12]
+ (-0.011285190200840936) [X5 Y6 Y11 X12]
+ (-0.007731425250775248) [Y0 Y1 X10 X11]
+ (-0.007731425250775248) [X0 X1 Y10 Y11]
+ (-0.007156934919856952) [Y4 Y5 X8 X9]
+ (-0.007156934919856952) [X4 X5 Y8 Y9]
+ (-0.006888194352970519) [Y0 Y1 X6 X7]
+ (-0.006888194352970519) [X0 X1 Y6 Y7]
+ (-0.006509361201177232) [Y0 Y1 X8 X9]
+ (-0.006509361201177232) [X0 X1 Y8 Y9]
+ (-0.006087822480561855) [Y8 Y9 X12 X13]
+ (-0.006087822480561855) [X8 X9 Y12 Y13]
+ (-0.0052837764884029505) [Y0 Y1 X12 X13]
+ (-0.0052837764884029505) [X0 X1 Y12 Y13]
+ (-0.005143391768825165) [Y3 X4 X5 Y6]
+ (-0.005143391768825165) [X3 Y4 Y5 X6]
+ (-0.00468490338815522) [Y1 X2 X6 Y7]
+ (-0.00468490338815522) [Y1 Y2 Y6 Y7]
+ (-0.00468490338815522) [X1 X2 X6 X7]
+ (-0.00468490338815522) [X1 Y2 Y6 X7]
+ (-0.004575007626639205) [Y1 X2 X12 Y13]
+ (-0.004575007626639205) [Y1 Y2 Y12 Y13]
+ (-0.004575007626639205) [X1 X2 X12 X13]
+ (-0.004575007626639205) [X1 Y2 Y12 X13]
+ (-0.004424855449441859) [Y1 X2 X4 Y5]
+ (-0.004424855449441859) [Y1 Y2 Y4 Y5]
+ (-0.004424855449441859) [X1 X2 X4 X5]
+ (-0.004424855449441859) [X1 Y2 Y4 X5]
+ (-0.003479511890334372) [Y2 Z3 Z5 Y6]
+ (-0.003479511890334372) [X2 Z3 Z5 X6]
+ (-0.003479511890334372) [Y3 Z4 Z6 Y7]
+ (-0.003479511890334372) [X3 Z4 Z6 X7]
+ (-0.002745836470186813) [Y0 Y1 X4 X5]
+ (-0.002745836470186813) [X0 X1 Y4 Y5]
+ (-0.001799219493663041) [Y1 X2 X10 Y11]
+ (-0.001799219493663041) [Y1 Y2 Y10 Y11]
+ (-0.001799219493663041) [X1 X2 X10 X11]
+ (-0.001799219493663041) [X1 Y2 Y10 X11]
+ (-0.00029219862611100774) [Y7 Y8 X9 X10]
+ (-0.00029219862611100774) [X7 X8 Y9 Y10]
+ (-8.19426137298764e-06) [Z10 Y11 Z12 Y13]
+ (-8.19426137298764e-06) [Z10 X11 Z12 X13]
+ (-7.801707501268766e-06) [Y2 Z3 Y4 Z11]
+ (-7.801707501268766e-06) [X2 Z3 X4 Z11]
+ (-7.801707501268766e-06) [Y3 Z4 Y5 Z10]
+ (-7.801707501268766e-06) [X3 Z4 X5 Z10]
+ (-4.6430510688976775e-06) [Y3 X4 X10 Y11]
+ (-4.6430510688976775e-06) [Y3 Y4 Y10 Y11]
+ (-4.6430510688976775e-06) [X3 X4 X10 X11]
+ (-4.6430510688976775e-06) [X3 Y4 Y10 X11]
+ (-4.5888551560244765e-06) [Y4 Z5 Y6 Z13]
+ (-4.5888551560244765e-06) [X4 Z5 X6 Z13]
+ (-4.5888551560244765e-06) [Y5 Z6 Y7 Z12]
+ (-4.5888551560244765e-06) [X5 Z6 X7 Z12]
+ (-4.5565692186228086e-06) [Y5 X6 X12 Y13]
+ (-4.5565692186228086e-06) [Y5 Y6 Y12 Y13]
+ (-4.5565692186228086e-06) [X5 X6 X12 X13]
+ (-4.5565692186228086e-06) [X5 Y6 Y12 X13]
+ (-3.6945132948112823e-06) [Y4 X5 X11 Y12]
+ (-3.6945132948112823e-06) [Y4 Y5 Y11 Y12]
+ (-3.6945132948112823e-06) [X4 X5 X11 X12]
+ (-3.6945132948112823e-06) [X4 Y5 Y11 X12]
+ (-3.344081556385234e-06) [Z0 Y5 Z6 Y7]
+ (-3.344081556385234e-06) [Z0 X5 Z6 X7]
+ (-3.344081556385234e-06) [Z1 Y4 Z5 Y6]
+ (-3.344081556385234e-06) [Z1 X4 Z5 X6]
+ (-3.158656432371087e-06) [Y2 Z3 Y4 Z10]
+ (-3.158656432371087e-06) [X2 Z3 X4 Z10]
+ (-3.158656432371087e-06) [Y3 Z4 Y5 Z11]
+ (-3.158656432371087e-06) [X3 Z4 X5 Z11]
+ (-3.0993492435152643e-06) [Z0 Y4 Z5 Y6]
+ (-3.0993492435152643e-06) [Z0 X4 Z5 X6]
+ (-3.0993492435152643e-06) [Z1 Y5 Z6 Y7]
+ (-3.0993492435152643e-06) [Z1 X5 Z6 X7]
+ (-2.8909678818779808e-06) [Z6 Y11 Z12 Y13]
+ (-2.8909678818779808e-06) [Z6 X11 Z12 X13]
+ (-2.8909678818779808e-06) [Z7 Y10 Z11 Y12]
+ (-2.8909678818779808e-06) [Z7 X10 Z11 X12]
+ (-2.1776646053192113e-06) [Z0 Y10 Z11 Y12]
+ (-2.1776646053192113e-06) [Z0 X10 Z11 X12]
+ (-2.1776646053192113e-06) [Z1 Y11 Z12 Y13]
+ (-2.1776646053192113e-06) [Z1 X11 Z12 X13]
+ (-1.8818501831735682e-06) [Y4 Z5 Y6 Z9]
+ (-1.8818501831735682e-06) [X4 Z5 X6 Z9]
+ (-1.8818501831735682e-06) [Y5 Z6 Y7 Z8]
+ (-1.8818501831735682e-06) [X5 Z6 X7 Z8]
+ (-1.8551201217817651e-06) [Z6 Y10 Z11 Y12]
+ (-1.8551201217817651e-06) [Z6 X10 Z11 X12]
+ (-1.8551201217817651e-06) [Z7 Y11 Z12 Y13]
+ (-1.8551201217817651e-06) [Z7 X11 Z12 X13]
+ (-1.8540608579209806e-06) [Y4 Z5 Y6 Z7]
+ (-1.8540608579209806e-06) [X4 Z5 X6 Z7]
+ (-1.8163031699078305e-06) [Z4 Y11 Z12 Y13]
+ (-1.8163031699078305e-06) [Z4 X11 Z12 X13]
+ (-1.8163031699078305e-06) [Z5 Y10 Z11 Y12]
+ (-1.8163031699078305e-06) [Z5 X10 Z11 X12]
+ (-1.69239782868596e-06) [Y4 Z5 Y6 Z10]
+ (-1.69239782868596e-06) [X4 Z5 X6 Z10]
+ (-1.69239782868596e-06) [Y5 Z6 Y7 Z11]
+ (-1.69239782868596e-06) [X5 Z6 X7 Z11]
+ (-1.614879414149115e-06) [Z0 Y11 Z12 Y13]
+ (-1.614879414149115e-06) [Z0 X11 Z12 X13]
+ (-1.614879414149115e-06) [Z1 Y10 Z11 Y12]
+ (-1.614879414149115e-06) [Z1 X10 Z11 X12]
+ (-1.5973171980038763e-06) [Z8 Y10 Z11 Y12]
+ (-1.5973171980038763e-06) [Z8 X10 Z11 X12]
+ (-1.5973171980038763e-06) [Z9 Y11 Z12 Y13]
+ (-1.5973171980038763e-06) [Z9 X11 Z12 X13]
+ (-1.4548424489957025e-06) [Y3 X4 X6 Y7]
+ (-1.4548424489957025e-06) [Y3 Y4 Y6 Y7]
+ (-1.4548424489957025e-06) [X3 X4 X6 X7]
+ (-1.4548424489957025e-06) [X3 Y4 Y6 X7]
+ (-1.3980449081038952e-06) [Y4 Z5 Y6 Z8]
+ (-1.3980449081038952e-06) [X4 Z5 X6 Z8]
+ (-1.3980449081038952e-06) [Y5 Z6 Y7 Z9]
+ (-1.3980449081038952e-06) [X5 Z6 X7 Z9]
+ (-1.195489009929967e-06) [Y2 Z3 Y4 Z7]
+ (-1.195489009929967e-06) [X2 Z3 X4 Z7]
+ (-1.195489009929967e-06) [Y3 Z4 Y5 Z6]
+ (-1.195489009929967e-06) [X3 Z4 X5 Z6]
+ (-1.190850808346385e-06) [Z0 Y3 Z4 Y5]
+ (-1.190850808346385e-06) [Z0 X3 Z4 X5]
+ (-1.190850808346385e-06) [Z1 Y2 Z3 Y4]
+ (-1.190850808346385e-06) [Z1 X2 Z3 X4]
+ (-1.1708301370199428e-06) [Z2 Y5 Z6 Y7]
+ (-1.1708301370199428e-06) [Z2 X5 Z6 X7]
+ (-1.1708301370199428e-06) [Z3 Y4 Z5 Y6]
+ (-1.1708301370199428e-06) [Z3 X4 Z5 X6]
+ (-1.0632283425650232e-06) [Z2 Y10 Z11 Y12]
+ (-1.0632283425650232e-06) [Z2 X10 Z11 X12]
+ (-1.0632283425650232e-06) [Z3 Y11 Z12 Y13]
+ (-1.0632283425650232e-06) [Z3 X11 Z12 X13]
+ (-1.0358477600962157e-06) [Y6 X7 X11 Y12]
+ (-1.0358477600962157e-06) [Y6 Y7 Y11 Y12]
+ (-1.0358477600962157e-06) [X6 X7 X11 X12]
+ (-1.0358477600962157e-06) [X6 Y7 Y11 X12]
+ (-9.50924975153256e-07) [Z2 Y4 Z5 Y6]
+ (-9.50924975153256e-07) [Z2 X4 Z5 X6]
+ (-9.50924975153256e-07) [Z3 Y5 Z6 Y7]
+ (-9.50924975153256e-07) [Z3 X5 Z6 X7]
+ (-9.344557777862528e-07) [Z8 Y11 Z12 Y13]
+ (-9.344557777862528e-07) [Z8 X11 Z12 X13]
+ (-9.344557777862528e-07) [Z9 Y10 Z11 Y12]
+ (-9.344557777862528e-07) [Z9 X10 Z11 X12]
+ (-8.337746754497725e-07) [Z0 Y2 Z3 Y4]
+ (-8.337746754497725e-07) [Z0 X2 Z3 X4]
+ (-8.337746754497725e-07) [Z1 Y3 Z4 Y5]
+ (-8.337746754497725e-07) [Z1 X3 Z4 X5]
+ (-7.956895372178396e-07) [Y3 X4 X8 Y9]
+ (-7.956895372178396e-07) [Y3 Y4 Y8 Y9]
+ (-7.956895372178396e-07) [X3 X4 X8 X9]
+ (-7.956895372178396e-07) [X3 Y4 Y8 X9]
+ (-7.764994118585462e-07) [Y2 Z3 Y4 Z5]
+ (-7.764994118585462e-07) [X2 Z3 X4 Z5]
+ (-5.929765815797967e-07) [Z4 Y5 Z6 Y7]
+ (-5.929765815797967e-07) [Z4 X5 Z6 X7]
+ (-5.770052995087671e-07) [Y2 Z3 Y4 Z9]
+ (-5.770052995087671e-07) [X2 Z3 X4 Z9]
+ (-5.770052995087671e-07) [Y3 Z4 Y5 Z8]
+ (-5.770052995087671e-07) [X3 Z4 X5 Z8]
+ (-5.471647745008922e-07) [Y1 Y2 X11 X12]
+ (-5.471647745008922e-07) [X1 X2 Y11 Y12]
+ (-4.838052750696728e-07) [Y5 X6 X8 Y9]
+ (-4.838052750696728e-07) [Y5 Y6 Y8 Y9]
+ (-4.838052750696728e-07) [X5 X6 X8 X9]
+ (-4.838052750696728e-07) [X5 Y6 Y8 X9]
+ (-3.570761328966124e-07) [Y0 X1 X3 Y4]
+ (-3.570761328966124e-07) [Y0 Y1 Y3 Y4]
+ (-3.570761328966124e-07) [X0 X1 X3 X4]
+ (-3.570761328966124e-07) [X0 Y1 Y3 X4]
+ (-2.447323128699698e-07) [Y0 X1 X5 Y6]
+ (-2.447323128699698e-07) [Y0 Y1 Y5 Y6]
+ (-2.447323128699698e-07) [X0 X1 X5 X6]
+ (-2.447323128699698e-07) [X0 Y1 Y5 X6]
+ (-2.199051618666867e-07) [Y2 X3 X5 Y6]
+ (-2.199051618666867e-07) [Y2 Y3 Y5 Y6]
+ (-2.199051618666867e-07) [X2 X3 X5 X6]
+ (-2.199051618666867e-07) [X2 Y3 Y5 X6]
+ (-1.933241276972481e-07) [Y1 X2 X3 Y4]
+ (-1.933241276972481e-07) [X1 Y2 Y3 X4]
+ (-1.2919694862091841e-07) [Y1 Z2 Z3 Y5]
+ (-1.2919694862091841e-07) [X1 Z2 Z3 X5]
+ (1.7379332622872893e-07) [Y0 Z1 Z3 Y4]
+ (1.7379332622872893e-07) [X0 Z1 Z3 X4]
+ (1.7379332622872893e-07) [Y1 Z2 Z4 Y5]
+ (1.7379332622872893e-07) [X1 Z2 Z4 X5]
+ (1.933241276972481e-07) [Y1 Y2 X3 X4]
+ (1.933241276972481e-07) [X1 X2 Y3 Y4]
+ (2.1868423770907252e-07) [Y2 Z3 Y4 Z8]
+ (2.1868423770907252e-07) [X2 Z3 X4 Z8]
+ (2.1868423770907252e-07) [Y3 Z4 Y5 Z9]
+ (2.1868423770907252e-07) [X3 Z4 X5 Z9]
+ (2.5935343906573556e-07) [Y2 Z3 Y4 Z6]
+ (2.5935343906573556e-07) [X2 Z3 X4 Z6]
+ (2.5935343906573556e-07) [Y3 Z4 Y5 Z7]
+ (2.5935343906573556e-07) [X3 Z4 X5 Z7]
+ (3.6060718676970107e-07) [Y0 Z1 Z2 Y4]
+ (3.6060718676970107e-07) [X0 Z1 Z2 X4]
+ (3.6060718676970107e-07) [Y1 Z3 Z4 Y5]
+ (3.6060718676970107e-07) [X1 Z3 Z4 X5]
+ (5.471647745008922e-07) [Y1 X2 X11 Y12]
+ (5.471647745008922e-07) [X1 Y2 Y11 X12]
+ (5.627851911700963e-07) [Y0 X1 X11 Y12]
+ (5.627851911700963e-07) [Y0 Y1 Y11 Y12]
+ (5.627851911700963e-07) [X0 X1 X11 X12]
+ (5.627851911700963e-07) [X0 Y1 Y11 X12]
+ (6.628614202176238e-07) [Y8 X9 X11 Y12]
+ (6.628614202176238e-07) [Y8 Y9 Y11 Y12]
+ (6.628614202176238e-07) [X8 X9 X11 X12]
+ (6.628614202176238e-07) [X8 Y9 Y11 X12]
+ (1.1094407590483146e-06) [Z2 Y11 Z12 Y13]
+ (1.1094407590483146e-06) [Z2 X11 Z12 X13]
+ (1.1094407590483146e-06) [Z3 Y10 Z11 Y12]
+ (1.1094407590483146e-06) [Z3 X10 Z11 X12]
+ (1.602116740510007e-06) [Z2 Y3 Z4 Y5]
+ (1.602116740510007e-06) [Z2 X3 Z4 X5]
+ (1.8782101249034516e-06) [Z4 Y10 Z11 Y12]
+ (1.8782101249034516e-06) [Z4 X10 Z11 X12]
+ (1.8782101249034516e-06) [Z5 Y11 Z12 Y13]
+ (1.8782101249034516e-06) [Z5 X11 Z12 X13]
+ (2.172669101613338e-06) [Y2 X3 X11 Y12]
+ (2.172669101613338e-06) [Y2 Y3 Y11 Y12]
+ (2.172669101613338e-06) [X2 X3 X11 X12]
+ (2.172669101613338e-06) [X2 Y3 Y11 X12]
+ (3.1174479459333394e-06) [Y0 Z2 Z3 Y4]
+ (3.1174479459333394e-06) [X0 Z2 Z3 X4]
+ (3.5390541848134163e-06) [Y2 Z3 Y4 Z12]
+ (3.5390541848134163e-06) [X2 Z3 X4 Z12]
+ (3.5390541848134163e-06) [Y3 Z4 Y5 Z13]
+ (3.5390541848134163e-06) [X3 Z4 X5 Z13]
+ (4.2819138851835545e-06) [Y4 Z5 Y6 Z11]
+ (4.2819138851835545e-06) [X4 Z5 X6 Z11]
+ (4.2819138851835545e-06) [Y5 Z6 Y7 Z10]
+ (4.2819138851835545e-06) [X5 Z6 X7 Z10]
+ (5.2758831224879995e-06) [Y3 X4 X12 Y13]
+ (5.2758831224879995e-06) [Y3 Y4 Y12 Y13]
+ (5.2758831224879995e-06) [X3 X4 X12 X13]
+ (5.2758831224879995e-06) [X3 Y4 Y12 X13]
+ (5.974311713869513e-06) [Y5 X6 X10 Y11]
+ (5.974311713869513e-06) [Y5 Y6 Y10 Y11]
+ (5.974311713869513e-06) [X5 X6 X10 X11]
+ (5.974311713869513e-06) [X5 Y6 Y10 X11]
+ (7.954413176813625e-06) [Y10 Z11 Y12 Z13]
+ (7.954413176813625e-06) [X10 Z11 X12 Z13]
+ (8.814937307301416e-06) [Y2 Z3 Y4 Z13]
+ (8.814937307301416e-06) [X2 Z3 X4 Z13]
+ (8.814937307301416e-06) [Y3 Z4 Y5 Z12]
+ (8.814937307301416e-06) [X3 Z4 X5 Z12]
+ (0.00029219862611100774) [Y7 X8 X9 Y10]
+ (0.00029219862611100774) [X7 Y8 Y9 X10]
+ (0.0004956762314915127) [Y2 Z4 Z5 Y6]
+ (0.0004956762314915127) [X2 Z4 Z5 X6]
+ (0.0011059037691897099) [Y0 Z1 Y2 Z5]
+ (0.0011059037691897099) [X0 Z1 X2 Z5]
+ (0.0011059037691897099) [Y1 Z2 Y3 Z4]
+ (0.0011059037691897099) [X1 Z2 X3 Z4]
+ (0.001663879878490793) [Y2 Z3 Z4 Y6]
+ (0.001663879878490793) [X2 Z3 Z4 X6]
+ (0.001663879878490793) [Y3 Z5 Z6 Y7]
+ (0.001663879878490793) [X3 Z5 Z6 X7]
+ (0.0017560707018412667) [Y0 Z1 Y2 Z11]
+ (0.0017560707018412667) [X0 Z1 X2 Z11]
+ (0.0017560707018412667) [Y1 Z2 Y3 Z10]
+ (0.0017560707018412667) [X1 Z2 X3 Z10]
+ (0.0023262306231581044) [Y0 Z1 Y2 Z13]
+ (0.0023262306231581044) [X0 Z1 X2 Z13]
+ (0.0023262306231581044) [Y1 Z2 Y3 Z12]
+ (0.0023262306231581044) [X1 Z2 X3 Z12]
+ (0.002745836470186813) [Y0 X1 X4 Y5]
+ (0.002745836470186813) [X0 Y1 Y4 X5]
+ (0.002929768674751078) [Y0 Z1 Y2 Z9]
+ (0.002929768674751078) [X0 Z1 X2 Z9]
+ (0.002929768674751078) [Y1 Z2 Y3 Z8]
+ (0.002929768674751078) [X1 Z2 X3 Z8]
+ (0.003276971931231706) [Y0 Z1 Y2 Z3]
+ (0.003276971931231706) [X0 Z1 X2 Z3]
+ (0.003347617530666186) [Y0 Z1 Y2 Z7]
+ (0.003347617530666186) [X0 Z1 X2 Z7]
+ (0.003347617530666186) [Y1 Z2 Y3 Z6]
+ (0.003347617530666186) [X1 Z2 X3 Z6]
+ (0.003555290195504308) [Y0 Z1 Y2 Z10]
+ (0.003555290195504308) [X0 Z1 X2 Z10]
+ (0.003555290195504308) [Y1 Z2 Y3 Z11]
+ (0.003555290195504308) [X1 Z2 X3 Z11]
+ (0.005143391768825165) [Y3 Y4 X5 X6]
+ (0.005143391768825165) [X3 X4 Y5 Y6]
+ (0.0052837764884029505) [Y0 X1 X12 Y13]
+ (0.0052837764884029505) [X0 Y1 Y12 X13]
+ (0.00553075921863157) [Y0 Z1 Y2 Z4]
+ (0.00553075921863157) [X0 Z1 X2 Z4]
+ (0.00553075921863157) [Y1 Z2 Y3 Z5]
+ (0.00553075921863157) [X1 Z2 X3 Z5]
+ (0.006087822480561855) [Y8 X9 X12 Y13]
+ (0.006087822480561855) [X8 Y9 Y12 X13]
+ (0.006509361201177232) [Y0 X1 X8 Y9]
+ (0.006509361201177232) [X0 Y1 Y8 X9]
+ (0.006888194352970519) [Y0 X1 X6 Y7]
+ (0.006888194352970519) [X0 Y1 Y6 X7]
+ (0.006901238249797309) [Y0 Z1 Y2 Z12]
+ (0.006901238249797309) [X0 Z1 X2 Z12]
+ (0.006901238249797309) [Y1 Z2 Y3 Z13]
+ (0.006901238249797309) [X1 Z2 X3 Z13]
+ (0.007156934919856952) [Y4 X5 X8 Y9]
+ (0.007156934919856952) [X4 Y5 Y8 X9]
+ (0.007731425250775248) [Y0 X1 X10 Y11]
+ (0.007731425250775248) [X0 Y1 Y10 X11]
+ (0.008032520918821406) [Y0 Z1 Y2 Z6]
+ (0.008032520918821406) [X0 Z1 X2 Z6]
+ (0.008032520918821406) [Y1 Z2 Y3 Z7]
+ (0.008032520918821406) [X1 Z2 X3 Z7]
+ (0.011055020596132108) [Y0 Z1 Y2 Z8]
+ (0.011055020596132108) [X0 Z1 X2 Z8]
+ (0.011055020596132108) [Y1 Z2 Y3 Z9]
+ (0.011055020596132108) [X1 Z2 X3 Z9]
+ (0.011285190200840936) [Y5 Y6 X11 X12]
+ (0.011285190200840936) [X5 X6 Y11 Y12]
+ (0.011307274008848274) [Y7 Z8 Z9 Y11]
+ (0.011307274008848274) [X7 Z8 Z9 X11]
+ (0.011982389010247986) [Y4 X5 X6 Y7]
+ (0.011982389010247986) [X4 Y5 Y6 X7]
+ (0.013873381748426034) [Y6 X7 X8 Y9]
+ (0.013873381748426034) [X6 Y7 Y8 X9]
+ (0.014583648907612708) [Y0 X1 X2 Y3]
+ (0.014583648907612708) [X0 Y1 Y2 X3]
+ (0.015577208063976469) [Y2 X3 X12 Y13]
+ (0.015577208063976469) [X2 Y3 Y12 X13]
+ (0.01736611899465139) [Y6 X7 X12 Y13]
+ (0.01736611899465139) [X6 Y7 Y12 X13]
+ (0.01768006795248148) [Y4 X5 X10 Y11]
+ (0.01768006795248148) [X4 Y5 Y10 X11]
+ (0.017825140995786585) [Y6 X7 X10 Y11]
+ (0.017825140995786585) [X6 Y7 Y10 X11]
+ (0.019028242443847203) [Y3 X4 X11 Y12]
+ (0.019028242443847203) [X3 Y4 Y11 X12]
+ (0.025384657508457288) [Y2 X3 X10 Y11]
+ (0.025384657508457288) [X2 Y3 Y10 X11]
+ (0.028685183716105876) [Y10 X11 X12 Y13]
+ (0.028685183716105876) [X10 Y11 Y12 X13]
+ (0.029812424517345927) [Y6 Z7 Z8 Y10]
+ (0.029812424517345927) [X6 Z7 Z8 X10]
+ (0.029812424517345927) [Y7 Z9 Z10 Y11]
+ (0.029812424517345927) [X7 Z9 Z10 X11]
+ (0.03010462314345693) [Y6 Z7 Z9 Y10]
+ (0.03010462314345693) [X6 Z7 Z9 X10]
+ (0.03010462314345693) [Y7 Z8 Z10 Y11]
+ (0.03010462314345693) [X7 Z8 Z10 X11]
+ (0.030787505389143977) [Y6 Z8 Z9 Y10]
+ (0.030787505389143977) [X6 Z8 Z9 X10]
+ (0.031143817988967235) [Y2 X3 X6 Y7]
+ (0.031143817988967235) [X2 Y3 Y6 X7]
+ (0.03583956795335344) [Y2 X3 X4 Y5]
+ (0.03583956795335344) [X2 Y3 Y4 X5]
+ (0.03619412355904267) [Y2 X3 X8 Y9]
+ (0.03619412355904267) [X2 Y3 Y8 X9]
+ (0.03831467029480389) [Y4 X5 X12 Y13]
+ (0.03831467029480389) [X4 Y5 Y12 X13]
+ (0.10433064780651419) [Z0 Y1 Z2 Y3]
+ (0.10433064780651419) [Z0 X1 Z2 X3]
+ (-0.12133276911042266) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042266) [X2 Z3 Z4 Z5 X6]
+ (-0.12133276911042264) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133276911042264) [X3 Z4 Z5 Z6 X7]
+ (3.2020768800148152e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.2020768800148152e-06) [X1 Z2 Z3 Z4 X5]
+ (3.202076880014816e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.202076880014816e-06) [X0 Z1 Z2 Z3 X4]
+ (0.2284810656491897) [Y6 Z7 Z8 Z9 Y10]
+ (0.2284810656491897) [X6 Z7 Z8 Z9 X10]
+ (0.2284810656491897) [Y7 Z8 Z9 Z10 Y11]
+ (0.2284810656491897) [X7 Z8 Z9 Z10 X11]
+ (-0.03276765782329046) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.03276765782329046) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.03276765782329046) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.03276765782329046) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.02711503684527315) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.02711503684527315) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.02711503684527315) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.02711503684527315) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.025996177598021065) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.025996177598021065) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.017561202409646138) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.017561202409646138) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.017561202409646138) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.017561202409646138) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.014564531231173012) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564531231173012) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564531231173012) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564531231173012) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.012215040997614012) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.012215040997614012) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.012215040997614012) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.012215040997614012) [X4 Z5 X6 X11 Z12 X13]
+ (-0.012215040997614012) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.012215040997614012) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.012215040997614012) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.012215040997614012) [X5 Z6 X7 X10 Z11 X12]
+ (-0.01175601341981919) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.01175601341981919) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.01175601341981919) [X3 Z4 Z5 X6 X8 X9]
+ (-0.01175601341981919) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.008764827575688727) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.008764827575688727) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.008764827575688727) [X2 Z3 Z4 X5 X11 X12]
+ (-0.008764827575688727) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.008764827575688727) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.008764827575688727) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.008764827575688727) [X3 X4 X10 Z11 Z12 X13]
+ (-0.008764827575688727) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.007306759928832979) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306759928832979) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306759928832979) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306759928832979) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005805188989826945) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.005805188989826945) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.005805188989826945) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.005805188989826945) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.0056526209780173205) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.0056526209780173205) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.0056526209780173205) [X0 X1 X3 Z4 Z5 X6]
+ (-0.0056526209780173205) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.005143391768825165) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.005143391768825165) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.005143391768825165) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.005143391768825165) [X2 Z3 X4 X5 Z6 X7]
+ (-0.0046849033881552204) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.0046849033881552204) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668620318776283) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668620318776283) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.004575007626639205) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.004575007626639205) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424855449441859) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424855449441859) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.004158797381840045) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.004158797381840045) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.004158797381840045) [X3 Z4 Z5 X6 X12 X13]
+ (-0.004158797381840045) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.003493790359890115) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.003493790359890115) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.003493790359890115) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.003493790359890115) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.002779026799025545) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.002779026799025545) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.0022939566113524567) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.0022939566113524567) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.001799219493663041) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.001799219493663041) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278753941369425) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.0017278753941369425) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298507967730775) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298507967730775) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298507967730775) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298507967730775) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533856254125381) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533856254125381) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008145313270956521) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008145313270956521) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008145313270956521) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008145313270956521) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.735036880586365e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735036880586365e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735036880586365e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735036880586365e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.774817865294386e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.774817865294386e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.774817865294386e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.774817865294386e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.518362216274245e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.518362216274245e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.518362216274245e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.518362216274245e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.444344676550153e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.444344676550153e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.444344676550153e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.444344676550153e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.52437384917388e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.52437384917388e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.52437384917388e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.52437384917388e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.2900284338437634e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.2900284338437634e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.2900284338437634e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.2900284338437634e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.974311713869514e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.974311713869514e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.2758831224879995e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.2758831224879995e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.6430510688976775e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.6430510688976775e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.556569218622809e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.556569218622809e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.253224225912431e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.253224225912431e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.7696594524638293e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.7696594524638293e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.6945132948112823e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.6945132948112823e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.6102971310630136e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.6102971310630136e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.6102971310630136e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.6102971310630136e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.3131455003510816e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.3131455003510816e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.2774831960070846e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.2774831960070846e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.2774831960070846e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.2774831960070846e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.211228348822798e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.211228348822798e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.211228348822798e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.211228348822798e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.1513463114486456e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.1513463114486456e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.0882507116055013e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.0882507116055013e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.172669101613338e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.172669101613338e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.4548424489957025e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.4548424489957025e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.330473188744232e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.330473188744232e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2283337824304815e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2283337824304815e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.0358477600962157e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.0358477600962157e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956895372178396e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956895372178396e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733197743228761e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733197743228761e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733197743228761e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733197743228761e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.628614202176238e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.628614202176238e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.55628191498708e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.55628191498708e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.55628191498708e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.55628191498708e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.41829157499296e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.41829157499296e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.41829157499296e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.41829157499296e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.927453083541542e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927453083541542e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927453083541542e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927453083541542e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627851911700963e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627851911700963e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660625025499e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660625025499e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660625025499e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660625025499e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660625025499e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660625025499e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660625025499e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660625025499e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.838052750696728e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.838052750696728e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.570761328966125e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.570761328966125e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.328139350559289e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.328139350559289e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.086826565106642e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.086826565106642e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.086826565106642e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.086826565106642e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.447323128699698e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.447323128699698e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.3713289477973268e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.3713289477973268e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.3713289477973268e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.3713289477973268e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.1990516186668664e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.1990516186668664e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.933241276972481e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.933241276972481e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.933241276972481e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.933241276972481e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.8394209153854212e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.8394209153854212e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.8394209153854212e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.8394209153854212e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.5510539175621003e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.5510539175621003e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.5510539175621003e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.5510539175621003e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.380778147994234e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.380778147994234e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.380778147994234e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.380778147994234e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.380778147994234e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.380778147994234e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.380778147994234e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.380778147994234e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.380778147994234e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.380778147994234e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.380778147994234e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.380778147994234e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.2919694862091844e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.2919694862091844e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.1076325599728088e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.1076325599728088e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.1076325599728088e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.1076325599728088e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.1076325599728088e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.1076325599728088e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.1076325599728088e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.1076325599728088e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.057446596872177e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057446596872177e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057446596872177e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057446596872177e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649310133828995e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649310133828995e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649310133828995e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649310133828995e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.8394209153854215e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.8394209153854215e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.8394209153854215e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.8394209153854215e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.1990516186668664e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.1990516186668664e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.447323128699698e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.447323128699698e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.2362599611802263e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.2362599611802263e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.2362599611802263e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.2362599611802263e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.328139350559289e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.328139350559289e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.570761328966125e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.570761328966125e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.838052750696728e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.838052750696728e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.627851911700963e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627851911700963e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.628614202176238e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.628614202176238e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956895372178396e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956895372178396e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.306536652500481e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.306536652500481e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.306536652500481e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.306536652500481e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.0358477600962157e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.0358477600962157e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2283337824304815e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2283337824304815e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.2393363217607123e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393363217607123e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393363217607123e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393363217607123e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.330473188744232e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.330473188744232e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.4548424489957025e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.4548424489957025e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.172669101613338e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.172669101613338e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.0882507116055013e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.0882507116055013e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.1174479459333394e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.1174479459333394e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.1513463114486456e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.1513463114486456e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.3131455003510816e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.3131455003510816e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.3343312897612576e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.3343312897612576e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.6945132948112823e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.6945132948112823e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.183932559620936e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.183932559620936e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.556569218622809e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.556569218622809e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.6430510688976775e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.6430510688976775e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.2758831224879995e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.2758831224879995e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.974311713869514e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.974311713869514e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.00029219862611100774) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.00029219862611100774) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.00029219862611100774) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.00029219862611100774) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004956762314915127) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004956762314915127) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.000665007021949931) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.000665007021949931) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.000665007021949931) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.000665007021949931) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533856254125381) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533856254125381) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.0016095313817213546) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.0016095313817213546) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.0016095313817213546) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.0016095313817213546) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.00166760418114401) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.00166760418114401) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.00166760418114401) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.00166760418114401) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.0017278753941369425) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.0017278753941369425) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.001799219493663041) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.001799219493663041) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.0022939566113524567) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.0022939566113524567) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.0024629170071338927) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.0024629170071338927) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.0024629170071338927) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.0024629170071338927) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.003961560792496467) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.003961560792496467) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.003961560792496467) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.003961560792496467) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.004424855449441859) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424855449441859) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004575007626639205) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.004575007626639205) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668620318776283) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668620318776283) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.0046849033881552204) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.0046849033881552204) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.0053248352342217115) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.0053248352342217115) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.0053248352342217115) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.0053248352342217115) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.005368659358109612) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368659358109612) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368659358109612) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368659358109612) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.007960880725921583) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.007960880725921583) [X4 Z5 X6 X10 Z11 X12]
+ (0.007960880725921583) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.007960880725921583) [X5 Z6 X7 X11 Z12 X13]
+ (0.008890731522694659) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.008890731522694659) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.008890731522694659) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.008890731522694659) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.010263414868158474) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.010263414868158474) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.010263414868158474) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.010263414868158474) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.010540425907671569) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.010540425907671569) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.010540425907671569) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.010540425907671569) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010960074940542693) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010960074940542693) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010960074940542693) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010960074940542693) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307274008848274) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307274008848274) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.014411099430130938) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411099430130938) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411099430130938) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411099430130938) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.015225630757226591) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.015225630757226591) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.015225630757226591) [X3 Z4 Z5 X6 X10 X11]
+ (0.015225630757226591) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.01826683486937567) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.01826683486937567) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.01826683486937567) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.01826683486937567) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.019020423173040063) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.019020423173040063) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.019020423173040063) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.019020423173040063) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.020175921723535595) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.020175921723535595) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.020175921723535595) [X4 Z5 Z6 X7 X11 X12]
+ (0.020175921723535595) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.020175921723535595) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.020175921723535595) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.020175921723535595) [X5 X6 X10 Z11 Z12 X13]
+ (0.020175921723535595) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.024353077678068914) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.024353077678068914) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.024353077678068914) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.024353077678068914) [X2 Z3 X4 X11 Z12 X13]
+ (0.024353077678068914) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.024353077678068914) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.024353077678068914) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.024353077678068914) [X3 Z4 X5 X10 Z11 X12]
+ (0.024389082531149672) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.024389082531149672) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.024389082531149672) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.024389082531149672) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.02510495713884458) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.02510495713884458) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.02510495713884458) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.02510495713884458) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.030787505389143977) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.030787505389143977) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.045879470781297595) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.045879470781297595) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.05600733087780792) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.05600733087780792) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.05600733087780792) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.05600733087780792) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661378) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.05608468124661378) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661378) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.05608468124661378) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.631277928985137e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631277928985137e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.631277928985136e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631277928985136e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.5950860073664348e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.5950860073664348e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.595086007366433e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.595086007366433e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.04274327701378123) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04274327701378123) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (0.04274327701378124) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.04274327701378124) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (-0.04764261217638316) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.04764261217638316) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.04764261217638316) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.04764261217638316) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.04171881383982181) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.04171881383982181) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.04171881383982181) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.04171881383982181) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.039564416322893266) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.039564416322893266) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.039564416322893266) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.039564416322893266) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.039359168022052915) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.039359168022052915) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.039359168022052915) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.039359168022052915) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.039318051947197556) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.039318051947197556) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.039318051947197556) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.039318051947197556) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.035608378988312386) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.035608378988312386) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.029903789512624724) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.029903789512624724) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.029903789512624724) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.029903789512624724) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.028730779551905422) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.028730779551905422) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.028730779551905422) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.028730779551905422) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.02563723829602687) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.02563723829602687) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.02563723829602687) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.02563723829602687) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.024755463292890828) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.024755463292890828) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.024755463292890828) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.024755463292890828) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.024282117354692916) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.024282117354692916) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.023145130929529065) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.023145130929529065) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.022528440196013105) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.022528440196013105) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02143381072160076) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.02143381072160076) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.02143381072160076) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.02143381072160076) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251606) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.019257505095251606) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019028242443847203) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.019028242443847203) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.018889030304942846) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.018889030304942846) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.018889030304942846) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.018889030304942846) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.016024603689179517) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.016024603689179517) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.015225630757226591) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225630757226591) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.014603704729162087) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.014603704729162087) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564531231173012) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564531231173012) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.011756013419819192) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.011756013419819192) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.011285190200840936) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.011285190200840936) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.009841749246962574) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009841749246962574) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.009612634606847352) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612634606847352) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612634606847352) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612634606847352) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.008469978791023968) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.008469978791023968) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.007306759928832979) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.007306759928832979) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0056526209780173205) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.0056526209780173205) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368659358109612) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368659358109612) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.004158797381840045) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.004158797381840045) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0033566705638328666) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.0033566705638328666) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.0033566705638328666) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.0033566705638328666) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.003267513854423534) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.003267513854423534) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.003267513854423534) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.003267513854423534) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.002779026799025545) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.002779026799025545) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.002686040977806595) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.002686040977806595) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.002686040977806595) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.002686040977806595) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.0022939566113524567) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524567) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.0022939566113524567) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524567) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.0009581655836696521) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.0009581655836696521) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0009581655836696521) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.0009581655836696521) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.0009581655836696521) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.0009581655836696521) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0009581655836696521) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.0009581655836696521) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.00024636437569571044) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00024636437569571044) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303549693) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.00013840177303549693) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.00013840177303549693) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.00013840177303549693) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.735036880586365e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735036880586365e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585307069906e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585307069906e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.6103585307069906e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.6103585307069906e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808796602463e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.5316808796602463e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808796602463e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5316808796602463e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.8061027759657e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.8061027759657e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.8061027759657e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.8061027759657e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.08979946804203e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.08979946804203e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.08979946804203e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.08979946804203e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.652209670374801e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.652209670374801e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.652209670374801e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.652209670374801e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851834839274e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.481851834839274e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851834839274e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.481851834839274e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.071480736833476e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.071480736833476e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.071480736833476e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.071480736833476e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.734622039132222e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.734622039132222e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.734622039132222e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.734622039132222e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.728843147572854e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.728843147572854e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.728843147572854e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.728843147572854e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.253224225912431e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.253224225912431e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.7696594524638293e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.7696594524638293e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.544395429556236e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.544395429556236e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.544395429556236e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.544395429556236e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.544395429556236e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.544395429556236e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.544395429556236e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.544395429556236e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.3609563204691774e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563204691774e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563204691774e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.3609563204691774e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.103215604984957e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.103215604984957e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.103215604984957e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.103215604984957e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.011122098660476e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.011122098660476e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.011122098660476e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.011122098660476e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.942946836800386e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.942946836800386e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.942946836800386e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.942946836800386e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174774270236e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.6541174774270236e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174774270236e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.6541174774270236e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.5224930677506355e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224930677506355e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224930677506355e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224930677506355e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224930677506355e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930677506355e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930677506355e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224930677506355e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.2283337824304817e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337824304817e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.2283337824304817e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337824304817e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.988770288599885e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988770288599885e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988770288599885e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988770288599885e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.867765104674475e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104674475e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104674475e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.867765104674475e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.189990976028723e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.189990976028723e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246207482779e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246207482779e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.471647745008922e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.471647745008922e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.561447179507352e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.561447179507352e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.561447179507352e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.561447179507352e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.5233896784461345e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.5233896784461345e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.4273231090925325e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.4273231090925325e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.4273231090925325e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.4273231090925325e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.328139350559289e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350559289e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.328139350559289e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350559289e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.086826565106642e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.086826565106642e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.8882935937336273e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935937336273e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935937336273e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.8882935937336273e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.3713289477973268e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.3713289477973268e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.8394209153854215e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.8394209153854215e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.057446596872177e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057446596872177e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.537178094521377e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.537178094521377e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.537178094521377e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.537178094521377e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.057446596872177e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057446596872177e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.209350632448028e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350632448028e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350632448028e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.209350632448028e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783553552703e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783553552703e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783553552703e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783553552703e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.8394209153854215e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.8394209153854215e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.3713289477973268e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.3713289477973268e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.086826565106642e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.086826565106642e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.5233896784461345e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.5233896784461345e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.471647745008922e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.471647745008922e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246207482779e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246207482779e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189990976028723e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.189990976028723e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.330473188744232e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.330473188744232e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.330473188744232e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.330473188744232e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.62885324369801e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.62885324369801e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.62885324369801e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.62885324369801e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.6893489516247585e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.6893489516247585e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.6893489516247585e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.6893489516247585e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.745518400696248e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.745518400696248e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.745518400696248e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.745518400696248e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.745518400696248e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.745518400696248e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.745518400696248e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.745518400696248e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.2118420193753944e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420193753944e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.2118420193753944e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420193753944e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.2118420193753944e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420193753944e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.2118420193753944e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420193753944e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.3131455003510816e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.3131455003510816e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.3131455003510816e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.3131455003510816e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.3343312897612576e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.3343312897612576e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.183932559620936e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.183932559620936e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.735036880586365e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735036880586365e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00024636437569571044) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00024636437569571044) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0004458535128841031) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.0004458535128841031) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.0004458535128841031) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.0004458535128841031) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940221543005588) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005588) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005588) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005588) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005588) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005588) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940221543005588) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005588) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533856254125381) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533856254125381) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533856254125381) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533856254125381) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0010435246534907334) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435246534907334) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435246534907334) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435246534907334) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.0012803060973496407) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.0012803060973496407) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.0012803060973496407) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.0012803060973496407) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.0013038004788126951) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.0013038004788126951) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.0013038004788126951) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.0013038004788126951) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.0022619660624823472) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.0022619660624823472) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.0022619660624823472) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.0022619660624823472) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.0022619660624823472) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.0022619660624823472) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.0022619660624823472) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.0022619660624823472) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.003989841456619289) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.003989841456619289) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.003989841456619289) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.003989841456619289) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.004158797381840045) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.004158797381840045) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.004311038507914268) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.004311038507914268) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.004311038507914268) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.004311038507914268) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.004636976661182506) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.004636976661182506) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.004636976661182506) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.004636976661182506) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.005114473831660387) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.005114473831660387) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.005114473831660387) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.005114473831660387) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.005114473831660387) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660387) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660387) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005114473831660387) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005241535382803838) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.005241535382803838) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.005241535382803838) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.005241535382803838) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.005262642473076843) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262642473076843) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262642473076843) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262642473076843) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368659358109612) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368659358109612) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005379937155839336) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.005379937155839336) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.005379937155839336) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.005379937155839336) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.0056526209780173205) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.0056526209780173205) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.005708495985960945) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708495985960945) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708495985960945) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708495985960945) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.007306759928832979) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.007306759928832979) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469978791023968) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.008469978791023968) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.009841749246962574) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.009841749246962574) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.011285190200840936) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.011285190200840936) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.011756013419819192) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.011756013419819192) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.014564531231173012) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564531231173012) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603704729162087) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.014603704729162087) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.015225630757226591) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225630757226591) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.016024603689179517) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.016024603689179517) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.019028242443847203) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.019028242443847203) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.019257505095251606) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.019257505095251606) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.045879470781297595) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.045879470781297595) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.3693708936615621) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.3693708936615621) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.369370893661562) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.369370893661562) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.2816425776702273) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.2816425776702273) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.28164257767022716) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.28164257767022716) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.09065144207036477) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.09065144207036477) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.09065144207036477) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.09065144207036477) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863625) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684737589863625) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863625) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684737589863625) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635021950634978) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635021950634978) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635021950634978) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635021950634978) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099213996) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.06752385099213996) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099213996) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.06752385099213996) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.035608378988312386) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.035608378988312386) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03490334337366192) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03490334337366192) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03490334337366192) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03490334337366192) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883830054) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.024591860883830054) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883830054) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.024591860883830054) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.024282117354692916) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.024282117354692916) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.023145130929529068) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.023145130929529068) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.022528440196013105) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.022528440196013105) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019538050311314618) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.019538050311314618) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.019538050311314618) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.019538050311314618) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.017091553155898765) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.017091553155898765) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.017091553155898765) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.017091553155898765) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.016024603689179517) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.016024603689179517) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.016024603689179517) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.016024603689179517) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.010311482489831865) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831865) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831865) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311482489831865) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.009841749246962574) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962574) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.009841749246962574) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962574) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209815) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209815) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209815) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008826368514209815) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008541996625454816) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454816) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454816) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454816) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454816) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454816) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454816) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008541996625454816) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008469978791023968) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023968) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.008469978791023968) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023968) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.004668620318776283) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668620318776283) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.003876470899336938) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.003876470899336938) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.0038040661717285394) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0038040661717285394) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0038040661717285394) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0038040661717285394) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178878) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178878) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003356670563832866) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.003356670563832866) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.0032675138544235333) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.0032675138544235333) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0021413612231016245) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.0021413612231016245) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.0017278753941369425) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.0017278753941369425) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.0016407548553124302) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0016407548553124302) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214169408) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0014528843214169408) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214169408) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0014528843214169408) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870896771024445) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007870896771024445) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192743499487698) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192743499487698) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.00019400857029757903) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00019400857029757903) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303549693) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.00013840177303549693) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.14162522115832e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.14162522115832e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.14162522115832e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.14162522115832e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.071480736833477e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.071480736833477e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.1513463114486456e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.1513463114486456e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0882507116055013e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.0882507116055013e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.98851170641361e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.98851170641361e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.874299071583031e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.874299071583031e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.3609563204691774e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.3609563204691774e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.3002946563997167e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3002946563997167e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.1468376508632083e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.1468376508632083e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.1468376508632083e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.1468376508632083e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.352332103919494e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.352332103919494e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.352332103919494e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.352332103919494e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091637199926667e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637199926667e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637199926667e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637199926667e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637199926667e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637199926667e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637199926667e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091637199926667e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.07430598671377e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.07430598671377e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.07430598671377e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.07430598671377e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128987151553e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128987151553e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128987151553e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128987151553e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867765104674474e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.867765104674474e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560692465641761e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692465641761e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692465641761e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692465641761e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692465641761e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692465641761e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692465641761e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560692465641761e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.997018422638844e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.997018422638844e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.997018422638844e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.997018422638844e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.997018422638844e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.997018422638844e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.997018422638844e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.997018422638844e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.56824752148053e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.56824752148053e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.56824752148053e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.56824752148053e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393087054176e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393087054176e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393087054176e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393087054176e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393087054176e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393087054176e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.3767393087054176e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393087054176e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.8882935937336273e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.8882935937336273e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.686381547083598e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.686381547083598e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035783553552708e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7035783553552708e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.209350632448028e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.209350632448028e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244813893e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773244813893e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244813893e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773244813893e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244813893e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773244813893e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773244813893e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773244813893e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.974225379613676e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.974225379613676e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.974225379613676e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.974225379613676e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0474716556025244e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.0474716556025244e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.0474716556025244e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.0474716556025244e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.209350632448028e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.209350632448028e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0717282185651057e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0717282185651057e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0717282185651057e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0717282185651057e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.2004287494584223e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.2004287494584223e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.2004287494584223e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.2004287494584223e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.7035783553552708e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7035783553552708e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.3120943053755349e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.3120943053755349e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.3120943053755349e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.3120943053755349e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.686381547083598e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.686381547083598e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.8882935937336273e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.8882935937336273e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.0922506164229963e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506164229963e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.0922506164229963e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506164229963e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.0922506164229963e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506164229963e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.0922506164229963e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506164229963e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.444597854223221e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.444597854223221e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.444597854223221e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.444597854223221e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.684915095459212e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.684915095459212e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.684915095459212e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.684915095459212e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246974426047886e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246974426047886e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246974426047886e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246974426047886e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246974426047886e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246974426047886e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246974426047886e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246974426047886e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.867765104674474e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.867765104674474e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3002946563997167e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3002946563997167e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.3609563204691774e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.3609563204691774e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.874299071583031e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.874299071583031e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.883676576164791e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.883676576164791e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.947356011941913e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.947356011941913e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.947356011941913e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.947356011941913e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.98851170641361e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.98851170641361e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.0882507116055013e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.0882507116055013e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.1513463114486456e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.1513463114486456e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.846201671500693e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846201671500693e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846201671500693e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846201671500693e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.071480736833477e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.071480736833477e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.1055267222561544e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.1055267222561544e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.1055267222561544e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.1055267222561544e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146496327900409e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146496327900409e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146496327900409e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146496327900409e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159350502128719e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159350502128719e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159350502128719e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159350502128719e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.4279886568370785e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.4279886568370785e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.4279886568370785e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.4279886568370785e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935867718355523e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935867718355523e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935867718355523e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935867718355523e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.2532733484951475e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.2532733484951475e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979825793839186e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979825793839186e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979825793839186e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979825793839186e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.205548411215232e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.205548411215232e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.205548411215232e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.205548411215232e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00013840177303549693) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.00013840177303549693) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.00018787053389548943) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.00018787053389548943) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.00018787053389548943) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00018787053389548943) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00019400857029757903) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00019400857029757903) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0002463643756957104) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0002463643756957104) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0002463643756957104) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0002463643756957104) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192743499487698) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192743499487698) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156734248908614) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156734248908614) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156734248908614) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156734248908614) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024445) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007870896771024445) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.0015324835230729969) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.0015324835230729969) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.0015324835230729969) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.0015324835230729969) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.0016407548553124302) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0016407548553124302) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0017278753941369425) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.0017278753941369425) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.0024464971554158522) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.0024464971554158522) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.0024464971554158522) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.0024464971554158522) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.0032675138544235333) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.0032675138544235333) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.003356670563832866) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.003356670563832866) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.0034841573002178878) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0034841573002178878) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.003876470899336938) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.003876470899336938) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.004668620318776283) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668620318776283) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.0047672721882780445) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.0047672721882780445) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.0047672721882780445) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.0047672721882780445) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.005286546538226814) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.005286546538226814) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.005286546538226814) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.005286546538226814) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.005408954422409934) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.005408954422409934) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.005408954422409934) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.005408954422409934) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.010715508469796792) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010715508469796792) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010715508469796792) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010715508469796792) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.010757563953908941) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010757563953908941) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010757563953908941) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010757563953908941) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.014603704729162087) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.014603704729162087) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.014603704729162087) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.014603704729162087) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.01929956057936376) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01929956057936376) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01929956057936376) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01929956057936376) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01929956057936376) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01929956057936376) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.01929956057936376) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01929956057936376) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.058591988733861525) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.058591988733861525) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.775950527587675e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.775950527587675e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.775950527587675e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.775950527587675e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.07165035181002377) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.07165035181002377) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0716503518100238) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0716503518100238) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.019257505095251606) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019257505095251606) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010311482489831865) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311482489831865) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008826368514209815) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209815) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
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
+ (-0.0053480515826766295) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0053480515826766295) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0053480515826766295) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0053480515826766295) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0038040661717285394) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0038040661717285394) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00298416616812192) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.00298416616812192) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.00298416616812192) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.00298416616812192) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.002446497155415853) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.002446497155415853) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.0022494124470939817) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0022494124470939817) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022494124470939817) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0022494124470939817) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0021413612231016245) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.0021413612231016245) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.001863894282458742) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.001863894282458742) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.001863894282458742) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.001863894282458742) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.001863894282458742) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.001863894282458742) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.001863894282458742) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.001863894282458742) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0016407548553124302) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124302) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0016407548553124302) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124302) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0012223378081538214) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538214) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538214) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538214) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538214) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538214) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538214) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0012223378081538214) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0010283292378562424) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0010283292378562424) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0010283292378562424) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0010283292378562424) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.146306145372307e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.146306145372307e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.874299071583031e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.874299071583031e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.874299071583031e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.874299071583031e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3002946563997167e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3002946563997167e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3002946563997167e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3002946563997167e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.0444941298488986e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.0444941298488986e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.0444941298488986e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.0444941298488986e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.95607923045314e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.95607923045314e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.95607923045314e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.95607923045314e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105515037547498e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105515037547498e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105515037547498e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105515037547498e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661347213485897e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661347213485897e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661347213485897e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661347213485897e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341414246811e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341414246811e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189990976028723e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.189990976028723e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876621658549943e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876621658549943e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876621658549943e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876621658549943e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246207482779e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246207482779e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.5233896784461345e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.5233896784461345e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.076732531723379e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.076732531723379e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.076732531723379e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.076732531723379e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.013471459167641e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.013471459167641e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.904599884242174e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.904599884242174e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.904599884242174e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.904599884242174e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.6667317543308087e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.6667317543308087e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.6667317543308087e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6667317543308087e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.850564192905642e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.850564192905642e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309316578118e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6569309316578118e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309316578118e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6569309316578118e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.850564192905642e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.850564192905642e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.686381547083598e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.686381547083598e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.686381547083598e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.686381547083598e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471459167641e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.013471459167641e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.5233896784461345e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.5233896784461345e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.6704023908254527e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.6704023908254527e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.6704023908254527e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.6704023908254527e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246207482779e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246207482779e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189990976028723e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.189990976028723e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341414246811e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341414246811e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949476488207814e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949476488207814e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.7924939578056179e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939578056179e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939578056179e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.7924939578056179e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.883676576164791e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.883676576164791e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.98851170641361e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.98851170641361e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.98851170641361e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.98851170641361e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.2532733484951475e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.2532733484951475e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4017109736019642e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4017109736019642e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4017109736019642e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4017109736019642e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603693825257e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.5809603693825257e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603693825257e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.5809603693825257e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192743499487698) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487698) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192743499487698) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487698) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.0007870896771024446) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024446) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024446) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024446) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0011726348316441926) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0011726348316441926) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0011726348316441926) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0011726348316441926) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012366478019245001) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.0012366478019245001) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.0012366478019245001) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.0012366478019245001) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.002200964069500435) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002200964069500435) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002200964069500435) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002200964069500435) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798014) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798014) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798014) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798014) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798014) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798014) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002394972639798014) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798014) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002446497155415853) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.002446497155415853) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.0038040661717285394) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0038040661717285394) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.003876470899336938) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.003876470899336938) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.003876470899336938) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.003876470899336938) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.00422081397004642) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.00422081397004642) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.00422081397004642) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.00422081397004642) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.008826368514209815) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.008826368514209815) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.010311482489831865) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311482489831865) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257505095251606) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019257505095251606) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.058591988733861525) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.058591988733861525) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.39870090148784e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.39870090148784e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.3987009014878396e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.3987009014878396e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178878) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178878) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00298416616812192) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.00298416616812192) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.00019400857029757903) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00019400857029757903) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453723072e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453723072e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7924939578056179e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7924939578056179e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.54034141424681e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.54034141424681e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.54034141424681e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.54034141424681e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.850564192905642e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.850564192905642e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.850564192905642e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.850564192905642e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471459167641e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471459167641e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471459167641e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471459167641e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949476488207814e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949476488207814e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.7924939578056179e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7924939578056179e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029757903) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029757903) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00298416616812192) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.00298416616812192) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.0034841573002178878) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0034841573002178878) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
  (-73.13873231352532) [I0]
+ (-0.18066792656583341) [Z7]
+ (-0.1596143250180988) [Z5]
+ (-0.15961432501809877) [Z4]
+ (0.1741995615505565) [Z2]
+ (0.17419956155055658) [Z3]
+ (0.22757269005453518) [Z0]
+ (0.2275726900545352) [Z1]
+ (-8.194261372162012e-06) [Y4 Y6]
+ (-8.194261372162012e-06) [X4 X6]
+ (7.954413176300601e-06) [Y5 Y7]
+ (7.954413176300601e-06) [X5 X7]
+ (0.11270386920332229) [Z4 Z6]
+ (0.11270386920332229) [Z5 Z7]
+ (0.11952438964682668) [Z0 Z4]
+ (0.11952438964682668) [Z1 Z5]
+ (0.13401715261963715) [Z0 Z6]
+ (0.13401715261963715) [Z1 Z7]
+ (0.13734953064261307) [Z0 Z5]
+ (0.13734953064261307) [Z1 Z4]
+ (0.13766872645852576) [Z2 Z4]
+ (0.13766872645852576) [Z3 Z5]
+ (0.14138905291942813) [Z4 Z7]
+ (0.14138905291942813) [Z5 Z6]
+ (0.1472294321876617) [Z2 Z5]
+ (0.1472294321876617) [Z3 Z4]
+ (0.14926355147388892) [Z4 Z5]
+ (0.14973486803496946) [Z2 Z6]
+ (0.14973486803496946) [Z3 Z7]
+ (0.15138327161428855) [Z0 Z7]
+ (0.15138327161428855) [Z1 Z6]
+ (0.15435748657223664) [Z6 Z7]
+ (0.15582269051553138) [Z2 Z7]
+ (0.15582269051553138) [Z3 Z6]
+ (0.16756653265461258) [Z0 Z2]
+ (0.16756653265461258) [Z1 Z3]
+ (0.19392534613270188) [Z0 Z1]
+ (-7.037887510693237e-06) [Y5 Z6 Y7]
+ (-7.037887510693237e-06) [X5 Z6 X7]
+ (-7.0378875106932336e-06) [Y4 Z5 Y6]
+ (-7.0378875106932336e-06) [X4 Z5 X6]
+ (-0.028685183716105837) [Y4 Y5 X6 X7]
+ (-0.028685183716105837) [X4 X5 Y6 Y7]
+ (-0.01782514099578642) [Y0 Y1 X4 X5]
+ (-0.01782514099578642) [X0 X1 Y4 Y5]
+ (-0.017366118994651427) [Y0 Y1 X6 X7]
+ (-0.017366118994651427) [X0 X1 Y6 Y7]
+ (-0.013873381748426132) [Y0 Y1 X2 X3]
+ (-0.013873381748426132) [X0 X1 Y2 Y3]
+ (-0.009560705729135947) [Y2 Y3 X4 X5]
+ (-0.009560705729135947) [X2 X3 Y4 Y5]
+ (-0.006087822480561883) [Y2 Y3 X6 X7]
+ (-0.006087822480561883) [X2 X3 Y6 Y7]
+ (-0.00029219862611108043) [Y1 Y2 X3 X4]
+ (-0.00029219862611108043) [X1 X2 Y3 Y4]
+ (-8.19426137216201e-06) [Z4 Y5 Z6 Y7]
+ (-8.19426137216201e-06) [Z4 X5 Z6 X7]
+ (-2.8909678816028014e-06) [Z0 Y5 Z6 Y7]
+ (-2.8909678816028014e-06) [Z0 X5 Z6 X7]
+ (-2.8909678816028014e-06) [Z1 Y4 Z5 Y6]
+ (-2.8909678816028014e-06) [Z1 X4 Z5 X6]
+ (-1.8551201215054782e-06) [Z0 Y4 Z5 Y6]
+ (-1.8551201215054782e-06) [Z0 X4 Z5 X6]
+ (-1.8551201215054782e-06) [Z1 Y5 Z6 Y7]
+ (-1.8551201215054782e-06) [Z1 X5 Z6 X7]
+ (-1.597317197784971e-06) [Z2 Y4 Z5 Y6]
+ (-1.597317197784971e-06) [Z2 X4 Z5 X6]
+ (-1.597317197784971e-06) [Z3 Y5 Z6 Y7]
+ (-1.597317197784971e-06) [Z3 X5 Z6 X7]
+ (-1.0358477600973232e-06) [Y0 X1 X5 Y6]
+ (-1.0358477600973232e-06) [Y0 Y1 Y5 Y6]
+ (-1.0358477600973232e-06) [X0 X1 X5 X6]
+ (-1.0358477600973232e-06) [X0 Y1 Y5 X6]
+ (-9.344557776182406e-07) [Z2 Y5 Z6 Y7]
+ (-9.344557776182406e-07) [Z2 X5 Z6 X7]
+ (-9.344557776182406e-07) [Z3 Y4 Z5 Y6]
+ (-9.344557776182406e-07) [Z3 X4 Z5 X6]
+ (6.628614201667303e-07) [Y2 X3 X5 Y6]
+ (6.628614201667303e-07) [Y2 Y3 Y5 Y6]
+ (6.628614201667303e-07) [X2 X3 X5 X6]
+ (6.628614201667303e-07) [X2 Y3 Y5 X6]
+ (7.954413176300601e-06) [Y4 Z5 Y6 Z7]
+ (7.954413176300601e-06) [X4 Z5 X6 Z7]
+ (0.00029219862611108043) [Y1 X2 X3 Y4]
+ (0.00029219862611108043) [X1 Y2 Y3 X4]
+ (0.006087822480561883) [Y2 X3 X6 Y7]
+ (0.006087822480561883) [X2 Y3 Y6 X7]
+ (0.009560705729135947) [Y2 X3 X4 Y5]
+ (0.009560705729135947) [X2 Y3 Y4 X5]
+ (0.011307274008848152) [Y1 Z2 Z3 Y5]
+ (0.011307274008848152) [X1 Z2 Z3 X5]
+ (0.013873381748426132) [Y0 X1 X2 Y3]
+ (0.013873381748426132) [X0 Y1 Y2 X3]
+ (0.017366118994651427) [Y0 X1 X6 Y7]
+ (0.017366118994651427) [X0 Y1 Y6 X7]
+ (0.01782514099578642) [Y0 X1 X4 Y5]
+ (0.01782514099578642) [X0 Y1 Y4 X5]
+ (0.028685183716105837) [Y4 X5 X6 Y7]
+ (0.028685183716105837) [X4 Y5 Y6 X7]
+ (0.029812424517345715) [Y0 Z1 Z2 Y4]
+ (0.029812424517345715) [X0 Z1 Z2 X4]
+ (0.029812424517345715) [Y1 Z3 Z4 Y5]
+ (0.029812424517345715) [X1 Z3 Z4 X5]
+ (0.030104623143456796) [Y0 Z1 Z3 Y4]
+ (0.030104623143456796) [X0 Z1 Z3 X4]
+ (0.030104623143456796) [Y1 Z2 Z4 Y5]
+ (0.030104623143456796) [X1 Z2 Z4 X5]
+ (0.030787505389143904) [Y0 Z2 Z3 Y4]
+ (0.030787505389143904) [X0 Z2 Z3 X4]
+ (0.0437526380106599) [Y0 Z1 Z2 Z3 Y4]
+ (0.0437526380106599) [X0 Z1 Z2 Z3 X4]
+ (0.0437526380106599) [Y1 Z2 Z3 Z4 Y5]
+ (0.0437526380106599) [X1 Z2 Z3 Z4 X5]
+ (-0.014564531231172956) [Y1 Z2 Z3 X4 X6 Y7]
+ (-0.014564531231172956) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-0.014564531231172956) [X1 Z2 Z3 X4 X6 X7]
+ (-0.014564531231172956) [X1 Z2 Z3 Y4 Y6 X7]
+ (-6.524373848536911e-06) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (-6.524373848536911e-06) [X0 Z1 Z2 Z3 Z4 X6]
+ (-6.524373848536911e-06) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (-6.524373848536911e-06) [X1 Z2 Z3 Z5 Z6 X7]
+ (-3.7696594519979014e-06) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (-3.7696594519979014e-06) [X0 Z2 Z3 Z4 Z5 X6]
+ (-3.610297130602983e-06) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (-3.610297130602983e-06) [X0 Z1 Z3 Z4 Z5 X6]
+ (-3.610297130602983e-06) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (-3.610297130602983e-06) [X1 Z2 Z4 Z5 Z6 X7]
+ (-3.3131455001219473e-06) [Y1 Z2 Z3 Y4 X5 X6]
+ (-3.3131455001219473e-06) [X1 Z2 Z3 X4 Y5 Y6]
+ (-3.277483195549735e-06) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (-3.277483195549735e-06) [X0 Z1 Z2 Z4 Z5 X6]
+ (-3.277483195549735e-06) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (-3.277483195549735e-06) [X1 Z3 Z4 Z5 Z6 X7]
+ (-3.211228348414964e-06) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (-3.211228348414964e-06) [X0 Z1 Z2 Z3 Z5 X6]
+ (-3.211228348414964e-06) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (-3.211228348414964e-06) [X1 Z2 Z3 Z4 Z6 X7]
+ (-1.035847760097323e-06) [Y0 Y1 X4 Z5 Z6 X7]
+ (-1.035847760097323e-06) [X0 X1 Y4 Z5 Z6 Y7]
+ (-6.628614201667305e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (-6.628614201667305e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (-3.3281393505324746e-07) [Y1 X2 X3 Z4 Z5 Y6]
+ (-3.3281393505324746e-07) [X1 Y2 Y3 Z4 Z5 X6]
+ (3.3281393505324746e-07) [Y1 Y2 X3 Z4 Z5 X6]
+ (3.3281393505324746e-07) [X1 X2 Y3 Z4 Z5 Y6]
+ (6.628614201667305e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (6.628614201667305e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (1.035847760097323e-06) [Y0 X1 X4 Z5 Z6 Y7]
+ (1.035847760097323e-06) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.3131455001219473e-06) [Y1 Z2 Z3 X4 X5 Y6]
+ (3.3131455001219473e-06) [X1 Z2 Z3 Y4 Y5 X6]
+ (4.183932559405235e-06) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (4.183932559405235e-06) [X1 Z2 Z3 Z4 Z5 X7]
+ (0.00029219862611108043) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (0.00029219862611108043) [Y0 Z1 Y2 X3 Z4 X5]
+ (0.00029219862611108043) [X0 Z1 X2 Y3 Z4 Y5]
+ (0.00029219862611108043) [X0 Z1 X2 X3 Z4 X5]
+ (0.01054042590767159) [Y0 Z1 Z2 Z3 Y4 Z7]
+ (0.01054042590767159) [X0 Z1 Z2 Z3 X4 Z7]
+ (0.01054042590767159) [Y1 Z2 Z3 Z4 Y5 Z6]
+ (0.01054042590767159) [X1 Z2 Z3 Z4 X5 Z6]
+ (0.011307274008848154) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (0.011307274008848154) [X0 Z1 Z2 Z3 X4 Z5]
+ (0.025104957138844548) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (0.025104957138844548) [X0 Z1 Z2 Z3 X4 Z6]
+ (0.025104957138844548) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (0.025104957138844548) [X1 Z2 Z3 Z4 X5 Z7]
+ (0.030787505389143904) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (0.030787505389143904) [Z0 X1 Z2 Z3 Z4 X5]
+ (-5.105396549673884e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (-5.105396549673884e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (-5.105396549673883e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-5.105396549673883e-06) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.014564531231172954) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-0.014564531231172954) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-3.7696594519979014e-06) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-3.7696594519979014e-06) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-3.3281393505324746e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-3.3281393505324746e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-3.3281393505324746e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-3.3281393505324746e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (3.3131455001219473e-06) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (3.3131455001219473e-06) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (3.3131455001219473e-06) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (3.3131455001219473e-06) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (4.183932559405235e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (4.183932559405235e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (0.014564531231172954) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (0.014564531231172954) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
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
(0.005708495985960927+0j) [X0 Z1 X2 X6 Z7 Z8 Z9 X10] +
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
(0.005708495985960927+0j) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10] +
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
(0.005708495985960927+0j) [X1 Z2 X3 X7 Z8 Z9 Z10 X11] +
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
(0.005708495985960927+0j) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11] +
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
(-9.509249751380598e-07+0j) [Z2 Y4 Z5 Y6] +
(-4.728843147417908e-06+0j) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(0.12495807739503227+0j) [Z2 Z4] +
(-1.1708301369988116e-06+0j) [Z2 X5 Z6 X7] +
(-7.089799467802333e-06+0j) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-1.1708301369988116e-06+0j) [Z2 Y5 Z6 Y7] +
(-7.089799467802333e-06+0j) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
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
(-1.1708301369988116e-06+0j) [Z3 Y4 Z5 Y6] +
(-7.089799467802333e-06+0j) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(0.16079764534838575+0j) [Z3 Z4] +
(-9.509249751380598e-07+0j) [Z3 X5 Z6 X7] +
(-4.728843147417908e-06+0j) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-9.509249751380598e-07+0j) [Z3 Y5 Z6 Y7] +
(-4.728843147417908e-06+0j) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
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
(-5.929765814984023e-07+0j) [Z4 X5 Z6 X7] +
(-4.253224225664189e-06+0j) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-5.929765814984023e-07+0j) [Z4 Y5 Z6 Y7] +
(-4.253224225664189e-06+0j) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
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
(0.2200397733437609+0j) [Z8 Z9] +
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
  (-46.46390678868897) [I0]
+ (0.7829661725950181) [Z10]
+ (0.7829661725950183) [Z11]
+ (0.8084581961720491) [Z12]
+ (0.8084581961720492) [Z13]
+ (1.203440228914562) [Z5]
+ (1.2034402289145625) [Z4]
+ (1.3096862988615408) [Z6]
+ (1.3096862988615414) [Z7]
+ (1.3693525634718167) [Z8]
+ (1.3693525634718167) [Z9]
+ (1.6538942226831703) [Z2]
+ (1.6538942226831705) [Z3]
+ (12.41263074211178) [Z0]
+ (12.41263074211178) [Z1]
+ (-8.194261372765662e-06) [Y10 Y12]
+ (-8.194261372765662e-06) [X10 X12]
+ (-1.854060857750056e-06) [Y5 Y7]
+ (-1.854060857750056e-06) [X5 X7]
+ (-7.7649941176132e-07) [Y3 Y5]
+ (-7.7649941176132e-07) [X3 X5]
+ (-5.929765814481299e-07) [Y4 Y6]
+ (-5.929765814481299e-07) [X4 X6]
+ (1.6021167406285531e-06) [Y2 Y4]
+ (1.6021167406285531e-06) [X2 X4]
+ (7.95441317690007e-06) [Y11 Y13]
+ (7.95441317690007e-06) [X11 X13]
+ (0.003276971931231721) [Y1 Y3]
+ (0.003276971931231721) [X1 X3]
+ (0.10433064780651438) [Y0 Y2]
+ (0.10433064780651438) [X0 X2]
+ (0.11270386920332245) [Z10 Z12]
+ (0.11270386920332245) [Z11 Z13]
+ (0.11383573679388662) [Z4 Z12]
+ (0.11383573679388662) [Z5 Z13]
+ (0.1195243896468268) [Z6 Z10]
+ (0.1195243896468268) [Z7 Z11]
+ (0.12489990917237606) [Z4 Z10]
+ (0.12489990917237606) [Z5 Z11]
+ (0.12495807739503204) [Z2 Z4]
+ (0.12495807739503204) [Z3 Z5]
+ (0.12799502492468418) [Z2 Z10]
+ (0.12799502492468418) [Z3 Z11]
+ (0.1340171526196371) [Z6 Z12]
+ (0.1340171526196371) [Z7 Z13]
+ (0.13701191674040733) [Z4 Z6]
+ (0.13701191674040733) [Z5 Z7]
+ (0.1373495306426134) [Z6 Z11]
+ (0.1373495306426134) [Z7 Z10]
+ (0.13739104762683216) [Z2 Z6]
+ (0.13739104762683216) [Z3 Z7]
+ (0.1376687264585259) [Z8 Z10]
+ (0.1376687264585259) [Z9 Z11]
+ (0.14011289865354817) [Z2 Z12]
+ (0.14011289865354817) [Z3 Z13]
+ (0.14138905291942835) [Z10 Z13]
+ (0.14138905291942835) [Z11 Z12]
+ (0.14257997712485757) [Z4 Z11]
+ (0.14257997712485757) [Z5 Z10]
+ (0.14722943218766182) [Z8 Z11]
+ (0.14722943218766182) [Z9 Z10]
+ (0.14899430575065534) [Z4 Z7]
+ (0.14899430575065534) [Z5 Z6]
+ (0.1492635514738892) [Z10 Z11]
+ (0.1496070268444529) [Z4 Z8]
+ (0.1496070268444529) [Z5 Z9]
+ (0.1497348680349694) [Z8 Z12]
+ (0.1497348680349694) [Z9 Z13]
+ (0.1507140812100828) [Z2 Z8]
+ (0.1507140812100828) [Z3 Z9]
+ (0.15138327161428855) [Z6 Z13]
+ (0.15138327161428855) [Z7 Z12]
+ (0.15215040708869054) [Z4 Z13]
+ (0.15215040708869054) [Z5 Z12]
+ (0.15337968243314154) [Z2 Z11]
+ (0.15337968243314154) [Z3 Z10]
+ (0.1543574865722366) [Z12 Z13]
+ (0.15569010671752465) [Z2 Z13]
+ (0.15569010671752465) [Z3 Z12]
+ (0.1558226905155313) [Z8 Z13]
+ (0.1558226905155313) [Z9 Z12]
+ (0.15676396176430984) [Z4 Z9]
+ (0.15676396176430984) [Z5 Z8]
+ (0.15755314797985648) [Z4 Z5]
+ (0.16079764534838545) [Z2 Z5]
+ (0.16079764534838545) [Z3 Z4]
+ (0.16756653265461255) [Z6 Z8]
+ (0.16756653265461255) [Z7 Z9]
+ (0.16853486561579933) [Z2 Z7]
+ (0.16853486561579933) [Z3 Z6]
+ (0.1814399144030386) [Z6 Z9]
+ (0.1814399144030386) [Z7 Z8]
+ (0.1818908579075135) [Z2 Z3]
+ (0.18690820476912548) [Z2 Z9]
+ (0.18690820476912548) [Z3 Z8]
+ (0.19299723935364274) [Z0 Z10]
+ (0.19299723935364274) [Z1 Z11]
+ (0.1939253461327016) [Z6 Z7]
+ (0.19661770890342148) [Z0 Z4]
+ (0.19661770890342148) [Z1 Z5]
+ (0.19936354537360831) [Z0 Z5]
+ (0.19936354537360831) [Z1 Z4]
+ (0.20072866460441802) [Z0 Z11]
+ (0.20072866460441802) [Z1 Z10]
+ (0.21102659849791555) [Z0 Z12]
+ (0.21102659849791555) [Z1 Z13]
+ (0.2163103749863185) [Z0 Z13]
+ (0.2163103749863185) [Z1 Z12]
+ (0.23671080783830442) [Z0 Z2]
+ (0.23671080783830442) [Z1 Z3]
+ (0.24164663936017192) [Z0 Z6]
+ (0.24164663936017192) [Z1 Z7]
+ (0.24853483371314244) [Z0 Z7]
+ (0.24853483371314244) [Z1 Z6]
+ (0.2512944567459171) [Z0 Z3]
+ (0.2512944567459171) [Z1 Z2]
+ (0.2723251830660571) [Z0 Z8]
+ (0.2723251830660571) [Z1 Z9]
+ (0.2788345442672343) [Z0 Z9]
+ (0.2788345442672343) [Z1 Z8]
+ (1.1861763734860522) [Z0 Z1]
+ (-1.2260484987608157e-05) [Y4 Z5 Y6]
+ (-1.2260484987608157e-05) [X4 Z5 X6]
+ (-1.2260484987608153e-05) [Y5 Z6 Y7]
+ (-1.2260484987608153e-05) [X5 Z6 X7]
+ (-1.072231215807589e-05) [Y10 Z11 Y12]
+ (-1.072231215807589e-05) [X10 Z11 X12]
+ (-1.0722312158075884e-05) [Y11 Z12 Y13]
+ (-1.0722312158075884e-05) [X11 Z12 X13]
+ (-3.8870516712890866e-06) [Y3 Z4 Y5]
+ (-3.8870516712890866e-06) [X3 Z4 X5]
+ (-3.887051671289086e-06) [Y2 Z3 Y4]
+ (-3.887051671289086e-06) [X2 Z3 X4]
+ (0.1250703257977213) [Y0 Z1 Y2]
+ (0.1250703257977213) [X0 Z1 X2]
+ (0.1250703257977213) [Y1 Z2 Y3]
+ (0.1250703257977213) [X1 Z2 X3]
+ (-0.038314670294803906) [Y4 Y5 X12 X13]
+ (-0.038314670294803906) [X4 X5 Y12 Y13]
+ (-0.03619412355904265) [Y2 Y3 X8 X9]
+ (-0.03619412355904265) [X2 X3 Y8 Y9]
+ (-0.035839567953353406) [Y2 Y3 X4 X5]
+ (-0.035839567953353406) [X2 X3 Y4 Y5]
+ (-0.031143817988967173) [Y2 Y3 X6 X7]
+ (-0.031143817988967173) [X2 X3 Y6 Y7]
+ (-0.028685183716105907) [Y10 Y11 X12 X13]
+ (-0.028685183716105907) [X10 X11 Y12 Y13]
+ (-0.02599617759802106) [Y3 Z4 Z5 Y7]
+ (-0.02599617759802106) [X3 Z4 Z5 X7]
+ (-0.02538465750845734) [Y2 Y3 X10 X11]
+ (-0.02538465750845734) [X2 X3 Y10 Y11]
+ (-0.019028242443847213) [Y3 Y4 X11 X12]
+ (-0.019028242443847213) [X3 X4 Y11 Y12]
+ (-0.017825140995786592) [Y6 Y7 X10 X11]
+ (-0.017825140995786592) [X6 X7 Y10 Y11]
+ (-0.0176800679524815) [Y4 Y5 X10 X11]
+ (-0.0176800679524815) [X4 X5 Y10 Y11]
+ (-0.017366118994651444) [Y6 Y7 X12 X13]
+ (-0.017366118994651444) [X6 X7 Y12 Y13]
+ (-0.015577208063976476) [Y2 Y3 X12 X13]
+ (-0.015577208063976476) [X2 X3 Y12 Y13]
+ (-0.014583648907612712) [Y0 Y1 X2 X3]
+ (-0.014583648907612712) [X0 X1 Y2 Y3]
+ (-0.01387338174842606) [Y6 Y7 X8 X9]
+ (-0.01387338174842606) [X6 X7 Y8 Y9]
+ (-0.011982389010247991) [Y4 Y5 X6 X7]
+ (-0.011982389010247991) [X4 X5 Y6 Y7]
+ (-0.011285190200840962) [Y5 X6 X11 Y12]
+ (-0.011285190200840962) [X5 Y6 Y11 X12]
+ (-0.007731425250775289) [Y0 Y1 X10 X11]
+ (-0.007731425250775289) [X0 X1 Y10 Y11]
+ (-0.0071569349198569426) [Y4 Y5 X8 X9]
+ (-0.0071569349198569426) [X4 X5 Y8 Y9]
+ (-0.006888194352970548) [Y0 Y1 X6 X7]
+ (-0.006888194352970548) [X0 X1 Y6 Y7]
+ (-0.006509361201177239) [Y0 Y1 X8 X9]
+ (-0.006509361201177239) [X0 X1 Y8 Y9]
+ (-0.0060878224805618626) [Y8 Y9 X12 X13]
+ (-0.0060878224805618626) [X8 X9 Y12 Y13]
+ (-0.005283776488402967) [Y0 Y1 X12 X13]
+ (-0.005283776488402967) [X0 X1 Y12 Y13]
+ (-0.005143391768825169) [Y3 X4 X5 Y6]
+ (-0.005143391768825169) [X3 Y4 Y5 X6]
+ (-0.004684903388155202) [Y1 X2 X6 Y7]
+ (-0.004684903388155202) [Y1 Y2 Y6 Y7]
+ (-0.004684903388155202) [X1 X2 X6 X7]
+ (-0.004684903388155202) [X1 Y2 Y6 X7]
+ (-0.004575007626639213) [Y1 X2 X12 Y13]
+ (-0.004575007626639213) [Y1 Y2 Y12 Y13]
+ (-0.004575007626639213) [X1 X2 X12 X13]
+ (-0.004575007626639213) [X1 Y2 Y12 X13]
+ (-0.004424855449441851) [Y1 X2 X4 Y5]
+ (-0.004424855449441851) [Y1 Y2 Y4 Y5]
+ (-0.004424855449441851) [X1 X2 X4 X5]
+ (-0.004424855449441851) [X1 Y2 Y4 X5]
+ (-0.003479511890334363) [Y2 Z3 Z5 Y6]
+ (-0.003479511890334363) [X2 Z3 Z5 X6]
+ (-0.003479511890334363) [Y3 Z4 Z6 Y7]
+ (-0.003479511890334363) [X3 Z4 Z6 X7]
+ (-0.0027458364701868103) [Y0 Y1 X4 X5]
+ (-0.0027458364701868103) [X0 X1 Y4 Y5]
+ (-0.0017992194936630275) [Y1 X2 X10 Y11]
+ (-0.0017992194936630275) [Y1 Y2 Y10 Y11]
+ (-0.0017992194936630275) [X1 X2 X10 X11]
+ (-0.0017992194936630275) [X1 Y2 Y10 X11]
+ (-0.0002921986261110184) [Y7 Y8 X9 X10]
+ (-0.0002921986261110184) [X7 X8 Y9 Y10]
+ (-8.194261372765662e-06) [Z10 Y11 Z12 Y13]
+ (-8.194261372765662e-06) [Z10 X11 Z12 X13]
+ (-7.801707501119841e-06) [Y2 Z3 Y4 Z11]
+ (-7.801707501119841e-06) [X2 Z3 X4 Z11]
+ (-7.801707501119841e-06) [Y3 Z4 Y5 Z10]
+ (-7.801707501119841e-06) [X3 Z4 X5 Z10]
+ (-4.64305106888275e-06) [Y3 X4 X10 Y11]
+ (-4.64305106888275e-06) [Y3 Y4 Y10 Y11]
+ (-4.64305106888275e-06) [X3 X4 X10 X11]
+ (-4.64305106888275e-06) [X3 Y4 Y10 X11]
+ (-4.588855155921227e-06) [Y4 Z5 Y6 Z13]
+ (-4.588855155921227e-06) [X4 Z5 X6 Z13]
+ (-4.588855155921227e-06) [Y5 Z6 Y7 Z12]
+ (-4.588855155921227e-06) [X5 Z6 X7 Z12]
+ (-4.556569218634285e-06) [Y5 X6 X12 Y13]
+ (-4.556569218634285e-06) [Y5 Y6 Y12 Y13]
+ (-4.556569218634285e-06) [X5 X6 X12 X13]
+ (-4.556569218634285e-06) [X5 Y6 Y12 X13]
+ (-3.694513294818083e-06) [Y4 X5 X11 Y12]
+ (-3.694513294818083e-06) [Y4 Y5 Y11 Y12]
+ (-3.694513294818083e-06) [X4 X5 X11 X12]
+ (-3.694513294818083e-06) [X4 Y5 Y11 X12]
+ (-3.3440815561341335e-06) [Z0 Y5 Z6 Y7]
+ (-3.3440815561341335e-06) [Z0 X5 Z6 X7]
+ (-3.3440815561341335e-06) [Z1 Y4 Z5 Y6]
+ (-3.3440815561341335e-06) [Z1 X4 Z5 X6]
+ (-3.1586564322370907e-06) [Y2 Z3 Y4 Z10]
+ (-3.1586564322370907e-06) [X2 Z3 X4 Z10]
+ (-3.1586564322370907e-06) [Y3 Z4 Y5 Z11]
+ (-3.1586564322370907e-06) [X3 Z4 X5 Z11]
+ (-3.0993492432834415e-06) [Z0 Y4 Z5 Y6]
+ (-3.0993492432834415e-06) [Z0 X4 Z5 X6]
+ (-3.0993492432834415e-06) [Z1 Y5 Z6 Y7]
+ (-3.0993492432834415e-06) [Z1 X5 Z6 X7]
+ (-2.8909678817133396e-06) [Z6 Y11 Z12 Y13]
+ (-2.8909678817133396e-06) [Z6 X11 Z12 X13]
+ (-2.8909678817133396e-06) [Z7 Y10 Z11 Y12]
+ (-2.8909678817133396e-06) [Z7 X10 Z11 X12]
+ (-2.17766460521573e-06) [Z0 Y10 Z11 Y12]
+ (-2.17766460521573e-06) [Z0 X10 Z11 X12]
+ (-2.17766460521573e-06) [Z1 Y11 Z12 Y13]
+ (-2.17766460521573e-06) [Z1 X11 Z12 X13]
+ (-1.881850183003616e-06) [Y4 Z5 Y6 Z9]
+ (-1.881850183003616e-06) [X4 Z5 X6 Z9]
+ (-1.881850183003616e-06) [Y5 Z6 Y7 Z8]
+ (-1.881850183003616e-06) [X5 Z6 X7 Z8]
+ (-1.8551201216922172e-06) [Z6 Y10 Z11 Y12]
+ (-1.8551201216922172e-06) [Z6 X10 Z11 X12]
+ (-1.8551201216922172e-06) [Z7 Y11 Z12 Y13]
+ (-1.8551201216922172e-06) [Z7 X11 Z12 X13]
+ (-1.8540608577500559e-06) [Y4 Z5 Y6 Z7]
+ (-1.8540608577500559e-06) [X4 Z5 X6 Z7]
+ (-1.8163031698332941e-06) [Z4 Y11 Z12 Y13]
+ (-1.8163031698332941e-06) [Z4 X11 Z12 X13]
+ (-1.8163031698332941e-06) [Z5 Y10 Z11 Y12]
+ (-1.8163031698332941e-06) [Z5 X10 Z11 X12]
+ (-1.6923978286205239e-06) [Y4 Z5 Y6 Z10]
+ (-1.6923978286205239e-06) [X4 Z5 X6 Z10]
+ (-1.6923978286205239e-06) [Y5 Z6 Y7 Z11]
+ (-1.6923978286205239e-06) [X5 Z6 X7 Z11]
+ (-1.6148794140428648e-06) [Z0 Y11 Z12 Y13]
+ (-1.6148794140428648e-06) [Z0 X11 Z12 X13]
+ (-1.6148794140428648e-06) [Z1 Y10 Z11 Y12]
+ (-1.6148794140428648e-06) [Z1 X10 Z11 X12]
+ (-1.5973171979038351e-06) [Z8 Y10 Z11 Y12]
+ (-1.5973171979038351e-06) [Z8 X10 Z11 X12]
+ (-1.5973171979038351e-06) [Z9 Y11 Z12 Y13]
+ (-1.5973171979038351e-06) [Z9 X11 Z12 X13]
+ (-1.4548424489026373e-06) [Y3 X4 X6 Y7]
+ (-1.4548424489026373e-06) [Y3 Y4 Y6 Y7]
+ (-1.4548424489026373e-06) [X3 X4 X6 X7]
+ (-1.4548424489026373e-06) [X3 Y4 Y6 X7]
+ (-1.3980449079636316e-06) [Y4 Z5 Y6 Z8]
+ (-1.3980449079636316e-06) [X4 Z5 X6 Z8]
+ (-1.3980449079636316e-06) [Y5 Z6 Y7 Z9]
+ (-1.3980449079636316e-06) [X5 Z6 X7 Z9]
+ (-1.1954890097353473e-06) [Y2 Z3 Y4 Z7]
+ (-1.1954890097353473e-06) [X2 Z3 X4 Z7]
+ (-1.1954890097353473e-06) [Y3 Z4 Y5 Z6]
+ (-1.1954890097353473e-06) [X3 Z4 X5 Z6]
+ (-1.190850808032541e-06) [Z0 Y3 Z4 Y5]
+ (-1.190850808032541e-06) [Z0 X3 Z4 X5]
+ (-1.190850808032541e-06) [Z1 Y2 Z3 Y4]
+ (-1.190850808032541e-06) [Z1 X2 Z3 X4]
+ (-1.170830136879832e-06) [Z2 Y5 Z6 Y7]
+ (-1.170830136879832e-06) [Z2 X5 Z6 X7]
+ (-1.170830136879832e-06) [Z3 Y4 Z5 Y6]
+ (-1.170830136879832e-06) [Z3 X4 Z5 X6]
+ (-1.0632283424891886e-06) [Z2 Y10 Z11 Y12]
+ (-1.0632283424891886e-06) [Z2 X10 Z11 X12]
+ (-1.0632283424891886e-06) [Z3 Y11 Z12 Y13]
+ (-1.0632283424891886e-06) [Z3 X11 Z12 X13]
+ (-1.0358477600211222e-06) [Y6 X7 X11 Y12]
+ (-1.0358477600211222e-06) [Y6 Y7 Y11 Y12]
+ (-1.0358477600211222e-06) [X6 X7 X11 X12]
+ (-1.0358477600211222e-06) [X6 Y7 Y11 X12]
+ (-9.509249750585443e-07) [Z2 Y4 Z5 Y6]
+ (-9.509249750585443e-07) [Z2 X4 Z5 X6]
+ (-9.509249750585443e-07) [Z3 Y5 Z6 Y7]
+ (-9.509249750585443e-07) [Z3 X5 Z6 X7]
+ (-9.344557777032418e-07) [Z8 Y11 Z12 Y13]
+ (-9.344557777032418e-07) [Z8 X11 Z12 X13]
+ (-9.344557777032418e-07) [Z9 Y10 Z11 Y12]
+ (-9.344557777032418e-07) [Z9 X10 Z11 X12]
+ (-8.337746751770017e-07) [Z0 Y2 Z3 Y4]
+ (-8.337746751770017e-07) [Z0 X2 Z3 X4]
+ (-8.337746751770017e-07) [Z1 Y3 Z4 Y5]
+ (-8.337746751770017e-07) [Z1 X3 Z4 X5]
+ (-7.956895371495506e-07) [Y3 X4 X8 Y9]
+ (-7.956895371495506e-07) [Y3 Y4 Y8 Y9]
+ (-7.956895371495506e-07) [X3 X4 X8 X9]
+ (-7.956895371495506e-07) [X3 Y4 Y8 X9]
+ (-7.7649941176132e-07) [Y2 Z3 Y4 Z5]
+ (-7.7649941176132e-07) [X2 Z3 X4 Z5]
+ (-5.929765814481299e-07) [Z4 Y5 Z6 Y7]
+ (-5.929765814481299e-07) [Z4 X5 Z6 X7]
+ (-5.77005299320742e-07) [Y2 Z3 Y4 Z9]
+ (-5.77005299320742e-07) [X2 Z3 X4 Z9]
+ (-5.77005299320742e-07) [Y3 Z4 Y5 Z8]
+ (-5.77005299320742e-07) [X3 Z4 X5 Z8]
+ (-5.471647744953659e-07) [Y1 Y2 X11 X12]
+ (-5.471647744953659e-07) [X1 X2 Y11 Y12]
+ (-4.838052750399841e-07) [Y5 X6 X8 Y9]
+ (-4.838052750399841e-07) [Y5 Y6 Y8 Y9]
+ (-4.838052750399841e-07) [X5 X6 X8 X9]
+ (-4.838052750399841e-07) [X5 Y6 Y8 X9]
+ (-3.5707613285553945e-07) [Y0 X1 X3 Y4]
+ (-3.5707613285553945e-07) [Y0 Y1 Y3 Y4]
+ (-3.5707613285553945e-07) [X0 X1 X3 X4]
+ (-3.5707613285553945e-07) [X0 Y1 Y3 X4]
+ (-2.4473231285069204e-07) [Y0 X1 X5 Y6]
+ (-2.4473231285069204e-07) [Y0 Y1 Y5 Y6]
+ (-2.4473231285069204e-07) [X0 X1 X5 X6]
+ (-2.4473231285069204e-07) [X0 Y1 Y5 X6]
+ (-2.1990516182128787e-07) [Y2 X3 X5 Y6]
+ (-2.1990516182128787e-07) [Y2 Y3 Y5 Y6]
+ (-2.1990516182128787e-07) [X2 X3 X5 X6]
+ (-2.1990516182128787e-07) [X2 Y3 Y5 X6]
+ (-1.9332412769187862e-07) [Y1 X2 X3 Y4]
+ (-1.9332412769187862e-07) [X1 Y2 Y3 X4]
+ (-1.2919694861407063e-07) [Y1 Z2 Z3 Y5]
+ (-1.2919694861407063e-07) [X1 Z2 Z3 X5]
+ (1.7379332622097704e-07) [Y0 Z1 Z3 Y4]
+ (1.7379332622097704e-07) [X0 Z1 Z3 X4]
+ (1.7379332622097704e-07) [Y1 Z2 Z4 Y5]
+ (1.7379332622097704e-07) [X1 Z2 Z4 X5]
+ (1.9332412769187862e-07) [Y1 Y2 X3 X4]
+ (1.9332412769187862e-07) [X1 X2 Y3 Y4]
+ (2.186842378288086e-07) [Y2 Z3 Y4 Z8]
+ (2.186842378288086e-07) [X2 Z3 X4 Z8]
+ (2.186842378288086e-07) [Y3 Z4 Y5 Z9]
+ (2.186842378288086e-07) [X3 Z4 X5 Z9]
+ (2.5935343916729037e-07) [Y2 Z3 Y4 Z6]
+ (2.5935343916729037e-07) [X2 Z3 X4 Z6]
+ (2.5935343916729037e-07) [Y3 Z4 Y5 Z7]
+ (2.5935343916729037e-07) [X3 Z4 X5 Z7]
+ (3.6060718675579147e-07) [Y0 Z1 Z2 Y4]
+ (3.6060718675579147e-07) [X0 Z1 Z2 X4]
+ (3.6060718675579147e-07) [Y1 Z3 Z4 Y5]
+ (3.6060718675579147e-07) [X1 Z3 Z4 X5]
+ (5.471647744953659e-07) [Y1 X2 X11 Y12]
+ (5.471647744953659e-07) [X1 Y2 Y11 X12]
+ (5.627851911728656e-07) [Y0 X1 X11 Y12]
+ (5.627851911728656e-07) [Y0 Y1 Y11 Y12]
+ (5.627851911728656e-07) [X0 X1 X11 X12]
+ (5.627851911728656e-07) [X0 Y1 Y11 X12]
+ (6.628614202005932e-07) [Y8 X9 X11 Y12]
+ (6.628614202005932e-07) [Y8 Y9 Y11 Y12]
+ (6.628614202005932e-07) [X8 X9 X11 X12]
+ (6.628614202005932e-07) [X8 Y9 Y11 X12]
+ (1.1094407590817735e-06) [Z2 Y11 Z12 Y13]
+ (1.1094407590817735e-06) [Z2 X11 Z12 X13]
+ (1.1094407590817735e-06) [Z3 Y10 Z11 Y12]
+ (1.1094407590817735e-06) [Z3 X10 Z11 X12]
+ (1.6021167406285531e-06) [Z2 Y3 Z4 Y5]
+ (1.6021167406285531e-06) [Z2 X3 Z4 X5]
+ (1.8782101249847884e-06) [Z4 Y10 Z11 Y12]
+ (1.8782101249847884e-06) [Z4 X10 Z11 X12]
+ (1.8782101249847884e-06) [Z5 Y11 Z12 Y13]
+ (1.8782101249847884e-06) [Z5 X11 Z12 X13]
+ (2.172669101570962e-06) [Y2 X3 X11 Y12]
+ (2.172669101570962e-06) [Y2 Y3 Y11 Y12]
+ (2.172669101570962e-06) [X2 X3 X11 X12]
+ (2.172669101570962e-06) [X2 Y3 Y11 X12]
+ (3.1174479456358975e-06) [Y0 Z2 Z3 Y4]
+ (3.1174479456358975e-06) [X0 Z2 Z3 X4]
+ (3.539054184854151e-06) [Y2 Z3 Y4 Z12]
+ (3.539054184854151e-06) [X2 Z3 X4 Z12]
+ (3.539054184854151e-06) [Y3 Z4 Y5 Z13]
+ (3.539054184854151e-06) [X3 Z4 X5 Z13]
+ (4.28191388518842e-06) [Y4 Z5 Y6 Z11]
+ (4.28191388518842e-06) [X4 Z5 X6 Z11]
+ (4.28191388518842e-06) [Y5 Z6 Y7 Z10]
+ (4.28191388518842e-06) [X5 Z6 X7 Z10]
+ (5.275883122459954e-06) [Y3 X4 X12 Y13]
+ (5.275883122459954e-06) [Y3 Y4 Y12 Y13]
+ (5.275883122459954e-06) [X3 X4 X12 X13]
+ (5.275883122459954e-06) [X3 Y4 Y12 X13]
+ (5.974311713808944e-06) [Y5 X6 X10 Y11]
+ (5.974311713808944e-06) [Y5 Y6 Y10 Y11]
+ (5.974311713808944e-06) [X5 X6 X10 X11]
+ (5.974311713808944e-06) [X5 Y6 Y10 X11]
+ (7.95441317690007e-06) [Y10 Z11 Y12 Z13]
+ (7.95441317690007e-06) [X10 Z11 X12 Z13]
+ (8.814937307314104e-06) [Y2 Z3 Y4 Z13]
+ (8.814937307314104e-06) [X2 Z3 X4 Z13]
+ (8.814937307314104e-06) [Y3 Z4 Y5 Z12]
+ (8.814937307314104e-06) [X3 Z4 X5 Z12]
+ (0.0002921986261110184) [Y7 X8 X9 Y10]
+ (0.0002921986261110184) [X7 Y8 Y9 X10]
+ (0.0004956762314915497) [Y2 Z4 Z5 Y6]
+ (0.0004956762314915497) [X2 Z4 Z5 X6]
+ (0.0011059037691897233) [Y0 Z1 Y2 Z5]
+ (0.0011059037691897233) [X0 Z1 X2 Z5]
+ (0.0011059037691897233) [Y1 Z2 Y3 Z4]
+ (0.0011059037691897233) [X1 Z2 X3 Z4]
+ (0.001663879878490806) [Y2 Z3 Z4 Y6]
+ (0.001663879878490806) [X2 Z3 Z4 X6]
+ (0.001663879878490806) [Y3 Z5 Z6 Y7]
+ (0.001663879878490806) [X3 Z5 Z6 X7]
+ (0.001756070701841283) [Y0 Z1 Y2 Z11]
+ (0.001756070701841283) [X0 Z1 X2 Z11]
+ (0.001756070701841283) [Y1 Z2 Y3 Z10]
+ (0.001756070701841283) [X1 Z2 X3 Z10]
+ (0.00232623062315812) [Y0 Z1 Y2 Z13]
+ (0.00232623062315812) [X0 Z1 X2 Z13]
+ (0.00232623062315812) [Y1 Z2 Y3 Z12]
+ (0.00232623062315812) [X1 Z2 X3 Z12]
+ (0.0027458364701868103) [Y0 X1 X4 Y5]
+ (0.0027458364701868103) [X0 Y1 Y4 X5]
+ (0.002929768674751103) [Y0 Z1 Y2 Z9]
+ (0.002929768674751103) [X0 Z1 X2 Z9]
+ (0.002929768674751103) [Y1 Z2 Y3 Z8]
+ (0.002929768674751103) [X1 Z2 X3 Z8]
+ (0.0032769719312317207) [Y0 Z1 Y2 Z3]
+ (0.0032769719312317207) [X0 Z1 X2 Z3]
+ (0.003347617530666207) [Y0 Z1 Y2 Z7]
+ (0.003347617530666207) [X0 Z1 X2 Z7]
+ (0.003347617530666207) [Y1 Z2 Y3 Z6]
+ (0.003347617530666207) [X1 Z2 X3 Z6]
+ (0.0035552901955043107) [Y0 Z1 Y2 Z10]
+ (0.0035552901955043107) [X0 Z1 X2 Z10]
+ (0.0035552901955043107) [Y1 Z2 Y3 Z11]
+ (0.0035552901955043107) [X1 Z2 X3 Z11]
+ (0.005143391768825169) [Y3 Y4 X5 X6]
+ (0.005143391768825169) [X3 X4 Y5 Y6]
+ (0.005283776488402967) [Y0 X1 X12 Y13]
+ (0.005283776488402967) [X0 Y1 Y12 X13]
+ (0.005530759218631574) [Y0 Z1 Y2 Z4]
+ (0.005530759218631574) [X0 Z1 X2 Z4]
+ (0.005530759218631574) [Y1 Z2 Y3 Z5]
+ (0.005530759218631574) [X1 Z2 X3 Z5]
+ (0.0060878224805618626) [Y8 X9 X12 Y13]
+ (0.0060878224805618626) [X8 Y9 Y12 X13]
+ (0.006509361201177239) [Y0 X1 X8 Y9]
+ (0.006509361201177239) [X0 Y1 Y8 X9]
+ (0.006888194352970548) [Y0 X1 X6 Y7]
+ (0.006888194352970548) [X0 Y1 Y6 X7]
+ (0.006901238249797332) [Y0 Z1 Y2 Z12]
+ (0.006901238249797332) [X0 Z1 X2 Z12]
+ (0.006901238249797332) [Y1 Z2 Y3 Z13]
+ (0.006901238249797332) [X1 Z2 X3 Z13]
+ (0.0071569349198569426) [Y4 X5 X8 Y9]
+ (0.0071569349198569426) [X4 Y5 Y8 X9]
+ (0.007731425250775289) [Y0 X1 X10 Y11]
+ (0.007731425250775289) [X0 Y1 Y10 X11]
+ (0.008032520918821407) [Y0 Z1 Y2 Z6]
+ (0.008032520918821407) [X0 Z1 X2 Z6]
+ (0.008032520918821407) [Y1 Z2 Y3 Z7]
+ (0.008032520918821407) [X1 Z2 X3 Z7]
+ (0.011055020596132132) [Y0 Z1 Y2 Z8]
+ (0.011055020596132132) [X0 Z1 X2 Z8]
+ (0.011055020596132132) [Y1 Z2 Y3 Z9]
+ (0.011055020596132132) [X1 Z2 X3 Z9]
+ (0.011285190200840962) [Y5 Y6 X11 X12]
+ (0.011285190200840962) [X5 X6 Y11 Y12]
+ (0.01130727400884824) [Y7 Z8 Z9 Y11]
+ (0.01130727400884824) [X7 Z8 Z9 X11]
+ (0.011982389010247991) [Y4 X5 X6 Y7]
+ (0.011982389010247991) [X4 Y5 Y6 X7]
+ (0.01387338174842606) [Y6 X7 X8 Y9]
+ (0.01387338174842606) [X6 Y7 Y8 X9]
+ (0.014583648907612712) [Y0 X1 X2 Y3]
+ (0.014583648907612712) [X0 Y1 Y2 X3]
+ (0.015577208063976476) [Y2 X3 X12 Y13]
+ (0.015577208063976476) [X2 Y3 Y12 X13]
+ (0.017366118994651444) [Y6 X7 X12 Y13]
+ (0.017366118994651444) [X6 Y7 Y12 X13]
+ (0.0176800679524815) [Y4 X5 X10 Y11]
+ (0.0176800679524815) [X4 Y5 Y10 X11]
+ (0.017825140995786592) [Y6 X7 X10 Y11]
+ (0.017825140995786592) [X6 Y7 Y10 X11]
+ (0.019028242443847213) [Y3 X4 X11 Y12]
+ (0.019028242443847213) [X3 Y4 Y11 X12]
+ (0.02538465750845734) [Y2 X3 X10 Y11]
+ (0.02538465750845734) [X2 Y3 Y10 X11]
+ (0.028685183716105907) [Y10 X11 X12 Y13]
+ (0.028685183716105907) [X10 Y11 Y12 X13]
+ (0.029812424517345892) [Y6 Z7 Z8 Y10]
+ (0.029812424517345892) [X6 Z7 Z8 X10]
+ (0.029812424517345892) [Y7 Z9 Z10 Y11]
+ (0.029812424517345892) [X7 Z9 Z10 X11]
+ (0.03010462314345691) [Y6 Z7 Z9 Y10]
+ (0.03010462314345691) [X6 Z7 Z9 X10]
+ (0.03010462314345691) [Y7 Z8 Z10 Y11]
+ (0.03010462314345691) [X7 Z8 Z10 X11]
+ (0.030787505389143956) [Y6 Z8 Z9 Y10]
+ (0.030787505389143956) [X6 Z8 Z9 X10]
+ (0.031143817988967173) [Y2 X3 X6 Y7]
+ (0.031143817988967173) [X2 Y3 Y6 X7]
+ (0.035839567953353406) [Y2 X3 X4 Y5]
+ (0.035839567953353406) [X2 Y3 Y4 X5]
+ (0.03619412355904265) [Y2 X3 X8 Y9]
+ (0.03619412355904265) [X2 Y3 Y8 X9]
+ (0.038314670294803906) [Y4 X5 X12 Y13]
+ (0.038314670294803906) [X4 Y5 Y12 X13]
+ (0.1043306478065144) [Z0 Y1 Z2 Y3]
+ (0.1043306478065144) [Z0 X1 Z2 X3]
+ (-0.12133276911042261) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042261) [X2 Z3 Z4 Z5 X6]
+ (-0.12133276911042258) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133276911042258) [X3 Z4 Z5 Z6 X7]
+ (3.2020768789092767e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.2020768789092767e-06) [X1 Z2 Z3 Z4 X5]
+ (3.202076878909277e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.202076878909277e-06) [X0 Z1 Z2 Z3 X4]
+ (0.22848106564918982) [Y6 Z7 Z8 Z9 Y10]
+ (0.22848106564918982) [X6 Z7 Z8 Z9 X10]
+ (0.22848106564918982) [Y7 Z8 Z9 Z10 Y11]
+ (0.22848106564918982) [X7 Z8 Z9 Z10 X11]
+ (-0.03276765782329054) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.03276765782329054) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.03276765782329054) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.03276765782329054) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273183) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.027115036845273183) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273183) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.027115036845273183) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.02599617759802106) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.02599617759802106) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.017561202409646152) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.017561202409646152) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.017561202409646152) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.017561202409646152) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.014564531231173043) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564531231173043) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564531231173043) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564531231173043) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.01221504099761403) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.01221504099761403) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.01221504099761403) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.01221504099761403) [X4 Z5 X6 X11 Z12 X13]
+ (-0.01221504099761403) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.01221504099761403) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.01221504099761403) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.01221504099761403) [X5 Z6 X7 X10 Z11 X12]
+ (-0.011756013419819224) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.011756013419819224) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.011756013419819224) [X3 Z4 Z5 X6 X8 X9]
+ (-0.011756013419819224) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.008764827575688722) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.008764827575688722) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.008764827575688722) [X2 Z3 Z4 X5 X11 X12]
+ (-0.008764827575688722) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.008764827575688722) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.008764827575688722) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.008764827575688722) [X3 X4 X10 Z11 Z12 X13]
+ (-0.008764827575688722) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.007306759928832996) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306759928832996) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306759928832996) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306759928832996) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.00580518898982693) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.00580518898982693) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.00580518898982693) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.00580518898982693) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.005652620978017352) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.005652620978017352) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.005652620978017352) [X0 X1 X3 Z4 Z5 X6]
+ (-0.005652620978017352) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.005143391768825169) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.005143391768825169) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.005143391768825169) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.005143391768825169) [X2 Z3 X4 X5 Z6 X7]
+ (-0.004684903388155202) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.004684903388155202) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668620318776298) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668620318776298) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.004575007626639213) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.004575007626639213) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424855449441851) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424855449441851) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.0041587973818400575) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.0041587973818400575) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.0041587973818400575) [X3 Z4 Z5 X6 X12 X13]
+ (-0.0041587973818400575) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.0034937903598901304) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.0034937903598901304) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.0034937903598901304) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.0034937903598901304) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.0027790267990255276) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.0027790267990255276) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.0022939566113524637) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.0022939566113524637) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.0017992194936630275) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.0017992194936630275) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.001727875394136972) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.001727875394136972) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298507967730662) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298507967730662) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298507967730662) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298507967730662) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533856254125409) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533856254125409) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008145313270956711) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008145313270956711) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008145313270956711) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008145313270956711) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.735036880589034e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735036880589034e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735036880589034e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735036880589034e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.774817865256418e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.774817865256418e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.774817865256418e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.774817865256418e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.518362216222638e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.518362216222638e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.518362216222638e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.518362216222638e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.4443446765372e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.4443446765372e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.4443446765372e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.4443446765372e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.524373849098145e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.524373849098145e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.524373849098145e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.524373849098145e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.290028433856645e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.290028433856645e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.290028433856645e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.290028433856645e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.9743117138089446e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.9743117138089446e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.275883122459954e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.275883122459954e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.64305106888275e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.64305106888275e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.556569218634285e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.556569218634285e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.253224225856047e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.253224225856047e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.7696594525330416e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.7696594525330416e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.6945132948180827e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.6945132948180827e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.6102971311042095e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.6102971311042095e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.6102971311042095e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.6102971311042095e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.3131455002863005e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.3131455002863005e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.277483196069515e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.277483196069515e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.277483196069515e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.277483196069515e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.2112283488118444e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.2112283488118444e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.2112283488118444e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.2112283488118444e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.1513463114346734e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.1513463114346734e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.088250711594721e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.088250711594721e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.172669101570962e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.172669101570962e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.4548424489026373e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.4548424489026373e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.3304731887192182e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.3304731887192182e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2283337823659933e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2283337823659933e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.0358477600211224e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.0358477600211224e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956895371495506e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956895371495506e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.73319774330198e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.73319774330198e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.73319774330198e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.73319774330198e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.628614202005932e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.628614202005932e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.556281914964969e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.556281914964969e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.556281914964969e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.556281914964969e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.418291575047824e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.418291575047824e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.418291575047824e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.418291575047824e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.927453083626368e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927453083626368e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927453083626368e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927453083626368e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627851911728656e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627851911728656e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660625075058e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660625075058e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660625075058e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660625075058e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660625075058e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660625075058e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660625075058e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660625075058e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.838052750399841e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.838052750399841e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.5707613285553945e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.5707613285553945e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.3281393503469453e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.3281393503469453e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.086826565123655e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.086826565123655e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.086826565123655e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.086826565123655e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.4473231285069204e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.4473231285069204e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.371328947623609e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.371328947623609e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.371328947623609e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.371328947623609e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.1990516182128787e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.1990516182128787e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.9332412769187856e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.9332412769187856e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.9332412769187856e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.9332412769187856e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.8394209152565335e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.8394209152565335e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.8394209152565335e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.8394209152565335e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.5510539175105374e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.5510539175105374e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.5510539175105374e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.5510539175105374e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.3807781479137316e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.3807781479137316e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.3807781479137316e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.3807781479137316e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.3807781479137316e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.3807781479137316e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.3807781479137316e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.3807781479137316e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.3807781479137316e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.3807781479137316e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.3807781479137316e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.3807781479137316e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.2919694861407066e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.2919694861407066e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.1076325599920385e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.1076325599920385e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.1076325599920385e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.1076325599920385e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.1076325599920385e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.1076325599920385e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.1076325599920385e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.1076325599920385e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.057446596756127e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057446596756127e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057446596756127e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057446596756127e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649310132835984e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649310132835984e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649310132835984e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649310132835984e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.8394209152565335e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.8394209152565335e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.8394209152565335e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.8394209152565335e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.1990516182128787e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.1990516182128787e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.4473231285069204e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.4473231285069204e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.2362599609072075e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.2362599609072075e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.2362599609072075e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.2362599609072075e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.3281393503469453e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.3281393503469453e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.5707613285553945e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.5707613285553945e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.838052750399841e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.838052750399841e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.627851911728656e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627851911728656e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.628614202005932e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.628614202005932e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956895371495506e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956895371495506e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.306536652315913e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.306536652315913e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.306536652315913e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.306536652315913e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.0358477600211224e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.0358477600211224e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2283337823659933e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2283337823659933e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.2393363217439567e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393363217439567e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393363217439567e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393363217439567e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.3304731887192182e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.3304731887192182e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.4548424489026373e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.4548424489026373e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.172669101570962e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.172669101570962e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.088250711594721e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.088250711594721e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.1174479456358975e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.1174479456358975e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.1513463114346734e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.1513463114346734e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.3131455002863005e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.3131455002863005e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.33433128968128e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.33433128968128e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.6945132948180827e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.6945132948180827e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.183932559559492e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.183932559559492e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.556569218634285e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.556569218634285e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.64305106888275e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.64305106888275e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.275883122459954e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.275883122459954e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.9743117138089446e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.9743117138089446e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.0002921986261110184) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.0002921986261110184) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.0002921986261110184) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.0002921986261110184) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004956762314915497) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004956762314915497) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650070219499275) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650070219499275) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650070219499275) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650070219499275) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533856254125409) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533856254125409) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.0016095313817213756) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.0016095313817213756) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.0016095313817213756) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.0016095313817213756) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0016676041811440434) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.0016676041811440434) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.0016676041811440434) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.0016676041811440434) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.001727875394136972) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.001727875394136972) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.0017992194936630275) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.0017992194936630275) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.0022939566113524637) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.0022939566113524637) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.002462917007133917) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.002462917007133917) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.002462917007133917) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.002462917007133917) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.003961560792496507) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.003961560792496507) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.003961560792496507) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.003961560792496507) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.004424855449441851) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424855449441851) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004575007626639213) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.004575007626639213) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668620318776298) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668620318776298) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.004684903388155202) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.004684903388155202) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005324835234221695) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.005324835234221695) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.005324835234221695) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.005324835234221695) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.005368659358109601) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368659358109601) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368659358109601) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368659358109601) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.007960880725921575) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.007960880725921575) [X4 Z5 X6 X10 Z11 X12]
+ (0.007960880725921575) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.007960880725921575) [X5 Z6 X7 X11 Z12 X13]
+ (0.008890731522694643) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.008890731522694643) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.008890731522694643) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.008890731522694643) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.010263414868158495) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.010263414868158495) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.010263414868158495) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.010263414868158495) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.010540425907671574) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.010540425907671574) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.010540425907671574) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.010540425907671574) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010960074940542613) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010960074940542613) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010960074940542613) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010960074940542613) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307274008848239) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307274008848239) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.014411099430130907) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411099430130907) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411099430130907) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411099430130907) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.015225630757226579) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.015225630757226579) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.015225630757226579) [X3 Z4 Z5 X6 X10 X11]
+ (0.015225630757226579) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.018266834869375605) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.018266834869375605) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.018266834869375605) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.018266834869375605) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.019020423173039983) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.019020423173039983) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.019020423173039983) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.019020423173039983) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.020175921723535606) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.020175921723535606) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.020175921723535606) [X4 Z5 Z6 X7 X11 X12]
+ (0.020175921723535606) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.020175921723535606) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.020175921723535606) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.020175921723535606) [X5 X6 X10 Z11 Z12 X13]
+ (0.020175921723535606) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.02435307767806891) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.02435307767806891) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.02435307767806891) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.02435307767806891) [X2 Z3 X4 X11 Z12 X13]
+ (0.02435307767806891) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.02435307767806891) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.02435307767806891) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.02435307767806891) [X3 Z4 X5 X10 Z11 X12]
+ (0.024389082531149586) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.024389082531149586) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.024389082531149586) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.024389082531149586) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.025104957138844617) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.025104957138844617) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.025104957138844617) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.025104957138844617) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.030787505389143956) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.030787505389143956) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.04587947078129795) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.04587947078129795) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.05600733087780793) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.05600733087780793) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.05600733087780793) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.05600733087780793) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661382) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.05608468124661382) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661382) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.05608468124661382) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.631277928929958e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631277928929958e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.631277928929958e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631277928929958e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.5950860074171392e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.5950860074171392e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.595086007417138e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.595086007417138e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.04274327701378264) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.04274327701378264) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (0.04274327701378264) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04274327701378264) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.04764261217638313) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.04764261217638313) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.04764261217638313) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.04764261217638313) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.04171881383982178) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.04171881383982178) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.04171881383982178) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.04171881383982178) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.03956441632289324) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.03956441632289324) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.03956441632289324) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03956441632289324) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.039359168022052964) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.039359168022052964) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.039359168022052964) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.039359168022052964) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.039318051947197535) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.039318051947197535) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.039318051947197535) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.039318051947197535) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03560837898831247) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.03560837898831247) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.029903789512624786) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.029903789512624786) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.029903789512624786) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.029903789512624786) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.028730779551905457) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.028730779551905457) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.028730779551905457) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.028730779551905457) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.025637238296026904) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.025637238296026904) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.025637238296026904) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.025637238296026904) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.024755463292890887) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.024755463292890887) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.024755463292890887) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.024755463292890887) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.024282117354692993) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.024282117354692993) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.023145130929529092) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.023145130929529092) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.022528440196013008) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.022528440196013008) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.021433810721600773) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.021433810721600773) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.021433810721600773) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.021433810721600773) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251617) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.019257505095251617) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019028242443847213) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.019028242443847213) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.018889030304942864) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.018889030304942864) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.018889030304942864) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.018889030304942864) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.016024603689179538) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.016024603689179538) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.015225630757226577) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225630757226577) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.014603704729162078) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.014603704729162078) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564531231173043) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564531231173043) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.011756013419819222) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.011756013419819222) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.011285190200840962) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.011285190200840962) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.009841749246962595) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009841749246962595) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.009612634606847364) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612634606847364) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612634606847364) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612634606847364) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.008469978791024013) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.008469978791024013) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.007306759928832996) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.007306759928832996) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005652620978017352) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.005652620978017352) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368659358109601) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368659358109601) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.0041587973818400575) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.0041587973818400575) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0033566705638328775) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.0033566705638328775) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.0033566705638328775) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.0033566705638328775) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.003267513854423549) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.003267513854423549) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.003267513854423549) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.003267513854423549) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.0027790267990255276) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.0027790267990255276) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.0026860409778066193) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.0026860409778066193) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.0026860409778066193) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.0026860409778066193) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.0022939566113524637) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524637) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.0022939566113524637) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524637) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.0009581655836696482) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.0009581655836696482) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0009581655836696482) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.0009581655836696482) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.0009581655836696482) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.0009581655836696482) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0009581655836696482) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.0009581655836696482) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.00024636437569570176) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00024636437569570176) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303551501) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.00013840177303551501) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.00013840177303551501) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.00013840177303551501) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.735036880589034e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735036880589034e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585306970607e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585306970607e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.6103585306970607e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.6103585306970607e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808796524566e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.5316808796524566e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808796524566e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5316808796524566e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.806102775933428e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.806102775933428e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.806102775933428e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.806102775933428e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.089799467987445e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.089799467987445e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.089799467987445e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.089799467987445e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.652209670490781e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.652209670490781e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.652209670490781e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.652209670490781e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851834958756e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.481851834958756e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851834958756e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.481851834958756e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.071480736789165e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.071480736789165e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.071480736789165e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.071480736789165e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.734622039144263e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.734622039144263e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.734622039144263e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.734622039144263e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.728843147589545e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.728843147589545e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.728843147589545e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.728843147589545e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.253224225856047e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.253224225856047e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.7696594525330416e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.7696594525330416e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.5443954295301144e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.5443954295301144e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.5443954295301144e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.5443954295301144e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.5443954295301144e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.5443954295301144e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.5443954295301144e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.5443954295301144e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.360956320397899e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.360956320397899e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.360956320397899e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.360956320397899e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.1032156050307416e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.1032156050307416e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.1032156050307416e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.1032156050307416e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.011122098723013e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.011122098723013e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.011122098723013e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.011122098723013e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9429468368016825e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.9429468368016825e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9429468368016825e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9429468368016825e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174774841445e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.6541174774841445e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174774841445e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.6541174774841445e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.522493067746977e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.522493067746977e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.522493067746977e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.522493067746977e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.522493067746977e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.522493067746977e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.522493067746977e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.522493067746977e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.2283337823659933e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337823659933e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.2283337823659933e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337823659933e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.988770288135398e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988770288135398e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988770288135398e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988770288135398e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.867765104460438e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104460438e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104460438e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.867765104460438e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.189990976002735e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.189990976002735e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246207347024e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246207347024e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.471647744953659e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.471647744953659e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.561447179353921e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.561447179353921e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.561447179353921e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.561447179353921e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.523389678541588e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.523389678541588e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.4273231087814745e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.4273231087814745e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.4273231087814745e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.4273231087814745e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.3281393503469453e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.3281393503469453e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.3281393503469453e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.3281393503469453e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.086826565123655e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.086826565123655e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.8882935931753796e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935931753796e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935931753796e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.8882935931753796e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.371328947623609e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.371328947623609e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.8394209152565338e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.8394209152565338e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.057446596756126e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057446596756126e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.5371780966515614e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.5371780966515614e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.5371780966515614e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.5371780966515614e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.057446596756126e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057446596756126e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.209350630772868e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350630772868e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350630772868e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.209350630772868e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.703578355320245e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.703578355320245e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.703578355320245e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.703578355320245e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.8394209152565338e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.8394209152565338e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.371328947623609e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.371328947623609e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.086826565123655e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.086826565123655e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.523389678541588e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.523389678541588e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.471647744953659e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.471647744953659e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246207347024e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246207347024e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189990976002735e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.189990976002735e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.3304731887192182e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.3304731887192182e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.3304731887192182e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.3304731887192182e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288532436876962e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288532436876962e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288532436876962e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288532436876962e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.6893489516436469e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.6893489516436469e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.6893489516436469e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.6893489516436469e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.745518400716574e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.745518400716574e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.745518400716574e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.745518400716574e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.745518400716574e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.745518400716574e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.745518400716574e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.745518400716574e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.2118420193906244e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420193906244e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.2118420193906244e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420193906244e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.2118420193906244e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420193906244e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.2118420193906244e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420193906244e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.3131455002863e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.3131455002863e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.3131455002863e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.3131455002863e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.33433128968128e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.33433128968128e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.183932559559492e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.183932559559492e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.735036880589034e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735036880589034e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00024636437569570176) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00024636437569570176) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0004458535128840967) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.0004458535128840967) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.0004458535128840967) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.0004458535128840967) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940221543005589) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005589) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005589) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005589) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005589) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005589) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940221543005589) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005589) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533856254125409) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533856254125409) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533856254125409) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533856254125409) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0010435246534907573) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435246534907573) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435246534907573) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435246534907573) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.0012803060973496706) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.0012803060973496706) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.0012803060973496706) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.0012803060973496706) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.0013038004788126943) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.0013038004788126943) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.0013038004788126943) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.0013038004788126943) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.0022619660624823425) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.0022619660624823425) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.0022619660624823425) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.0022619660624823425) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.0022619660624823425) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.0022619660624823425) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.0022619660624823425) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.0022619660624823425) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.0039898414566193145) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.0039898414566193145) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.0039898414566193145) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.0039898414566193145) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.0041587973818400575) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.0041587973818400575) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.004311038507914307) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.004311038507914307) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.004311038507914307) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.004311038507914307) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.004636976661182549) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.004636976661182549) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.004636976661182549) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.004636976661182549) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.005114473831660395) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.005114473831660395) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.005114473831660395) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.005114473831660395) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.005114473831660395) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660395) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660395) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005114473831660395) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00524153538280386) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.00524153538280386) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.00524153538280386) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.00524153538280386) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.005262642473076857) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262642473076857) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262642473076857) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262642473076857) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368659358109601) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368659358109601) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005379937155839376) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.005379937155839376) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.005379937155839376) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.005379937155839376) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.005652620978017352) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.005652620978017352) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.005708495985960954) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708495985960954) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708495985960954) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708495985960954) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.007306759928832996) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.007306759928832996) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469978791024013) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.008469978791024013) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.009841749246962595) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.009841749246962595) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.011285190200840962) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.011285190200840962) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.011756013419819222) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.011756013419819222) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.014564531231173043) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564531231173043) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603704729162078) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.014603704729162078) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.015225630757226577) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225630757226577) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.016024603689179538) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.016024603689179538) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.019028242443847213) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.019028242443847213) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.019257505095251617) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.019257505095251617) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.045879470781297955) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.045879470781297955) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.36937089366156156) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.36937089366156156) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.36937089366156156) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.36937089366156156) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.28164257767022755) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.28164257767022755) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.28164257767022743) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.28164257767022743) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.09065144207036488) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.09065144207036488) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.09065144207036488) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.09065144207036488) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863635) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684737589863635) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863635) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684737589863635) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635021950634997) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635021950634997) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635021950634997) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635021950634997) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214012) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.06752385099214012) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214012) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.06752385099214012) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03560837898831247) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.03560837898831247) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03490334337366186) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03490334337366186) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03490334337366186) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03490334337366186) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883830002) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.024591860883830002) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883830002) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.024591860883830002) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.024282117354692993) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.024282117354692993) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.02314513092952909) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.02314513092952909) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.022528440196013008) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.022528440196013008) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019538050311314663) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.019538050311314663) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.019538050311314663) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.019538050311314663) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.017091553155898817) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.017091553155898817) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.017091553155898817) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.017091553155898817) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.016024603689179538) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.016024603689179538) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.016024603689179538) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.016024603689179538) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.010311482489831856) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831856) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831856) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311482489831856) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.009841749246962595) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962595) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.009841749246962595) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962595) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.00882636851420985) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.00882636851420985) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.00882636851420985) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.00882636851420985) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008541996625454825) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454825) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454825) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454825) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454825) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454825) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454825) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008541996625454825) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008469978791024013) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791024013) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.008469978791024013) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791024013) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.004668620318776298) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668620318776298) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.0038764708993369364) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.0038764708993369364) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.003804066171728543) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728543) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728543) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003804066171728543) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484157300217887) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484157300217887) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003356670563832877) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.003356670563832877) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.003267513854423549) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.003267513854423549) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.002141361223101574) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.002141361223101574) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.001727875394136972) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.001727875394136972) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.001640754855312429) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.001640754855312429) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.001452884321416885) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.001452884321416885) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.001452884321416885) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.001452884321416885) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.000787089677102442) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.000787089677102442) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192743499487625) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192743499487625) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.0001940085702975639) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0001940085702975639) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303551501) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.00013840177303551501) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.1416252211532e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.1416252211532e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.1416252211532e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.1416252211532e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.071480736789165e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.071480736789165e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.1513463114346734e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.1513463114346734e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.088250711594721e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.088250711594721e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.9885117062588877e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.9885117062588877e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8742990715354606e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.8742990715354606e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.360956320397899e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.360956320397899e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.3002946563854565e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3002946563854565e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.1468376508656397e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.1468376508656397e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.1468376508656397e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.1468376508656397e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.352332104031923e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.352332104031923e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.352332104031923e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.352332104031923e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091637199998814e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637199998814e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637199998814e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637199998814e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637199998814e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637199998814e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637199998814e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091637199998814e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.074305986760822e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.074305986760822e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.074305986760822e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.074305986760822e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128987199173e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128987199173e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128987199173e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128987199173e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867765104460438e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.867765104460438e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560692465701058e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692465701058e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692465701058e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692465701058e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692465701058e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692465701058e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692465701058e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560692465701058e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.997018422540365e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.997018422540365e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.997018422540365e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.997018422540365e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.997018422540365e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.997018422540365e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.997018422540365e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.997018422540365e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.568247521457226e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.568247521457226e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.568247521457226e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.568247521457226e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393086575835e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393086575835e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393086575835e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393086575835e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393086575835e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393086575835e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.3767393086575835e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393086575835e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.88829359317538e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.88829359317538e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.686381547933619e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.686381547933619e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035783553202455e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7035783553202455e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.209350630772868e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.209350630772868e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773245444755e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773245444755e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773245444755e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773245444755e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773245444755e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773245444755e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773245444755e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773245444755e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.974225379940653e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.974225379940653e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.974225379940653e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.974225379940653e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0474716556865726e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.0474716556865726e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.0474716556865726e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.0474716556865726e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.209350630772868e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.209350630772868e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.071728218677438e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.071728218677438e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.071728218677438e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.071728218677438e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.2004287494469114e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.2004287494469114e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.2004287494469114e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.2004287494469114e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.7035783553202455e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7035783553202455e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.3120943054211407e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.3120943054211407e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.3120943054211407e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.3120943054211407e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.686381547933619e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.686381547933619e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.88829359317538e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.88829359317538e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.0922506163612317e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506163612317e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.0922506163612317e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506163612317e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.0922506163612317e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506163612317e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.0922506163612317e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506163612317e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.444597854041408e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.444597854041408e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.444597854041408e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.444597854041408e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.6849150953468585e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.6849150953468585e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.6849150953468585e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.6849150953468585e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246974426024462e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246974426024462e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246974426024462e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246974426024462e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246974426024462e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246974426024462e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246974426024462e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246974426024462e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.867765104460438e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.867765104460438e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3002946563854565e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3002946563854565e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.360956320397899e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.360956320397899e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.8742990715354606e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.8742990715354606e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.8836765761191227e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.8836765761191227e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.9473560119748385e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9473560119748385e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.9473560119748385e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9473560119748385e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117062588877e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9885117062588877e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.088250711594721e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.088250711594721e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.1513463114346734e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.1513463114346734e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.84620167144296e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.84620167144296e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.84620167144296e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.84620167144296e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.071480736789165e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.071480736789165e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.105526722216182e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.105526722216182e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.105526722216182e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.105526722216182e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.1464963278284165e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.1464963278284165e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.1464963278284165e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.1464963278284165e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159350502035221e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159350502035221e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159350502035221e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159350502035221e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427988656828584e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427988656828584e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427988656828584e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427988656828584e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935867718233725e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935867718233725e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935867718233725e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935867718233725e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348417977e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.253273348417977e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979825793751644e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979825793751644e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979825793751644e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979825793751644e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.205548411217617e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.205548411217617e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.205548411217617e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.205548411217617e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00013840177303551501) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.00013840177303551501) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.00018787053389554407) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.00018787053389554407) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.00018787053389554407) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00018787053389554407) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0001940085702975639) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.0001940085702975639) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0002463643756957018) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0002463643756957018) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0002463643756957018) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0002463643756957018) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192743499487625) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192743499487625) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156734248909101) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156734248909101) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156734248909101) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156734248909101) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.000787089677102442) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.000787089677102442) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.0015324835230730658) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.0015324835230730658) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.0015324835230730658) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.0015324835230730658) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.001640754855312429) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.001640754855312429) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.001727875394136972) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.001727875394136972) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.0024464971554158457) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.0024464971554158457) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.0024464971554158457) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.0024464971554158457) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.003267513854423549) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.003267513854423549) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.003356670563832877) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.003356670563832877) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.003484157300217887) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.003484157300217887) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0038764708993369364) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.0038764708993369364) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.004668620318776298) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668620318776298) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.0047672721882781165) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.0047672721882781165) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.0047672721882781165) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.0047672721882781165) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.005286546538226879) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.005286546538226879) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.005286546538226879) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.005286546538226879) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.005408954422410002) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.005408954422410002) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.005408954422410002) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.005408954422410002) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.010715508469796787) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010715508469796787) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010715508469796787) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010715508469796787) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.010757563953908965) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010757563953908965) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010757563953908965) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010757563953908965) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.014603704729162078) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.014603704729162078) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.014603704729162078) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.014603704729162078) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.019299560579363786) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363786) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363786) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363786) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363786) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363786) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.019299560579363786) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363786) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.05859198873386193) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.05859198873386193) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.7759505274210235e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.7759505274210235e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.775950527421024e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.775950527421024e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.07165035181002695) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07165035181002695) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.071650351810027) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.071650351810027) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251617) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019257505095251617) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010311482489831856) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311482489831856) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00882636851420985) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.00882636851420985) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.007597464029770625) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.007597464029770625) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.007597464029770625) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.007597464029770625) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311873) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311873) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311873) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311873) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311873) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311873) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311873) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311873) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676638) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005348051582676638) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005348051582676638) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676638) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.003804066171728543) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003804066171728543) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219247) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.0029841661681219247) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.0029841661681219247) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.0029841661681219247) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.0024464971554158457) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.0024464971554158457) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.002249412447093986) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.002249412447093986) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.002249412447093986) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002249412447093986) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0021413612231015737) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.0021413612231015737) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.0018638942824587518) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587518) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587518) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587518) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587518) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587518) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0018638942824587518) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587518) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.001640754855312429) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.001640754855312429) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.001640754855312429) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.001640754855312429) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0012223378081538255) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538255) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538255) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538255) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538255) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538255) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538255) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0012223378081538255) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0010283292378562615) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0010283292378562615) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0010283292378562615) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0010283292378562615) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1463061453780102e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1463061453780102e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.8742990715354606e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990715354606e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.8742990715354606e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990715354606e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3002946563854565e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3002946563854565e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3002946563854565e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3002946563854565e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.0444941298732934e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.0444941298732934e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.0444941298732934e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.0444941298732934e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.956079230700605e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.956079230700605e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.956079230700605e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.956079230700605e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105515037914875e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105515037914875e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105515037914875e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105515037914875e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661347213687505e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661347213687505e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661347213687505e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661347213687505e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341414127589e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341414127589e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189990976002735e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.189990976002735e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876621658774222e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876621658774222e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876621658774222e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876621658774222e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246207347024e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246207347024e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.523389678541588e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.523389678541588e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0767325318403453e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.0767325318403453e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0767325318403453e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.0767325318403453e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.013471459108772e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.013471459108772e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.904599884605344e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.904599884605344e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.904599884605344e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.904599884605344e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.6667317544377133e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.6667317544377133e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.6667317544377133e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6667317544377133e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641927857296e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.8505641927857296e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.656930931324407e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.656930931324407e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.656930931324407e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.656930931324407e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.8505641927857296e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.8505641927857296e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.686381547933619e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.686381547933619e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.686381547933619e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.686381547933619e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471459108772e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.013471459108772e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.523389678541588e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.523389678541588e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.670402390433179e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.670402390433179e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.670402390433179e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.670402390433179e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246207347024e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246207347024e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189990976002735e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.189990976002735e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341414127589e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341414127589e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949476487792095e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949476487792095e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.7924939578066616e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939578066616e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939578066616e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.7924939578066616e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.8836765761191222e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.8836765761191222e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.9885117062588877e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117062588877e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117062588877e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117062588877e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348417977e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253273348417977e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.401710973589851e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.401710973589851e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.401710973589851e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.401710973589851e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.580960369370517e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.580960369370517e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.580960369370517e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.580960369370517e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192743499487625) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487625) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192743499487625) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487625) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.000787089677102442) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.000787089677102442) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.000787089677102442) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.000787089677102442) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0011726348316441918) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0011726348316441918) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0011726348316441918) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0011726348316441918) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.001236647801924559) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.001236647801924559) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.001236647801924559) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.001236647801924559) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.0022009640695004537) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0022009640695004537) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0022009640695004537) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0022009640695004537) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980175) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980175) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0023949726397980175) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980175) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0023949726397980175) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0023949726397980175) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0023949726397980175) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0023949726397980175) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0024464971554158457) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.0024464971554158457) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.003804066171728543) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.003804066171728543) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0038764708993369364) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.0038764708993369364) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.0038764708993369364) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.0038764708993369364) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.004220813970046484) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.004220813970046484) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.004220813970046484) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.004220813970046484) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.00882636851420985) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00882636851420985) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.010311482489831856) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311482489831856) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257505095251617) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019257505095251617) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.05859198873386193) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.05859198873386193) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.398700901706782e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.398700901706782e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.398700901706782e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.398700901706782e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003484157300217887) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003484157300217887) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219247) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.0029841661681219247) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.0001940085702975639) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0001940085702975639) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453780102e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453780102e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7924939578066616e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7924939578066616e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540341414127589e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341414127589e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540341414127589e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341414127589e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.85056419278573e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.85056419278573e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.85056419278573e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.85056419278573e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471459108772e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471459108772e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471459108772e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471459108772e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949476487792095e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949476487792095e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.7924939578066616e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7924939578066616e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0001940085702975639) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0001940085702975639) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0029841661681219247) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.0029841661681219247) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.003484157300217887) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.003484157300217887) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
  (-73.13873231352534) [I0]
+ (-0.18066792656583353) [Z7]
+ (-0.1596143250180991) [Z5]
+ (-0.15961432501809908) [Z4]
+ (0.17419956155055616) [Z3]
+ (0.1741995615505562) [Z2]
+ (0.22757269005453454) [Z0]
+ (0.22757269005453468) [Z1]
+ (-8.194261371431502e-06) [Y4 Y6]
+ (-8.194261371431502e-06) [X4 X6]
+ (7.954413175504442e-06) [Y5 Y7]
+ (7.954413175504442e-06) [X5 X7]
+ (0.11270386920332197) [Z4 Z6]
+ (0.11270386920332197) [Z5 Z7]
+ (0.11952438964682663) [Z0 Z4]
+ (0.11952438964682663) [Z1 Z5]
+ (0.1340171526196369) [Z0 Z6]
+ (0.1340171526196369) [Z1 Z7]
+ (0.13734953064261313) [Z0 Z5]
+ (0.13734953064261313) [Z1 Z4]
+ (0.13766872645852563) [Z2 Z4]
+ (0.13766872645852563) [Z3 Z5]
+ (0.14138905291942788) [Z4 Z7]
+ (0.14138905291942788) [Z5 Z6]
+ (0.14722943218766155) [Z2 Z5]
+ (0.14722943218766155) [Z3 Z4]
+ (0.14926355147388864) [Z4 Z5]
+ (0.1497348680349691) [Z2 Z6]
+ (0.1497348680349691) [Z3 Z7]
+ (0.15138327161428827) [Z0 Z7]
+ (0.15138327161428827) [Z1 Z6]
+ (0.1543574865722361) [Z6 Z7]
+ (0.15582269051553094) [Z2 Z7]
+ (0.15582269051553094) [Z3 Z6]
+ (0.16756653265461266) [Z0 Z2]
+ (0.16756653265461266) [Z1 Z3]
+ (0.19392534613270182) [Z0 Z1]
+ (-7.037887511004269e-06) [Y5 Z6 Y7]
+ (-7.037887511004269e-06) [X5 Z6 X7]
+ (-7.0378875110042675e-06) [Y4 Z5 Y6]
+ (-7.0378875110042675e-06) [X4 Z5 X6]
+ (-0.02868518371610592) [Y4 Y5 X6 X7]
+ (-0.02868518371610592) [X4 X5 Y6 Y7]
+ (-0.01782514099578653) [Y0 Y1 X4 X5]
+ (-0.01782514099578653) [X0 X1 Y4 Y5]
+ (-0.01736611899465138) [Y0 Y1 X6 X7]
+ (-0.01736611899465138) [X0 X1 Y6 Y7]
+ (-0.013873381748426063) [Y0 Y1 X2 X3]
+ (-0.013873381748426063) [X0 X1 Y2 Y3]
+ (-0.009560705729135916) [Y2 Y3 X4 X5]
+ (-0.009560705729135916) [X2 X3 Y4 Y5]
+ (-0.006087822480561835) [Y2 Y3 X6 X7]
+ (-0.006087822480561835) [X2 X3 Y6 Y7]
+ (-0.00029219862611102313) [Y1 Y2 X3 X4]
+ (-0.00029219862611102313) [X1 X2 Y3 Y4]
+ (-8.194261371431502e-06) [Z4 Y5 Z6 Y7]
+ (-8.194261371431502e-06) [Z4 X5 Z6 X7]
+ (-2.890967881426623e-06) [Z0 Y5 Z6 Y7]
+ (-2.890967881426623e-06) [Z0 X5 Z6 X7]
+ (-2.890967881426623e-06) [Z1 Y4 Z5 Y6]
+ (-2.890967881426623e-06) [Z1 X4 Z5 X6]
+ (-1.85512012124749e-06) [Z0 Y4 Z5 Y6]
+ (-1.85512012124749e-06) [Z0 X4 Z5 X6]
+ (-1.85512012124749e-06) [Z1 Y5 Z6 Y7]
+ (-1.85512012124749e-06) [Z1 X5 Z6 X7]
+ (-1.5973171975468116e-06) [Z2 Y4 Z5 Y6]
+ (-1.5973171975468116e-06) [Z2 X4 Z5 X6]
+ (-1.5973171975468116e-06) [Z3 Y5 Z6 Y7]
+ (-1.5973171975468116e-06) [Z3 X5 Z6 X7]
+ (-1.035847760179133e-06) [Y0 X1 X5 Y6]
+ (-1.035847760179133e-06) [Y0 Y1 Y5 Y6]
+ (-1.035847760179133e-06) [X0 X1 X5 X6]
+ (-1.035847760179133e-06) [X0 Y1 Y5 X6]
+ (-9.344557774289378e-07) [Z2 Y5 Z6 Y7]
+ (-9.344557774289378e-07) [Z2 X5 Z6 X7]
+ (-9.344557774289378e-07) [Z3 Y4 Z5 Y6]
+ (-9.344557774289378e-07) [Z3 X4 Z5 X6]
+ (6.62861420117874e-07) [Y2 X3 X5 Y6]
+ (6.62861420117874e-07) [Y2 Y3 Y5 Y6]
+ (6.62861420117874e-07) [X2 X3 X5 X6]
+ (6.62861420117874e-07) [X2 Y3 Y5 X6]
+ (7.954413175504442e-06) [Y4 Z5 Y6 Z7]
+ (7.954413175504442e-06) [X4 Z5 X6 Z7]
+ (0.00029219862611102313) [Y1 X2 X3 Y4]
+ (0.00029219862611102313) [X1 Y2 Y3 X4]
+ (0.006087822480561835) [Y2 X3 X6 Y7]
+ (0.006087822480561835) [X2 Y3 Y6 X7]
+ (0.009560705729135916) [Y2 X3 X4 Y5]
+ (0.009560705729135916) [X2 Y3 Y4 X5]
+ (0.011307274008848206) [Y1 Z2 Z3 Y5]
+ (0.011307274008848206) [X1 Z2 Z3 X5]
+ (0.013873381748426063) [Y0 X1 X2 Y3]
+ (0.013873381748426063) [X0 Y1 Y2 X3]
+ (0.01736611899465138) [Y0 X1 X6 Y7]
+ (0.01736611899465138) [X0 Y1 Y6 X7]
+ (0.01782514099578653) [Y0 X1 X4 Y5]
+ (0.01782514099578653) [X0 Y1 Y4 X5]
+ (0.02868518371610592) [Y4 X5 X6 Y7]
+ (0.02868518371610592) [X4 Y5 Y6 X7]
+ (0.02981242451734585) [Y0 Z1 Z2 Y4]
+ (0.02981242451734585) [X0 Z1 Z2 X4]
+ (0.02981242451734585) [Y1 Z3 Z4 Y5]
+ (0.02981242451734585) [X1 Z3 Z4 X5]
+ (0.030104623143456875) [Y0 Z1 Z3 Y4]
+ (0.030104623143456875) [X0 Z1 Z3 X4]
+ (0.030104623143456875) [Y1 Z2 Z4 Y5]
+ (0.030104623143456875) [X1 Z2 Z4 X5]
+ (0.030787505389143953) [Y0 Z2 Z3 Y4]
+ (0.030787505389143953) [X0 Z2 Z3 X4]
+ (0.043752638010659976) [Y0 Z1 Z2 Z3 Y4]
+ (0.043752638010659976) [X0 Z1 Z2 Z3 X4]
+ (0.043752638010660046) [Y1 Z2 Z3 Z4 Y5]
+ (0.043752638010660046) [X1 Z2 Z3 Z4 X5]
+ (-0.014564531231173024) [Y1 Z2 Z3 X4 X6 Y7]
+ (-0.014564531231173024) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-0.014564531231173024) [X1 Z2 Z3 X4 X6 X7]
+ (-0.014564531231173024) [X1 Z2 Z3 Y4 Y6 X7]
+ (-6.524373847939656e-06) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (-6.524373847939656e-06) [X0 Z1 Z2 Z3 Z4 X6]
+ (-6.524373847939656e-06) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (-6.524373847939656e-06) [X1 Z2 Z3 Z5 Z6 X7]
+ (-3.7696594514574885e-06) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (-3.7696594514574885e-06) [X0 Z2 Z3 Z4 Z5 X6]
+ (-3.610297130057856e-06) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (-3.610297130057856e-06) [X0 Z1 Z3 Z4 Z5 X6]
+ (-3.610297130057856e-06) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (-3.610297130057856e-06) [X1 Z2 Z4 Z5 Z6 X7]
+ (-3.313145499938609e-06) [Y1 Z2 Z3 Y4 X5 X6]
+ (-3.313145499938609e-06) [X1 Z2 Z3 X4 Y5 Y6]
+ (-3.2774831949975085e-06) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (-3.2774831949975085e-06) [X0 Z1 Z2 Z4 Z5 X6]
+ (-3.2774831949975085e-06) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (-3.2774831949975085e-06) [X1 Z3 Z4 Z5 Z6 X7]
+ (-3.2112283480010467e-06) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (-3.2112283480010467e-06) [X0 Z1 Z2 Z3 Z5 X6]
+ (-3.2112283480010467e-06) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (-3.2112283480010467e-06) [X1 Z2 Z3 Z4 Z6 X7]
+ (-1.035847760179133e-06) [Y0 Y1 X4 Z5 Z6 X7]
+ (-1.035847760179133e-06) [X0 X1 Y4 Z5 Z6 Y7]
+ (-6.62861420117874e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (-6.62861420117874e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (-3.328139350603475e-07) [Y1 X2 X3 Z4 Z5 Y6]
+ (-3.328139350603475e-07) [X1 Y2 Y3 Z4 Z5 X6]
+ (3.328139350603475e-07) [Y1 Y2 X3 Z4 Z5 X6]
+ (3.328139350603475e-07) [X1 X2 Y3 Z4 Z5 Y6]
+ (6.62861420117874e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (6.62861420117874e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (1.035847760179133e-06) [Y0 X1 X4 Z5 Z6 Y7]
+ (1.035847760179133e-06) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.313145499938609e-06) [Y1 Z2 Z3 X4 X5 Y6]
+ (3.313145499938609e-06) [X1 Z2 Z3 Y4 Y5 X6]
+ (4.183932559074278e-06) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (4.183932559074278e-06) [X1 Z2 Z3 Z4 Z5 X7]
+ (0.00029219862611102313) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (0.00029219862611102313) [Y0 Z1 Y2 X3 Z4 X5]
+ (0.00029219862611102313) [X0 Z1 X2 Y3 Z4 Y5]
+ (0.00029219862611102313) [X0 Z1 X2 X3 Z4 X5]
+ (0.010540425907671482) [Y0 Z1 Z2 Z3 Y4 Z7]
+ (0.010540425907671482) [X0 Z1 Z2 Z3 X4 Z7]
+ (0.010540425907671482) [Y1 Z2 Z3 Z4 Y5 Z6]
+ (0.010540425907671482) [X1 Z2 Z3 Z4 X5 Z6]
+ (0.011307274008848206) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (0.011307274008848206) [X0 Z1 Z2 Z3 X4 Z5]
+ (0.025104957138844503) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (0.025104957138844503) [X0 Z1 Z2 Z3 X4 Z6]
+ (0.025104957138844503) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (0.025104957138844503) [X1 Z2 Z3 Z4 X5 Z7]
+ (0.030787505389143953) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (0.030787505389143953) [Z0 X1 Z2 Z3 Z4 X5]
+ (-5.1053965491472875e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (-5.1053965491472875e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (-5.10539654914728e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-5.10539654914728e-06) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.014564531231173026) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-0.014564531231173026) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-3.7696594514574885e-06) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-3.7696594514574885e-06) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-3.328139350603475e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-3.328139350603475e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-3.328139350603475e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-3.328139350603475e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (3.313145499938609e-06) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (3.313145499938609e-06) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (3.313145499938609e-06) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (3.313145499938609e-06) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (4.183932559074278e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (4.183932559074278e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (0.014564531231173026) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (0.014564531231173026) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
 </code>
 </pre>
 </details>

---

## 14. tutorial_falqon.html <a name="demo13"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_falqon.html):

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
Step 28, Cost = -7.123480825409055
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

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_falqon.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Step 1, Cost = -2.426543619778344
Step 2, Cost = -5.451838418111179
Step 3, Cost = -5.058939064534098
Step 4, Cost = 0.6663779891077335
Step 5, Cost = -3.961765919151038
Step 6, Cost = -6.012336027057515
Step 7, Cost = -6.383828240291048
Step 8, Cost = -6.568581722318148
Step 9, Cost = -6.65276742671039
Step 10, Cost = -6.718062615729156
Step 11, Cost = -6.763947743609314
Step 12, Cost = -6.804857466609755
Step 13, Cost = -6.839403058736208
Step 14, Cost = -6.8714592635528735
Step 15, Cost = -6.899746975480975
Step 16, Cost = -6.925884328592675
Step 17, Cost = -6.949229507885615
Step 18, Cost = -6.970594125057237
Step 19, Cost = -6.9899073299213175
Step 20, Cost = -7.007623105822653
Step 21, Cost = -7.023986049880349
Step 22, Cost = -7.039304856521966
Step 23, Cost = -7.053894937286057
Step 24, Cost = -7.067988454154538
Step 25, Cost = -7.081842534715249
Step 26, Cost = -7.095617260802669
Step 27, Cost = -7.109472588274388
Step 28, Cost = -7.123480825409052
Step 29, Cost = -7.137684426026866
Step 30, Cost = -7.152041022693099
Step 31, Cost = -7.166453310287264
Step 32, Cost = -7.180748341609386
Step 33, Cost = -7.194694917926686
Step 34, Cost = -7.208028603663323
Step 35, Cost = -7.2204563958703405
Step 36, Cost = -7.231727330032183
Step 37, Cost = -7.241565955502999
Step 38, Cost = -7.249767410209223
Step 39, Cost = -7.255782895664833
Step 40, Cost = -7.258987907014051
 </code>
 </pre>
 </details>

---

## 15. tutorial_general_parshift.html <a name="demo14"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_general_parshift.html):

```
Second-order finite difference:    [ 0.26814   1.696854 -2.055918 -7.236953]
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_general_parshift.html):

```
Second-order finite difference:    [ 0.268141  1.696854 -2.055918 -7.236953]
```

---

## 16. tutorial_qnn_module_tf.html <a name="demo15"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_qnn_module_tf.html):

```
30/30 - 16s - loss: 0.4997 - accuracy: 0.5000 - val_loss: 0.5081 - val_accuracy: 0.4400
30/30 - 16s - loss: 0.1800 - accuracy: 0.8933 - val_loss: 0.1841 - val_accuracy: 0.8400
30/30 - 16s - loss: 0.1593 - accuracy: 0.8667 - val_loss: 0.2177 - val_accuracy: 0.8400
30/30 - 32s - loss: 0.5189 - accuracy: 0.4000 - val_loss: 0.4945 - val_accuracy: 0.5400
30/30 - 33s - loss: 0.4822 - accuracy: 0.6200 - val_loss: 0.4412 - val_accuracy: 0.7200
30/30 - 33s - loss: 0.2720 - accuracy: 0.7867 - val_loss: 0.2185 - val_accuracy: 0.8200
30/30 - 33s - loss: 0.2056 - accuracy: 0.8400 - val_loss: 0.1893 - val_accuracy: 0.8400
30/30 - 33s - loss: 0.1753 - accuracy: 0.8400 - val_loss: 0.1856 - val_accuracy: 0.8400
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_qnn_module_tf.html):

```
30/30 - 17s - loss: 0.4997 - accuracy: 0.5000 - val_loss: 0.5081 - val_accuracy: 0.4400
30/30 - 17s - loss: 0.1800 - accuracy: 0.8933 - val_loss: 0.1841 - val_accuracy: 0.8400
30/30 - 17s - loss: 0.1593 - accuracy: 0.8667 - val_loss: 0.2177 - val_accuracy: 0.8400
30/30 - 34s - loss: 0.5189 - accuracy: 0.4000 - val_loss: 0.4945 - val_accuracy: 0.5400
30/30 - 34s - loss: 0.4822 - accuracy: 0.6200 - val_loss: 0.4412 - val_accuracy: 0.7200
30/30 - 34s - loss: 0.2720 - accuracy: 0.7867 - val_loss: 0.2185 - val_accuracy: 0.8200
30/30 - 34s - loss: 0.2056 - accuracy: 0.8400 - val_loss: 0.1893 - val_accuracy: 0.8400
30/30 - 34s - loss: 0.1753 - accuracy: 0.8400 - val_loss: 0.1856 - val_accuracy: 0.8400
```

---

## 17. tutorial_vqt.html <a name="demo16"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_vqt.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Cost at Step 0: -0.660535466652201
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
Cost at Step 1450: -14.375809556816055
Cost at Step 1500: -14.3895964168573
Cost at Step 1550: -14.503528638687158
Trace Distance: 0.09990470891807307
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
Cost at Step 0: -0.6605354666522012
Cost at Step 100: -4.64206718424408
Cost at Step 150: -5.127022428216522
Cost at Step 200: -6.529247997026342
Cost at Step 250: -7.072845531891933
Cost at Step 300: -8.118130270464723
Cost at Step 350: -8.990671111922008
Cost at Step 400: -10.626718967317709
Cost at Step 450: -10.932125299927723
Cost at Step 500: -11.322301913287113
Cost at Step 550: -11.356649170752883
Cost at Step 600: -12.034663278412415
Cost at Step 650: -12.027498428513415
Cost at Step 700: -12.385435567318007
Cost at Step 750: -13.04826452228186
Cost at Step 800: -13.145350801891551
Cost at Step 850: -13.309835521794035
Cost at Step 900: -13.625405779370135
Cost at Step 950: -13.837472564507221
Cost at Step 1000: -13.988625999650086
Cost at Step 1050: -14.03023532339978
Cost at Step 1100: -14.113337091247931
Cost at Step 1150: -14.161607053623618
Cost at Step 1200: -14.291333053815237
Cost at Step 1250: -14.396762677838964
Cost at Step 1300: -14.494544274363772
Cost at Step 1350: -14.589884323874257
Cost at Step 1400: -14.644291952582286
Cost at Step 1450: -14.685816456712367
Cost at Step 1500: -14.767916642748473
Cost at Step 1550: -14.768865972303828
Trace Distance: 0.08820083754276989
 </code>
 </pre>
 </details>

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
   (-46.46390678868895) [I0]
+ (0.7829661725950193) [Z10]
+ (0.7829661725950199) [Z11]
+ (0.8084581961720488) [Z13]
+ (0.808458196172049) [Z12]
+ (1.2034402289145603) [Z4]
+ (1.203440228914561) [Z5]
+ (1.3096862988615436) [Z6]
+ (1.3096862988615439) [Z7]
+ (1.3693525634718162) [Z8]
+ (1.3693525634718167) [Z9]
+ (1.6538942226831712) [Z2]
+ (1.6538942226831714) [Z3]
+ (12.41263074211178) [Z0]
+ (12.41263074211178) [Z1]
+ (-8.194261372883674e-06) [Y10 Y12]
+ (-8.194261372883674e-06) [X10 X12]
+ (-1.8540608579868293e-06) [Y5 Y7]
+ (-1.8540608579868293e-06) [X5 X7]
+ (-7.764994118844025e-07) [Y3 Y5]
+ (-7.764994118844025e-07) [X3 X5]
+ (-5.929765816267435e-07) [Y4 Y6]
+ (-5.929765816267435e-07) [X4 X6]
+ (1.6021167405450306e-06) [Y2 Y4]
+ (1.6021167405450306e-06) [X2 X4]
+ (7.954413176950966e-06) [Y11 Y13]
+ (7.954413176950966e-06) [X11 X13]
+ (0.0032769719312315602) [Y1 Y3]
+ (0.0032769719312315602) [X1 X3]
+ (0.1043306478065137) [Y0 Y2]
+ (0.1043306478065137) [X0 X2]
+ (0.11270386920332234) [Z10 Z12]
+ (0.11270386920332234) [Z11 Z13]
+ (0.11383573679388664) [Z4 Z12]
+ (0.11383573679388664) [Z5 Z13]
+ (0.11952438964682667) [Z6 Z10]
+ (0.11952438964682667) [Z7 Z11]
+ (0.12489990917237592) [Z4 Z10]
+ (0.12489990917237592) [Z5 Z11]
+ (0.1249580773950319) [Z2 Z4]
+ (0.1249580773950319) [Z3 Z5]
+ (0.12799502492468395) [Z2 Z10]
+ (0.12799502492468395) [Z3 Z11]
+ (0.13401715261963715) [Z6 Z12]
+ (0.13401715261963715) [Z7 Z13]
+ (0.13701191674040725) [Z4 Z6]
+ (0.13701191674040725) [Z5 Z7]
+ (0.13734953064261313) [Z6 Z11]
+ (0.13734953064261313) [Z7 Z10]
+ (0.13739104762683194) [Z2 Z6]
+ (0.13739104762683194) [Z3 Z7]
+ (0.1376687264585258) [Z8 Z10]
+ (0.1376687264585258) [Z9 Z11]
+ (0.14011289865354812) [Z2 Z12]
+ (0.14011289865354812) [Z3 Z13]
+ (0.14138905291942816) [Z10 Z13]
+ (0.14138905291942816) [Z11 Z12]
+ (0.14257997712485748) [Z4 Z11]
+ (0.14257997712485748) [Z5 Z10]
+ (0.14722943218766177) [Z8 Z11]
+ (0.14722943218766177) [Z9 Z10]
+ (0.1489943057506552) [Z4 Z7]
+ (0.1489943057506552) [Z5 Z6]
+ (0.14926355147388898) [Z10 Z11]
+ (0.14960702684445282) [Z4 Z8]
+ (0.14960702684445282) [Z5 Z9]
+ (0.14973486803496946) [Z8 Z12]
+ (0.14973486803496946) [Z9 Z13]
+ (0.1507140812100827) [Z2 Z8]
+ (0.1507140812100827) [Z3 Z9]
+ (0.15138327161428855) [Z6 Z13]
+ (0.15138327161428855) [Z7 Z12]
+ (0.15215040708869054) [Z4 Z13]
+ (0.15215040708869054) [Z5 Z12]
+ (0.15337968243314137) [Z2 Z11]
+ (0.15337968243314137) [Z3 Z10]
+ (0.1543574865722366) [Z12 Z13]
+ (0.15569010671752465) [Z2 Z13]
+ (0.15569010671752465) [Z3 Z12]
+ (0.15582269051553138) [Z8 Z13]
+ (0.15582269051553138) [Z9 Z12]
+ (0.15676396176430973) [Z4 Z9]
+ (0.15676396176430973) [Z5 Z8]
+ (0.15755314797985645) [Z4 Z5]
+ (0.1607976453483853) [Z2 Z5]
+ (0.1607976453483853) [Z3 Z4]
+ (0.16756653265461252) [Z6 Z8]
+ (0.16756653265461252) [Z7 Z9]
+ (0.16853486561579906) [Z2 Z7]
+ (0.16853486561579906) [Z3 Z6]
+ (0.18143991440303864) [Z6 Z9]
+ (0.18143991440303864) [Z7 Z8]
+ (0.18189085790751322) [Z2 Z3]
+ (0.18690820476912526) [Z2 Z9]
+ (0.18690820476912526) [Z3 Z8]
+ (0.19299723935364263) [Z0 Z10]
+ (0.19299723935364263) [Z1 Z11]
+ (0.1939253461327017) [Z6 Z7]
+ (0.19661770890342126) [Z0 Z4]
+ (0.19661770890342126) [Z1 Z5]
+ (0.19936354537360806) [Z0 Z5]
+ (0.19936354537360806) [Z1 Z4]
+ (0.2007286646044179) [Z0 Z11]
+ (0.2007286646044179) [Z1 Z10]
+ (0.2110265984979158) [Z0 Z12]
+ (0.2110265984979158) [Z1 Z13]
+ (0.2163103749863188) [Z0 Z13]
+ (0.2163103749863188) [Z1 Z12]
+ (0.2200397733437609) [Z8 Z9]
+ (0.23671080783830414) [Z0 Z2]
+ (0.23671080783830414) [Z1 Z3]
+ (0.24164663936017208) [Z0 Z6]
+ (0.24164663936017208) [Z1 Z7]
+ (0.2512944567459167) [Z0 Z3]
+ (0.2512944567459167) [Z1 Z2]
+ (0.27232518306605713) [Z0 Z8]
+ (0.27232518306605713) [Z1 Z9]
+ (0.2788345442672344) [Z0 Z9]
+ (0.2788345442672344) [Z1 Z8]
+ (1.1861763734860522) [Z0 Z1]
+ (-1.2260484989144468e-05) [Y4 Z5 Y6]
+ (-1.2260484989144468e-05) [X4 Z5 X6]
+ (-1.226048498914446e-05) [Y5 Z6 Y7]
+ (-1.226048498914446e-05) [X5 Z6 X7]
+ (-1.0722312157054071e-05) [Y10 Z11 Y12]
+ (-1.0722312157054071e-05) [X10 Z11 X12]
+ (-1.0722312157054066e-05) [Y11 Z12 Y13]
+ (-1.0722312157054066e-05) [X11 Z12 X13]
+ (-3.887051673683753e-06) [Y2 Z3 Y4]
+ (-3.887051673683753e-06) [X2 Z3 X4]
+ (-3.887051673683753e-06) [Y3 Z4 Y5]
+ (-3.887051673683753e-06) [X3 Z4 X5]
+ (0.12507032579771582) [Y0 Z1 Y2]
+ (0.12507032579771582) [X0 Z1 X2]
+ (0.1250703257977159) [Y1 Z2 Y3]
+ (0.1250703257977159) [X1 Z2 X3]
+ (-0.03831467029480389) [Y4 Y5 X12 X13]
+ (-0.03831467029480389) [X4 X5 Y12 Y13]
+ (-0.036194123559042564) [Y2 Y3 X8 X9]
+ (-0.036194123559042564) [X2 X3 Y8 Y9]
+ (-0.03583956795335341) [Y2 Y3 X4 X5]
+ (-0.03583956795335341) [X2 X3 Y4 Y5]
+ (-0.031143817988967086) [Y2 Y3 X6 X7]
+ (-0.031143817988967086) [X2 X3 Y6 Y7]
+ (-0.028685183716105837) [Y10 Y11 X12 X13]
+ (-0.028685183716105837) [X10 X11 Y12 Y13]
+ (-0.02599617759802115) [Y3 Z4 Z5 Y7]
+ (-0.02599617759802115) [X3 Z4 Z5 X7]
+ (-0.025384657508457396) [Y2 Y3 X10 X11]
+ (-0.025384657508457396) [X2 X3 Y10 Y11]
+ (-0.01902824244384729) [Y3 Y4 X11 X12]
+ (-0.01902824244384729) [X3 X4 Y11 Y12]
+ (-0.017825140995786453) [Y6 Y7 X10 X11]
+ (-0.017825140995786453) [X6 X7 Y10 Y11]
+ (-0.01768006795248156) [Y4 Y5 X10 X11]
+ (-0.01768006795248156) [X4 X5 Y10 Y11]
+ (-0.017366118994651403) [Y6 Y7 X12 X13]
+ (-0.017366118994651403) [X6 X7 Y12 Y13]
+ (-0.015577208063976503) [Y2 Y3 X12 X13]
+ (-0.015577208063976503) [X2 X3 Y12 Y13]
+ (-0.014583648907612577) [Y0 Y1 X2 X3]
+ (-0.014583648907612577) [X0 X1 Y2 Y3]
+ (-0.0138733817484261) [Y6 Y7 X8 X9]
+ (-0.0138733817484261) [X6 X7 Y8 Y9]
+ (-0.01198238901024795) [Y4 Y5 X6 X7]
+ (-0.01198238901024795) [X4 X5 Y6 Y7]
+ (-0.011285190200840909) [Y5 X6 X11 Y12]
+ (-0.011285190200840909) [X5 Y6 Y11 X12]
+ (-0.009560705729135961) [Y8 Y9 X10 X11]
+ (-0.009560705729135961) [X8 X9 Y10 Y11]
+ (-0.008125251921381025) [Y1 X2 X8 Y9]
+ (-0.008125251921381025) [Y1 Y2 Y8 Y9]
+ (-0.008125251921381025) [X1 X2 X8 X9]
+ (-0.008125251921381025) [X1 Y2 Y8 X9]
+ (-0.007731425250775288) [Y0 Y1 X10 X11]
+ (-0.007731425250775288) [X0 X1 Y10 Y11]
+ (-0.007156934919856918) [Y4 Y5 X8 X9]
+ (-0.007156934919856918) [X4 X5 Y8 Y9]
+ (-0.0068881943529705645) [Y0 Y1 X6 X7]
+ (-0.0068881943529705645) [X0 X1 Y6 Y7]
+ (-0.006509361201177241) [Y0 Y1 X8 X9]
+ (-0.006509361201177241) [X0 X1 Y8 Y9]
+ (-0.0060878224805618825) [Y8 Y9 X12 X13]
+ (-0.0060878224805618825) [X8 X9 Y12 Y13]
+ (-0.005283776488402981) [Y0 Y1 X12 X13]
+ (-0.005283776488402981) [X0 X1 Y12 Y13]
+ (-0.005143391768825178) [Y3 X4 X5 Y6]
+ (-0.005143391768825178) [X3 Y4 Y5 X6]
+ (-0.0046849033881552) [Y1 X2 X6 Y7]
+ (-0.0046849033881552) [Y1 Y2 Y6 Y7]
+ (-0.0046849033881552) [X1 X2 X6 X7]
+ (-0.0046849033881552) [X1 Y2 Y6 X7]
+ (-0.004575007626639222) [Y1 X2 X12 Y13]
+ (-0.004575007626639222) [Y1 Y2 Y12 Y13]
+ (-0.004575007626639222) [X1 X2 X12 X13]
+ (-0.004575007626639222) [X1 Y2 Y12 X13]
+ (-0.0044248554494418346) [Y1 X2 X4 Y5]
+ (-0.0044248554494418346) [Y1 Y2 Y4 Y5]
+ (-0.0044248554494418346) [X1 X2 X4 X5]
+ (-0.0044248554494418346) [X1 Y2 Y4 X5]
+ (-0.0034795118903343004) [Y2 Z3 Z5 Y6]
+ (-0.0034795118903343004) [X2 Z3 Z5 X6]
+ (-0.0034795118903343004) [Y3 Z4 Z6 Y7]
+ (-0.0034795118903343004) [X3 Z4 Z6 X7]
+ (-0.002745836470186797) [Y0 Y1 X4 X5]
+ (-0.002745836470186797) [X0 X1 Y4 Y5]
+ (-0.0017992194936630311) [Y1 X2 X10 Y11]
+ (-0.0017992194936630311) [Y1 Y2 Y10 Y11]
+ (-0.0017992194936630311) [X1 X2 X10 X11]
+ (-0.0017992194936630311) [X1 Y2 Y10 X11]
+ (-0.0002921986261110668) [Y7 Y8 X9 X10]
+ (-0.0002921986261110668) [X7 X8 Y9 Y10]
+ (-8.194261372883674e-06) [Z10 Y11 Z12 Y13]
+ (-8.194261372883674e-06) [Z10 X11 Z12 X13]
+ (-7.801707501245015e-06) [Y2 Z3 Y4 Z11]
+ (-7.801707501245015e-06) [X2 Z3 X4 Z11]
+ (-7.801707501245015e-06) [Y3 Z4 Y5 Z10]
+ (-7.801707501245015e-06) [X3 Z4 X5 Z10]
+ (-4.643051068868427e-06) [Y3 X4 X10 Y11]
+ (-4.643051068868427e-06) [Y3 Y4 Y10 Y11]
+ (-4.643051068868427e-06) [X3 X4 X10 X11]
+ (-4.643051068868427e-06) [X3 Y4 Y10 X11]
+ (-4.588855156052471e-06) [Y4 Z5 Y6 Z13]
+ (-4.588855156052471e-06) [X4 Z5 X6 Z13]
+ (-4.588855156052471e-06) [Y5 Z6 Y7 Z12]
+ (-4.588855156052471e-06) [X5 Z6 X7 Z12]
+ (-4.556569218584335e-06) [Y5 X6 X12 Y13]
+ (-4.556569218584335e-06) [Y5 Y6 Y12 Y13]
+ (-4.556569218584335e-06) [X5 X6 X12 X13]
+ (-4.556569218584335e-06) [X5 Y6 Y12 X13]
+ (-3.69451329476438e-06) [Y4 X5 X11 Y12]
+ (-3.69451329476438e-06) [Y4 Y5 Y11 Y12]
+ (-3.69451329476438e-06) [X4 X5 X11 X12]
+ (-3.69451329476438e-06) [X4 Y5 Y11 X12]
+ (-3.344081556519935e-06) [Z0 Y5 Z6 Y7]
+ (-3.344081556519935e-06) [Z0 X5 Z6 X7]
+ (-3.344081556519935e-06) [Z1 Y4 Z5 Y6]
+ (-3.344081556519935e-06) [Z1 X4 Z5 X6]
+ (-3.158656432376588e-06) [Y2 Z3 Y4 Z10]
+ (-3.158656432376588e-06) [X2 Z3 X4 Z10]
+ (-3.158656432376588e-06) [Y3 Z4 Y5 Z11]
+ (-3.158656432376588e-06) [X3 Z4 X5 Z11]
+ (-3.0993492436377626e-06) [Z0 Y4 Z5 Y6]
+ (-3.0993492436377626e-06) [Z0 X4 Z5 X6]
+ (-3.0993492436377626e-06) [Z1 Y5 Z6 Y7]
+ (-3.0993492436377626e-06) [Z1 X5 Z6 X7]
+ (-2.8909678818060245e-06) [Z6 Y11 Z12 Y13]
+ (-2.8909678818060245e-06) [Z6 X11 Z12 X13]
+ (-2.8909678818060245e-06) [Z7 Y10 Z11 Y12]
+ (-2.8909678818060245e-06) [Z7 X10 Z11 X12]
+ (-2.1776646051087113e-06) [Z0 Y10 Z11 Y12]
+ (-2.1776646051087113e-06) [Z0 X10 Z11 X12]
+ (-2.1776646051087113e-06) [Z1 Y11 Z12 Y13]
+ (-2.1776646051087113e-06) [Z1 X11 Z12 X13]
+ (-1.8818501832449744e-06) [Y4 Z5 Y6 Z9]
+ (-1.8818501832449744e-06) [X4 Z5 X6 Z9]
+ (-1.8818501832449744e-06) [Y5 Z6 Y7 Z8]
+ (-1.8818501832449744e-06) [X5 Z6 X7 Z8]
+ (-1.855120121668481e-06) [Z6 Y10 Z11 Y12]
+ (-1.855120121668481e-06) [Z6 X10 Z11 X12]
+ (-1.855120121668481e-06) [Z7 Y11 Z12 Y13]
+ (-1.855120121668481e-06) [Z7 X11 Z12 X13]
+ (-1.8540608579868293e-06) [Y4 Z5 Y6 Z7]
+ (-1.8540608579868293e-06) [X4 Z5 X6 Z7]
+ (-1.8163031697705097e-06) [Z4 Y11 Z12 Y13]
+ (-1.8163031697705097e-06) [Z4 X11 Z12 X13]
+ (-1.8163031697705097e-06) [Z5 Y10 Z11 Y12]
+ (-1.8163031697705097e-06) [Z5 X10 Z11 X12]
+ (-1.692397828701653e-06) [Y4 Z5 Y6 Z10]
+ (-1.692397828701653e-06) [X4 Z5 X6 Z10]
+ (-1.692397828701653e-06) [Y5 Z6 Y7 Z11]
+ (-1.692397828701653e-06) [X5 Z6 X7 Z11]
+ (-1.6148794139210583e-06) [Z0 Y11 Z12 Y13]
+ (-1.6148794139210583e-06) [Z0 X11 Z12 X13]
+ (-1.6148794139210583e-06) [Z1 Y10 Z11 Y12]
+ (-1.6148794139210583e-06) [Z1 X10 Z11 X12]
+ (-1.597317197908079e-06) [Z8 Y10 Z11 Y12]
+ (-1.597317197908079e-06) [Z8 X10 Z11 X12]
+ (-1.597317197908079e-06) [Z9 Y11 Z12 Y13]
+ (-1.597317197908079e-06) [Z9 X11 Z12 X13]
+ (-1.4548424490507874e-06) [Y3 X4 X6 Y7]
+ (-1.4548424490507874e-06) [Y3 Y4 Y6 Y7]
+ (-1.4548424490507874e-06) [X3 X4 X6 X7]
+ (-1.4548424490507874e-06) [X3 Y4 Y6 X7]
+ (-1.3980449081554113e-06) [Y4 Z5 Y6 Z8]
+ (-1.3980449081554113e-06) [X4 Z5 X6 Z8]
+ (-1.3980449081554113e-06) [Y5 Z6 Y7 Z9]
+ (-1.3980449081554113e-06) [X5 Z6 X7 Z9]
+ (-1.1954890100123857e-06) [Y2 Z3 Y4 Z7]
+ (-1.1954890100123857e-06) [X2 Z3 X4 Z7]
+ (-1.1954890100123857e-06) [Y3 Z4 Y5 Z6]
+ (-1.1954890100123857e-06) [X3 Z4 X5 Z6]
+ (-1.190850808483161e-06) [Z0 Y3 Z4 Y5]
+ (-1.190850808483161e-06) [Z0 X3 Z4 X5]
+ (-1.190850808483161e-06) [Z1 Y2 Z3 Y4]
+ (-1.190850808483161e-06) [Z1 X2 Z3 X4]
+ (-1.1708301370692143e-06) [Z2 Y5 Z6 Y7]
+ (-1.1708301370692143e-06) [Z2 X5 Z6 X7]
+ (-1.1708301370692143e-06) [Z3 Y4 Z5 Y6]
+ (-1.1708301370692143e-06) [Z3 X4 Z5 X6]
+ (-1.063228342470897e-06) [Z2 Y10 Z11 Y12]
+ (-1.063228342470897e-06) [Z2 X10 Z11 X12]
+ (-1.063228342470897e-06) [Z3 Y11 Z12 Y13]
+ (-1.063228342470897e-06) [Z3 X11 Z12 X13]
+ (-1.0358477601375435e-06) [Y6 X7 X11 Y12]
+ (-1.0358477601375435e-06) [Y6 Y7 Y11 Y12]
+ (-1.0358477601375435e-06) [X6 X7 X11 X12]
+ (-1.0358477601375435e-06) [X6 Y7 Y11 X12]
+ (-9.509249751675222e-07) [Z2 Y4 Z5 Y6]
+ (-9.509249751675222e-07) [Z2 X4 Z5 X6]
+ (-9.509249751675222e-07) [Z3 Y5 Z6 Y7]
+ (-9.509249751675222e-07) [Z3 X5 Z6 X7]
+ (-9.344557776705624e-07) [Z8 Y11 Z12 Y13]
+ (-9.344557776705624e-07) [Z8 X11 Z12 X13]
+ (-9.344557776705624e-07) [Z9 Y10 Z11 Y12]
+ (-9.344557776705624e-07) [Z9 X10 Z11 X12]
+ (-8.337746755638611e-07) [Z0 Y2 Z3 Y4]
+ (-8.337746755638611e-07) [Z0 X2 Z3 X4]
+ (-8.337746755638611e-07) [Z1 Y3 Z4 Y5]
+ (-8.337746755638611e-07) [Z1 X3 Z4 X5]
+ (-7.956895372600225e-07) [Y3 X4 X8 Y9]
+ (-7.956895372600225e-07) [Y3 Y4 Y8 Y9]
+ (-7.956895372600225e-07) [X3 X4 X8 X9]
+ (-7.956895372600225e-07) [X3 Y4 Y8 X9]
+ (-7.764994118844025e-07) [Y2 Z3 Y4 Z5]
+ (-7.764994118844025e-07) [X2 Z3 X4 Z5]
+ (-5.929765816267435e-07) [Z4 Y5 Z6 Y7]
+ (-5.929765816267435e-07) [Z4 X5 Z6 X7]
+ (-5.770052995668664e-07) [Y2 Z3 Y4 Z9]
+ (-5.770052995668664e-07) [X2 Z3 X4 Z9]
+ (-5.770052995668664e-07) [Y3 Z4 Y5 Z8]
+ (-5.770052995668664e-07) [X3 Z4 X5 Z8]
+ (-5.471647744986291e-07) [Y1 Y2 X11 X12]
+ (-5.471647744986291e-07) [X1 X2 Y11 Y12]
+ (-4.83805275089563e-07) [Y5 X6 X8 Y9]
+ (-4.83805275089563e-07) [Y5 Y6 Y8 Y9]
+ (-4.83805275089563e-07) [X5 X6 X8 X9]
+ (-4.83805275089563e-07) [X5 Y6 Y8 X9]
+ (-3.570761329192996e-07) [Y0 X1 X3 Y4]
+ (-3.570761329192996e-07) [Y0 Y1 Y3 Y4]
+ (-3.570761329192996e-07) [X0 X1 X3 X4]
+ (-3.570761329192996e-07) [X0 Y1 Y3 X4]
+ (-2.447323128821725e-07) [Y0 X1 X5 Y6]
+ (-2.447323128821725e-07) [Y0 Y1 Y5 Y6]
+ (-2.447323128821725e-07) [X0 X1 X5 X6]
+ (-2.447323128821725e-07) [X0 Y1 Y5 X6]
+ (-2.1990516190169207e-07) [Y2 X3 X5 Y6]
+ (-2.1990516190169207e-07) [Y2 Y3 Y5 Y6]
+ (-2.1990516190169207e-07) [X2 X3 X5 X6]
+ (-2.1990516190169207e-07) [X2 Y3 Y5 X6]
+ (-1.9332412770511448e-07) [Y1 X2 X3 Y4]
+ (-1.9332412770511448e-07) [X1 Y2 Y3 X4]
+ (-1.2919694861428975e-07) [Y1 Z2 Z3 Y5]
+ (-1.2919694861428975e-07) [X1 Z2 Z3 X5]
+ (1.737933262424616e-07) [Y0 Z1 Z3 Y4]
+ (1.737933262424616e-07) [X0 Z1 Z3 X4]
+ (1.737933262424616e-07) [Y1 Z2 Z4 Y5]
+ (1.737933262424616e-07) [X1 Z2 Z4 X5]
+ (1.9332412770511448e-07) [Y1 Y2 X3 X4]
+ (1.9332412770511448e-07) [X1 X2 Y3 Y4]
+ (2.186842376931562e-07) [Y2 Z3 Y4 Z8]
+ (2.186842376931562e-07) [X2 Z3 X4 Z8]
+ (2.186842376931562e-07) [Y3 Z4 Y5 Z9]
+ (2.186842376931562e-07) [X3 Z4 X5 Z9]
+ (2.593534390384019e-07) [Y2 Z3 Y4 Z6]
+ (2.593534390384019e-07) [X2 Z3 X4 Z6]
+ (2.593534390384019e-07) [Y3 Z4 Y5 Z7]
+ (2.593534390384019e-07) [X3 Z4 X5 Z7]
+ (3.6060718679329104e-07) [Y0 Z1 Z2 Y4]
+ (3.6060718679329104e-07) [X0 Z1 Z2 X4]
+ (3.6060718679329104e-07) [Y1 Z3 Z4 Y5]
+ (3.6060718679329104e-07) [X1 Z3 Z4 X5]
+ (5.471647744986291e-07) [Y1 X2 X11 Y12]
+ (5.471647744986291e-07) [X1 Y2 Y11 X12]
+ (5.627851911876528e-07) [Y0 X1 X11 Y12]
+ (5.627851911876528e-07) [Y0 Y1 Y11 Y12]
+ (5.627851911876528e-07) [X0 X1 X11 X12]
+ (5.627851911876528e-07) [X0 Y1 Y11 X12]
+ (6.628614202375168e-07) [Y8 X9 X11 Y12]
+ (6.628614202375168e-07) [Y8 Y9 Y11 Y12]
+ (6.628614202375168e-07) [X8 X9 X11 X12]
+ (6.628614202375168e-07) [X8 Y9 Y11 X12]
+ (1.1094407591958404e-06) [Z2 Y11 Z12 Y13]
+ (1.1094407591958404e-06) [Z2 X11 Z12 X13]
+ (1.1094407591958404e-06) [Z3 Y10 Z11 Y12]
+ (1.1094407591958404e-06) [Z3 X10 Z11 X12]
+ (1.6021167405450308e-06) [Z2 Y3 Z4 Y5]
+ (1.6021167405450308e-06) [Z2 X3 Z4 X5]
+ (1.8782101249938703e-06) [Z4 Y10 Z11 Y12]
+ (1.8782101249938703e-06) [Z4 X10 Z11 X12]
+ (1.8782101249938703e-06) [Z5 Y11 Z12 Y13]
+ (1.8782101249938703e-06) [Z5 X11 Z12 X13]
+ (2.1726691016667378e-06) [Y2 X3 X11 Y12]
+ (2.1726691016667378e-06) [Y2 Y3 Y11 Y12]
+ (2.1726691016667378e-06) [X2 X3 X11 X12]
+ (2.1726691016667378e-06) [X2 Y3 Y11 X12]
+ (3.117447946125198e-06) [Y0 Z2 Z3 Y4]
+ (3.117447946125198e-06) [X0 Z2 Z3 X4]
+ (3.539054184785432e-06) [Y2 Z3 Y4 Z12]
+ (3.539054184785432e-06) [X2 Z3 X4 Z12]
+ (3.539054184785432e-06) [Y3 Z4 Y5 Z13]
+ (3.539054184785432e-06) [X3 Z4 X5 Z13]
+ (4.281913885177783e-06) [Y4 Z5 Y6 Z11]
+ (4.281913885177783e-06) [X4 Z5 X6 Z11]
+ (4.281913885177783e-06) [Y5 Z6 Y7 Z10]
+ (4.281913885177783e-06) [X5 Z6 X7 Z10]
+ (5.275883122558622e-06) [Y3 X4 X12 Y13]
+ (5.275883122558622e-06) [Y3 Y4 Y12 Y13]
+ (5.275883122558622e-06) [X3 X4 X12 X13]
+ (5.275883122558622e-06) [X3 Y4 Y12 X13]
+ (5.9743117138794355e-06) [Y5 X6 X10 Y11]
+ (5.9743117138794355e-06) [Y5 Y6 Y10 Y11]
+ (5.9743117138794355e-06) [X5 X6 X10 X11]
+ (5.9743117138794355e-06) [X5 Y6 Y10 X11]
+ (7.954413176950966e-06) [Y10 Z11 Y12 Z13]
+ (7.954413176950966e-06) [X10 Z11 X12 Z13]
+ (8.814937307344054e-06) [Y2 Z3 Y4 Z13]
+ (8.814937307344054e-06) [X2 Z3 X4 Z13]
+ (8.814937307344054e-06) [Y3 Z4 Y5 Z12]
+ (8.814937307344054e-06) [X3 Z4 X5 Z12]
+ (0.0002921986261110668) [Y7 X8 X9 Y10]
+ (0.0002921986261110668) [X7 Y8 Y9 X10]
+ (0.0004956762314916446) [Y2 Z4 Z5 Y6]
+ (0.0004956762314916446) [X2 Z4 Z5 X6]
+ (0.0011059037691896108) [Y0 Z1 Y2 Z5]
+ (0.0011059037691896108) [X0 Z1 X2 Z5]
+ (0.0011059037691896108) [Y1 Z2 Y3 Z4]
+ (0.0011059037691896108) [X1 Z2 X3 Z4]
+ (0.0016638798784908773) [Y2 Z3 Z4 Y6]
+ (0.0016638798784908773) [X2 Z3 Z4 X6]
+ (0.0016638798784908773) [Y3 Z5 Z6 Y7]
+ (0.0016638798784908773) [X3 Z5 Z6 X7]
+ (0.001756070701841169) [Y0 Z1 Y2 Z11]
+ (0.001756070701841169) [X0 Z1 X2 Z11]
+ (0.001756070701841169) [Y1 Z2 Y3 Z10]
+ (0.001756070701841169) [X1 Z2 X3 Z10]
+ (0.0023262306231580142) [Y0 Z1 Y2 Z13]
+ (0.0023262306231580142) [X0 Z1 X2 Z13]
+ (0.0023262306231580142) [Y1 Z2 Y3 Z12]
+ (0.0023262306231580142) [X1 Z2 X3 Z12]
+ (0.002745836470186797) [Y0 X1 X4 Y5]
+ (0.002745836470186797) [X0 Y1 Y4 X5]
+ (0.0029297686747509575) [Y0 Z1 Y2 Z9]
+ (0.0029297686747509575) [X0 Z1 X2 Z9]
+ (0.0029297686747509575) [Y1 Z2 Y3 Z8]
+ (0.0029297686747509575) [X1 Z2 X3 Z8]
+ (0.0032769719312315607) [Y0 Z1 Y2 Z3]
+ (0.0032769719312315607) [X0 Z1 X2 Z3]
+ (0.003347617530666098) [Y0 Z1 Y2 Z7]
+ (0.003347617530666098) [X0 Z1 X2 Z7]
+ (0.003347617530666098) [Y1 Z2 Y3 Z6]
+ (0.003347617530666098) [X1 Z2 X3 Z6]
+ (0.0035552901955041996) [Y0 Z1 Y2 Z10]
+ (0.0035552901955041996) [X0 Z1 X2 Z10]
+ (0.0035552901955041996) [Y1 Z2 Y3 Z11]
+ (0.0035552901955041996) [X1 Z2 X3 Z11]
+ (0.005143391768825178) [Y3 Y4 X5 X6]
+ (0.005143391768825178) [X3 X4 Y5 Y6]
+ (0.005283776488402981) [Y0 X1 X12 Y13]
+ (0.005283776488402981) [X0 Y1 Y12 X13]
+ (0.005530759218631444) [Y0 Z1 Y2 Z4]
+ (0.005530759218631444) [X0 Z1 X2 Z4]
+ (0.005530759218631444) [Y1 Z2 Y3 Z5]
+ (0.005530759218631444) [X1 Z2 X3 Z5]
+ (0.0060878224805618825) [Y8 X9 X12 Y13]
+ (0.0060878224805618825) [X8 Y9 Y12 X13]
+ (0.006509361201177241) [Y0 X1 X8 Y9]
+ (0.006509361201177241) [X0 Y1 Y8 X9]
+ (0.0068881943529705645) [Y0 X1 X6 Y7]
+ (0.0068881943529705645) [X0 Y1 Y6 X7]
+ (0.006901238249797236) [Y0 Z1 Y2 Z12]
+ (0.006901238249797236) [X0 Z1 X2 Z12]
+ (0.006901238249797236) [Y1 Z2 Y3 Z13]
+ (0.006901238249797236) [X1 Z2 X3 Z13]
+ (0.007156934919856918) [Y4 X5 X8 Y9]
+ (0.007156934919856918) [X4 Y5 Y8 X9]
+ (0.007731425250775288) [Y0 X1 X10 Y11]
+ (0.007731425250775288) [X0 Y1 Y10 X11]
+ (0.008032520918821298) [Y0 Z1 Y2 Z6]
+ (0.008032520918821298) [X0 Z1 X2 Z6]
+ (0.008032520918821298) [Y1 Z2 Y3 Z7]
+ (0.008032520918821298) [X1 Z2 X3 Z7]
+ (0.009560705729135961) [Y8 X9 X10 Y11]
+ (0.009560705729135961) [X8 Y9 Y10 X11]
+ (0.01105502059613198) [Y0 Z1 Y2 Z8]
+ (0.01105502059613198) [X0 Z1 X2 Z8]
+ (0.01105502059613198) [Y1 Z2 Y3 Z9]
+ (0.01105502059613198) [X1 Z2 X3 Z9]
+ (0.011285190200840909) [Y5 Y6 X11 X12]
+ (0.011285190200840909) [X5 X6 Y11 Y12]
+ (0.011307274008848192) [Y7 Z8 Z9 Y11]
+ (0.011307274008848192) [X7 Z8 Z9 X11]
+ (0.01198238901024795) [Y4 X5 X6 Y7]
+ (0.01198238901024795) [X4 Y5 Y6 X7]
+ (0.0138733817484261) [Y6 X7 X8 Y9]
+ (0.0138733817484261) [X6 Y7 Y8 X9]
+ (0.014583648907612577) [Y0 X1 X2 Y3]
+ (0.014583648907612577) [X0 Y1 Y2 X3]
+ (0.015577208063976503) [Y2 X3 X12 Y13]
+ (0.015577208063976503) [X2 Y3 Y12 X13]
+ (0.017366118994651403) [Y6 X7 X12 Y13]
+ (0.017366118994651403) [X6 Y7 Y12 X13]
+ (0.01768006795248156) [Y4 X5 X10 Y11]
+ (0.01768006795248156) [X4 Y5 Y10 X11]
+ (0.017825140995786453) [Y6 X7 X10 Y11]
+ (0.017825140995786453) [X6 Y7 Y10 X11]
+ (0.01902824244384729) [Y3 X4 X11 Y12]
+ (0.01902824244384729) [X3 Y4 Y11 X12]
+ (0.025384657508457396) [Y2 X3 X10 Y11]
+ (0.025384657508457396) [X2 Y3 Y10 X11]
+ (0.028685183716105837) [Y10 X11 X12 Y13]
+ (0.028685183716105837) [X10 Y11 Y12 X13]
+ (0.029812424517345747) [Y6 Z7 Z8 Y10]
+ (0.029812424517345747) [X6 Z7 Z8 X10]
+ (0.029812424517345747) [Y7 Z9 Z10 Y11]
+ (0.029812424517345747) [X7 Z9 Z10 X11]
+ (0.030104623143456816) [Y6 Z7 Z9 Y10]
+ (0.030104623143456816) [X6 Z7 Z9 X10]
+ (0.030104623143456816) [Y7 Z8 Z10 Y11]
+ (0.030104623143456816) [X7 Z8 Z10 X11]
+ (0.030787505389143877) [Y6 Z8 Z9 Y10]
+ (0.030787505389143877) [X6 Z8 Z9 X10]
+ (0.031143817988967086) [Y2 X3 X6 Y7]
+ (0.031143817988967086) [X2 Y3 Y6 X7]
+ (0.03583956795335341) [Y2 X3 X4 Y5]
+ (0.03583956795335341) [X2 Y3 Y4 X5]
+ (0.036194123559042564) [Y2 X3 X8 Y9]
+ (0.036194123559042564) [X2 Y3 Y8 X9]
+ (0.03831467029480389) [Y4 X5 X12 Y13]
+ (0.03831467029480389) [X4 Y5 Y12 X13]
+ (0.1043306478065137) [Z0 Y1 Z2 Y3]
+ (0.1043306478065137) [Z0 X1 Z2 X3]
+ (-0.12133276911042361) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042361) [X2 Z3 Z4 Z5 X6]
+ (-0.12133276911042361) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133276911042361) [X3 Z4 Z5 Z6 X7]
+ (3.202076880589368e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.202076880589368e-06) [X0 Z1 Z2 Z3 X4]
+ (3.2020768805893683e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.2020768805893683e-06) [X1 Z2 Z3 Z4 X5]
+ (0.22848106564918808) [Y7 Z8 Z9 Z10 Y11]
+ (0.22848106564918808) [X7 Z8 Z9 Z10 X11]
+ (0.22848106564918813) [Y6 Z7 Z8 Z9 Y10]
+ (0.22848106564918813) [X6 Z7 Z8 Z9 X10]
+ (-0.0327676578232905) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.0327676578232905) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.0327676578232905) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.0327676578232905) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273166) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.027115036845273166) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273166) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.027115036845273166) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.02599617759802115) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.02599617759802115) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.017561202409646155) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.017561202409646155) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.017561202409646155) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.017561202409646155) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.014564531231172956) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564531231172956) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564531231172956) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564531231172956) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.012215040997613957) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.012215040997613957) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.012215040997613957) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.012215040997613957) [X4 Z5 X6 X11 Z12 X13]
+ (-0.012215040997613957) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.012215040997613957) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.012215040997613957) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.012215040997613957) [X5 Z6 X7 X10 Z11 X12]
+ (-0.011756013419819253) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.011756013419819253) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.011756013419819253) [X3 Z4 Z5 X6 X8 X9]
+ (-0.011756013419819253) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.00876482757568877) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.00876482757568877) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.00876482757568877) [X2 Z3 Z4 X5 X11 X12]
+ (-0.00876482757568877) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.00876482757568877) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.00876482757568877) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.00876482757568877) [X3 X4 X10 Z11 Z12 X13]
+ (-0.00876482757568877) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.008125251921381025) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125251921381025) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.007306759928833009) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306759928833009) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306759928833009) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306759928833009) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005805188989826904) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.005805188989826904) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.005805188989826904) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.005805188989826904) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.005652620978017334) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.005652620978017334) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.005652620978017334) [X0 X1 X3 Z4 Z5 X6]
+ (-0.005652620978017334) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.005143391768825178) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.005143391768825178) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.005143391768825178) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.005143391768825178) [X2 Z3 X4 X5 Z6 X7]
+ (-0.0046849033881552005) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.0046849033881552005) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668620318776294) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668620318776294) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.004575007626639222) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.004575007626639222) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.0044248554494418346) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.0044248554494418346) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.0041587973818401) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.0041587973818401) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.0041587973818401) [X3 Z4 Z5 X6 X12 X13]
+ (-0.0041587973818401) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.0034937903598902223) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.0034937903598902223) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.0034937903598902223) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.0034937903598902223) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.0027790267990255493) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.0027790267990255493) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.002293956611352447) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.002293956611352447) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.0017992194936630311) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.0017992194936630311) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.00172787539413695) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.00172787539413695) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298507967730484) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298507967730484) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298507967730484) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298507967730484) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533856254125412) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533856254125412) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008145313270956612) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008145313270956612) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008145313270956612) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008145313270956612) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.735036880590382e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735036880590382e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735036880590382e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735036880590382e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.774817865274418e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.774817865274418e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.774817865274418e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.774817865274418e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.5183622162527614e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.5183622162527614e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.5183622162527614e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.5183622162527614e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.444344676520591e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.444344676520591e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.444344676520591e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.444344676520591e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.524373849190024e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.524373849190024e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.524373849190024e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.524373849190024e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.290028433773122e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.290028433773122e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.290028433773122e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.290028433773122e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.974311713879434e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.974311713879434e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.275883122558622e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.275883122558622e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.643051068868428e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.643051068868428e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.556569218584335e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.556569218584335e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.253224225930398e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.253224225930398e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.769659452415901e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.769659452415901e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.69451329476438e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.69451329476438e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.6102971310189632e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.6102971310189632e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.6102971310189632e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.6102971310189632e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.3131455004142406e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.3131455004142406e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.2774831959432157e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.2774831959432157e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.2774831959432157e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.2774831959432157e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.2112283487757837e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.2112283487757837e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.2112283487757837e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.2112283487757837e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.1513463114178305e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.1513463114178305e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.0882507115986276e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.0882507115986276e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.1726691016667373e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.1726691016667373e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.4548424490507874e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.4548424490507874e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.3304731887538258e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.3304731887538258e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2283337824796397e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2283337824796397e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.0358477601375435e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.0358477601375435e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956895372600226e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956895372600226e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733197743008794e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733197743008794e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733197743008794e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733197743008794e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.628614202375168e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.628614202375168e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.556281914954776e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.556281914954776e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.556281914954776e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.556281914954776e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.418291574998256e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.418291574998256e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.418291574998256e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.418291574998256e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.927453083351202e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927453083351202e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927453083351202e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927453083351202e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627851911876528e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627851911876528e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660625054149e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660625054149e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660625054149e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660625054149e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660625054149e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660625054149e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660625054149e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660625054149e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.83805275089563e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.83805275089563e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.570761329192996e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.570761329192996e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.3281393507574757e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.3281393507574757e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.0868265651927427e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.0868265651927427e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.0868265651927427e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.0868265651927427e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.447323128821725e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.447323128821725e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.371328947903965e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.371328947903965e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.371328947903965e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.371328947903965e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.1990516190169207e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.1990516190169207e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.9332412770511448e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.9332412770511448e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.9332412770511448e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.9332412770511448e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.839420915456305e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.839420915456305e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.839420915456305e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.839420915456305e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.55105391762028e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.55105391762028e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.55105391762028e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.55105391762028e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.380778148044603e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.380778148044603e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.380778148044603e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.380778148044603e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.380778148044603e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.380778148044603e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.380778148044603e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.380778148044603e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.380778148044603e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.380778148044603e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.380778148044603e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.380778148044603e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.2919694861428975e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.2919694861428975e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.1076325599562972e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.1076325599562972e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.1076325599562972e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.1076325599562972e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.1076325599562972e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.1076325599562972e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.1076325599562972e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.1076325599562972e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.057446596575921e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057446596575921e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057446596575921e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057446596575921e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.6493101358287e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.6493101358287e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.6493101358287e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.6493101358287e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.839420915456305e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.839420915456305e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.839420915456305e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.839420915456305e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.1990516190169207e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.1990516190169207e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.447323128821725e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.447323128821725e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.236259961486835e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.236259961486835e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.236259961486835e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.236259961486835e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.3281393507574757e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.3281393507574757e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.570761329192996e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.570761329192996e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.83805275089563e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.83805275089563e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.627851911876528e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627851911876528e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.628614202375168e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.628614202375168e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956895372600226e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956895372600226e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.306536652748122e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.306536652748122e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.306536652748122e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.306536652748122e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.0358477601375435e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.0358477601375435e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2283337824796397e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2283337824796397e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.2393363217940865e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393363217940865e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393363217940865e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393363217940865e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.3304731887538258e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.3304731887538258e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.4548424490507874e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.4548424490507874e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.1726691016667373e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.1726691016667373e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.0882507115986276e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.0882507115986276e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.117447946125198e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.117447946125198e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.1513463114178305e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.1513463114178305e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.3131455004142406e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.3131455004142406e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.334331289759791e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.334331289759791e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.69451329476438e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.69451329476438e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.1839325596675525e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.1839325596675525e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.556569218584335e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.556569218584335e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.643051068868428e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.643051068868428e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.275883122558622e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.275883122558622e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.974311713879434e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.974311713879434e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.0002921986261110668) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.0002921986261110668) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.0002921986261110668) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.0002921986261110668) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004956762314916446) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004956762314916446) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650070219498778) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650070219498778) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650070219498778) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650070219498778) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533856254125412) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533856254125412) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.0016095313817213704) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.0016095313817213704) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.0016095313817213704) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.0016095313817213704) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0016676041811440607) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.0016676041811440607) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.0016676041811440607) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.0016676041811440607) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.00172787539413695) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.00172787539413695) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.0017992194936630311) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.0017992194936630311) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.002293956611352447) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.002293956611352447) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.0024629170071339117) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.0024629170071339117) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.0024629170071339117) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.0024629170071339117) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.003961560792496507) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.003961560792496507) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.003961560792496507) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.003961560792496507) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.0044248554494418346) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.0044248554494418346) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004575007626639222) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.004575007626639222) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668620318776294) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668620318776294) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.0046849033881552005) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.0046849033881552005) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005324835234221665) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.005324835234221665) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.005324835234221665) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.005324835234221665) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.005368659358109515) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368659358109515) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368659358109515) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368659358109515) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.007960880725921535) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.007960880725921535) [X4 Z5 X6 X10 Z11 X12]
+ (0.007960880725921535) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.007960880725921535) [X5 Z6 X7 X11 Z12 X13]
+ (0.008125251921381025) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125251921381025) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.008890731522694581) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.008890731522694581) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.008890731522694581) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.008890731522694581) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.010263414868158521) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.010263414868158521) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.010263414868158521) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.010263414868158521) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.01054042590767159) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.01054042590767159) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.01054042590767159) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.01054042590767159) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010960074940542493) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010960074940542493) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010960074940542493) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010960074940542493) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307274008848192) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307274008848192) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.01441109943013091) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.01441109943013091) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.01441109943013091) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.01441109943013091) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.01522563075722657) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.01522563075722657) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.01522563075722657) [X3 Z4 Z5 X6 X10 X11]
+ (0.01522563075722657) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.01558825010238019) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.01558825010238019) [X2 Z3 X4 X10 Z11 X12]
+ (0.01558825010238019) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.01558825010238019) [X3 Z4 X5 X11 Z12 X13]
+ (0.0182668348693755) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.0182668348693755) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.0182668348693755) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.0182668348693755) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.0190204231730399) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.0190204231730399) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.0190204231730399) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.0190204231730399) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.02017592172353549) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.02017592172353549) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.02017592172353549) [X4 Z5 Z6 X7 X11 X12]
+ (0.02017592172353549) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.02017592172353549) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.02017592172353549) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.02017592172353549) [X5 X6 X10 Z11 Z12 X13]
+ (0.02017592172353549) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.02435307767806896) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.02435307767806896) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.02435307767806896) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.02435307767806896) [X2 Z3 X4 X11 Z12 X13]
+ (0.02435307767806896) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.02435307767806896) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.02435307767806896) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.02435307767806896) [X3 Z4 X5 X10 Z11 X12]
+ (0.024389082531149416) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.024389082531149416) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.024389082531149416) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.024389082531149416) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.025104957138844544) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.025104957138844544) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.025104957138844544) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.025104957138844544) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.030787505389143877) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.030787505389143877) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.04587947078129806) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.04587947078129806) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.05600733087780768) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.05600733087780768) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.05600733087780768) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.05600733087780768) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661359) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.05608468124661359) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661359) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.05608468124661359) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.631277928933097e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631277928933097e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-6.631277928933091e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631277928933091e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.595086007298911e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.595086007298911e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.5950860072989063e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.5950860072989063e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.042743277013782666) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.042743277013782666) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (0.04274327701378268) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.04274327701378268) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (-0.0476426121763831) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.0476426121763831) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.0476426121763831) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.0476426121763831) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.04171881383982176) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.04171881383982176) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.04171881383982176) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.04171881383982176) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.03956441632289333) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.03956441632289333) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.03956441632289333) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03956441632289333) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03935916802205305) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.03935916802205305) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.03935916802205305) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.03935916802205305) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.03931805194719755) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03931805194719755) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.03931805194719755) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03931805194719755) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03560837898831247) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.03560837898831247) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.029903789512624783) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.029903789512624783) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.029903789512624783) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.029903789512624783) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.028730779551905464) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.028730779551905464) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.028730779551905464) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.028730779551905464) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.025637238296026894) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.025637238296026894) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.025637238296026894) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.025637238296026894) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.02475546329289093) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.02475546329289093) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.02475546329289093) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.02475546329289093) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.02428211735469299) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.02428211735469299) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.023145130929529252) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.023145130929529252) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.02252844019601287) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.02252844019601287) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02143381072160086) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.02143381072160086) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.02143381072160086) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.02143381072160086) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.01925750509525159) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.01925750509525159) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.01902824244384729) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.01902824244384729) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.01888903030494279) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.01888903030494279) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.01888903030494279) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.01888903030494279) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.016024603689179538) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.016024603689179538) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.01522563075722657) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.01522563075722657) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.014603704729162118) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.014603704729162118) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564531231172954) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564531231172954) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.01175601341981925) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.01175601341981925) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.011285190200840909) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.011285190200840909) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.009841749246962671) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009841749246962671) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.009612634606847355) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612634606847355) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612634606847355) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612634606847355) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.008469978791023923) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.008469978791023923) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.007306759928833009) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.007306759928833009) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005923798336561348) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.005923798336561348) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.0056526209780173335) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.0056526209780173335) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368659358109515) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368659358109515) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.0041587973818401) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.0041587973818401) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.003356670563832891) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.003356670563832891) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.003356670563832891) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.003356670563832891) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.00326751385442356) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.00326751385442356) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.00326751385442356) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.00326751385442356) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.0027790267990255493) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.0027790267990255493) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.0026860409778066015) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.0026860409778066015) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.0026860409778066015) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.0026860409778066015) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.002293956611352447) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.002293956611352447) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.002293956611352447) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.002293956611352447) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.0009581655836696521) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.0009581655836696521) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0009581655836696521) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.0009581655836696521) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.0009581655836696521) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.0009581655836696521) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0009581655836696521) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.0009581655836696521) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.00024636437569577674) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00024636437569577674) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0001384017730354987) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.0001384017730354987) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.0001384017730354987) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.0001384017730354987) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.735036880590382e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735036880590382e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585307047016e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585307047016e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.6103585307047016e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.6103585307047016e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808796570462e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.5316808796570462e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808796570462e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5316808796570462e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.806102775972842e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.806102775972842e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.806102775972842e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.806102775972842e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.089799468053156e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.089799468053156e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.089799468053156e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.089799468053156e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.652209670258151e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.652209670258151e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.652209670258151e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.652209670258151e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851834710287e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.481851834710287e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851834710287e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.481851834710287e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.071480736820343e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.071480736820343e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.071480736820343e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.071480736820343e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.734622039152499e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.734622039152499e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.734622039152499e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.734622039152499e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.728843147552359e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.728843147552359e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.728843147552359e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.728843147552359e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.253224225930398e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.253224225930398e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.769659452415901e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.769659452415901e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.5443954295793313e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.5443954295793313e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.5443954295793313e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.5443954295793313e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.5443954295793313e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.5443954295793313e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.5443954295793313e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.5443954295793313e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.3609563205007967e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563205007967e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563205007967e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.3609563205007967e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.103215604972196e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.103215604972196e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.103215604972196e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.103215604972196e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.011122098611565e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.011122098611565e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.011122098611565e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.011122098611565e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9429468367921017e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.9429468367921017e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9429468367921017e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9429468367921017e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.654117477370505e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.654117477370505e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.654117477370505e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.654117477370505e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.5224930677413162e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224930677413162e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224930677413162e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224930677413162e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224930677413162e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930677413162e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930677413162e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224930677413162e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.2283337824796395e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337824796395e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.2283337824796395e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337824796395e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.988770288824677e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988770288824677e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988770288824677e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988770288824677e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.867765104765538e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104765538e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104765538e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.867765104765538e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.189990976009894e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.189990976009894e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246207537664e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246207537664e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.471647744986291e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.471647744986291e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.56144717980704e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.56144717980704e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.56144717980704e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.56144717980704e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.5233896783453847e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.5233896783453847e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.427323109017637e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.427323109017637e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.427323109017637e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.427323109017637e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.3281393507574757e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.3281393507574757e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.3281393507574757e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.3281393507574757e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.0868265651927427e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.0868265651927427e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.8882935942159665e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935942159665e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935942159665e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.8882935942159665e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.371328947903965e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.371328947903965e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.839420915456305e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.839420915456305e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.057446596575921e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057446596575921e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.537178095672063e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.537178095672063e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.537178095672063e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.537178095672063e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.057446596575921e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057446596575921e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.209350636063088e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350636063088e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350636063088e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.209350636063088e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783554786354e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783554786354e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783554786354e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783554786354e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.839420915456305e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.839420915456305e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.371328947903965e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.371328947903965e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.0868265651927427e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.0868265651927427e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.5233896783453847e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.5233896783453847e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.471647744986291e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.471647744986291e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246207537664e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246207537664e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189990976009894e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.189990976009894e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.3304731887538258e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.3304731887538258e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.3304731887538258e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.3304731887538258e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288532436765143e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288532436765143e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288532436765143e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288532436765143e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.6893489516183555e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.6893489516183555e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.6893489516183555e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.6893489516183555e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.7455184006968642e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.7455184006968642e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.7455184006968642e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.7455184006968642e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.7455184006968642e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.7455184006968642e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.7455184006968642e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.7455184006968642e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.2118420193596726e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420193596726e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.2118420193596726e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420193596726e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.2118420193596726e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420193596726e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.2118420193596726e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420193596726e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.3131455004142406e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.3131455004142406e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.3131455004142406e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.3131455004142406e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.334331289759791e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.334331289759791e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.1839325596675525e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.1839325596675525e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.735036880590382e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735036880590382e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00024636437569577674) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00024636437569577674) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0004458535128840854) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.0004458535128840854) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.0004458535128840854) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.0004458535128840854) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940221543005255) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005255) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005255) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005255) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005255) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005255) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940221543005255) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005255) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533856254125412) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533856254125412) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533856254125412) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533856254125412) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0010435246534907505) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435246534907505) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435246534907505) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435246534907505) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.0012803060973496636) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.0012803060973496636) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.0012803060973496636) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.0012803060973496636) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.0013038004788126934) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.0013038004788126934) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.0013038004788126934) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.0013038004788126934) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.002261966062482345) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.002261966062482345) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.002261966062482345) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.002261966062482345) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.002261966062482345) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.002261966062482345) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.002261966062482345) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.002261966062482345) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.003989841456619296) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.003989841456619296) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.003989841456619296) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.003989841456619296) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.0041587973818401) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.0041587973818401) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.00431103850791431) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.00431103850791431) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.00431103850791431) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.00431103850791431) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.004636976661182555) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.004636976661182555) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.004636976661182555) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.004636976661182555) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.005114473831660379) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.005114473831660379) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.005114473831660379) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.005114473831660379) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.005114473831660379) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660379) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660379) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005114473831660379) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.0052415353828038636) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.0052415353828038636) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.0052415353828038636) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.0052415353828038636) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.005262642473076819) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262642473076819) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262642473076819) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262642473076819) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368659358109515) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368659358109515) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005379937155839363) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.005379937155839363) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.005379937155839363) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.005379937155839363) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.0056526209780173335) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.0056526209780173335) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.005708495985960904) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708495985960904) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708495985960904) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708495985960904) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.005923798336561348) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.005923798336561348) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.007306759928833009) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.007306759928833009) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469978791023923) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.008469978791023923) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.009841749246962671) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.009841749246962671) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.011285190200840909) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.011285190200840909) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.01175601341981925) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.01175601341981925) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.014564531231172954) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564531231172954) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603704729162118) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.014603704729162118) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.01522563075722657) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.01522563075722657) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.016024603689179538) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.016024603689179538) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.01902824244384729) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.01902824244384729) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.01925750509525159) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.01925750509525159) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.04587947078129806) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04587947078129806) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.36937089366156195) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.36937089366156195) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.36937089366156195) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.36937089366156195) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.2816425776702295) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.2816425776702295) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.28164257767022943) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.28164257767022943) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.09065144207036485) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.09065144207036485) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.09065144207036485) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.09065144207036485) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863632) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684737589863632) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863632) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684737589863632) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635021950635008) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635021950635008) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635021950635008) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635021950635008) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214027) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.06752385099214027) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214027) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.06752385099214027) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03560837898831247) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.03560837898831247) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03490334337366172) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03490334337366172) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03490334337366172) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03490334337366172) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883829926) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.024591860883829926) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883829926) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.024591860883829926) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.024282117354692993) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.024282117354692993) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.023145130929529256) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.023145130929529256) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.02252844019601287) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.02252844019601287) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019538050311314732) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.019538050311314732) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.019538050311314732) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.019538050311314732) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.017091553155898918) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.017091553155898918) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.017091553155898918) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.017091553155898918) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.016024603689179538) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.016024603689179538) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.016024603689179538) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.016024603689179538) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.010311482489831797) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831797) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831797) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311482489831797) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.009841749246962671) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962671) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.009841749246962671) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962671) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209816) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209816) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209816) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008826368514209816) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008541996625454825) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454825) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454825) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454825) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454825) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454825) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454825) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008541996625454825) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008469978791023923) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023923) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.008469978791023923) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023923) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.004668620318776294) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668620318776294) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.003876470899336949) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.003876470899336949) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.003804066171728539) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728539) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728539) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003804066171728539) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484157300217881) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484157300217881) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003356670563832891) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.003356670563832891) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.00326751385442356) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.00326751385442356) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0021413612231016114) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.0021413612231016114) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.00172787539413695) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.00172787539413695) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.0016407548553123794) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0016407548553123794) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214168972) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0014528843214168972) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214168972) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0014528843214168972) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870896771024411) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007870896771024411) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192743499487689) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192743499487689) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.0001940085702975656) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0001940085702975656) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0001384017730354987) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.0001384017730354987) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.141625221157718e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.141625221157718e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.141625221157718e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.141625221157718e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.071480736820343e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.071480736820343e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.1513463114178305e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.1513463114178305e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0882507115986276e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.0882507115986276e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.98851170649276e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.98851170649276e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8742990716240056e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.8742990716240056e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.3609563205007967e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.3609563205007967e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.300294656408087e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.300294656408087e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.1468376508538543e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.1468376508538543e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.1468376508538543e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.1468376508538543e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.352332103870293e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.352332103870293e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.352332103870293e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.352332103870293e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091637199832162e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637199832162e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637199832162e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637199832162e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637199832162e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637199832162e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637199832162e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091637199832162e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.074305986653847e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.074305986653847e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.074305986653847e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.074305986653847e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128987051764e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128987051764e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128987051764e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128987051764e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867765104765538e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.867765104765538e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560692465637466e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692465637466e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692465637466e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692465637466e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692465637466e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692465637466e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692465637466e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560692465637466e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.997018422589549e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.997018422589549e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.997018422589549e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.997018422589549e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.997018422589549e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.997018422589549e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.997018422589549e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.997018422589549e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.568247521486778e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.568247521486778e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.568247521486778e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.568247521486778e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393087063784e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393087063784e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393087063784e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393087063784e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393087063784e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393087063784e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.3767393087063784e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393087063784e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.8882935942159665e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.8882935942159665e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6863815465477444e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.6863815465477444e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035783554786354e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7035783554786354e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.20935063606309e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.20935063606309e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244383903e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773244383903e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244383903e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773244383903e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244383903e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773244383903e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773244383903e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773244383903e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253796096287e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253796096287e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9742253796096287e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9742253796096287e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.04747165555499e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.04747165555499e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.04747165555499e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.04747165555499e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.20935063606309e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.20935063606309e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0717282185447439e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0717282185447439e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0717282185447439e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0717282185447439e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.2004287494936088e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.2004287494936088e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.2004287494936088e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.2004287494936088e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.7035783554786354e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7035783554786354e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.3120943054328504e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.3120943054328504e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.3120943054328504e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.3120943054328504e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.6863815465477444e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815465477444e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.8882935942159665e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.8882935942159665e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.092250616451051e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.092250616451051e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.092250616451051e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.092250616451051e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.092250616451051e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.092250616451051e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.092250616451051e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.092250616451051e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.44459785433961e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.44459785433961e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.44459785433961e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.44459785433961e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.684915095531342e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.684915095531342e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.684915095531342e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.684915095531342e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246974426082409e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246974426082409e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246974426082409e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246974426082409e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246974426082409e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246974426082409e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246974426082409e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246974426082409e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.867765104765538e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.867765104765538e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.300294656408087e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.300294656408087e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.3609563205007967e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.3609563205007967e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.8742990716240056e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.8742990716240056e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.883676576259922e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.883676576259922e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.9473560119668607e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9473560119668607e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.9473560119668607e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9473560119668607e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.98851170649276e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.98851170649276e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.0882507115986276e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.0882507115986276e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.1513463114178305e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.1513463114178305e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.846201671537267e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846201671537267e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846201671537267e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846201671537267e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.071480736820343e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.071480736820343e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.1055267223313125e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.1055267223313125e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.1055267223313125e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.1055267223313125e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.1464963279453545e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.1464963279453545e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.1464963279453545e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.1464963279453545e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.15935050223141e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.15935050223141e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.15935050223141e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.15935050223141e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427988656886184e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427988656886184e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427988656886184e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427988656886184e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935867718459621e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935867718459621e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935867718459621e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935867718459621e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.2532733485746025e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.2532733485746025e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979825793955318e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979825793955318e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979825793955318e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979825793955318e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.205548411217487e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.205548411217487e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.205548411217487e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.205548411217487e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0001384017730354987) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.0001384017730354987) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.0001878705338954823) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0001878705338954823) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0001878705338954823) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0001878705338954823) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0001940085702975656) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.0001940085702975656) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00024636437569577674) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569577674) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00024636437569577674) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569577674) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192743499487689) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192743499487689) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156734248908637) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156734248908637) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156734248908637) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156734248908637) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024411) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007870896771024411) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.0015324835230730212) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.0015324835230730212) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.0015324835230730212) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.0015324835230730212) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.0016407548553123794) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0016407548553123794) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.00172787539413695) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.00172787539413695) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.002446497155415812) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.002446497155415812) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.002446497155415812) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.002446497155415812) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.00326751385442356) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.00326751385442356) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.003356670563832891) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.003356670563832891) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.003484157300217881) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.003484157300217881) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.003876470899336949) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.003876470899336949) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.004668620318776294) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668620318776294) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278098) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.004767272188278098) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.004767272188278098) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278098) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.005286546538226867) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.005286546538226867) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.005286546538226867) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.005286546538226867) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.00540895442240997) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.00540895442240997) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.00540895442240997) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.00540895442240997) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.005923798336561348) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561348) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.005923798336561348) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561348) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.010757563953908937) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010757563953908937) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010757563953908937) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010757563953908937) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.014603704729162118) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.014603704729162118) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.014603704729162118) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.014603704729162118) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.019299560579363766) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363766) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363766) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363766) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363766) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363766) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.019299560579363766) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363766) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.05859198873386188) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.05859198873386188) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.7759505277475934e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.7759505277475934e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.775950527747597e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.775950527747597e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.07165035181002505) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07165035181002505) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0716503518100251) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0716503518100251) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.01925750509525159) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.01925750509525159) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010311482489831797) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311482489831797) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008826368514209816) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209816) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.007597464029770573) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.007597464029770573) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.007597464029770573) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.007597464029770573) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311873) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311873) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311873) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311873) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311873) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311873) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311873) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311873) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676582) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005348051582676582) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005348051582676582) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676582) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.003804066171728539) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003804066171728539) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00298416616812194) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.00298416616812194) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.00298416616812194) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.00298416616812194) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.0024464971554158123) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.0024464971554158123) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.0022494124470939913) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0022494124470939913) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022494124470939913) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0022494124470939913) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0021413612231016114) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.0021413612231016114) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.001863894282458701) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.001863894282458701) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.001863894282458701) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.001863894282458701) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.001863894282458701) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.001863894282458701) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.001863894282458701) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.001863894282458701) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0016407548553123794) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553123794) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0016407548553123794) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553123794) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0012223378081538314) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538314) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538314) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538314) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538314) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538314) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538314) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0012223378081538314) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0010283292378562659) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0010283292378562659) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0010283292378562659) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0010283292378562659) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1463061453927793e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1463061453927793e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.8742990716240056e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990716240056e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.8742990716240056e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990716240056e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.300294656408087e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.300294656408087e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.300294656408087e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.300294656408087e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.0444941298782e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.0444941298782e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.0444941298782e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.0444941298782e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.956079230708508e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.956079230708508e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.956079230708508e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.956079230708508e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105515037736159e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105515037736159e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105515037736159e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105515037736159e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661347213655739e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661347213655739e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661347213655739e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661347213655739e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341414341033e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341414341033e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189990976009894e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.189990976009894e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876621658678963e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876621658678963e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876621658678963e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876621658678963e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246207537664e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246207537664e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.5233896783453847e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.5233896783453847e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.07673253200088e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.07673253200088e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.07673253200088e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.07673253200088e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.013471459197837e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.013471459197837e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.904599884440967e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.904599884440967e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.904599884440967e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.904599884440967e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.666731754564373e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.666731754564373e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.666731754564373e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.666731754564373e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.850564192972347e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.850564192972347e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309315176559e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6569309315176559e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309315176559e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6569309315176559e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.850564192972347e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.850564192972347e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815465477444e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815465477444e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.6863815465477444e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815465477444e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471459197837e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.013471459197837e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.5233896783453847e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.5233896783453847e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.6704023907154924e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.6704023907154924e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.6704023907154924e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.6704023907154924e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246207537664e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246207537664e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189990976009894e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.189990976009894e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341414341033e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341414341033e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.94947648819433e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.94947648819433e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.792493957830162e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.792493957830162e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.792493957830162e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.792493957830162e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.8836765762599224e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.8836765762599224e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.98851170649276e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.98851170649276e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.98851170649276e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.98851170649276e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.2532733485746025e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.2532733485746025e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4017109736203532e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4017109736203532e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4017109736203532e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4017109736203532e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603694033694e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.5809603694033694e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603694033694e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.5809603694033694e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192743499487689) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487689) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192743499487689) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487689) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.0007870896771024411) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024411) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024411) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024411) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0011726348316441863) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0011726348316441863) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0011726348316441863) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0011726348316441863) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012366478019245283) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.0012366478019245283) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.0012366478019245283) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.0012366478019245283) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.002200964069500453) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002200964069500453) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002200964069500453) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002200964069500453) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798018) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798018) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798018) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798018) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798018) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798018) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002394972639798018) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798018) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0024464971554158123) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.0024464971554158123) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.003804066171728539) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.003804066171728539) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.003876470899336949) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.003876470899336949) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.003876470899336949) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.003876470899336949) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.004220813970046469) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.004220813970046469) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.004220813970046469) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.004220813970046469) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.008826368514209816) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.008826368514209816) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.010311482489831797) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311482489831797) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01925750509525159) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01925750509525159) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.05859198873386188) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.05859198873386188) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.398700901582537e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.398700901582537e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.398700901582537e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.398700901582537e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003484157300217881) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003484157300217881) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219403) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.0029841661681219403) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.0001940085702975656) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0001940085702975656) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453927793e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453927793e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.792493957830162e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.792493957830162e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540341414341033e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341414341033e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540341414341033e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341414341033e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.850564192972347e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.850564192972347e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.850564192972347e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.850564192972347e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714591978366e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0134714591978366e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714591978366e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0134714591978366e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.94947648819433e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.94947648819433e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.792493957830162e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.792493957830162e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0001940085702975656) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0001940085702975656) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0029841661681219403) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.0029841661681219403) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.003484157300217881) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.003484157300217881) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
Expectation value of XIZ =  0.07715357869738937
Expectation value of XIZ =  0.07715357869738931
<H> =  3.87682591686312
3.87682591686312
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
   (-46.46390678868893) [I0]
+ (0.7829661725950192) [Z10]
+ (0.7829661725950193) [Z11]
+ (0.808458196172049) [Z13]
+ (0.8084581961720492) [Z12]
+ (1.2034402289145627) [Z4]
+ (1.203440228914563) [Z5]
+ (1.3096862988615432) [Z6]
+ (1.3096862988615432) [Z7]
+ (1.369352563471819) [Z8]
+ (1.369352563471819) [Z9]
+ (1.6538942226831703) [Z3]
+ (1.6538942226831705) [Z2]
+ (12.412630742111759) [Z0]
+ (12.412630742111759) [Z1]
+ (-8.194261372282402e-06) [Y10 Y12]
+ (-8.194261372282402e-06) [X10 X12]
+ (-1.8540608578992652e-06) [Y5 Y7]
+ (-1.8540608578992652e-06) [X5 X7]
+ (-7.764994119774016e-07) [Y3 Y5]
+ (-7.764994119774016e-07) [X3 X5]
+ (-5.92976581490258e-07) [Y4 Y6]
+ (-5.92976581490258e-07) [X4 X6]
+ (1.6021167406634068e-06) [Y2 Y4]
+ (1.6021167406634068e-06) [X2 X4]
+ (7.95441317632806e-06) [Y11 Y13]
+ (7.95441317632806e-06) [X11 X13]
+ (0.0032769719312315797) [Y1 Y3]
+ (0.0032769719312315797) [X1 X3]
+ (0.10433064780651384) [Y0 Y2]
+ (0.10433064780651384) [X0 X2]
+ (0.11270386920332229) [Z10 Z12]
+ (0.11270386920332229) [Z11 Z13]
+ (0.11383573679388673) [Z4 Z12]
+ (0.11383573679388673) [Z5 Z13]
+ (0.11952438964682684) [Z6 Z10]
+ (0.11952438964682684) [Z7 Z11]
+ (0.12489990917237608) [Z4 Z10]
+ (0.12489990917237608) [Z5 Z11]
+ (0.12495807739503229) [Z2 Z4]
+ (0.12495807739503229) [Z3 Z5]
+ (0.1279950249246842) [Z2 Z10]
+ (0.1279950249246842) [Z3 Z11]
+ (0.13401715261963723) [Z6 Z12]
+ (0.13401715261963723) [Z7 Z13]
+ (0.13701191674040764) [Z4 Z6]
+ (0.13701191674040764) [Z5 Z7]
+ (0.1373495306426133) [Z6 Z11]
+ (0.1373495306426133) [Z7 Z10]
+ (0.1373910476268324) [Z2 Z6]
+ (0.1373910476268324) [Z3 Z7]
+ (0.13766872645852588) [Z8 Z10]
+ (0.13766872645852588) [Z9 Z11]
+ (0.14011289865354828) [Z2 Z12]
+ (0.14011289865354828) [Z3 Z13]
+ (0.14138905291942822) [Z10 Z13]
+ (0.14138905291942822) [Z11 Z12]
+ (0.1425799771248576) [Z4 Z11]
+ (0.1425799771248576) [Z5 Z10]
+ (0.14722943218766182) [Z8 Z11]
+ (0.14722943218766182) [Z9 Z10]
+ (0.1489943057506556) [Z4 Z7]
+ (0.1489943057506556) [Z5 Z6]
+ (0.14926355147388912) [Z10 Z11]
+ (0.1496070268444531) [Z4 Z8]
+ (0.1496070268444531) [Z5 Z9]
+ (0.1497348680349694) [Z8 Z12]
+ (0.1497348680349694) [Z9 Z13]
+ (0.1507140812100831) [Z2 Z8]
+ (0.1507140812100831) [Z3 Z9]
+ (0.15138327161428866) [Z6 Z13]
+ (0.15138327161428866) [Z7 Z12]
+ (0.15215040708869065) [Z4 Z13]
+ (0.15215040708869065) [Z5 Z12]
+ (0.15337968243314165) [Z2 Z11]
+ (0.15337968243314165) [Z3 Z10]
+ (0.15435748657223647) [Z12 Z13]
+ (0.15569010671752476) [Z2 Z13]
+ (0.15569010671752476) [Z3 Z12]
+ (0.15582269051553127) [Z8 Z13]
+ (0.15582269051553127) [Z9 Z12]
+ (0.15676396176431007) [Z4 Z9]
+ (0.15676396176431007) [Z5 Z8]
+ (0.15755314797985676) [Z4 Z5]
+ (0.1607976453483858) [Z2 Z5]
+ (0.1607976453483858) [Z3 Z4]
+ (0.16756653265461288) [Z6 Z8]
+ (0.16756653265461288) [Z7 Z9]
+ (0.16853486561579956) [Z2 Z7]
+ (0.16853486561579956) [Z3 Z6]
+ (0.181439914403039) [Z6 Z9]
+ (0.181439914403039) [Z7 Z8]
+ (0.18189085790751383) [Z2 Z3]
+ (0.18690820476912573) [Z2 Z9]
+ (0.18690820476912573) [Z3 Z8]
+ (0.19299723935364232) [Z0 Z10]
+ (0.19299723935364232) [Z1 Z11]
+ (0.19392534613270232) [Z6 Z7]
+ (0.19661770890342137) [Z0 Z4]
+ (0.19661770890342137) [Z1 Z5]
+ (0.19936354537360818) [Z0 Z5]
+ (0.19936354537360818) [Z1 Z4]
+ (0.20072866460441763) [Z0 Z11]
+ (0.20072866460441763) [Z1 Z10]
+ (0.21102659849791516) [Z0 Z12]
+ (0.21102659849791516) [Z1 Z13]
+ (0.2163103749863181) [Z0 Z13]
+ (0.2163103749863181) [Z1 Z12]
+ (0.22003977334376112) [Z8 Z9]
+ (0.2367108078383042) [Z0 Z2]
+ (0.2367108078383042) [Z1 Z3]
+ (0.24164663936017206) [Z0 Z6]
+ (0.24164663936017206) [Z1 Z7]
+ (0.2512944567459169) [Z0 Z3]
+ (0.2512944567459169) [Z1 Z2]
+ (0.27232518306605674) [Z0 Z8]
+ (0.27232518306605674) [Z1 Z9]
+ (0.278834544267234) [Z0 Z9]
+ (0.278834544267234) [Z1 Z8]
+ (1.1861763734860484) [Z0 Z1]
+ (-1.2260484988493164e-05) [Y5 Z6 Y7]
+ (-1.2260484988493164e-05) [X5 Z6 X7]
+ (-1.2260484988493157e-05) [Y4 Z5 Y6]
+ (-1.2260484988493157e-05) [X4 Z5 X6]
+ (-1.0722312157868114e-05) [Y11 Z12 Y13]
+ (-1.0722312157868114e-05) [X11 Z12 X13]
+ (-1.072231215786811e-05) [Y10 Z11 Y12]
+ (-1.072231215786811e-05) [X10 Z11 X12]
+ (-3.887051673330075e-06) [Y3 Z4 Y5]
+ (-3.887051673330075e-06) [X3 Z4 X5]
+ (-3.887051673330074e-06) [Y2 Z3 Y4]
+ (-3.887051673330074e-06) [X2 Z3 X4]
+ (0.12507032579771954) [Y0 Z1 Y2]
+ (0.12507032579771954) [X0 Z1 X2]
+ (0.1250703257977196) [Y1 Z2 Y3]
+ (0.1250703257977196) [X1 Z2 X3]
+ (-0.038314670294803906) [Y4 Y5 X12 X13]
+ (-0.038314670294803906) [X4 X5 Y12 Y13]
+ (-0.03619412355904263) [Y2 Y3 X8 X9]
+ (-0.03619412355904263) [X2 X3 Y8 Y9]
+ (-0.03583956795335352) [Y2 Y3 X4 X5]
+ (-0.03583956795335352) [X2 X3 Y4 Y5]
+ (-0.031143817988967145) [Y2 Y3 X6 X7]
+ (-0.031143817988967145) [X2 X3 Y6 Y7]
+ (-0.02868518371610595) [Y10 Y11 X12 X13]
+ (-0.02868518371610595) [X10 X11 Y12 Y13]
+ (-0.025996177598021246) [Y3 Z4 Z5 Y7]
+ (-0.025996177598021246) [X3 Z4 Z5 X7]
+ (-0.025384657508457455) [Y2 Y3 X10 X11]
+ (-0.025384657508457455) [X2 X3 Y10 Y11]
+ (-0.01902824244384732) [Y3 Y4 X11 X12]
+ (-0.01902824244384732) [X3 X4 Y11 Y12]
+ (-0.01782514099578644) [Y6 Y7 X10 X11]
+ (-0.01782514099578644) [X6 X7 Y10 Y11]
+ (-0.017680067952481535) [Y4 Y5 X10 X11]
+ (-0.017680067952481535) [X4 X5 Y10 Y11]
+ (-0.017366118994651427) [Y6 Y7 X12 X13]
+ (-0.017366118994651427) [X6 X7 Y12 Y13]
+ (-0.015577208063976474) [Y2 Y3 X12 X13]
+ (-0.015577208063976474) [X2 X3 Y12 Y13]
+ (-0.014583648907612656) [Y0 Y1 X2 X3]
+ (-0.014583648907612656) [X0 X1 Y2 Y3]
+ (-0.013873381748426129) [Y6 Y7 X8 X9]
+ (-0.013873381748426129) [X6 X7 Y8 Y9]
+ (-0.011982389010247962) [Y4 Y5 X6 X7]
+ (-0.011982389010247962) [X4 X5 Y6 Y7]
+ (-0.01128519020084091) [Y5 X6 X11 Y12]
+ (-0.01128519020084091) [X5 Y6 Y11 X12]
+ (-0.009560705729135954) [Y8 Y9 X10 X11]
+ (-0.009560705729135954) [X8 X9 Y10 Y11]
+ (-0.008125251921381043) [Y1 X2 X8 Y9]
+ (-0.008125251921381043) [Y1 Y2 Y8 Y9]
+ (-0.008125251921381043) [X1 X2 X8 X9]
+ (-0.008125251921381043) [X1 Y2 Y8 X9]
+ (-0.0077314252507752835) [Y0 Y1 X10 X11]
+ (-0.0077314252507752835) [X0 X1 Y10 Y11]
+ (-0.00715693491985695) [Y4 Y5 X8 X9]
+ (-0.00715693491985695) [X4 X5 Y8 Y9]
+ (-0.006888194352970583) [Y0 Y1 X6 X7]
+ (-0.006888194352970583) [X0 X1 Y6 Y7]
+ (-0.0065093612011772415) [Y0 Y1 X8 X9]
+ (-0.0065093612011772415) [X0 X1 Y8 Y9]
+ (-0.006087822480561868) [Y8 Y9 X12 X13]
+ (-0.006087822480561868) [X8 X9 Y12 Y13]
+ (-0.005283776488402963) [Y0 Y1 X12 X13]
+ (-0.005283776488402963) [X0 X1 Y12 Y13]
+ (-0.005143391768825107) [Y3 X4 X5 Y6]
+ (-0.005143391768825107) [X3 Y4 Y5 X6]
+ (-0.00468490338815521) [Y1 X2 X6 Y7]
+ (-0.00468490338815521) [Y1 Y2 Y6 Y7]
+ (-0.00468490338815521) [X1 X2 X6 X7]
+ (-0.00468490338815521) [X1 Y2 Y6 X7]
+ (-0.004575007626639211) [Y1 X2 X12 Y13]
+ (-0.004575007626639211) [Y1 Y2 Y12 Y13]
+ (-0.004575007626639211) [X1 X2 X12 X13]
+ (-0.004575007626639211) [X1 Y2 Y12 X13]
+ (-0.004424855449441864) [Y1 X2 X4 Y5]
+ (-0.004424855449441864) [Y1 Y2 Y4 Y5]
+ (-0.004424855449441864) [X1 X2 X4 X5]
+ (-0.004424855449441864) [X1 Y2 Y4 X5]
+ (-0.0034795118903343204) [Y2 Z3 Z5 Y6]
+ (-0.0034795118903343204) [X2 Z3 Z5 X6]
+ (-0.0034795118903343204) [Y3 Z4 Z6 Y7]
+ (-0.0034795118903343204) [X3 Z4 Z6 X7]
+ (-0.002745836470186813) [Y0 Y1 X4 X5]
+ (-0.002745836470186813) [X0 X1 Y4 Y5]
+ (-0.001799219493663029) [Y1 X2 X10 Y11]
+ (-0.001799219493663029) [Y1 Y2 Y10 Y11]
+ (-0.001799219493663029) [X1 X2 X10 X11]
+ (-0.001799219493663029) [X1 Y2 Y10 X11]
+ (-0.0002921986261110701) [Y7 Y8 X9 X10]
+ (-0.0002921986261110701) [X7 X8 Y9 Y10]
+ (-8.1942613722824e-06) [Z10 Y11 Z12 Y13]
+ (-8.1942613722824e-06) [Z10 X11 Z12 X13]
+ (-7.801707500643188e-06) [Y2 Z3 Y4 Z11]
+ (-7.801707500643188e-06) [X2 Z3 X4 Z11]
+ (-7.801707500643188e-06) [Y3 Z4 Y5 Z10]
+ (-7.801707500643188e-06) [X3 Z4 X5 Z10]
+ (-4.643051068575135e-06) [Y3 X4 X10 Y11]
+ (-4.643051068575135e-06) [Y3 Y4 Y10 Y11]
+ (-4.643051068575135e-06) [X3 X4 X10 X11]
+ (-4.643051068575135e-06) [X3 Y4 Y10 X11]
+ (-4.588855155687443e-06) [Y4 Z5 Y6 Z13]
+ (-4.588855155687443e-06) [X4 Z5 X6 Z13]
+ (-4.588855155687443e-06) [Y5 Z6 Y7 Z12]
+ (-4.588855155687443e-06) [X5 Z6 X7 Z12]
+ (-4.556569218247175e-06) [Y5 X6 X12 Y13]
+ (-4.556569218247175e-06) [Y5 Y6 Y12 Y13]
+ (-4.556569218247175e-06) [X5 X6 X12 X13]
+ (-4.556569218247175e-06) [X5 Y6 Y12 X13]
+ (-3.6945132946322484e-06) [Y4 X5 X11 Y12]
+ (-3.6945132946322484e-06) [Y4 Y5 Y11 Y12]
+ (-3.6945132946322484e-06) [X4 X5 X11 X12]
+ (-3.6945132946322484e-06) [X4 Y5 Y11 X12]
+ (-3.3440815563566876e-06) [Z0 Y5 Z6 Y7]
+ (-3.3440815563566876e-06) [Z0 X5 Z6 X7]
+ (-3.3440815563566876e-06) [Z1 Y4 Z5 Y6]
+ (-3.3440815563566876e-06) [Z1 X4 Z5 X6]
+ (-3.1586564320680526e-06) [Y2 Z3 Y4 Z10]
+ (-3.1586564320680526e-06) [X2 Z3 X4 Z10]
+ (-3.1586564320680526e-06) [Y3 Z4 Y5 Z11]
+ (-3.1586564320680526e-06) [X3 Z4 X5 Z11]
+ (-3.0993492434814473e-06) [Z0 Y4 Z5 Y6]
+ (-3.0993492434814473e-06) [Z0 X4 Z5 X6]
+ (-3.0993492434814473e-06) [Z1 Y5 Z6 Y7]
+ (-3.0993492434814473e-06) [Z1 X5 Z6 X7]
+ (-2.890967881630772e-06) [Z6 Y11 Z12 Y13]
+ (-2.890967881630772e-06) [Z6 X11 Z12 X13]
+ (-2.890967881630772e-06) [Z7 Y10 Z11 Y12]
+ (-2.890967881630772e-06) [Z7 X10 Z11 X12]
+ (-2.177664605042258e-06) [Z0 Y10 Z11 Y12]
+ (-2.177664605042258e-06) [Z0 X10 Z11 X12]
+ (-2.177664605042258e-06) [Z1 Y11 Z12 Y13]
+ (-2.177664605042258e-06) [Z1 X11 Z12 X13]
+ (-1.8818501831378656e-06) [Y4 Z5 Y6 Z9]
+ (-1.8818501831378656e-06) [X4 Z5 X6 Z9]
+ (-1.8818501831378656e-06) [Y5 Z6 Y7 Z8]
+ (-1.8818501831378656e-06) [X5 Z6 X7 Z8]
+ (-1.8551201215480016e-06) [Z6 Y10 Z11 Y12]
+ (-1.8551201215480016e-06) [Z6 X10 Z11 X12]
+ (-1.8551201215480016e-06) [Z7 Y11 Z12 Y13]
+ (-1.8551201215480016e-06) [Z7 X11 Z12 X13]
+ (-1.854060857899265e-06) [Y4 Z5 Y6 Z7]
+ (-1.854060857899265e-06) [X4 Z5 X6 Z7]
+ (-1.8163031698642447e-06) [Z4 Y11 Z12 Y13]
+ (-1.8163031698642447e-06) [Z4 X11 Z12 X13]
+ (-1.8163031698642447e-06) [Z5 Y10 Z11 Y12]
+ (-1.8163031698642447e-06) [Z5 X10 Z11 X12]
+ (-1.692397828588753e-06) [Y4 Z5 Y6 Z10]
+ (-1.692397828588753e-06) [X4 Z5 X6 Z10]
+ (-1.692397828588753e-06) [Y5 Z6 Y7 Z11]
+ (-1.692397828588753e-06) [X5 Z6 X7 Z11]
+ (-1.6148794138984012e-06) [Z0 Y11 Z12 Y13]
+ (-1.6148794138984012e-06) [Z0 X11 Z12 X13]
+ (-1.6148794138984012e-06) [Z1 Y10 Z11 Y12]
+ (-1.6148794138984012e-06) [Z1 X10 Z11 X12]
+ (-1.5973171978409703e-06) [Z8 Y10 Z11 Y12]
+ (-1.5973171978409703e-06) [Z8 X10 Z11 X12]
+ (-1.5973171978409703e-06) [Z9 Y11 Z12 Y13]
+ (-1.5973171978409703e-06) [Z9 X11 Z12 X13]
+ (-1.4548424489921242e-06) [Y3 X4 X6 Y7]
+ (-1.4548424489921242e-06) [Y3 Y4 Y6 Y7]
+ (-1.4548424489921242e-06) [X3 X4 X6 X7]
+ (-1.4548424489921242e-06) [X3 Y4 Y6 X7]
+ (-1.3980449080567684e-06) [Y4 Z5 Y6 Z8]
+ (-1.3980449080567684e-06) [X4 Z5 X6 Z8]
+ (-1.3980449080567684e-06) [Y5 Z6 Y7 Z9]
+ (-1.3980449080567684e-06) [X5 Z6 X7 Z9]
+ (-1.1954890099013397e-06) [Y2 Z3 Y4 Z7]
+ (-1.1954890099013397e-06) [X2 Z3 X4 Z7]
+ (-1.1954890099013397e-06) [Y3 Z4 Y5 Z6]
+ (-1.1954890099013397e-06) [X3 Z4 X5 Z6]
+ (-1.1908508083570703e-06) [Z0 Y3 Z4 Y5]
+ (-1.1908508083570703e-06) [Z0 X3 Z4 X5]
+ (-1.1908508083570703e-06) [Z1 Y2 Z3 Y4]
+ (-1.1908508083570703e-06) [Z1 X2 Z3 X4]
+ (-1.1708301369838795e-06) [Z2 Y5 Z6 Y7]
+ (-1.1708301369838795e-06) [Z2 X5 Z6 X7]
+ (-1.1708301369838795e-06) [Z3 Y4 Z5 Y6]
+ (-1.1708301369838795e-06) [Z3 X4 Z5 X6]
+ (-1.0632283423709644e-06) [Z2 Y10 Z11 Y12]
+ (-1.0632283423709644e-06) [Z2 X10 Z11 X12]
+ (-1.0632283423709644e-06) [Z3 Y11 Z12 Y13]
+ (-1.0632283423709644e-06) [Z3 X11 Z12 X13]
+ (-1.0358477600827703e-06) [Y6 X7 X11 Y12]
+ (-1.0358477600827703e-06) [Y6 Y7 Y11 Y12]
+ (-1.0358477600827703e-06) [X6 X7 X11 X12]
+ (-1.0358477600827703e-06) [X6 Y7 Y11 X12]
+ (-9.509249750982672e-07) [Z2 Y4 Z5 Y6]
+ (-9.509249750982672e-07) [Z2 X4 Z5 X6]
+ (-9.509249750982672e-07) [Z3 Y5 Z6 Y7]
+ (-9.509249750982672e-07) [Z3 X5 Z6 X7]
+ (-9.344557776652907e-07) [Z8 Y11 Z12 Y13]
+ (-9.344557776652907e-07) [Z8 X11 Z12 X13]
+ (-9.344557776652907e-07) [Z9 Y10 Z11 Y12]
+ (-9.344557776652907e-07) [Z9 X10 Z11 X12]
+ (-8.337746754467109e-07) [Z0 Y2 Z3 Y4]
+ (-8.337746754467109e-07) [Z0 X2 Z3 X4]
+ (-8.337746754467109e-07) [Z1 Y3 Z4 Y5]
+ (-8.337746754467109e-07) [Z1 X3 Z4 X5]
+ (-7.956895372502105e-07) [Y3 X4 X8 Y9]
+ (-7.956895372502105e-07) [Y3 Y4 Y8 Y9]
+ (-7.956895372502105e-07) [X3 X4 X8 X9]
+ (-7.956895372502105e-07) [X3 Y4 Y8 X9]
+ (-7.764994119774015e-07) [Y2 Z3 Y4 Z5]
+ (-7.764994119774015e-07) [X2 Z3 X4 Z5]
+ (-5.92976581490258e-07) [Z4 Y5 Z6 Y7]
+ (-5.92976581490258e-07) [Z4 X5 Z6 X7]
+ (-5.770052994987465e-07) [Y2 Z3 Y4 Z9]
+ (-5.770052994987465e-07) [X2 Z3 X4 Z9]
+ (-5.770052994987465e-07) [Y3 Z4 Y5 Z8]
+ (-5.770052994987465e-07) [X3 Z4 X5 Z8]
+ (-5.471647744714756e-07) [Y1 Y2 X11 X12]
+ (-5.471647744714756e-07) [X1 X2 Y11 Y12]
+ (-4.838052750810973e-07) [Y5 X6 X8 Y9]
+ (-4.838052750810973e-07) [Y5 Y6 Y8 Y9]
+ (-4.838052750810973e-07) [X5 X6 X8 X9]
+ (-4.838052750810973e-07) [X5 Y6 Y8 X9]
+ (-3.5707613291035953e-07) [Y0 X1 X3 Y4]
+ (-3.5707613291035953e-07) [Y0 Y1 Y3 Y4]
+ (-3.5707613291035953e-07) [X0 X1 X3 X4]
+ (-3.5707613291035953e-07) [X0 Y1 Y3 X4]
+ (-2.4473231287524075e-07) [Y0 X1 X5 Y6]
+ (-2.4473231287524075e-07) [Y0 Y1 Y5 Y6]
+ (-2.4473231287524075e-07) [X0 X1 X5 X6]
+ (-2.4473231287524075e-07) [X0 Y1 Y5 X6]
+ (-2.1990516188561224e-07) [Y2 X3 X5 Y6]
+ (-2.1990516188561224e-07) [Y2 Y3 Y5 Y6]
+ (-2.1990516188561224e-07) [X2 X3 X5 X6]
+ (-2.1990516188561224e-07) [X2 Y3 Y5 X6]
+ (-1.9332412770657797e-07) [Y1 X2 X3 Y4]
+ (-1.9332412770657797e-07) [X1 Y2 Y3 X4]
+ (-1.291969486330071e-07) [Y1 Z2 Z3 Y5]
+ (-1.291969486330071e-07) [X1 Z2 Z3 X5]
+ (1.737933262400666e-07) [Y0 Z1 Z3 Y4]
+ (1.737933262400666e-07) [X0 Z1 Z3 X4]
+ (1.737933262400666e-07) [Y1 Z2 Z4 Y5]
+ (1.737933262400666e-07) [X1 Z2 Z4 X5]
+ (1.9332412770657797e-07) [Y1 Y2 X3 X4]
+ (1.9332412770657797e-07) [X1 X2 Y3 Y4]
+ (2.1868423775146393e-07) [Y2 Z3 Y4 Z8]
+ (2.1868423775146393e-07) [X2 Z3 X4 Z8]
+ (2.1868423775146393e-07) [Y3 Z4 Y5 Z9]
+ (2.1868423775146393e-07) [X3 Z4 X5 Z9]
+ (2.5935343909078466e-07) [Y2 Z3 Y4 Z6]
+ (2.5935343909078466e-07) [X2 Z3 X4 Z6]
+ (2.5935343909078466e-07) [Y3 Z4 Y5 Z7]
+ (2.5935343909078466e-07) [X3 Z4 X5 Z7]
+ (3.6060718679088436e-07) [Y0 Z1 Z2 Y4]
+ (3.6060718679088436e-07) [X0 Z1 Z2 X4]
+ (3.6060718679088436e-07) [Y1 Z3 Z4 Y5]
+ (3.6060718679088436e-07) [X1 Z3 Z4 X5]
+ (5.471647744714756e-07) [Y1 X2 X11 Y12]
+ (5.471647744714756e-07) [X1 Y2 Y11 X12]
+ (5.627851911438568e-07) [Y0 X1 X11 Y12]
+ (5.627851911438568e-07) [Y0 Y1 Y11 Y12]
+ (5.627851911438568e-07) [X0 X1 X11 X12]
+ (5.627851911438568e-07) [X0 Y1 Y11 X12]
+ (6.628614201756798e-07) [Y8 X9 X11 Y12]
+ (6.628614201756798e-07) [Y8 Y9 Y11 Y12]
+ (6.628614201756798e-07) [X8 X9 X11 X12]
+ (6.628614201756798e-07) [X8 Y9 Y11 X12]
+ (1.1094407591750189e-06) [Z2 Y11 Z12 Y13]
+ (1.1094407591750189e-06) [Z2 X11 Z12 X13]
+ (1.1094407591750189e-06) [Z3 Y10 Z11 Y12]
+ (1.1094407591750189e-06) [Z3 X10 Z11 X12]
+ (1.6021167406634066e-06) [Z2 Y3 Z4 Y5]
+ (1.6021167406634066e-06) [Z2 X3 Z4 X5]
+ (1.878210124768003e-06) [Z4 Y10 Z11 Y12]
+ (1.878210124768003e-06) [Z4 X10 Z11 X12]
+ (1.878210124768003e-06) [Z5 Y11 Z12 Y13]
+ (1.878210124768003e-06) [Z5 X11 Z12 X13]
+ (2.172669101545983e-06) [Y2 X3 X11 Y12]
+ (2.172669101545983e-06) [Y2 Y3 Y11 Y12]
+ (2.172669101545983e-06) [X2 X3 X11 X12]
+ (2.172669101545983e-06) [X2 Y3 Y11 X12]
+ (3.117447946076816e-06) [Y0 Z2 Z3 Y4]
+ (3.117447946076816e-06) [X0 Z2 Z3 X4]
+ (3.5390541845148905e-06) [Y2 Z3 Y4 Z12]
+ (3.5390541845148905e-06) [X2 Z3 X4 Z12]
+ (3.5390541845148905e-06) [Y3 Z4 Y5 Z13]
+ (3.5390541845148905e-06) [X3 Z4 X5 Z13]
+ (4.281913884919525e-06) [Y4 Z5 Y6 Z11]
+ (4.281913884919525e-06) [X4 Z5 X6 Z11]
+ (4.281913884919525e-06) [Y5 Z6 Y7 Z10]
+ (4.281913884919525e-06) [X5 Z6 X7 Z10]
+ (5.275883122220327e-06) [Y3 X4 X12 Y13]
+ (5.275883122220327e-06) [Y3 Y4 Y12 Y13]
+ (5.275883122220327e-06) [X3 X4 X12 X13]
+ (5.275883122220327e-06) [X3 Y4 Y12 X13]
+ (5.9743117135082775e-06) [Y5 X6 X10 Y11]
+ (5.9743117135082775e-06) [Y5 Y6 Y10 Y11]
+ (5.9743117135082775e-06) [X5 X6 X10 X11]
+ (5.9743117135082775e-06) [X5 Y6 Y10 X11]
+ (7.95441317632806e-06) [Y10 Z11 Y12 Z13]
+ (7.95441317632806e-06) [X10 Z11 X12 Z13]
+ (8.814937306735218e-06) [Y2 Z3 Y4 Z13]
+ (8.814937306735218e-06) [X2 Z3 X4 Z13]
+ (8.814937306735218e-06) [Y3 Z4 Y5 Z12]
+ (8.814937306735218e-06) [X3 Z4 X5 Z12]
+ (0.0002921986261110701) [Y7 X8 X9 Y10]
+ (0.0002921986261110701) [X7 Y8 Y9 X10]
+ (0.0004956762314916702) [Y2 Z4 Z5 Y6]
+ (0.0004956762314916702) [X2 Z4 Z5 X6]
+ (0.001105903769189622) [Y0 Z1 Y2 Z5]
+ (0.001105903769189622) [X0 Z1 X2 Z5]
+ (0.001105903769189622) [Y1 Z2 Y3 Z4]
+ (0.001105903769189622) [X1 Z2 X3 Z4]
+ (0.0016638798784907865) [Y2 Z3 Z4 Y6]
+ (0.0016638798784907865) [X2 Z3 Z4 X6]
+ (0.0016638798784907865) [Y3 Z5 Z6 Y7]
+ (0.0016638798784907865) [X3 Z5 Z6 X7]
+ (0.0017560707018411793) [Y0 Z1 Y2 Z11]
+ (0.0017560707018411793) [X0 Z1 X2 Z11]
+ (0.0017560707018411793) [Y1 Z2 Y3 Z10]
+ (0.0017560707018411793) [X1 Z2 X3 Z10]
+ (0.0023262306231580207) [Y0 Z1 Y2 Z13]
+ (0.0023262306231580207) [X0 Z1 X2 Z13]
+ (0.0023262306231580207) [Y1 Z2 Y3 Z12]
+ (0.0023262306231580207) [X1 Z2 X3 Z12]
+ (0.002745836470186813) [Y0 X1 X4 Y5]
+ (0.002745836470186813) [X0 Y1 Y4 X5]
+ (0.002929768674750974) [Y0 Z1 Y2 Z9]
+ (0.002929768674750974) [X0 Z1 X2 Z9]
+ (0.002929768674750974) [Y1 Z2 Y3 Z8]
+ (0.002929768674750974) [X1 Z2 X3 Z8]
+ (0.00327697193123158) [Y0 Z1 Y2 Z3]
+ (0.00327697193123158) [X0 Z1 X2 Z3]
+ (0.00334761753066612) [Y0 Z1 Y2 Z7]
+ (0.00334761753066612) [X0 Z1 X2 Z7]
+ (0.00334761753066612) [Y1 Z2 Y3 Z6]
+ (0.00334761753066612) [X1 Z2 X3 Z6]
+ (0.0035552901955042083) [Y0 Z1 Y2 Z10]
+ (0.0035552901955042083) [X0 Z1 X2 Z10]
+ (0.0035552901955042083) [Y1 Z2 Y3 Z11]
+ (0.0035552901955042083) [X1 Z2 X3 Z11]
+ (0.005143391768825107) [Y3 Y4 X5 X6]
+ (0.005143391768825107) [X3 X4 Y5 Y6]
+ (0.005283776488402963) [Y0 X1 X12 Y13]
+ (0.005283776488402963) [X0 Y1 Y12 X13]
+ (0.005530759218631487) [Y0 Z1 Y2 Z4]
+ (0.005530759218631487) [X0 Z1 X2 Z4]
+ (0.005530759218631487) [Y1 Z2 Y3 Z5]
+ (0.005530759218631487) [X1 Z2 X3 Z5]
+ (0.006087822480561868) [Y8 X9 X12 Y13]
+ (0.006087822480561868) [X8 Y9 Y12 X13]
+ (0.0065093612011772415) [Y0 X1 X8 Y9]
+ (0.0065093612011772415) [X0 Y1 Y8 X9]
+ (0.006888194352970583) [Y0 X1 X6 Y7]
+ (0.006888194352970583) [X0 Y1 Y6 X7]
+ (0.00690123824979723) [Y0 Z1 Y2 Z12]
+ (0.00690123824979723) [X0 Z1 X2 Z12]
+ (0.00690123824979723) [Y1 Z2 Y3 Z13]
+ (0.00690123824979723) [X1 Z2 X3 Z13]
+ (0.00715693491985695) [Y4 X5 X8 Y9]
+ (0.00715693491985695) [X4 Y5 Y8 X9]
+ (0.0077314252507752835) [Y0 X1 X10 Y11]
+ (0.0077314252507752835) [X0 Y1 Y10 X11]
+ (0.00803252091882133) [Y0 Z1 Y2 Z6]
+ (0.00803252091882133) [X0 Z1 X2 Z6]
+ (0.00803252091882133) [Y1 Z2 Y3 Z7]
+ (0.00803252091882133) [X1 Z2 X3 Z7]
+ (0.009560705729135954) [Y8 X9 X10 Y11]
+ (0.009560705729135954) [X8 Y9 Y10 X11]
+ (0.011055020596132017) [Y0 Z1 Y2 Z8]
+ (0.011055020596132017) [X0 Z1 X2 Z8]
+ (0.011055020596132017) [Y1 Z2 Y3 Z9]
+ (0.011055020596132017) [X1 Z2 X3 Z9]
+ (0.01128519020084091) [Y5 Y6 X11 X12]
+ (0.01128519020084091) [X5 X6 Y11 Y12]
+ (0.011307274008848133) [Y7 Z8 Z9 Y11]
+ (0.011307274008848133) [X7 Z8 Z9 X11]
+ (0.011982389010247962) [Y4 X5 X6 Y7]
+ (0.011982389010247962) [X4 Y5 Y6 X7]
+ (0.013873381748426129) [Y6 X7 X8 Y9]
+ (0.013873381748426129) [X6 Y7 Y8 X9]
+ (0.014583648907612656) [Y0 X1 X2 Y3]
+ (0.014583648907612656) [X0 Y1 Y2 X3]
+ (0.015577208063976474) [Y2 X3 X12 Y13]
+ (0.015577208063976474) [X2 Y3 Y12 X13]
+ (0.017366118994651427) [Y6 X7 X12 Y13]
+ (0.017366118994651427) [X6 Y7 Y12 X13]
+ (0.017680067952481535) [Y4 X5 X10 Y11]
+ (0.017680067952481535) [X4 Y5 Y10 X11]
+ (0.01782514099578644) [Y6 X7 X10 Y11]
+ (0.01782514099578644) [X6 Y7 Y10 X11]
+ (0.01902824244384732) [Y3 X4 X11 Y12]
+ (0.01902824244384732) [X3 Y4 Y11 X12]
+ (0.025384657508457455) [Y2 X3 X10 Y11]
+ (0.025384657508457455) [X2 Y3 Y10 X11]
+ (0.02868518371610595) [Y10 X11 X12 Y13]
+ (0.02868518371610595) [X10 Y11 Y12 X13]
+ (0.02981242451734574) [Y6 Z7 Z8 Y10]
+ (0.02981242451734574) [X6 Z7 Z8 X10]
+ (0.02981242451734574) [Y7 Z9 Z10 Y11]
+ (0.02981242451734574) [X7 Z9 Z10 X11]
+ (0.03010462314345681) [Y6 Z7 Z9 Y10]
+ (0.03010462314345681) [X6 Z7 Z9 X10]
+ (0.03010462314345681) [Y7 Z8 Z10 Y11]
+ (0.03010462314345681) [X7 Z8 Z10 X11]
+ (0.030787505389143925) [Y6 Z8 Z9 Y10]
+ (0.030787505389143925) [X6 Z8 Z9 X10]
+ (0.031143817988967145) [Y2 X3 X6 Y7]
+ (0.031143817988967145) [X2 Y3 Y6 X7]
+ (0.03583956795335352) [Y2 X3 X4 Y5]
+ (0.03583956795335352) [X2 Y3 Y4 X5]
+ (0.03619412355904263) [Y2 X3 X8 Y9]
+ (0.03619412355904263) [X2 Y3 Y8 X9]
+ (0.038314670294803906) [Y4 X5 X12 Y13]
+ (0.038314670294803906) [X4 Y5 Y12 X13]
+ (0.10433064780651384) [Z0 Y1 Z2 Y3]
+ (0.10433064780651384) [Z0 X1 Z2 X3]
+ (-0.1213327691104237) [Y2 Z3 Z4 Z5 Y6]
+ (-0.1213327691104237) [X2 Z3 Z4 Z5 X6]
+ (-0.12133276911042369) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133276911042369) [X3 Z4 Z5 Z6 X7]
+ (3.20207688045785e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.20207688045785e-06) [X0 Z1 Z2 Z3 X4]
+ (3.20207688045785e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.20207688045785e-06) [X1 Z2 Z3 Z4 X5]
+ (0.22848106564918766) [Y6 Z7 Z8 Z9 Y10]
+ (0.22848106564918766) [X6 Z7 Z8 Z9 X10]
+ (0.22848106564918771) [Y7 Z8 Z9 Z10 Y11]
+ (0.22848106564918771) [X7 Z8 Z9 Z10 X11]
+ (-0.03276765782329051) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.03276765782329051) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.03276765782329051) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.03276765782329051) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.02711503684527316) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.02711503684527316) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.02711503684527316) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.02711503684527316) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.02599617759802125) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.02599617759802125) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.017561202409646183) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.017561202409646183) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.017561202409646183) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.017561202409646183) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.014564531231173026) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564531231173026) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564531231173026) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564531231173026) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.012215040997613943) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.012215040997613943) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.012215040997613943) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.012215040997613943) [X4 Z5 X6 X11 Z12 X13]
+ (-0.012215040997613943) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.012215040997613943) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.012215040997613943) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.012215040997613943) [X5 Z6 X7 X10 Z11 X12]
+ (-0.011756013419819272) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.011756013419819272) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.011756013419819272) [X3 Z4 Z5 X6 X8 X9]
+ (-0.011756013419819272) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.008764827575688791) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.008764827575688791) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.008764827575688791) [X2 Z3 Z4 X5 X11 X12]
+ (-0.008764827575688791) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.008764827575688791) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.008764827575688791) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.008764827575688791) [X3 X4 X10 Z11 Z12 X13]
+ (-0.008764827575688791) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.008125251921381043) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125251921381043) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.007306759928832969) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306759928832969) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306759928832969) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306759928832969) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005805188989826911) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.005805188989826911) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.005805188989826911) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.005805188989826911) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.005652620978017353) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.005652620978017353) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.005652620978017353) [X0 X1 X3 Z4 Z5 X6]
+ (-0.005652620978017353) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.005143391768825107) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.005143391768825107) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.005143391768825107) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.005143391768825107) [X2 Z3 X4 X5 Z6 X7]
+ (-0.00468490338815521) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.00468490338815521) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668620318776311) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668620318776311) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.004575007626639211) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.004575007626639211) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424855449441864) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424855449441864) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.004158797381840055) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.004158797381840055) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.004158797381840055) [X3 Z4 Z5 X6 X12 X13]
+ (-0.004158797381840055) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.0034937903598901547) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.0034937903598901547) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.0034937903598901547) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.0034937903598901547) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.002779026799025558) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.002779026799025558) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.00229395661135247) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.00229395661135247) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.001799219493663029) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.001799219493663029) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278753941369525) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.0017278753941369525) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298507967730315) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298507967730315) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298507967730315) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298507967730315) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533856254125517) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533856254125517) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008145313270956716) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008145313270956716) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008145313270956716) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008145313270956716) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.735036880591118e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735036880591118e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735036880591118e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735036880591118e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.774817864831235e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.774817864831235e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.774817864831235e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.774817864831235e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.518362215877677e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.518362215877677e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.518362215877677e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.518362215877677e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.444344676137002e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.444344676137002e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.444344676137002e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.444344676137002e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.524373848671525e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.524373848671525e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.524373848671525e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.524373848671525e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.2900284334510646e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.2900284334510646e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.2900284334510646e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.2900284334510646e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.974311713508278e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.974311713508278e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.275883122220327e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.275883122220327e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.643051068575135e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.643051068575135e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.556569218247174e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.556569218247174e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.2532242256376586e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.2532242256376586e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.7696594521258432e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.7696594521258432e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.6945132946322484e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.6945132946322484e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.6102971307280087e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.6102971307280087e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.6102971307280087e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.6102971307280087e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.313145500159117e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.313145500159117e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.277483195679268e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.277483195679268e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.277483195679268e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.277483195679268e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.2112283485124073e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.2112283485124073e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.2112283485124073e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.2112283485124073e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.151346311180102e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.151346311180102e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.088250711406215e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.088250711406215e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.172669101545983e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.172669101545983e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.4548424489921247e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.4548424489921247e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.330473188694232e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.330473188694232e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2283337824266118e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2283337824266118e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.0358477600827703e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.0358477600827703e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956895372502105e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956895372502105e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733197742537358e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733197742537358e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733197742537358e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733197742537358e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.628614201756798e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.628614201756798e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.556281914629969e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.556281914629969e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.556281914629969e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.556281914629969e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.418291574605165e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.418291574605165e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.418291574605165e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.418291574605165e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.927453083008435e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927453083008435e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927453083008435e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927453083008435e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627851911438568e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627851911438568e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660624682167e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660624682167e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660624682167e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660624682167e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660624682167e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660624682167e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660624682167e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660624682167e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.838052750810973e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.838052750810973e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.5707613291035953e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.5707613291035953e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.3281393504874045e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.3281393504874045e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.086826565048128e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.086826565048128e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.086826565048128e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.086826565048128e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.4473231287524075e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.4473231287524075e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.371328947882606e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.371328947882606e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.371328947882606e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.371328947882606e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.1990516188561227e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.1990516188561227e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.9332412770657797e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.9332412770657797e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.9332412770657797e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.9332412770657797e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.839420915418176e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.839420915418176e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.839420915418176e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.839420915418176e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.5510539175734317e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.5510539175734317e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.5510539175734317e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.5510539175734317e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.3807781480485453e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.3807781480485453e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.3807781480485453e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.3807781480485453e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.3807781480485453e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.3807781480485453e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.3807781480485453e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.3807781480485453e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.3807781480485453e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.3807781480485453e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.3807781480485453e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.3807781480485453e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.291969486330071e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.291969486330071e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.1076325599191062e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.1076325599191062e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.1076325599191062e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.1076325599191062e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.1076325599191062e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.1076325599191062e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.1076325599191062e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.1076325599191062e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.057446595289238e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057446595289238e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057446595289238e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057446595289238e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649310134966374e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649310134966374e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649310134966374e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649310134966374e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.8394209154181763e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.8394209154181763e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.8394209154181763e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.8394209154181763e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.1990516188561227e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.1990516188561227e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.4473231287524075e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.4473231287524075e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.236259961379243e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.236259961379243e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.236259961379243e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.236259961379243e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.3281393504874045e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.3281393504874045e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.5707613291035953e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.5707613291035953e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.838052750810973e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.838052750810973e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.627851911438568e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627851911438568e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.628614201756798e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.628614201756798e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956895372502105e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956895372502105e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.306536652006789e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.306536652006789e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.306536652006789e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.306536652006789e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.0358477600827703e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.0358477600827703e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2283337824266118e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2283337824266118e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.2393363217054917e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393363217054917e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393363217054917e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393363217054917e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.330473188694232e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.330473188694232e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.4548424489921247e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.4548424489921247e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.172669101545983e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.172669101545983e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.088250711406215e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.088250711406215e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.117447946076816e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.117447946076816e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.151346311180102e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.151346311180102e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.313145500159117e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.313145500159117e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.334331289370106e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.334331289370106e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.6945132946322484e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.6945132946322484e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.183932559383917e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.183932559383917e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.556569218247174e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.556569218247174e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.643051068575135e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.643051068575135e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.275883122220327e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.275883122220327e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.974311713508278e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.974311713508278e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.00029219862611107013) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.00029219862611107013) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.00029219862611107013) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.00029219862611107013) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004956762314916702) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004956762314916702) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650070219498998) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650070219498998) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650070219498998) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650070219498998) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533856254125517) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533856254125517) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.0016095313817213776) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.0016095313817213776) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.0016095313817213776) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.0016095313817213776) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0016676041811440694) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.0016676041811440694) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.0016676041811440694) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.0016676041811440694) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.0017278753941369525) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.0017278753941369525) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.001799219493663029) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.001799219493663029) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.00229395661135247) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.00229395661135247) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.0024629170071339295) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.0024629170071339295) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.0024629170071339295) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.0024629170071339295) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.003961560792496539) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.003961560792496539) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.003961560792496539) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.003961560792496539) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.004424855449441864) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424855449441864) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004575007626639211) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.004575007626639211) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668620318776311) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668620318776311) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.00468490338815521) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.00468490338815521) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005324835234221684) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.005324835234221684) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.005324835234221684) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.005324835234221684) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.005368659358109517) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368659358109517) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368659358109517) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368659358109517) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.007960880725921557) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.007960880725921557) [X4 Z5 X6 X10 Z11 X12]
+ (0.007960880725921557) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.007960880725921557) [X5 Z6 X7 X11 Z12 X13]
+ (0.008125251921381043) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125251921381043) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.008890731522694588) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.008890731522694588) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.008890731522694588) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.008890731522694588) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.01026341486815853) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.01026341486815853) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.01026341486815853) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.01026341486815853) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.010540425907671484) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.010540425907671484) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.010540425907671484) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.010540425907671484) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010960074940542552) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010960074940542552) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010960074940542552) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010960074940542552) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307274008848133) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307274008848133) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.014411099430130917) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411099430130917) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411099430130917) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411099430130917) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.01522563075722659) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.01522563075722659) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.01522563075722659) [X3 Z4 Z5 X6 X10 X11]
+ (0.01522563075722659) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.015588250102380217) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.015588250102380217) [X2 Z3 X4 X10 Z11 X12]
+ (0.015588250102380217) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.015588250102380217) [X3 Z4 X5 X11 Z12 X13]
+ (0.018266834869375522) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.018266834869375522) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.018266834869375522) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.018266834869375522) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.01902042317303994) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.01902042317303994) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.01902042317303994) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.01902042317303994) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.020175921723535502) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.020175921723535502) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.020175921723535502) [X4 Z5 Z6 X7 X11 X12]
+ (0.020175921723535502) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.020175921723535502) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.020175921723535502) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.020175921723535502) [X5 X6 X10 Z11 Z12 X13]
+ (0.020175921723535502) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.024353077678069004) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.024353077678069004) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.024353077678069004) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.024353077678069004) [X2 Z3 X4 X11 Z12 X13]
+ (0.024353077678069004) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.024353077678069004) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.024353077678069004) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.024353077678069004) [X3 Z4 X5 X10 Z11 X12]
+ (0.02438908253114946) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.02438908253114946) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.02438908253114946) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.02438908253114946) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.025104957138844506) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.025104957138844506) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.025104957138844506) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.025104957138844506) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.030787505389143925) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.030787505389143925) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.04587947078129807) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.04587947078129807) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.05600733087780753) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.05600733087780753) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.05600733087780753) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.05600733087780753) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661344) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.05608468124661344) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661344) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.05608468124661344) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.63127792860353e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.63127792860353e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.631277928603529e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631277928603529e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.5950860070907513e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.5950860070907513e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.59508600709075e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.59508600709075e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0427432770137834) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.0427432770137834) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (0.042743277013783415) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.042743277013783415) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.04764261217638316) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.04764261217638316) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.04764261217638316) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.04764261217638316) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.0417188138398218) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.0417188138398218) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.0417188138398218) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.0417188138398218) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.039564416322893425) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.039564416322893425) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.039564416322893425) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.039564416322893425) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.039359168022053165) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.039359168022053165) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.039359168022053165) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.039359168022053165) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.03931805194719762) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03931805194719762) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.03931805194719762) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03931805194719762) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03560837898831262) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.03560837898831262) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.029903789512624884) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.029903789512624884) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.029903789512624884) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.029903789512624884) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.028730779551905547) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.028730779551905547) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.028730779551905547) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.028730779551905547) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.02563723829602686) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.02563723829602686) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.02563723829602686) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.02563723829602686) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.024755463292891033) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.024755463292891033) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.024755463292891033) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.024755463292891033) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.024282117354693024) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.024282117354693024) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.023145130929529065) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.023145130929529065) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.02252844019601303) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.02252844019601303) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.021433810721600957) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.021433810721600957) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.021433810721600957) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.021433810721600957) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251617) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.019257505095251617) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.01902824244384732) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.01902824244384732) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.01888903030494292) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.01888903030494292) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.01888903030494292) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.01888903030494292) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.016024603689179566) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.016024603689179566) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.015225630757226591) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225630757226591) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.014603704729162127) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.014603704729162127) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564531231173026) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564531231173026) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.011756013419819272) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.011756013419819272) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.01128519020084091) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.01128519020084091) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.009841749246962626) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009841749246962626) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.009612634606847293) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612634606847293) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612634606847293) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612634606847293) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.008469978791023925) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.008469978791023925) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.007306759928832969) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.007306759928832969) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.00592379833656135) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.00592379833656135) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.005652620978017355) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.005652620978017355) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368659358109518) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368659358109518) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.004158797381840055) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.004158797381840055) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.003356670563832898) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.003356670563832898) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.003356670563832898) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.003356670563832898) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.003267513854423561) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.003267513854423561) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.003267513854423561) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.003267513854423561) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.0027790267990255575) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.0027790267990255575) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.0026860409778066037) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.0026860409778066037) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.0026860409778066037) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.0026860409778066037) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.00229395661135247) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.00229395661135247) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.00229395661135247) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.00229395661135247) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.0009581655836696516) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.0009581655836696516) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0009581655836696516) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.0009581655836696516) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.0009581655836696516) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.0009581655836696516) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0009581655836696516) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.0009581655836696516) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.00024636437569580905) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00024636437569580905) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303549252) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.00013840177303549252) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.00013840177303549252) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.00013840177303549252) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.735036880591118e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735036880591118e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585306152563e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585306152563e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.6103585306152563e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.6103585306152563e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808795717483e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.5316808795717483e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808795717483e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5316808795717483e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.80610277534811e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.80610277534811e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.80610277534811e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.80610277534811e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.0897994677474635e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.0897994677474635e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.0897994677474635e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.0897994677474635e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.652209669759234e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.652209669759234e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.652209669759234e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.652209669759234e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851834226397e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.481851834226397e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851834226397e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.481851834226397e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.071480736456765e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.071480736456765e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.071480736456765e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.071480736456765e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.734622038891345e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.734622038891345e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.734622038891345e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.734622038891345e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.728843147348688e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.728843147348688e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.728843147348688e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.728843147348688e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.2532242256376586e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.2532242256376586e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.7696594521258432e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.7696594521258432e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.5443954293653767e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.5443954293653767e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.5443954293653767e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.5443954293653767e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.5443954293653767e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.5443954293653767e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.5443954293653767e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.5443954293653767e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.3609563203987754e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563203987754e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563203987754e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.3609563203987754e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.103215604780577e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.103215604780577e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.103215604780577e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.103215604780577e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220983366744e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.0111220983366744e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220983366744e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.0111220983366744e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9429468367667924e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.9429468367667924e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9429468367667924e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9429468367667924e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174772698183e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.6541174772698183e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174772698183e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.6541174772698183e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.5224930676177536e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224930676177536e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224930676177536e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224930676177536e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224930676177536e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930676177536e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930676177536e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224930676177536e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.2283337824266118e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337824266118e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.2283337824266118e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337824266118e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.988770288277098e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988770288277098e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988770288277098e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988770288277098e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.86776510435079e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.86776510435079e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.86776510435079e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.86776510435079e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.189990975467485e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.189990975467485e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246207060447e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246207060447e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.471647744714756e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.471647744714756e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.5614471795916186e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.5614471795916186e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.5614471795916186e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.5614471795916186e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.523389678001538e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.523389678001538e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.42732310868548e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.42732310868548e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.42732310868548e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.42732310868548e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.3281393504874045e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.3281393504874045e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.3281393504874045e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.3281393504874045e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.086826565048128e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.086826565048128e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.8882935949697395e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935949697395e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935949697395e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.8882935949697395e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.371328947882606e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.371328947882606e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.8394209154181763e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.8394209154181763e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.057446595289238e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057446595289238e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.5371780955101156e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.5371780955101156e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.5371780955101156e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.5371780955101156e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.057446595289238e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057446595289238e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.209350644390285e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350644390285e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350644390285e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.209350644390285e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.703578355328367e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.703578355328367e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.703578355328367e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.703578355328367e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.8394209154181763e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.8394209154181763e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.371328947882606e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.371328947882606e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.086826565048128e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.086826565048128e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.523389678001538e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.523389678001538e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.471647744714756e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.471647744714756e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246207060447e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246207060447e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189990975467485e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.189990975467485e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.330473188694232e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.330473188694232e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.330473188694232e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.330473188694232e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288532435623478e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288532435623478e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288532435623478e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288532435623478e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.689348951502331e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.689348951502331e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.689348951502331e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.689348951502331e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.7455184005376678e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.7455184005376678e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.7455184005376678e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.7455184005376678e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.7455184005376678e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.7455184005376678e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.7455184005376678e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.7455184005376678e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.211842019120085e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.211842019120085e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.211842019120085e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.211842019120085e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.211842019120085e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.211842019120085e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.211842019120085e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.211842019120085e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.313145500159117e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.313145500159117e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.313145500159117e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.313145500159117e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.3343312893701054e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.3343312893701054e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.183932559383917e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.183932559383917e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.735036880591118e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735036880591118e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00024636437569580905) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00024636437569580905) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00044585351288408195) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.00044585351288408195) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.00044585351288408195) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.00044585351288408195) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940221543005248) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005248) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005248) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005248) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005248) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005248) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940221543005248) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005248) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533856254125517) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533856254125517) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533856254125517) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533856254125517) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0010435246534907584) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435246534907584) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435246534907584) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435246534907584) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.0012803060973496706) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.0012803060973496706) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.0012803060973496706) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.0012803060973496706) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.0013038004788127006) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.0013038004788127006) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.0013038004788127006) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.0013038004788127006) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.0022619660624823516) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.0022619660624823516) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.0022619660624823516) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.0022619660624823516) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.0022619660624823516) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.0022619660624823516) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.0022619660624823516) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.0022619660624823516) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.003989841456619303) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.003989841456619303) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.003989841456619303) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.003989841456619303) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.004158797381840055) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.004158797381840055) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.004311038507914319) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.004311038507914319) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.004311038507914319) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.004311038507914319) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.004636976661182568) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.004636976661182568) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.004636976661182568) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.004636976661182568) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.005114473831660393) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.005114473831660393) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.005114473831660393) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.005114473831660393) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.005114473831660393) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660393) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660393) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005114473831660393) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005241535382803876) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.005241535382803876) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.005241535382803876) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.005241535382803876) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.005262642473076836) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262642473076836) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262642473076836) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262642473076836) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368659358109518) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368659358109518) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005379937155839369) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.005379937155839369) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.005379937155839369) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.005379937155839369) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.005652620978017355) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.005652620978017355) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.00570849598596092) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.00570849598596092) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.00570849598596092) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.00570849598596092) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.00592379833656135) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.00592379833656135) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.007306759928832969) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.007306759928832969) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469978791023925) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.008469978791023925) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.009841749246962626) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.009841749246962626) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.01128519020084091) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.01128519020084091) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.011756013419819272) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.011756013419819272) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.014564531231173026) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564531231173026) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603704729162127) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.014603704729162127) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.015225630757226591) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225630757226591) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.016024603689179566) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.016024603689179566) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.01902824244384732) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.01902824244384732) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.019257505095251617) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.019257505095251617) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.04587947078129808) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04587947078129808) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.369370893661562) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.369370893661562) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.369370893661562) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.369370893661562) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.28164257767022977) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.28164257767022977) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.28164257767022965) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.28164257767022965) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.09065144207036475) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.09065144207036475) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.09065144207036475) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.09065144207036475) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0868473758986362) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0868473758986362) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0868473758986362) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0868473758986362) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635021950635017) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635021950635017) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635021950635017) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635021950635017) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214031) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.06752385099214031) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214031) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.06752385099214031) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03560837898831262) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.03560837898831262) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0349033433736618) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0349033433736618) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0349033433736618) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0349033433736618) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02459186088383002) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.02459186088383002) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02459186088383002) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.02459186088383002) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.024282117354693024) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.024282117354693024) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.023145130929529065) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.023145130929529065) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.02252844019601303) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.02252844019601303) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019538050311314767) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.019538050311314767) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.019538050311314767) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.019538050311314767) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.017091553155898886) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.017091553155898886) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.017091553155898886) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.017091553155898886) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.016024603689179566) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.016024603689179566) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.016024603689179566) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.016024603689179566) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.010311482489831788) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831788) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831788) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311482489831788) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.009841749246962626) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962626) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.009841749246962626) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962626) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209846) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209846) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209846) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008826368514209846) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008541996625454849) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454849) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454849) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454849) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454849) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454849) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454849) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008541996625454849) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008469978791023925) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023925) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.008469978791023925) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023925) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.004668620318776311) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668620318776311) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.0038764708993369546) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.0038764708993369546) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.0038040661717285433) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0038040661717285433) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0038040661717285433) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0038040661717285433) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178886) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178886) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003356670563832898) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.003356670563832898) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.003267513854423561) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.003267513854423561) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.00214136122310163) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.00214136122310163) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.0017278753941369525) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.0017278753941369525) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.001640754855312395) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.001640754855312395) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214169382) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0014528843214169382) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214169382) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0014528843214169382) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.000787089677102447) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.000787089677102447) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192743499487698) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192743499487698) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.00019400857029756564) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00019400857029756564) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303549252) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.00013840177303549252) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.141625221159449e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.141625221159449e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.141625221159449e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.141625221159449e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.071480736456765e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.071480736456765e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.151346311180102e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.151346311180102e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.088250711406215e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.088250711406215e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.9885117062732894e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.9885117062732894e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8742990714174194e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.8742990714174194e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.3609563203987754e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.3609563203987754e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.3002946562351854e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3002946562351854e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.1468376507760043e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.1468376507760043e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.1468376507760043e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.1468376507760043e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.352332103322363e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.352332103322363e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.352332103322363e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.352332103322363e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091637199219599e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637199219599e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637199219599e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637199219599e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637199219599e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637199219599e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637199219599e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091637199219599e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.074305986121887e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.074305986121887e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.074305986121887e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.074305986121887e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128986541983e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128986541983e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128986541983e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128986541983e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.86776510435079e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.86776510435079e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560692465069604e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692465069604e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692465069604e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692465069604e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692465069604e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692465069604e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692465069604e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560692465069604e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.997018422246007e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.997018422246007e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.997018422246007e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.997018422246007e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.997018422246007e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.997018422246007e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.997018422246007e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.997018422246007e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.5682475212180616e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.5682475212180616e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.5682475212180616e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.5682475212180616e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376739308540444e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.376739308540444e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376739308540444e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.376739308540444e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376739308540444e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.376739308540444e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.376739308540444e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.376739308540444e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.88829359496974e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.88829359496974e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6863815464536835e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.6863815464536835e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.703578355328367e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.703578355328367e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.209350644390285e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.209350644390285e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244535795e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773244535795e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244535795e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773244535795e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244535795e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773244535795e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773244535795e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773244535795e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253795693827e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253795693827e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9742253795693827e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9742253795693827e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0474716555794414e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.0474716555794414e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.0474716555794414e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.0474716555794414e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.209350644390285e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.209350644390285e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0717282184794673e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0717282184794673e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0717282184794673e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0717282184794673e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.200428749390195e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.200428749390195e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.200428749390195e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.200428749390195e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.703578355328367e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.703578355328367e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.3120943052398757e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.3120943052398757e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.3120943052398757e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.3120943052398757e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.6863815464536835e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815464536835e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.88829359496974e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.88829359496974e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.0922506161740935e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506161740935e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.0922506161740935e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506161740935e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.0922506161740935e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506161740935e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.0922506161740935e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506161740935e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.444597854112892e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.444597854112892e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.444597854112892e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.444597854112892e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.684915095281545e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.684915095281545e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.684915095281545e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.684915095281545e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246974425539914e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246974425539914e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246974425539914e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246974425539914e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246974425539914e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246974425539914e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246974425539914e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246974425539914e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.86776510435079e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.86776510435079e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3002946562351854e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3002946562351854e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.3609563203987754e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.3609563203987754e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.8742990714174194e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.8742990714174194e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.8836765760718476e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.8836765760718476e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.947356011737309e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.947356011737309e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.947356011737309e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.947356011737309e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117062732894e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9885117062732894e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.088250711406215e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.088250711406215e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.151346311180102e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.151346311180102e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.8462016713068296e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.8462016713068296e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.8462016713068296e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.8462016713068296e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.071480736456765e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.071480736456765e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.105526722033182e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.105526722033182e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.105526722033182e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.105526722033182e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146496327542016e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146496327542016e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146496327542016e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146496327542016e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159350501920718e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159350501920718e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159350501920718e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159350501920718e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427988656566086e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427988656566086e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427988656566086e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427988656566086e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935867718010598e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935867718010598e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935867718010598e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935867718010598e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348140938e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.253273348140938e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979825793450601e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979825793450601e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979825793450601e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979825793450601e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.205548411218528e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.205548411218528e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.205548411218528e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.205548411218528e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00013840177303549252) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.00013840177303549252) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.00018787053389545647) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.00018787053389545647) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.00018787053389545647) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00018787053389545647) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00019400857029756564) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00019400857029756564) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00024636437569580905) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569580905) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00024636437569580905) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569580905) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192743499487698) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192743499487698) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156734248908526) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156734248908526) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156734248908526) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156734248908526) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.000787089677102447) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.000787089677102447) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.0015324835230729962) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.0015324835230729962) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.0015324835230729962) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.0015324835230729962) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.001640754855312395) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.001640754855312395) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0017278753941369525) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.0017278753941369525) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.0024464971554158835) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.0024464971554158835) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.0024464971554158835) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.0024464971554158835) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.003267513854423561) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.003267513854423561) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.003356670563832898) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.003356670563832898) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.0034841573002178886) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0034841573002178886) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0038764708993369546) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.0038764708993369546) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.004668620318776311) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668620318776311) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278083) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.004767272188278083) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.004767272188278083) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278083) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.005286546538226853) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.005286546538226853) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.005286546538226853) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.005286546538226853) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.005408954422409951) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.005408954422409951) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.005408954422409951) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.005408954422409951) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.00592379833656135) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.00592379833656135) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.00592379833656135) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.00592379833656135) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.010757563953908951) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010757563953908951) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010757563953908951) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010757563953908951) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.014603704729162127) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.014603704729162127) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.014603704729162127) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.014603704729162127) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.0192995605793638) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0192995605793638) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0192995605793638) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0192995605793638) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0192995605793638) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0192995605793638) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0192995605793638) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0192995605793638) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.058591988733861795) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.058591988733861795) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.775950527310584e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.775950527310584e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.7759505273105854e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.7759505273105854e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0716503518100258) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0716503518100258) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.07165035181002584) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.07165035181002584) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251617) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019257505095251617) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010311482489831788) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311482489831788) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008826368514209846) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209846) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.007597464029770601) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.007597464029770601) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.007597464029770601) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.007597464029770601) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311885) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311885) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311885) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311885) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311885) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311885) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311885) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311885) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0053480515826766035) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0053480515826766035) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0053480515826766035) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0053480515826766035) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0038040661717285433) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0038040661717285433) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219416) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.0029841661681219416) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.0029841661681219416) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.0029841661681219416) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.0024464971554158835) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.0024464971554158835) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.0022494124470939965) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0022494124470939965) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022494124470939965) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0022494124470939965) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00214136122310163) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.00214136122310163) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.001863894282458715) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.001863894282458715) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.001863894282458715) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.001863894282458715) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.001863894282458715) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.001863894282458715) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.001863894282458715) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.001863894282458715) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.001640754855312395) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.001640754855312395) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.001640754855312395) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.001640754855312395) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0012223378081538385) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538385) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538385) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538385) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538385) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538385) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538385) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0012223378081538385) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0010283292378562728) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0010283292378562728) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0010283292378562728) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0010283292378562728) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1463061453049059e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1463061453049059e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.8742990714174194e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990714174194e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.8742990714174194e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990714174194e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3002946562351854e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3002946562351854e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3002946562351854e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3002946562351854e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.0444941297760369e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.0444941297760369e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.0444941297760369e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.0444941297760369e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.956079229765873e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.956079229765873e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.956079229765873e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.956079229765873e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105515036930539e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105515036930539e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105515036930539e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105515036930539e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661347212885456e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661347212885456e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661347212885456e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661347212885456e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341413782613e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341413782613e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189990975467485e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.189990975467485e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876621657963968e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876621657963968e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876621657963968e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876621657963968e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246207060447e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246207060447e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.523389678001538e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.523389678001538e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.076732531771019e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.076732531771019e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.076732531771019e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.076732531771019e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.0134714589175084e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0134714589175084e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.904599883977756e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.904599883977756e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.904599883977756e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.904599883977756e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.6667317543865276e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.6667317543865276e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.6667317543865276e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6667317543865276e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.850564192835333e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.850564192835333e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309319899238e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6569309319899238e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309319899238e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6569309319899238e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.850564192835333e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.850564192835333e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815464536835e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815464536835e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.6863815464536835e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815464536835e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714589175084e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.0134714589175084e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.523389678001538e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.523389678001538e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.6704023909074317e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.6704023909074317e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.6704023909074317e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.6704023909074317e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246207060447e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246207060447e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189990975467485e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.189990975467485e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341413782613e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341413782613e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.94947648777133e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.94947648777133e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.792493957710613e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.792493957710613e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.792493957710613e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.792493957710613e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.883676576071848e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.883676576071848e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.9885117062732894e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117062732894e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117062732894e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117062732894e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348140937e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253273348140937e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4017109735367616e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4017109735367616e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4017109735367616e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4017109735367616e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603693078228e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.5809603693078228e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603693078228e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.5809603693078228e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192743499487698) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487698) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192743499487698) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487698) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.000787089677102447) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.000787089677102447) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.000787089677102447) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.000787089677102447) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.001172634831644188) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.001172634831644188) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.001172634831644188) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.001172634831644188) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012366478019244975) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.0012366478019244975) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.0012366478019244975) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.0012366478019244975) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.002200964069500461) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002200964069500461) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002200964069500461) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002200964069500461) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798026) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798026) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798026) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798026) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798026) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798026) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002394972639798026) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798026) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0024464971554158835) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.0024464971554158835) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.0038040661717285433) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0038040661717285433) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.003876470899336955) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.003876470899336955) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.003876470899336955) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.003876470899336955) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.0042208139700464385) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.0042208139700464385) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.0042208139700464385) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.0042208139700464385) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.008826368514209846) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.008826368514209846) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.010311482489831788) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311482489831788) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257505095251617) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019257505095251617) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.058591988733861795) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.058591988733861795) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3987009014632985e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3987009014632985e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.3987009014632982e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.3987009014632982e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178886) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178886) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219416) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.0029841661681219416) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.00019400857029756564) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00019400857029756564) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453049059e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453049059e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.792493957710613e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.792493957710613e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540341413782613e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413782613e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540341413782613e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413782613e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.850564192835333e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.850564192835333e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.850564192835333e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.850564192835333e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458917508e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471458917508e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458917508e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471458917508e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.94947648777133e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.94947648777133e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.792493957710613e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.792493957710613e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756564) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756564) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0029841661681219416) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.0029841661681219416) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.0034841573002178886) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0034841573002178886) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
Expectation value of XIZ =  0.07715357869738931
Expectation value of XIZ =  0.07715357869738954
<H> =  3.8768259168631207
3.8768259168631203
 </code>
 </pre>
 </details>

---

## 19. tutorial_jax_transformations.html <a name="demo18"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_jax_transformations.html):

```
No jit time: 0.0118 seconds
First run time: 0.0900 seconds
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_jax_transformations.html):

```
No jit time: 0.0140 seconds
First run time: 0.0840 seconds
```

---

