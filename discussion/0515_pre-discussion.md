## 0513pre-discussion

#### 实验1

- r1_ADS 有点疑惑，写出他 **defense** 的 loss function :

$$
\ell = \ell_{\text{ce}}(f(x_{fgsm}),y) + \lambda \Vert g(x_{fgsm}) - g(x_{ifgsm})) \Vert_2^2
$$

若在 **Attack** 中加入 **reg** ，$x_{fgsm}$ 刚开始随机噪音。和 defense 一样的式子的话，再去算一个 $x_{ifgsm}$ 有一点奇怪，我是这么算的，但效果好像不太好。

- r2_ADS 在等待结果
- r3_ADS 还没改

#### 实验2 未改

#### os1-reg0 

- 效果不好，正则项起不了作用，还是会发生 grad masking，我这里理解为， 单独约束 两个 $loss$ 的差值好像没什么效果，所以这个lab的作者后面去 约束他们的 logits 这样的过程把。