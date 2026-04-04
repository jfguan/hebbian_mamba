# Removing Q/K Projections for Gated Delta Net Slighly Improves Performance at ~15% Less Parameters
Suprisingly, we can remove the query and key projections in Gated Delta Net by directly using:

1. Current hidden state as the query vector
2. Previous hidden state as the key vector

This results in both faster convergence, marginally better performance despite strictly less parameters, and saves ~12.5% to ~25% of a layer's parameters.

For a ~100M parameter model trained for 300M tokens on The Stack(Code), a shifted key Gated Delta Net has a fitted training loss of 1.02 compared to 1.03 of a normal Gated Delta net model.
The shift is similar to RWKV token lerp, but removes Q/K projections completely.

We also show the same concept does not apply to softmax attention. Mechanism discovered by Opus 4.6.

# Attention Basic Review
Attention uses x_t (hidden state at position t) to generate the key k_t and value v_t vectors, one per previous token, as well as the current query vector q_t.

In a simplified example with word tokens, we need to predict the blank:
0    x_1  x_2      x_3  x_4  x_5  x_6  x_7  x_8
The  dog  barked.  The  man  saw  the  dog  ____?
q_0  q_1  q_2      q_3  q_4  q_5  q_6  q_7  q_8
k_0  k_1  k_2      k_3  k_4  k_5  k_6  k_7  k_8
v_0  v_1  v_2      v_3  v_4  v_5  v_6  v_7  v_8

Key vectors encode what for a token, "what am I", value vectors encode for a token "what I mean in context", and the query vector encodes for my current prediction, what other tokens are relevant to me?

In our example, our query vector q_8, and the dot product q_8 · k_t tells us the relevance of previous token t. For example, `dog` and `barked` are more relevant than `The`.
After calculating relevance scores, normalized by softmax, we get a weighed average of all the previous value vectors that inform our final prediction.

# Linear Attention Basic Review
Because attention requires all previous K, V vectors, cost grows with sequence length. Linear attention use a fixed state size to represent the past K,V vectors.

Pros: no growing memory/compute costs.
Cons: no free lunch - compression is inherently lossy and recall is worse.

Explanation:
With two k,v vectors, first take the outer product v⊗k, written also as (v · k^T).
Afterwards, multiplying v⊗k by k again, we get v · (k^T @ k) = v · ‖k‖². You get a scaled v back.

Instead of storing each k, v separately we store the combined v⊗k in a fixed size matrix M by doing M += v⊗k. To get back v, just do M * k.  M as a lossy fixed size KV cache, which we continually add new v⊗k

However, because M is fixed size, eventually all the keys start to overlap, so if two keys were similar, querying will get return a combination of the two corresponding values.

In practice various gating and decay mechanisms mitigate the key collision/capacity issues.

# Shifted Key Trick
Normally, the q, k vectors are generated from learned q, k projections, but the shifted key trick skips the learned projections entirely. Instead we directly use (with x_t is the hidden state at position t):
1. x_{t-1} as the key vector k_t, for v_t. This binds the previous state to the current value.
2. x_t as the query vector. Due to the key shift, querying the memory matrix with x_t returns "for positions similar to x_t, what came after?"

Going back to our example:
                                            x_8
The  dog  barked.  The  man  saw  the  dog  ____?
0    0    x_1      x_2  x_3  x_4  x_5  x_6  x_7
v_0  v_1  v_2      v_3  v_4  v_5  v_6  v_7  v_8

In terms of key/value pairs:
The -> dog
dog -> barked
barked. -> The
The -> man
man -> saw
...

In predict the blank, we just predicted that token "dog" so our hidden state x_8 is similar x_1, which strengthen's the v_2 representation for "barked".

This hard prior fixes the symmetric memory matrix issue of linear attention normally solved by learned Q/K projections. Because the hidden state x_t is input to both the k_t, v_t vectors, the symmetric key value pairs don't encode what comes next: e.g. the key might represent "I am the dog token" and value might represent "meaning of dog". In our example prediction, without the shifted key, our current hidden state is "dog", so when we query the matrix, we get "meaning of dog" back, when we actually wanted "meaning of bark". 

This symmetry issue doesn't apply to softmax attention, which retains all previous keys to query against.

We can also think of the token shift as copy paste - after I see x, think of y whcih does seem extremely limiting - associations are restricted to neighboring tokens.
However,  empirically at 100M parameter sizes it still seems to work, perhaps suggesting that for linear attention models, the q, k projections are mostly about:
1. Learning to break the symmetry in the memory matrix
2. Forming good orthogonal keys to fully utilize the keyspace
3. Associating abstract concepts rather than raw words.

It seems that the raw hidden states serve these responsiblities well enough or better.

# Experiments
Disclaimer - all models are decently undertrained. Curves are fit on data on last 80% of training to avoid too much early training influence. Sequence length is 2048, vocab of 1024.

## 18M Scale Testing
We train a baseline 17.9M parameter Gated Delta Net and 14.7M Shifted Key Gated Delta Net models for 30M tokens, batch size 4, sequence length 2048 on coding tokens (The Stack). Shifted key GDN is the same, just removing the Q, K projections.

for the training losses with smoothed data points, and see the token shift performs better despite having less parameters and expressiveness.

However for transformers, the shifted key transformer performs worse. This suggests while softmax attention and linear attention have the same root, they do behave differently. While both are doing pattern matching, perhaps softmax attention does it through querying/recalling exact past keys, while linear attention does a fuzzier general pattern matching.

## 100M Scale Testing
We scale up to a 105M for Gated Delta Net and 86.2M Shifted Key Gated Delta Net, trained for 300M tokens, batch size 1.

The shifted key model maintains a small lead despite ~15% less parameters, as well as faster convergence due to not needing to learn QK projections.

Lastly, the shifted key model seems to produce "better" keys across it's layers:

1. Effective rank - how many different keys are being stored.
2. Avg Pairwise Cosine - how close and "jumbled" keys are for clean retrieval
3. Condition number - how well the keys as a whole use the dimensional "storage" space

The shifted key model performs better all metrics except 3. at layer 0 which is an artificat of addng a padding key since at position 0 there's no previous hidden state to use as the key.

# Conclusions
I'm not exactly sure why this works. While it seems to make intuitive sense that associations can be chained together to form memory, it is confusing that restriction of only associating directly neighboring tokens doesn't impact performance more. Perhaps this is too restrictive at scale, although it does seem to demonstrate linear attention related models are genuinely differnt in some way.
