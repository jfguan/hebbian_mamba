# Shifted Key Architectures
For architectures that utilize query and key projections like transformer and Gated Delta Net, we can replace the projections by simply using the previous hidden state directly as both key and query. This saves ~12.5% to 25% of a layer's parameters depending on expand parameter for free, which can be reinvested in more layers.

For a 100M parameter model trained for 200M tokens on The Stack(Code), a shifted key Gated Delta Net has a training loss of 1.08, .04 nats better than versus Gated Delta Net(1.12). We further decrease loss with a hybrid of sliding window + gated delta net to 1.04.

Afterwards is some discussion on the purpose of convolution and memory blocks in the model.

# Attention Basic Review + Shifted Key Trick
In a simplified example with word tokens, we need to predict the blank:
The dog barked. The man saw the dog ____?

Attention uses x_t (hidden state at position t) to generate the key k_t and value v_t vectors, one per previous token, where the key represents a word's "relevance", and value represents the word's "abstract meaning".

0    x_1  x_2      x_3  x_4  x_5  x_6  x_7  x_8
The  dog  barked.  The  man  saw  the  dog  ____?
k_0  k_1  k_2      k_3  k_4  k_5  k_6  k_7  k_8
v_0  v_1  v_2      v_3  v_4  v_5  v_6  v_7  v_8

The model creates a query vector q_8 and the dot product q_8 · k_t tells us the relevance of previous token t. For example, `dog` and `barked` are more relevant than `The`.

The shifted key trick directly uses x_t (hidden state at position t) as the query and x_{t-1} as the key, so that the previous hidden state corresponds to what the model is thinking currently.

In our example:
state:   x_0  x_1  x_2     x_3  x_4  x_5  x_6  x_7
         The  dog  barked. The  man  saw  the  dog  ____?
keys:    0    x_0  x_1     x_2  x_3  x_4  x_5  x_6
values:  v_0  v_1  v_2     v_3  v_4  v_5  v_6  v_7


At x_7, the model is thinking about "dog" - because x_7 is similar to x_1, v_2 is weighted very high, which has the abstract concept of  "barked" and influences the model to predict "bark".

Basically, if your current token is `dog`, look back to what was predicted right after `dog`. While the bigram structure seems to only be able to encode neighboring tokens as related, adding a depth 4 conv before attention adds local information to the chain.

The idea is similar to RWKV-4's token lerp.

# Simplified Architectures - Sliding Window Attention and Gated Delta Net
Applying the trick, we remove the Q/K projections from Sliding Window Attention and Gated Delta Net.

For Gated delta net, we also remove the convolution on the Value projection, and remove the per dimension output gate as well, opting for a simpler per head output gate.

We make two types of layers in the same pattern:
1. Convolution
2. MLP
3. Memory block

The memory block can either be Sliding Window Attention or Gated Delta Net.

# Results
## 18M Scale Testing
Disclaimer - all models are heavily undertrained, and limited windows, so noise variation could be significant. However, the nat gaps seem quite large and are at least indicative.

We train a baselines Mamba and convolution models to compare against Linear Hebbian on a coding dataset, The Stack. Coding datasets help exaggerate recall differences since code is high density variable and function associations.

We see that the convolution + MLP model performs .3 nats worse than Mamba, but Linear Hebbian performs almost ~.9 nats better than Mamba. 

As we see on prose pg19, Linear Hebbian performs only ~.11 nats better, which makes sense, prose needs less recall.

## 100M Scale Testing
Scaling up to 100M, again we show that hebbian maintains a sigifnicant nat improvement around ~.5 over baseline Mamba, theorized just due to larger state to store memory in.

In addition, we create a new layer type delta hebbian, which is just hebbian with the delta rule. It's very close to Gated Delta Net, and splits the matrix block diagonally with heads for hardware efficiency. Placing a few delta hebbian layers at the end of the model further improves performance, theorizing they play a role similar to full attention layers in hybrid models like Olmo. The delta rule allows erasing values before rewriting, allowing cleaner key writes as well as preserving keys indefinitely for critical long term keywords.

## TODO - train multihead attention and GDN as well

## TODO

## Generalization Testing
To test that the architecture generalizes, we use Karpathy's nano-gpt setup, training a 124M model on fineweb-edu data for 10B tokens. With no tuning we see validation loss is ~.10 worse than the GPT-2.

# Personal Thoughts

## Convolutions are half of what you need
The MambaOut paper and Canon layer paper showed for both language and vision models, the SSM block doesn't contribute much, most of the strength comes from just stacking convolutions. Why? Stacked convolutions provide the immediate local context. 

Stacking 6 convolution + MLP layers with conv width of 4 tokens, you actually have a 4 + (4-1) * 5 = 19 nonlinear local token window to work with, which is quite a lot - maybe a full sentence.

If layer 1 has information on token t and past 3 tokens, layer 2 mixes t and past 6 tokens
Layer 1: [t to t-3]
Layer 2: [t to t-3, t-1 to t-4, t-2 to t-5, t-3 to t-6]
...

However, we obviously need longer range memory, which is why the other half is "associative recall" - remembering things. Mamba's state and effective memory is quite small. This however doesn't get exposed on short context general english because it doens't stress associative recall, so it performs similarly to Gated Delta Net. However, on code, which is almost purely associative recall, Gated Delta Net performs much better.

 While Mamba state updates are smarter, the small state fundementally cannot store as much information as full attention.


# Linear Attention Basic Review
Linear attention tries
For each k,v vectors, take the outer product v⊗k or (v · k^T).
Multiplying v⊗k by k again, we get v · (k^T @ k) = v · ‖k‖². You get v back.
Essentially, storing kv together allows retrieving v with k. 

After adding v⊗k to M, multiplying M by k retrieves v. M is your KV cache.
However, M is fixed size, so continually adding v⊗k start to overlap. Instead of a clean v retrieval, you get a weighted combination of all v's in M, which could be useful.

Every new token, we multiple M by some decay γ so old keys fade and "make room" for new keys.

## Full Attention is a tempting local minimum
Many challengers to Tranformers have tried and "failed", but the bitter lesson points toward linear architectures:
1. Learned designs beat human designed at scale. Attention sinks, strided, sliding, etc. are designed compression to managed cost. For strided attention, why every 4th token? For sliding window, what if the important token slides right outside the window?  Global layers plug some gaps but you can tell the models are referencing something they shouldn't be.
2. Linear complexity best utilizes compute. We're already hitting limits at 1M tokens, and image/video scales even worse. Hybrid models that patch the recall issue with full attention is still a smaller quadratic.
1. The whole point of deep learning is learned approximations eventually approach exact at scale.

Emphasis on scale, we can't see out of the local minimum that recall might not be as imporant as we think it is.

It’s quite hard to invest in linear architectures when all benchmarks suggests they are strictly worse, but benchmark design is biased:
1. Benchmarks effectively disallow rereading, a very unnatural test setting that biases hard for recall. 
Full attention remembers the question/context/answer choices clearly. Great!
Linear attnetion forgets, but humans don't read once then shoot from the hip - we re-read whatever we need.
2. Short context testing uses full attention, but real architectures use sliding window.
3. Recall and needle in a haystack are misleading metrics. Perfect recall can be mitigated if the model can externalize thought storage via notes.

A fair comparison requires agentic models that actionably re-request information, like how humans can ask “what was the question again?”. Unfortunately, agentic scale requires a lot of investment.

Fundementally, recall and runtime complexity are in tension. Good recall naturally requires more state, but more state is costly.


