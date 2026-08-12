

<!-- page: ? -->

### **Scaling Laws for Neural Language Models**



**Jared Kaplan** <sup>_∗_</sup>


Johns Hopkins University, OpenAI

```
   jaredk@jhu.edu

```


**Sam McCandlish** <sup>_∗_</sup>


OpenAI

```
sam@openai.com

```


**Tom Henighan**


OpenAI

```
henighan@openai.com

```

**Scott Gray**


OpenAI

```
scott@openai.com

```


**Tom B. Brown**


OpenAI

```
  tom@openai.com

```

**Alec Radford**


OpenAI

```
alec@openai.com

```


**Benjamin Chess**


OpenAI

```
 bchess@openai.com

```

**Jeffrey Wu**


OpenAI

```
jeffwu@openai.com

```


**Rewon Child**


OpenAI

```
 rewon@openai.com

```

**Dario Amodei**


OpenAI

```
damodei@openai.com

```


**Abstract**


We study empirical scaling laws for language model performance on the cross-entropy loss.
The loss scales as a power-law with model size, dataset size, and the amount of compute
used for training, with some trends spanning more than seven orders of magnitude. Other
architectural details such as network width or depth have minimal effects within a wide
range. Simple equations govern the dependence of overfitting on model/dataset size and the
dependence of training speed on model size. These relationships allow us to determine the
optimal allocation of a fixed compute budget. Larger models are significantly more sampleefficient, such that optimally compute-efficient training involves training very large models
on a relatively modest amount of data and stopping significantly before convergence.


_∗_ Equal contribution.


Contributions: Jared Kaplan and Sam McCandlish led the research. Tom Henighan contributed the LSTM experiments. Tom Brown, Rewon Child, and Scott Gray, and Alec Radford developed the optimized Transformer
implementation. Jeff Wu, Benjamin Chess, and Alec Radford developed the text datasets. Dario Amodei provided
guidance throughout the project.




<!-- page: ? -->

**Contents**


**1** **Introduction** **2**


**2** **Background and Methods** **6**


**3** **Empirical Results and Basic Power Laws** **7**


**4** **Charting the Infinite Data Limit and Overfitting** **10**


**5** **Scaling Laws with Model Size and Training Time** **12**


**6** **Optimal Allocation of the Compute Budget** **14**


**7** **Related Work** **18**


**8** **Discussion** **18**


**Appendices** **20**


**A** **Summary of Power Laws** **20**


**B** **Empirical Model of Compute-Efficient Frontier** **20**


**C** **Caveats** **22**


**D** **Supplemental Figures** **23**


**1** **Introduction**


Language provides a natural domain for the study of artificial intelligence, as the vast majority of reasoning tasks can be efficiently expressed and evaluated in language, and the world’s text provides a wealth of
data for unsupervised learning via generative modeling. Deep learning has recently seen rapid progress in language modeling, with state of the art models [RNSS18, DCLT18, YDY <sup>+</sup> 19, LOG <sup>+</sup> 19, RSR <sup>+</sup> 19] approaching
human-level performance on many specific tasks [WPN <sup>+</sup> 19], including the composition of coherent multiparagraph prompted text samples [RWC <sup>+</sup> 19].


One might expect language modeling performance to depend on model architecture, the size of neural models,
the computing power used to train them, and the data available for this training process. In this work we will
empirically investigate the dependence of language modeling loss on all of these factors, focusing on the
Transformer architecture [VSP <sup>+</sup> 17, LSP <sup>+</sup> 18]. The high ceiling and low floor for performance on language
tasks allows us to study trends over more than seven orders of magnitude in scale.


Throughout we will observe precise power-law scalings for performance as a function of training time, context length, dataset size, model size, and compute budget.


**1.1** **Summary**


Our key findings for Transformer language models are are as follows:


2Here we display predicted compute when using a sufficiently small batch size. See Figure 13 for comparison to the
purely empirical data.


2




<!-- page: ? -->

![](results/scaling_laws_llm/images/scaling_laws_llm.pdf-2-0.png)

PF-days, non-embedding tokens non-embedding





tokens



**Figure** **1** Language modeling performance improves smoothly as we increase the model size, datasetset
size, and amount of compute <sup>2</sup> used for training. For optimal performance all three factors must be scaled
up in tandem. Empirical performance has a power-law relationship with each individual factor when not
bottlenecked by the other two.


**Performance** **depends** **strongly** **on** **scale,** **weakly** **on** **model** **shape:** Model performance depends most
strongly on scale, which consists of three factors: the number of model parameters _N_ (excluding embeddings), the size of the dataset _D_, and the amount of compute _C_ used for training. Within reasonable limits,
performance depends very weakly on other architectural hyperparameters such as depth vs. width. (Section
3)


**Smooth** **power** **laws:** Performance has a power-law relationship with each of the three scale factors
_N, D, C_ when not bottlenecked by the other two, with trends spanning more than six orders of magnitude
(see Figure 1). We observe no signs of deviation from these trends on the upper end, though performance
must flatten out eventually before reaching zero loss. (Section 3)


**Universality of overfitting:** Performance improves predictably as long as we scale up _N_ and _D_ in tandem,
but enters a regime of diminishing returns if either _N_ or _D_ is held fixed while the other increases. The
performance penalty depends predictably on the ratio _N_ <sup>0</sup> <sup>_._</sup> <sup>74</sup> _/D_, meaning that every time we increase the
model size 8x, we only need to increase the data by roughly 5x to avoid a penalty. (Section 4)


**Universality** **of** **training:** Training curves follow predictable power-laws whose parameters are roughly
independent of the model size. By extrapolating the early part of a training curve, we can roughly predict the
loss that would be achieved if we trained for much longer. (Section 5)


**Transfer improves with test performance:** When we evaluate models on text with a different distribution
than they were trained on, the results are strongly correlated to those on the training validation set with
a roughly constant offset in the loss - in other words, transfer to a different distribution incurs a constant
penalty but otherwise improves roughly in line with performance on the training set. (Section 3.2.2)


**Sample efficiency:** Large models are more sample-efficient than small models, reaching the same level of
performance with fewer optimization steps (Figure 2) and using fewer data points (Figure 4).


**Convergence is inefficient:** When working within a fixed compute budget _C_ but without any other restrictions on the model size _N_ or available data _D_, we attain optimal performance by training _very large models_
and stopping _significantly short of convergence_ (see Figure 3). Maximally compute-efficient training would
therefore be far more sample efficient than one might expect based on training small models to convergence,
with data requirements growing very slowly as _D_ _∼_ _C_ <sup>0</sup> <sup>_._</sup> <sup>27</sup> with training compute. (Section 6)


**Optimal** **batch** **size:** The ideal batch size for training these models is roughly a power of the loss only,
and continues to be determinable by measuring the gradient noise scale [MKAT18]; it is roughly 1-2 million
tokens at convergence for the largest models we can train. (Section 5.1)


Taken together, these results show that language modeling performance improves smoothly and predictably
as we appropriately scale up model size, data, and compute. We expect that larger language models will
perform better and be more sample efficient than current models.


3




<!-- page: ? -->

Test Loss



10


4



Larger models require **fewer samples**
to reach the same performance



10


8


6


4



The optimal model size grows smoothly
with the loss target and compute budget




|Col1|Col2|
|---|---|
|||
||103 Params|
|109 Params||
|||



![](results/scaling_laws_llm/images/scaling_laws_llm.pdf-3-10.png)

**Figure 2** We show a series of language model training runs, with models ranging in size from 10 <sup>3</sup> to 10 <sup>9</sup>
parameters (excluding embeddings).



![](results/scaling_laws_llm/images/scaling_laws_llm.pdf-3-22.png)



Data requirements
grow relatively slowly


Optimal model size
increases very quickly



**Figure 3** As more compute becomes available, we can choose how much to allocate towards training larger
models, using larger batches, and training for more steps. We illustrate this for a billion-fold increase in
compute. For optimally compute-efficient training, most of the increase should go towards increased model
size. A relatively small increase in data is needed to avoid reuse. Of the increase in data, most can be used to
increase parallelism through larger batch sizes, with only a very small increase in serial training time required.


**1.2** **Summary of Scaling Laws**


The test loss of a Transformer trained to autoregressively model language can be predicted using a power-law
when performance is limited by only either the number of non-embedding parameters _N_, the dataset size _D_,
or the optimally allocated compute budget _C_ min (see Figure 1):


1. For models with a limited number of parameters, trained to convergence on sufficiently large
datasets:

_L_ ( _N_ ) = ( _N_ c _/N_ ) <sup>_αN_</sup> ; _αN_ _∼_ 0 _._ 076 _,_ _N_ c _∼_ 8 _._ 8 _×_ 10 <sup>13</sup> (non-embedding parameters) (1.1)


2. For large models trained with a limited dataset with early stopping:

_L_ ( _D_ ) = ( _D_ c _/D_ ) <sup>_αD_</sup> ; _αD_ _∼_ 0 _._ 095 _,_ _D_ c _∼_ 5 _._ 4 _×_ 10 <sup>13</sup> (tokens) (1.2)


3. When training with a limited amount of compute, a sufficiently large dataset, an optimally-sized
model, and a sufficiently small batch size (making optimal <sup>3</sup> use of compute):



c <sup>min</sup> _∼_ 3 _._ 1 _×_ 10 <sup>8</sup> (PF-days) (1.3)




- _αC_ <sup>min</sup>
; _αC_ <sup>min</sup>



_C_ <sup>min</sup> _∼_ 0 _._ 050 _,_ _C_ c <sup>min</sup>



_L_ ( _C_ min) =




- _C_ <u>c</u> <sup>min</sup> _/C_ min



3We also observe an empirical power-law trend with the training compute _C_ (Figure 1) while training at fixed batch
size, but it is the trend with _C_ min that should be used to make predictions. They are related by equation (5.5).


4




<!-- page: ? -->

![](results/scaling_laws_llm/images/scaling_laws_llm.pdf-4-0.png)

10 <sup>4</sup> 10 <sup>5</sup>

Estimated Smin



4.5


4.0


3.5


3.0


2.5



|Loss vs Model and Dataset Size|Col2|
|---|---|
|||
||Params<br>708M<br>302M<br>85M<br>3M<br>25M<br>393.2K|
|||


10 <sup>7</sup> 10 <sup>8</sup> 10 <sup>9</sup> 10 <sup>10</sup>

Tokens in Dataset







10 <sup>8</sup>


10 <sup>7</sup>


10 <sup>6</sup>



**Figure 4** **Left** : The early-stopped test loss _L_ ( _N, D_ ) varies predictably with the dataset size _D_ and model
size _N_ according to Equation (1.5). **Right** : After an initial transient period, learning curves for all model
sizes _N_ can be fit with Equation (1.6), which is parameterized in terms of _S_ min, the number of steps when
training at large batch size (details in Section 5.1).



These relations hold across eight orders of magnitude in _C_ min, six orders of magnitude in _N_, and over two
orders of magnitude in _D_ . They depend very weakly on model shape and other Transformer hyperparameters
(depth, width, number of self-attention heads), with specific numerical values associated with the Webtext2
training set [RWC <sup>+</sup> 19]. The power laws _α_ N _, α_ D _, αC_ <sup>min</sup> specify the degree of performance improvement

expected as we scale up _N_, _D_, or _C_ min; for example, doubling the number of parameters yields a loss that
is smaller by a factor 2 <sup>_−αN_</sup> = 0 _._ 95. The precise numerical values of _N_ c _, C_ c <sup>min</sup> _,_ and _D_ c depend on the

vocabulary size and tokenization and hence do not have a fundamental meaning.


The critical batch size, which determines the speed/efficiency tradeoff for data parallelism ([MKAT18]), also
roughly obeys a power law in _L_ :


_<u>B∗</u>_
_B_ crit ( _L_ ) = _L_ <sup>1</sup> <sup>_/αB_</sup> <sup>_,_</sup> _B∗_ _∼_ 2 _·_ 10 <sup>8</sup> tokens _,_ _αB_ _∼_ 0 _._ 21 (1.4)



Equation (1.1) and (1.2) together suggest that as we increase the model size, we should increase the dataset

_<u>αN</u>_

size sublinearly according to _D_ _∝_ _N_ _αD_ _∼_ _N_ <sup>0</sup> <sup>_._</sup> <sup>74</sup> . In fact, we find that there is a single equation combining

(1.1) and (1.2) that governs the simultaneous dependence on _N_ and _D_ and governs the degree of overfitting:




- _αD_
(1.5)




- <sup>_<u>α</u>_</sup> _<u>N</u>_
_αD_ + _<u>Dc</u>_

_D_



_L_ ( _N, D_ ) =



�� _<u>Nc</u>_
_N_



with fits pictured on the left in figure 4. We conjecture that this functional form may also parameterize the
trained log-likelihood for other generative modeling tasks.


When training a given model for a finite number of parameter update steps _S_ in the infinite data limit, after
an initial transient period, the learning curves can be accurately fit by (see the right of figure 4)




- _<u>Sc</u>_
_S_ min( _S_ )




- _αS_
(1.6)




- _αN_
+



_L_ ( _N, S_ ) =




- _<u>Nc</u>_
_N_



where _Sc_ _≈_ 2 _._ 1 _×_ 10 <sup>3</sup> and _αS_ _≈_ 0 _._ 76, and _S_ min( _S_ ) is the minimum possible number of optimization steps
(parameter updates) estimated using Equation (5.4).


When training within a fixed compute budget _C_, but with no other constraints, Equation (1.6) leads to the
prediction that the optimal model size _N_, optimal batch size _B_, optimal number of steps _S_, and dataset size
_D_ should grow as



_C_

_N_ _∝_ _C_ <sup>_α_</sup> <sup>min</sup>



_C_ _C_

<sup>min</sup> _/αN,_ _B_ _∝_ _C_ <sup>_α_</sup> <sup>min</sup>



_C_ _C_

<sup>min</sup> _/αB_ _,_ _S_ _∝_ _C_ <sup>_α_</sup> <sup>min</sup>



_C_

<sup>min</sup> _/αS_ _,_ _D_ = _B · S_ (1.7)



with



_αC_ <sup>min</sup>



_αC_ <sup>min</sup> = 1 _/_ (1 _/αS_ + 1 _/αB_ + 1 _/αN_ ) (1.8)

which closely matches the empirically optimal results _N_ _∝_ _C_ min <sup>0</sup> <sup>_._</sup> <sup>73,</sup> <sup>_B_</sup> <sup>_∝_</sup> <sup>_C_</sup> min <sup>0</sup> <sup>_._</sup> <sup>24,</sup> <sup>and</sup> <sup>_S_</sup> <sup>_∝_</sup> <sup>_C_</sup> min <sup>0</sup> <sup>_._</sup> <sup>03.</sup> <sup>As</sup> <sup>the</sup>

computational budget _C_ increases, it should be spent primarily on larger models, without dramatic increases
in training time or dataset size (see Figure 3). This also implies that as models grow larger, they become
increasingly sample efficient. In practice, researchers typically train smaller models for longer than would



min <sup>0</sup> <sup>_._</sup> <sup>73,</sup> <sup>_B_</sup> <sup>_∝_</sup> <sup>_C_</sup> min <sup>0</sup> <sup>_._</sup> <sup>24</sup>



min <sup>0</sup> <sup>_._</sup> <sup>24,</sup> <sup>and</sup> <sup>_S_</sup> <sup>_∝_</sup> <sup>_C_</sup> min <sup>0</sup> <sup>_._</sup> <sup>03</sup>



5




<!-- page: ? -->

be maximally compute-efficient because of hardware constraints. Optimal performance depends on total
compute as a power law (see Equation (1.3)).


We provide some basic theoretical motivation for Equation (1.5), an analysis of learning curve fits and their
implications for training time, and a breakdown of our results per token. We also make some brief comparisons to LSTMs and recurrent Transformers [DGV <sup>+</sup> 18].


**1.3** **Notation**


We use the following notation:


_•_ _L_   - the cross entropy loss in nats. Typically it will be averaged over the tokens in a context, but in
some cases we report the loss for specific tokens within the context.

_•_ _N_   - the number of model parameters, _excluding all vocabulary and positional embeddings_

_•_ _C_ _≈_ 6 _NBS_   - an estimate of the total non-embedding training compute, where _B_ is the batch size,
and _S_ is the number of training steps (ie parameter updates). We quote numerical values in PF-days,
where one PF-day = 10 <sup>15</sup> _×_ 24 _×_ 3600 = 8 _._ 64 _×_ 10 <sup>19</sup> floating point operations.

_•_ _D_   - the dataset size in tokens

_•_ _B_ crit   - the critical batch size [MKAT18], defined and discussed in Section 5.1. Training at the
critical batch size provides a roughly optimal compromise between time and compute efficiency.

_•_ _C_ min   - an estimate of the minimum amount of non-embedding compute to reach a given value of
the loss. This is the training compute that would be used if the model were trained at a batch size
much less than the critical batch size.

_•_ _S_ min – an estimate of the minimal number of training steps needed to reach a given value of the loss.
This is also the number of training steps that would be used if the model were trained at a batch size
much greater than the critical batch size.

_•_ _αX_   - power-law exponents for the scaling of the loss as _L_ ( _X_ ) _∝_ 1 _/X_ <sup>_αX_</sup> where _X_ can be any of
_N, D, C, S, B, C_ <sup>min</sup> .


**2** **Background and Methods**


We train language models on WebText2, an extended version of the WebText [RWC <sup>+</sup> 19] dataset, tokenized
using byte-pair encoding [SHB15] with a vocabulary size _n_ vocab = 50257. We optimize the autoregressive log-likelihood (i.e. cross-entropy loss) averaged over a 1024-token context, which is also our principal
performance metric. We record the loss on the WebText2 test distribution and on a selection of other text
distributions. We primarily train decoder-only [LSP <sup>+</sup> 18, RNSS18] Transformer [VSP <sup>+</sup> 17] models, though
we also train LSTM models and Universal Transformers [DGV <sup>+</sup> 18] for comparison.


**2.1** **Parameter and Compute Scaling of Transformers**


We parameterize the Transformer architecture using hyperparameters _n_ layer (number of layers), _d_ model (dimension of the residual stream), _d_ ff (dimension of the intermediate feed-forward layer), _d_ attn (dimension of
the attention output), and _n_ heads (number of attention heads per layer). We include _n_ ctx tokens in the input
context, with _n_ ctx = 1024 except where otherwise noted.


We use _N_ to denote the model size, which we define as the number of _non-embedding_ parameters


_N_ _≈_ 2 _d_ model _n_ layer (2 _d_ attn + _d_ ff )

= 12 _n_ layer _d_ <sup>2</sup> model with the standard _d_ attn = _d_ ff _/_ 4 = _d_ model (2.1)

where we have excluded biases and other sub-leading terms. Our models also have _n_ vocab _d_ model parameters
in an embedding matrix, and use _n_ ctx _d_ model parameters for positional embeddings, but we do not include
these when discussing the ‘model size’ _N_ ; we will see that this produces significantly cleaner scaling laws.


Evaluating a forward pass of the Transformer involves roughly


_C_ forward _≈_ 2 _N_ + 2 _n_ layer _n_ ctx _d_ model (2.2)


add-multiply operations, where the factor of two comes from the multiply-accumulate operation used in
matrix multiplication. A more detailed per-operation parameter and compute count is included in Table 1.


6




<!-- page: ? -->

|Operation|Parameters|FLOPs per Token|
|---|---|---|
|Embed|(_n_vocab +_ n_ctx)_ d_model|4_d_model|
|Attention: QKV|_n_layer_d_model3_d_attn|2_n_layer_d_model3_d_attn|
|Attention: Mask|—|2_n_layer_n_ctx_d_attn|
|Attention: Project|_n_layer_d_attn_d_model|2_n_layer_d_attn_d_embd|
|Feedforward|_n_layer2_d_model_d_ﬀ|2_n_layer2_d_model_d_ﬀ|
|De-embed|—|2_d_model_n_vocab|
|**Total (Non-Embedding)**|_N_ = 2_d_model_n_layer (2_d_attn +_ d_ﬀ)|_C_forward = 2_N_ + 2_n_layer_n_ctx_d_attn|


**Table** **1** Parameter counts and compute (forward pass) estimates for a Transformer model. Sub-leading
terms such as nonlinearities, biases, and layer normalization are omitted.


For contexts and models with _d_ model _>_ _n_ ctx _/_ 12, the context-dependent computational cost per token is a
relatively small fraction of the total compute. Since we primarily study models where _d_ model _≫_ _n_ ctx _/_ 12,
we do not include context-dependent terms in our training compute estimate. Accounting for the backwards
pass (approximately twice the compute as the forwards pass), we then define the estimated non-embedding
compute as _C_ _≈_ 6 _N_ floating point operators per training token.


**2.2** **Training Procedures**


Unless otherwise noted, we train models with the Adam optimizer [KB14] for a fixed 2 _._ 5 _×_ 10 <sup>5</sup> steps with
a batch size of 512 sequences of 1024 tokens. Due to memory constraints, our largest models (more than
1B parameters) were trained with Adafactor [SS18]. We experimented with a variety of learning rates and
schedules, as discussed in Appendix D.6. We found that results at convergence were largely independent of
learning rate schedule. Unless otherwise noted, all training runs included in our data used a learning rate
schedule with a 3000 step linear warmup followed by a cosine decay to zero.


**2.3** **Datasets**


We train our models on an extended version of the WebText dataset described in [RWC <sup>+</sup> 19]. The original
WebText dataset was a web scrape of outbound links from Reddit through December 2017 which received at
least 3 karma. In the second version, WebText2, we added outbound Reddit links from the period of January
to October 2018, also with a minimum of 3 karma. The karma threshold served as a heuristic for whether
people found the link interesting or useful. The text of the new links was extracted with the Newspaper3k
python library. In total, the dataset consists of 20.3M documents containing 96 GB of text and 1 _._ 62 _×_ 10 <sup>10</sup>
words (as defined by `wc` ). We then apply the reversible tokenizer described in [RWC <sup>+</sup> 19], which yields
2 _._ 29 _×_ 10 <sup>10</sup> tokens. We reserve 6 _._ 6 _×_ 10 <sup>8</sup> of these tokens for use as a test set, and we also test on similarlyprepared samples of Books Corpus [ZKZ <sup>+</sup> 15], Common Crawl [Fou], English Wikipedia, and a collection
of publicly-available Internet Books.


**3** **Empirical Results and Basic Power Laws**


To characterize language model scaling we train a wide variety of models, varying a number of factors
including:


_•_ Model size (ranging in size from 768 to 1.5 billion non-embedding parameters)


_•_ Dataset size (ranging from 22 million to 23 billion tokens)


_•_ Shape (including depth, width, attention heads, and feed-forward dimension)


_•_ Context length (1024 for most runs, though we also experiment with shorter contexts)

_•_ Batch size (2 <sup>19</sup> for most runs, but we also vary it to measure the critical batch size)


7




<!-- page: ? -->

10%


8%


6%


4%


2%


0%





![](results/scaling_laws_llm/images/scaling_laws_llm.pdf-7-0.png)


|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|~~A wide range of architectures~~<br>|~~A wide range of architectures~~<br>|~~A wide range of architectures~~<br>|~~A wide range of architectures~~<br>|~~A wide range of architectures~~<br>|~~A wide range of architectures~~<br>|
|~~achieve similar performance~~|~~achieve similar performance~~|~~achieve similar performance~~|~~achieve similar performance~~|~~achieve similar performance~~|~~achieve similar performance~~|
|||||||
|||||||
|||||||
|||||||
|||||||


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
|~~22% additional compute~~|~~22% additional compute~~|~~22% additional compute~~|~~22% additional compute~~|~~22% additional compute~~|
||<br>|<br>|<br>|<br>|
|~~compensates for 1% loss increase~~|~~compensates for 1% loss increase~~|~~compensates for 1% loss increase~~|~~compensates for 1% loss increase~~|~~compensates for 1% loss increase~~|
||||||
||||||
||||||



![](results/scaling_laws_llm/images/scaling_laws_llm.pdf-7-1.png)

**Feed-Forward Ratio** (dff / dmodel)



**Feed-Forward Ratio** (dff / dmodel) **Attention Head Dimension** (dmodel / nhead)

**Aspect Ratio** (dmodel / nlayer)
50M Parameters 25M Parameters



25M Parameters



**Figure** **5** Performance depends very mildly on model shape when the total number of non-embedding
parameters _N_ is held fixed. The loss varies only a few percent over a wide range of shapes. Small differences
in parameter counts are compensated for by using the fit to _L_ ( _N_ ) as a baseline. Aspect ratio in particular can
vary by a factor of 40 while only slightly impacting performance; an ( _n_ layer _, d_ model) = (6 _,_ 4288) reaches a
loss within 3% of the (48 _,_ 1600) model used in [RWC <sup>+</sup> 19].



7

6

5


4


3


2



![](results/scaling_laws_llm/images/scaling_laws_llm.pdf-7-2.png)

10 <sup>6</sup> 10 <sup>7</sup> 10 <sup>8</sup> 10 <sup>9</sup>

Parameters (with embedding)



![](results/scaling_laws_llm/images/scaling_laws_llm.pdf-7-3.png)

10 <sup>3</sup> 10 <sup>4</sup> 10 <sup>5</sup> 10 <sup>6</sup> 10 <sup>7</sup> 10 <sup>8</sup> 10 <sup>9</sup>

Parameters (non-embedding)







7

6

5


4


3


2



**Figure** **6** **Left:** When we include embedding parameters, performance appears to depend strongly on the
number of layers in addition to the number of parameters. **Right:** When we exclude embedding parameters,
the performance of models with different depths converge to a single trend. Only models with fewer than 2
layers or with extreme depth-to-width ratios deviate significantly from the trend.


In this section we will display data along with empirically-motivated fits, deferring theoretical analysis to
later sections.


**3.1** **Approximate Transformer Shape and Hyperparameter Independence**


Transformer performance depends very weakly on the shape parameters _n_ layer _, n_ heads, and _d_ ff when we hold
the total non-embedding parameter count _N_ fixed. To establish these results we trained models with fixed
size while varying a single hyperparameter. This was simplest for the case of _n_ heads. When varying _n_ layer,
we simultaneously varied _d_ model while keeping _N_ _≈_ 12 _n_ layer _d_ <sup>2</sup> model <sup>fixed.</sup> <sup>Similarly,</sup> <sup>to</sup> <sup>vary</sup> <sup>_d_</sup> <sup>ff</sup> <sup>at</sup> <sup>fixed</sup>

model size we also simultaneously varied the _d_ model parameter, as required by the parameter counts in Table
1. Independence of _n_ layers would follow if deeper Transformers effectively behave as ensembles of shallower
models, as has been suggested for ResNets [VWB16]. The results are shown in Figure 5.


**3.2** **Performance with Non-Embedding Parameter Count** _N_


In Figure 6 we display the performance of a wide variety of models, ranging from small models with shape
( _n_ layer _, d_ model) = (2 _,_ 128) through billion-parameter models, ranging in shape from (6 _,_ 4288) through
(207 _,_ 768). Here we have trained to near convergence on the full WebText2 dataset and observe no overfitting (except possibly for the very largest models).


As shown in Figure 1, we find a steady trend with non-embedding parameter count _N_, which can be fit to the
first term of Equation (1.5), so that




- _αN_
(3.1)



_L_ ( _N_ ) _≈_




- _<u>Nc</u>_
_N_


8




<!-- page: ? -->

**LSTM** plateaus after <100 tokens
**Transformer** improves through the whole context



Test Loss



6


4


5


3


2





![](results/scaling_laws_llm/images/scaling_laws_llm.pdf-8-0.png)









10 <sup>1</sup> 10 <sup>2</sup> 10 <sup>3</sup>

Token Index in Context


|Transformers asymptotically outperform LSTMs due to improved use of long contexts|Col2|Col3|Col4|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
|3.6<br>4.2<br>3.0<br>2.4<br>4.8<br>5.4<br>105<br>108<br>106<br>107<br>109<br>Transformers<br>LSTMs<br>1 Layer<br>2 Layers<br>4 Layers<br> ss|3.6<br>4.2<br>3.0<br>2.4<br>4.8<br>5.4<br>105<br>108<br>106<br>107<br>109<br>Transformers<br>LSTMs<br>1 Layer<br>2 Layers<br>4 Layers<br> ss|3.6<br>4.2<br>3.0<br>2.4<br>4.8<br>5.4<br>105<br>108<br>106<br>107<br>109<br>Transformers<br>LSTMs<br>1 Layer<br>2 Layers<br>4 Layers<br> ss|3.6<br>4.2<br>3.0<br>2.4<br>4.8<br>5.4<br>105<br>108<br>106<br>107<br>109<br>Transformers<br>LSTMs<br>1 Layer<br>2 Layers<br>4 Layers<br> ss|3.6<br>4.2<br>3.0<br>2.4<br>4.8<br>5.4<br>105<br>108<br>106<br>107<br>109<br>Transformers<br>LSTMs<br>1 Layer<br>2 Layers<br>4 Layers<br> ss|3.6<br>4.2<br>3.0<br>2.4<br>4.8<br>5.4<br>105<br>108<br>106<br>107<br>109<br>Transformers<br>LSTMs<br>1 Layer<br>2 Layers<br>4 Layers<br> ss|3.6<br>4.2<br>3.0<br>2.4<br>4.8<br>5.4<br>105<br>108<br>106<br>107<br>109<br>Transformers<br>LSTMs<br>1 Layer<br>2 Layers<br>4 Layers<br> ss|3.6<br>4.2<br>3.0<br>2.4<br>4.8<br>5.4<br>105<br>108<br>106<br>107<br>109<br>Transformers<br>LSTMs<br>1 Layer<br>2 Layers<br>4 Layers<br> ss|
|3.6<br>4.2<br>3.0<br>2.4<br>4.8<br>5.4<br>105<br>108<br>106<br>107<br>109<br>Transformers<br>LSTMs<br>1 Layer<br>2 Layers<br>4 Layers<br> ss|Transformers<br>LSTMs<br>1 Layer<br>2 Layers<br>4 Layers|Transformers<br>LSTMs<br>1 Layer<br>2 Layers<br>4 Layers|Transformers<br>LSTMs<br>1 Layer<br>2 Layers<br>4 Layers|Transformers<br>LSTMs<br>1 Layer<br>2 Layers<br>4 Layers|Transformers<br>LSTMs<br>1 Layer<br>2 Layers<br>4 Layers|Transformers<br>LSTMs<br>1 Layer<br>2 Layers<br>4 Layers|Transformers<br>LSTMs<br>1 Layer<br>2 Layers<br>4 Layers|
|3.6<br>4.2<br>3.0<br>2.4<br>4.8<br>5.4<br>105<br>108<br>106<br>107<br>109<br>Transformers<br>LSTMs<br>1 Layer<br>2 Layers<br>4 Layers<br> ss|Transformers<br>LSTMs<br>1 Layer<br>2 Layers<br>4 Layers|05|106<br>|0||08<br>|09|



**Figure 7**


To observe these trends it is crucial to study performance as a function of _N_ ; if we instead use the total
parameter count (including the embedding parameters) the trend is somewhat obscured (see Figure 6). This
suggests that the embedding matrix can be made smaller without impacting performance, as has been seen in
recent work [LCG <sup>+</sup> 19].


Although these models have been trained on the WebText2 dataset, their test loss on a variety of other datasets
is also a power-law in _N_ with nearly identical power, as shown in Figure 8.


**3.2.1** **Comparing to LSTMs and Universal Transformers**


In Figure 7 we compare LSTM and Transformer performance as a function of non-embedding parameter
count _N_ . The LSTMs were trained with the same dataset and context length. We see from these figures
that the LSTMs perform as well as Transformers for tokens appearing early in the context, but cannot match
the Transformer performance for later tokens. We present power-law relationships between performance and
context position Appendix D.5, where increasingly large powers for larger models suggest improved ability
to quickly recognize patterns.


We also compare the performance of standard Transformers to recurrent Transformers [DGV <sup>+</sup> 18] in Figure
17 in the appendix. These models re-use parameters, and so perform slightly better as a function of _N_, at the
cost of additional compute per-parameter.


**3.2.2** **Generalization Among Data Distributions**


We have also tested our models on a set of additional text data distributions. The test loss on these datasets
as a function of model size is shown in Figure 8; in all cases the models were trained only on the WebText2
dataset. We see that the loss on these other data distributions improves smoothly with model size, in direct
parallel with the improvement on WebText2. We find that generalization depends almost exclusively on the
in-distribution validation loss, and does not depend on the duration of training or proximity to convergence.
We also observe no dependence on model depth (see Appendix D.8).


**3.3** **Performance with Dataset Size and Compute**


We display empirical trends for the test loss as a function of dataset size _D_ (in tokens) and training compute
_C_ in Figure 1.


For the trend with _D_ we trained a model with ( _n_ layer _, n_ embd) = (36 _,_ 1280) on fixed subsets of the WebText2
dataset. We stopped training once the test loss ceased to decrease. We see that the resulting test losses can be
fit with simple power-law




- _αD_
(3.2)



_L_ ( _D_ ) _≈_




- _<u>Dc</u>_
_D_



in the dataset size. The data and fit appear in Figure 1.


The total amount of non-embedding compute used during training can be estimated as _C_ = 6 _NBS_, where
_B_ is the batch size, _S_ is the number of parameter updates, and the factor of 6 accounts for the forward and
backward passes. Thus for a given value of _C_ we can scan over all models with various _N_ to find the model


9




<!-- page: ? -->

7

6


5


4


3



![](results/scaling_laws_llm/images/scaling_laws_llm.pdf-9-0.png)



![](results/scaling_laws_llm/images/scaling_laws_llm.pdf-9-1.png)





10 <sup>4</sup> 10 <sup>5</sup> 10 <sup>6</sup> 10 <sup>7</sup> 10 <sup>8</sup> 10 <sup>9</sup>

Parameters (non-embedding)



5.0 4.5 4.0 3.5 3.0 2.5
Test Loss on Training Distribution



**Figure 8** **Left:** Generalization performance to other data distributions improves smoothly with model size,
with only a small and very slowly growing offset from the WebText2 training distribution. **Right:** Generalization performance depends only on training distribution performance, and not on the phase of training.
We compare generalization of converged models (points) to that of a single large model (dashed curves) as it
trains.


with the best performance on step _S_ = 6 _BSC_ <sup>.</sup> <sup>Note</sup> <sup>that</sup> <sup>in</sup> <sup>these</sup> <sup>results</sup> <sup>_the_</sup> <sup>_batch_</sup> <sup>_size_</sup> <sup>_B_</sup> <sup>_remains_</sup> <sup>_fixed_</sup> <sup>_for_</sup>
_all models_, which means that these empirical results are not truly optimal. We will account for this in later
sections using an adjusted _C_ min to produce cleaner trends.


The result appears as the heavy black line on the left-hand plot in Figure 1. It can be fit with




- _αC_
(3.3)



_L_ ( _C_ ) _≈_




- _<u>Cc</u>_
_C_



The figure also includes images of individual learning curves to clarify when individual models are optimal.
We will study the optimal allocation of compute more closely later on. The data strongly suggests that sample
efficiency improves with model size, and we also illustrate this directly in Figure 19 in the appendix.


**4** **Charting the Infinite Data Limit and Overfitting**


In Section 3 we found a number of basic scaling laws for language modeling performance. Here we will
study the performance of a model of size _N_ trained on a dataset with _D_ tokens while varying _N_ and _D_
simultaneously. We will empirically demonstrate that the optimally trained test loss accords with the scaling
law of Equation (1.5). This provides guidance on how much data we would need to train models of increasing
size while keeping overfitting under control.


**4.1** **Proposed** _L_ ( _N, D_ ) **Equation**


We have chosen the parameterization (1.5) (repeated here for convenience):




- _αD_
(4.1)




- <sup>_<u>α</u>_</sup> _<u>N</u>_
_αD_ + _<u>Dc</u>_

_D_



_L_ ( _N, D_ ) =



�� _<u>Nc</u>_
_N_



using three principles:


1. Changes in vocabulary size or tokenization are expected to rescale the loss by an overall factor. The
parameterization of _L_ ( _N, D_ ) (and all models of the loss) must naturally allow for such a rescaling.


2. Fixing _D_ and sending _N_ _→∞_, the overall loss should approach _L_ ( _D_ ). Conversely, fixing _N_ and
sending _D_ _→∞_ the loss must approach _L_ ( _N_ ).


3. _L_ ( _N, D_ ) should be analytic at _D_ = _∞_, so that it has a series expansion in 1 _/D_ with integer powers.
Theoretical support for this principle is significantly weaker than for the first two.


Our choice of _L_ ( _N, D_ ) satisfies the first requirement because we can rescale _Nc, Dc_ with changes in the
vocabulary. This also implies that the values of _Nc, Dc_ have no fundamental meaning.


10




<!-- page: ? -->

|Overfitting|Col2|
|---|---|
|||
||Data Size<br>21M<br>43M<br>86M<br>172M<br>344M<br>688M<br>1.4B<br>22.0B|
|||


N N <sup>/</sup> D/D



4.5


4.0


3.5


3.0


2.5



|Data Size Bottleneck|Col2|
|---|---|
|||
||Data Size<br>21M<br>43M<br>86M<br>172M<br>344M<br>688M<br>1.4B<br>22.0B|
|||


Params (non-embed)





0.5


0.4


0.3


0.2


0.1


0.0





**Figure 9** The early-stopped test loss _L_ ( _N, D_ ) depends predictably on the dataset size _D_ and model size _N_
according to Equation (1.5). **Left** : For large _D_, performance is a straight power law in _N_ . For a smaller fixed
_D_, performance stops improving as _N_ increases and the model begins to overfit. (The reverse is also true,

_<u>αN</u>_

see Figure 4.) **Right** : The extent of overfitting depends predominantly on the ratio _N_ _αD_ _/D_, as predicted in

equation (4.3). The line is our fit to that equation.


Since we stop training early when the test loss ceases to improve and optimize all models in the same way, we
expect that larger models should always perform better than smaller models. But with fixed finite _D_, we also
do not expect any model to be capable of approaching the best possible loss (ie the entropy of text). Similarly,
a model with fixed size will be capacity-limited. These considerations motivate our second principle. Note
that knowledge of _L_ ( _N_ ) at infinite _D_ and _L_ ( _D_ ) at infinite _N_ fully determines all the parameters in _L_ ( _N, D_ ).


The third principle is more speculative. There is a simple and general reason one might expect overfitting
to scale _∝_ 1 _/D_ at very large _D_ . Overfitting should be related to the variance or the signal-to-noise ratio
of the dataset [AS17], and this scales as 1 _/D_ . This expectation should hold for any smooth loss function,
since we expect to be able to expand the loss about the _D_ _→∞_ limit. However, this argument assumes that
1 _/D_ corrections dominate over other sources of variance, such as the finite batch size and other limits on the
efficacy of optimization. Without empirical confirmation, we would not be very confident of its applicability.


Our third principle explains the asymmetry between the roles of _N_ and _D_ in Equation (1.5). Very similar
symmetric expressions <sup>4</sup> are possible, but they would not have a 1 _/D_ expansion with integer powers, and
would require the introduction of an additional parameter.


In any case, we will see that our equation for _L_ ( _N, D_ ) fits the data well, which is the most important justification for our _L_ ( _N, D_ ) ansatz.


**4.2** **Results**


We regularize all our models with 10% dropout, and by tracking test loss and stopping once it is no longer
decreasing. The results are displayed in Figure 9, including a fit to the four parameters _αN_ _, αD, Nc, Dc_ in
Equation (1.5):

|Parameter|α<br>N|α<br>D|N<br>c|
|---|---|---|---|
|Value|0_._076|0_._103|6_._4_ ×_ 1013|



**Table 2** Fits to _L_ ( _N, D_ )


We obtain an excellent fit, with the exception of the runs where the dataset has been reduced by a factor of
1024, to about 2 _×_ 10 <sup>7</sup> tokens. With such a small dataset, an epoch consists of only 40 parameter updates.
Perhaps such a tiny dataset represents a different regime for language modeling, as overfitting happens very
early in training (see Figure 16). Also note that the parameters differ very slightly from those obtained in
Section 3, as here we are fitting the full _L_ ( _N, D_ ) rather than just _L_ ( _N, ∞_ ) or _L_ ( _∞, D_ ).


To chart the borderlands of the infinite data limit, we can directly study the extent of overfitting. For all but
the largest models, we see no sign of overfitting when training with the full 22B token WebText2 dataset,
so we can take it as representative of _D_ = _∞_ . Thus we can compare finite _D_ to the infinite data limit by



4For example, one might have used _L_ ( _N, D_ ) = �� _<u>NNc</u>_




- _αN_ + - _<u>DDc</u>_


11




- _αD_ - _β_, but this does not have a 1 _/D_ expansion.




<!-- page: ? -->

<u>Critical Batch Size vs. Performance</u>



10 <sup>6</sup>


10 <sup>5</sup>


10 <sup>4</sup>


10 <sup>3</sup>



![](results/scaling_laws_llm/images/scaling_laws_llm.pdf-11-0.png)

WebText2 Train Loss











**Figure 10** The critical batch size _B_ crit follows a power law in the loss as performance increase, and does
not depend directly on the model size. We find that the critical batch size approximately doubles for every
13% decrease in loss. _B_ crit is measured empirically from the data shown in Figure 18, but it is also roughly
predicted by the gradient noise scale, as in [MKAT18].


defining

_δL_ ( _N, D_ ) _≡_ _L_ <sup>_<u>L</u>_</sup> ( <sup><u>(</u></sup> _N,_ <sup>_<u>N, D</u>_</sup> _∞_ <sup><u>)</u></sup> ) <sup>_−_</sup> <sup>1</sup> (4.2)


and studying it as a function of _N, D_ . In fact, we see empirically that _δL_ depends only a specific combination
of _N_ and _D_, as shown in Figure 16. This follows from the scaling law of Equation (1.5), which implies




- _αD_

_−_ 1 (4.3)




- <sup>_<u>α</u>_</sup> _<u>N</u>_
_αD_ _<u>Dc</u>_

_D_



_δL ≈_




- - _<u>N</u>_

1 +
_Nc_



Note that at large _D_ this formula also has a series expansion in powers of 1 _/D_ .


We estimate that the variation in the loss with different random seeds is roughly 0 _._ 02, which means that to
avoid overfitting when training to within that threshold of convergence we require


_D_ ≳ (5 _×_ 10 <sup>3</sup> ) _N_ <sup>0</sup> <sup>_._</sup> <sup>74</sup> (4.4)

With this relation, models smaller than 10 <sup>9</sup> parameters can be trained with minimal overfitting on the 22B
token WebText2 dataset, but our largest models will encounter some mild overfitting. More generally, this
relation shows that dataset size may grow sub-linearly in model size while avoiding overfitting. Note however
that this does not typically represent maximally compute-efficient training. We should also emphasize that
we have not optimized regularization (eg the dropout probability) while varying dataset and model size.


**5** **Scaling Laws with Model Size and Training Time**


In this section we will demonstrate that a simple scaling law provides a good description for the loss as a
function of model size _N_ and training time. First we will explain how to use the results of [MKAT18] to
define a universal training step _S_ min, which accounts for the fact that most of our models have not been
trained at an optimal batch size. Then we will demonstrate that we can fit the model size and training time
dependence of the loss using Equation (1.6). Later we will use these results to predict the optimal allocation
of training compute between model size and training time, and then confirm that prediction.


**5.1** **Adjustment for Training at** _B_ crit( _L_ )


A simple empirical theory for the batch size dependence of training was developed in [MKAT18] (see also

[SLA <sup>+</sup> 18, ZLN <sup>+</sup> 19]). It was argued that there is a critical batch size _B_ crit for training; for _B_ up to _B_ crit
the batch size can be increased with very minimal degradation in compute-efficiency, whereas for _B_ _> B_ crit
increases in _B_ result in diminishing returns. It was also argued that the gradient noise scale provides a simple


12




<!-- page: ? -->

prediction for _B_ crit, and that neither depends directly on model size except through the value of the loss that
has been attained. These results can be used to predict how training time and compute will vary with the
batch size. To utilize both training time and compute as effectively as possible, it is best to train with a batch
size _B_ _≈_ _B_ crit. Training at _B_ _≫_ _B_ crit minimizes the number of training steps, while _B_ _≪_ _B_ crit minimizes
the use of compute.


More specifically, it was demonstrated that for a wide variety of neural network tasks, the number of training
steps _S_ and the number of data examples processed _E_ = _BS_ satisfy the simple relation




- _<u>S</u>_
_S_ min



�� _<u>E</u>_

_−_ 1

_E_ min




 
_−_ 1 = 1 (5.1)



when training to any fixed value of the loss _L_ . Here _S_ min is the minimum number of steps necessary to reach
_L_, while _E_ min is the minimum number of data examples that must be processed.


We demonstrate the relation (5.1) for Transformers in Figure 18 in the appendix. This relation defines the
critical batch size



_B_ crit( _L_ ) _≡_ <sup>_<u>E</u>_</sup> <sup><u>min</u></sup>



_S_ min



(5.2)



which is a function of the target value of the loss. Training at the critical batch size makes a roughly optimal
time/compute tradeoff, requiring 2 _S_ min training steps and processing _E_ = 2 _E_ min data examples.

In Figure 10 we have plotted the critical batch size and gradient noise scale <sup>5</sup> as a function of training loss for
two different models. We see that _B_ crit( _L_ ) is independent of model size, and only depends on the loss _L_ . So
the predictions of [MKAT18] continue to hold for Transformer language models. The critical batch size can
be fit with a power-law in the loss

_<u>B∗</u>_
_B_ crit( _L_ ) _≈_ (5.3)
_L_ <sup>1</sup> <sup>_/αB_</sup>

where _B∗_ _≈_ 2 _×_ 10 <sup>8</sup> and _αB_ _≈_ 0 _._ 21.


We have chosen this parameterization for _B_ crit( _L_ ) because as the loss approaches its minimum value _L_ min,
the gradient noise scale is expected to diverge, and we expect _B_ crit to track this noise scale. We do not
know _L_ min, as we see no sign that our models are approaching it, but _L_ min _>_ 0 since the entropy of natural
language is non-zero. Since apparently _L_ min is much smaller than the values of _L_ we have achieved, we used
a parameterization where _B_ crit diverges as _L →_ 0.


We will use _B_ crit( _L_ ) to estimate the relation between the number of training steps _S_ while training at batch
size _B_ = 2 <sup>19</sup> tokens and the number of training steps while training at _B_ _≫_ _B_ crit. This is simply


_<u>S</u>_
_S_ min( _S_ ) _≡_ (minimum steps, at _B_ _≫_ _B_ crit) (5.4)
1 + _B_ crit( _L_ ) _/B_


for any given target value _L_ for the loss. This also defines a critical value of the compute needed to train to _L_
with a model of size _N_ if we were to train at _B_ _≪_ _B_ crit( _L_ ). This is


_<u>C</u>_
_C_ min( _C_ ) _≡_ (minimum compute, at _B_ _≪_ _B_ crit) (5.5)
1 + _B/B_ crit( _L_ )


where _C_ = 6 _NBS_ estimates the (non-embedding) compute used at batch size _B_ .


**5.2** **Results for** _L_ ( _N, S_ min) **and Performance with Model Size and Compute**


Now we will use _S_ min defined in Equation (5.4) to obtain a simple and universal fit for the dependence of the
loss on model size and training time in the infinite data limit. We will fit the stable, Adam-optimized training
runs using Equation (1.6), repeated here for convenience:




- _αS_
(5.6)




- _αN_ - _<u>Sc</u>_
+

_S_ min



_L_ ( _N, S_ min) =




- _<u>Nc</u>_
_N_



for the loss. We include all training steps after the warmup period of the learning rate schedule, and find a fit
to the data with the parameters:


5Although the critical batch size roughly matches the gradient noise scale, we are using a direct measurements of
_B_ crit from Figures 18 and 10 for all our later analyses.


13




<!-- page: ? -->

8
7

6

5


4


3


2



![](results/scaling_laws_llm/images/scaling_laws_llm.pdf-13-0.png)

10 <sup>4</sup> 10 <sup>6</sup> 10 <sup>8</sup>

Parameters (non-embedding)



10 <sup>0</sup>

10 1

10 2

10 3

10 4

10 5





![](results/scaling_laws_llm/images/scaling_laws_llm.pdf-13-1.png)



10 <sup>6</sup> 10 <sup>7</sup> 10 <sup>8</sup> 10 <sup>9</sup>

Parameters (non-embedding)



10 <sup>5</sup>


10 <sup>4</sup>



**Figure** **11** When we hold either total compute or number of training steps fixed, performance follows
_L_ ( _N, S_ ) from Equation (5.6). Each value of compute budget has an associated optimal model size that
maximizes performance. Mediocre fits at small _S_ are unsurprising, as the power-law equation for the learning
curves breaks down very early in training.

|Parameter α<br>N|α<br>S|N<br>c|S<br>c|
|---|---|---|---|
|Value<br>0_._077|0_._76|6_._5_ ×_ 1013|2_._1_ ×_ 103|



**Table 3** Fits to _L_ ( _N, S_ )


With these parameters, we obtain the learning curve fits in Figure 4. Though the fits are imperfect, we believe
they are quite compelling given the simplicity of Equation (5.6).


The data and fits can be visualized in a different and more interesting way, as shown in Figure 11. There we
study the test loss as a function of model size while fixing either the total non-embedding compute _C_ used
in training, or the number of steps _S_ . For the fits we use Equation (5.5) and (5.4) along with the parameters
above and Equation (5.6).


The power-law dependence of the loss on _S_ min reflects the interplay of optimizer dynamics and the loss
landscape. Since the fits are best late in training, when the loss may be approximately quadratic, the powerlaw should provide information about the spectrum of the Hessian of the loss. Its universality suggests that
the Hessian eigenvalue density is roughly independent of model size.


**5.3** **Lower Bound on Early Stopping Step**


The results for _L_ ( _N, S_ min) can be used to derive a lower-bound (and rough estimate) of the step at which
early stopping should occur when training is data limited. It is motivated by the idea that finite and infinite _D_
learning curves for a given model will be very similar until we reach _S_ min _≈_ _S_ stop. Thus overfitting should
be proportional to the correction from simply ending training at _S_ stop. This will underestimate _S_ stop, because
in reality the test loss will decrease more slowly when we have a finite _D_, and therefore we will require more
training steps to reach the optimal test loss at finite _D_ . This line of reasoning leads to the inequality


_<u>Sc</u>_
_S_ stop( _N, D_ ) ≳ (5.7)

[ _L_ ( _N, D_ ) _−_ _L_ ( _N, ∞_ )] <sup>1</sup> <sup>_/αS_</sup>


where _L_ ( _N, ∞_ ) is the converged loss, evaluated with infinite available data. This inequality and its comparison to the empirical data is displayed in Figure 16 in the appendix. In that figure, the values of _S_ stop
and _L_ ( _N, D_ ) are empirical (though _S_ stop is adjusted to mimic training at _B_ _≫_ _B_ crit), while _L_ ( _N, ∞_ ) is
computed from the fit to _L_ ( _N, D_ ) evaluated at _D_ = _∞_ .


**6** **Optimal Allocation of the Compute Budget**


We displayed the _empirical_ trend of performance as a function of the computation used during training in
the top-right of Figure 1. However, this result involved training at a fixed batch size _B_, whereas we know


14




<!-- page: ? -->

|Col1|Col2|Col3|
|---|---|---|
|Models between 0.6x and 2.2x the<br>optimal size can be trained with a<br>20% larger compute budget|Models between 0.6x and 2.2x the<br>optimal size can be trained with a<br>20% larger compute budget|Models between 0.6x and 2.2x the<br>optimal size can be trained with a<br>20% larger compute budget|
||||


|Col1|Col2|Smaller models require|Col4|
|---|---|---|---|
|<br>more steps to train, while<br>larger models require fewer<br>Our framework does not<br>capture early training dynamics|<br>more steps to train, while<br>larger models require fewer<br>Our framework does not<br>capture early training dynamics|<br>more steps to train, while<br>larger models require fewer<br>Our framework does not<br>capture early training dynamics|<br>more steps to train, while<br>larger models require fewer<br>Our framework does not<br>capture early training dynamics|
|<br>more steps to train, while<br>larger models require fewer<br>Our framework does not<br>capture early training dynamics|<br>more steps to train, while<br>larger models require fewer<br>Our framework does not<br>capture early training dynamics|||



**Figure 12** **Left:** Given a fixed compute budget, a particular model size is optimal, though somewhat larger
or smaller models can be trained with minimal additional compute. **Right:** Models larger than the computeefficient size require fewer steps to train, allowing for potentially faster training if sufficient additional parallelism is possible. Note that this equation should not be trusted for very large models, as it is only valid in the
power-law region of the learning curve, after initial transient effects.



![](results/scaling_laws_llm/images/scaling_laws_llm.pdf-14-15.png)







10 8 10 6 10 4 10 2 10 <sup>0</sup>



Compute (PF-days), non-embedding



**Figure** **13** When adjusting performance to simulate training far below the critical batch size, we find a
somewhat altered power law for _L_ ( _C_ min) when compared with the fully empirical results. The conspicuous
lump at 10 <sup>_−_</sup> <sup>5</sup> PF-days marks the transition from 1-layer to 2-layer networks; we exclude 1-layer networks
in the power-law fits. It is the _L_ ( _C_ min) trend that we expect to provide a reliable extrapolation for larger
compute.


that in fact we could train more efficiently <sup>6</sup> by training at the batch size _B_ crit discussed in Section 5.1.
Large and small values of the loss could have been achieved with fewer samples or fewer steps, respectively,
and correcting for this inefficiency by standardizing to the critical batch size results in cleaner and more
predictable trends.


In this section we will adjust for this oversight. More importantly, we will use the results of Section 5
to determine the optimal _allocation_ of compute between model size _N_ and the quantity of data processed
during training, namely 2 _B_ crit _S_ min. We will determine this allocation both empirically and theoretically, by
using the equation for _L_ ( _N, S_ min), and we will demonstrate that these methods agree.


**6.1** **Optimal Performance and Allocations**


Let us first study the loss as a function of the optimally allocated compute from Equation (5.5). The result is
plotted in Figure 13, along with a power-law fit. We see that as compared to the compute plot of Figure 1, the
new fit with _C_ min is somewhat improved.


Given _L_ ( _C_ min), it is natural to ask for the optimal model size _N_ ( _C_ min) that provides the minimal loss with a
given quantity of training compute. The optimal model size is shown in Figure 14. We observe that _N_ ( _C_ min)


6One might ask why we did not simply train at _B_ crit in the first place. The reason is that it depends not only on the
model but also on the target value of the loss we wish to achieve, and so is a moving target.


15




<!-- page: ? -->

![](results/scaling_laws_llm/images/scaling_laws_llm.pdf-15-0.png)



![](results/scaling_laws_llm/images/scaling_laws_llm.pdf-15-1.png)



10 7 10 5 10 3 10 1

Compute (PF-days), excluding embeddings





10 <sup>7</sup>


10 <sup>5</sup>


10 <sup>3</sup>





10 7 10 5 10 3 10 1

Compute (PF-days), non-embedding



15000


10000


5000


0



**Figure 14** **Left:** Each value of the compute budget _C_ min has an associated optimal model size _N_ . Optimal
model size grows very rapidly with _C_ min, increasing by 5x for each 10x increase in compute. The number
of data examples processed makes up the remainder of the increase, growing relatively modestly by only 2x.
**Right:** The batch-adjusted number of optimization steps also grows very slowly, if at all, meaning that most
of the growth in data examples processed can be used for increased batch sizes.


can be fit very well with a power-law


_N_ ( _C_ min) _∝_ ( _C_ min) <sup>0</sup> <sup>_._</sup> <sup>73</sup> _._ (6.1)


In Figure 12, we show the effect of training models of sub-optimal sizes (see Appendix B.4).


By definition _C_ min _≡_ 6 _NB_ crit _S_, and so we can use _N_ ( _C_ min) to extract further results. In particular, since
prior fits show _B_ _∝_ _L_ <sup>_−_</sup> <sup>4</sup> <sup>_._</sup> <sup>8</sup> and _L_ _∝_ _C_ min <sup>_−_</sup> <sup>0</sup> <sup>_._</sup> <sup>05</sup>, we can conclude that _B_ crit _∝_ _C_ min <sup>0</sup> <sup>_._</sup> <sup>24.</sup> <sup>This leads us to conclude</sup>

that the optimal number of steps will only grow very slowly with compute, as


_S_ min _∝_ ( _C_ min) <sup>0</sup> <sup>_._</sup> <sup>03</sup> _,_ (6.2)


matching the empirical results in Figure 14. In fact the measured exponent is sufficiently small that our results
may even be consistent with an exponent of zero.


Thus we conclude that as we scale up language modeling with an optimal allocation of computation, we
should predominantly increase the model size _N_, while simultaneously scaling up the batch size via _B_ _∝_
_B_ crit with negligible increase in the number of serial steps. Since compute-efficient training uses relatively
few optimization steps, additional work on speeding up early training dynamics may be warranted.


**6.2** **Predictions from** _L_ ( _N, S_ min)


The results for _L_ ( _C_ min) and the allocations can be predicted from the _L_ ( _N, S_ min) equation obtained in
Section 5. Given our equation for _L_ ( _N, S_ min), we can substitute _S_ min = <sup>_<u>C</u>_</sup> 6 _NB_ <sup><u>min</u></sup> <sup>and then find the minimum</sup>

of the loss as a function of _N_, while fixing the training compute. We carry out this procedure in detail in
Appendix B, where we also provide some additional predictions.


For the loss as a function of training compute, we predict that




- _αC_ <sup>min</sup>



_L_ ( _C_ min) =




- _<u>Cc</u>_ min
_C_ min



(6.3)



where



<u>1</u>

_αC_ <sup>min</sup> _≡_ 1 _/αS_ + 1 _/αB_ + 1 _/αN_ _≈_ 0 _._ 054 (6.4)



in excellent agreement with the exponent of Figure 13. We also predict that


_N_ ( _C_ min) _∝_ ( _C_ min) <sup>_α_</sup> _C_ <sup>min</sup> _/αN_ _≈_ ( _C_ min)0 _._ 71 (6.5)


which also matches the scaling of Figure 14 to within a few percent. Our scaling laws provide a predictive
framework for the performance of language modeling.


16




<!-- page: ? -->

![](results/scaling_laws_llm/images/scaling_laws_llm.pdf-16-1.png)

The intersection point is sensitive to

the precise power-law parameters


**Figure 15** Far beyond the model sizes we study empirically, we find a contradiction between our equations
for _L_ ( _C_ min) and _L_ ( _D_ ) due to the slow growth of data needed for compute-efficient training. The intersection
marks the point before which we expect our predictions to break down. The location of this point is highly
sensitive to the precise exponents from our power-law fits.


**6.3** **Contradictions and a Conjecture**


We observe no signs of deviation from straight power-law trends at large values of compute, data, or model
size. Our trends must eventually level off, though, since natural language has non-zero entropy.


Indeed, the trends for compute-efficient training described in this section already contain an apparent contradiction. At scales several orders of magnitude above those documented here, the performance predicted by
the _L_ ( _C_ min) scaling law decreases below what should be possible given the slow growth in training data with
compute. This implies that our scaling laws must break down before this point, but we conjecture that the
intersection point has a deeper meaning: it provides an estimate of the point at which Transformer language
models reach maximal performance.


Since the amount of data used by compute-efficient training grows slowly with the compute budget, the
performance predicted by _L_ ( _C_ min) eventually hits a lower bound set by the _L_ ( _D_ ) power law (see Figure 15).
Let us work this out in more detail.


To keep overfitting under control, the results of Section 4 imply that we should scale the dataset size as

_D_ _∝_ _N_ <sup>0</sup> <sup>_._</sup> <sup>74</sup> _∝_ _C_ min <sup>0</sup> <sup>_._</sup> <sup>54</sup> (6.6)


where we have used the compute-efficient _N_ ( _C_ min) from Figure 14.


Let us compare this to the data requirements of compute-efficient training. If we train at the critical batch
size (i.e. _C_ = 2 _C_ min) and never re-use data during training, we find that data usage grows with compute as

_D_ ( _C_ min) = 6 _N_ <u>2</u> _<u>C</u>_ ( _C_ <u>minmin)</u> <sup>_≈_</sup> �4 _×_ 10 <sup>10</sup> tokens� ( _C_ min _/_ PF-Day) <sup>0</sup> <sup>_._</sup> <sup>26</sup> (6.7)


This is the maximum rate at which the dataset size can productively grow with compute, since it means that
we are only training for a single epoch. But it grows the dataset much more slowly than in Equation (6.6).
It appears to imply that compute-efficient training will eventually run into a problem with overfitting, even if
the training process never re-uses any data!


According to Figure 1, we expect that when we are bottlenecked by the dataset size (ie by overfitting), the
loss should scale as _L_ ( _D_ ) _∝_ _D_ <sup>_−_</sup> <sup>0</sup> <sup>_._</sup> <sup>095</sup> . This implies that the loss would scale with compute as _L_ ( _D_ ( _C_ min)) _∝_
_C_ min <sup>_−_</sup> <sup>0</sup> <sup>_._</sup> <sup>03</sup> once we are data-limited. Once again, we have a contradiction, as this will eventually intersect with

our prediction for _L_ ( _C_ min) from Figure 13, where we found a scaling _L_ ( _C_ min) _∝_ _C_ min <sup>_−_</sup> <sup>0</sup> <sup>_._</sup> <sup>050</sup> .


The intersection point of _L_ ( _D_ ( _C_ min)) and _L_ ( _C_ min) occurs at

_C_ <sup>_∗_</sup> _∼_ 10 <sup>4</sup> PF-Days _N_ <sup>_∗_</sup> _∼_ 10 <sup>12</sup> parameters _,_ _D_ <sup>_∗_</sup> _∼_ 10 <sup>12</sup> tokens _,_ _L_ <sup>_∗_</sup> _∼_ 1 _._ 7 nats/token (6.8)


though the numerical values are highly uncertain, varying by an order or magnitude in either direction depending on the precise values of the exponents from the power-law fits. The most obvious interpretation is
that our scaling laws break down at or before we reach this point, which is still many orders of magnitude
away in both compute and model size.


17




<!-- page: ? -->

One might also conjecture that this intersection point has a deeper meaning. If we cannot increase the model
size beyond _N_ <sup>_∗_</sup> without qualitatively different data requirements, perhaps this means that once we reach
_C_ min <sup>_∗_</sup> <sup>and</sup> <sup>_N ∗_</sup> <sup>,</sup> <sup>we</sup> <sup>have</sup> <sup>extracted</sup> <sup>all</sup> <sup>of</sup> <sup>the</sup> <sup>reliable</sup> <sup>information</sup> <sup>available</sup> <sup>in</sup> <sup>natural</sup> <sup>language</sup> <sup>data.</sup> <sup>In</sup> <sup>this</sup>

interpretation, _L_ <sup>_∗_</sup> would provide a rough estimate for the entropy-per-token <sup>7</sup> of natural language. In this
scenario, we would expect the loss trend to level off at or before _L_ <sup>_∗_</sup> .


We can guess at the functional form of _L_ ( _C_ min) as it levels off by considering a version of our training
dataset with added noise. For example, we could append a random string of tokens to each context shown
to the model to artificially boost the loss by a constant additive factor. Then, the distance from the noise
floor _L −_ _L_ noise would be a more meaningful performance metric, with even a small decrease in this distance
potentially representing a significant boost in qualitative performance. Since the artificial noise would affect
all of our trends equally, the critical point of 6.8 would not change (aside from the absolute value of _L_ <sup>_∗_</sup> ), and
may be meaningful even if it occurs after the leveling off.


**7** **Related Work**


Power laws can arise from a wide variety of sources [THK18]. Power-law scalings with model and dataset
size in density estimation [Was06] and in random forest models [Bia12] may be connected with our results.
These models suggest that power-law exponents may have a very rough interpretation as the inverse of the
number of relevant features in the data.


Some early [BB01, Goo01] work found power-law scalings between performance and dataset size. More
recent work [HNA <sup>+</sup> 17, HAD19] also investigated scaling between model size and data size; their work is
perhaps the closest to ours in the literature <sup>8</sup> . Note, however, that [HNA <sup>+</sup> 17] found super-linear scaling of
dataset size with model size, whereas we find a sub-linear scaling. There are some parallels between our
findings on optimal allocation of compute and [Kom19], including power-law learning curves. EfficientNets

[TL19] also appear to obey an approximate power-law relation between accuracy and model size. Very recent
work [RRBS19b] studies scaling with both dataset size and model size for a variety of datasets, and fits an
ansatz similar to ours.


EfficientNet [TL19] advocates scaling depth and width exponentially (with different coefficients) for optimal
performance of image models, resulting in a power-law scaling of width as a function of depth. We find that
for language models this power should be roughly one when scaling up (as width/depth should remain fixed).
But more importantly, we find that the precise architectural hyperparameters are unimportant compared to the
overall scale of the language model. In [VWB16] it was argued that deep models can function as ensembles
of shallower models, which could potentially explain this finding. Earlier work [ZK16] has compared width
and depth, and found that wide ResNets can outperform deep ResNets on image classification. Some studies
fix computation per data example, which tends to scale in proportion to the number of model parameters,
whereas we investigate scaling with both model size and the quantity of training computation.


Various works [AS17, BHMM18] have investigated generalization in highly overparameterized models, finding a “jamming transition” [GJS <sup>+</sup> 19] when the model size reaches the dataset size (this may require training
many orders of magnitude beyond typical practice, and in particular does not use early stopping). We do
not observe such a transition, and find that the necessary training data scales sublinearly in the model size.
Expansions in the model size, particularly at large width [JGH18, LXS <sup>+</sup> 19], may provide a useful framework
for thinking about some of our scaling relations. Our results on optimization, such as the shape of learning
curves, can likely be explained using a noisy quadratic model, which can provide quite accurate predictions

[ZLN <sup>+</sup> 19] in realistic settings. Making this connection quantitative will require a characterization of the
Hessian spectrum [Pap18, GKX19, GARD18].


**8** **Discussion**


We have observed consistent scalings of language model log-likelihood loss with non-embedding parameter
count _N_, dataset size _D_, and optimized training computation _C_ min, as encapsulated in Equations (1.5) and
(1.6). Conversely, we find very weak dependence on many architectural and optimization hyperparameters.
Since scalings with _N, D, C_ min are power-laws, there are diminishing returns with increasing scale.


7Defining words using the `wc` utility, the WebText2 dataset has 1 _._ 4 tokens per word and 4 _._ 3 characters per token.
8After this work was completed, [RRBS19a] also appeared, which makes similar predictions for the dependence of
loss on both model and dataset size.


18




<!-- page: ? -->

We were able to precisely model the dependence of the loss on _N_ and _D_, and alternatively on _N_ and _S_, when
these parameters are varied simultaneously. We used these relations to derive the compute scaling, magnitude
of overfitting, early stopping step, and data requirements when training large language models. So our scaling
relations go beyond mere observation to provide a predictive framework. One might interpret these relations
as analogues of the ideal gas law, which relates the macroscopic properties of a gas in a universal way,
independent of most of the details of its microscopic consituents.


It is natural to conjecture that the scaling relations will apply to other generative modeling tasks with a
maximum likelihood loss, and perhaps in other settings as well. To this purpose, it will be interesting to
test these relations on other domains, such as images, audio, and video models, and perhaps also for random
network distillation. At this point we do not know which of our results depend on the structure of natural
language data, and which are universal. It would also be exciting to find a theoretical framework from
which the scaling relations can be derived: a ‘statistical mechanics’ underlying the ‘thermodynamics’ we
have observed. Such a theory might make it possible to derive other more precise predictions, and provide a
systematic understanding of the limitations of the scaling laws.


In the domain of natural language, it will be important to investigate whether continued improvement on the
loss translates into improvement on relevant language tasks. Smooth quantitative change can mask major
qualitative improvements: “more is different”. For example, the smooth aggregate growth of the economy
provides no indication of the specific technological developments that underwrite it. Similarly, the smooth
improvements in language model loss may hide seemingly qualitative changes in capability.


Our results strongly suggest that larger models will continue to perform better, and will also be much more
sample efficient than has been previously appreciated. Big models may be more important than big data.
In this context, further investigation into model parallelism is warranted. Deep models can be trained using
pipelining [HCC <sup>+</sup> 18], which splits parameters depth-wise between devices, but eventually requires increased
batch sizes as more devices are used. Wide networks on the other hand are more amenable to parallelization

[SCP <sup>+</sup> 18], since large layers can be split between multiple workers with less serial dependency. Sparsity

[CGRS19, GRK17] or branching (e.g. [KSH12]) may allow for even faster training of large networks through
increased model parallelism. And using methods like [WRH17, WYL19], which grow networks as they train,
it might be possible to remain on the compute-efficient frontier for an entire training run.


**Acknowledgements**


We would like to thank Shan Carter, Paul Christiano, Jack Clark, Ajeya Cotra, Ethan Dyer, Jason Eisner,
Danny Hernandez, Jacob Hilton, Brice Menard, Chris Olah, and Ilya Sutskever for discussions and for feedback on drafts of this work.


19




<!-- page: ? -->

# **Appendices**

**A** **Summary of Power Laws**


For easier reference, we provide a summary below of the key trends described throughout the paper.





|Parameters|Data|Compute|Batch Size|Equation|
|---|---|---|---|---|
|_N_|_∞_|_∞_|Fixed|_L_ (_N_) = (_N_c_/N_)_αN_|
|_∞_|_D_|Early Stop|Fixed|_L_ (_D_) = (_D_c_/D_)_αD_|
|Optimal|_∞_|_C_|Fixed|_L_ (_C_) = (_C_c_/C_)_αC_ (naive)|
|_N_opt|_D_opt|_C_min|_B ≪B_crit|_L_ (_C_min) =<br>�<br>_C_min<br>c<br>_/C_min<br>_α_min<br>_C_<br>|
|_N_|_D_|Early Stop|Fixed|_L_ (_N, D_) =<br>�_ N_c<br>_N_<br> _αN_<br>_αD_ + _Dc_<br>_D_<br>~~_α_~~_D_<br> <br> <br>|
|_N_|_∞_|_S_ steps|_B_|_L_ (_N, S_) =<br>�_ N_c<br>_N_<br>_αN_ +<br><br>_S_c<br>_S_min(_S,B_)<br>~~_α_~~_S_|


**Table 4**


The empirical fitted values for these trends are:

























|Power Law|Scale (tokenization-dependent)|
|---|---|
|_αN_ = 0_._076|_N_c = 8_._8_ ×_ 1013 params (non-embed)|
|_αD_ = 0_._095|_D_c = 5_._4_ ×_ 1013 tokens|
|_αC_ = 0_._057|_C_c = 1_._6_ ×_ 107 PF-days|
|_α_min<br>_C_<br>= 0_._050|_C_min<br>c<br>= 3_._1_ ×_ 108 PF-days|
|_αB_ = 0_._21|_B∗_= 2_._1_ ×_ 108 tokens|
|_αS_ = 0_._76|_S_c = 2_._1_ ×_ 103 steps|


**Table 5**


The optimal parameters for compute efficient training are given by:

|Compute-Efficient Value|Power Law|Scale|
|---|---|---|
|_N_opt =_ Ne · CpN_<br>min|_pN_ = 0_._73|_Ne_ = 1_._3_ ·_ 109 params|
|_B ≪B_crit =<br>_B∗_<br>_L_1_/αB_ =_ BeCpB_<br>min|_pB_ = 0_._24|_Be_ = 2_._0_ ·_ 106 tokens|
|<br>_S_min =_ Se · CpS_<br>min (lower bound)|_pS_ = 0_._03|_Se_ = 5_._4_ ·_ 103 steps|
|_D_opt =_ De · CpD_<br>min (1 epoch)|_pD_ = 0_._27|_De_ = 2_ ·_ 1010 tokens|



**Table 6**


**B** **Empirical Model of Compute-Efficient Frontier**


Throughout this appendix all values of _C, S,_ and _αC_ are adjusted for training at the critical batch size _B_ crit.
We have left off the ‘adj’ label to avoid cluttering the notation.


**B.1** **Defining Equations**


The power-law fit to the learning curves implies a simple prescription for compute-efficient training. In this
appendix, we will derive the optimal performance, model size, and number of training steps as a function of


20




<!-- page: ? -->

the compute budget. We start with the Equation (1.6), repeated here for convenience:




- _<u>Sc</u>_
_S_




- _αN_
+




- _αS_
_._ (B.1)



_L_ ( _N, S_ ) =




- _<u>Nc</u>_
_N_



Here, _S_ represents the number of parameter updates when training **at** **the** **critical** **batch** **size** [MKAT18],
which was defined in Equation (5.2) <sup>9</sup> :

_<u>B∗</u>_
_B_ ( _L_ ) = _L_ <sup>1</sup> <sup>_/αB_</sup> <sup>_._</sup> (B.2)

We would like to determine optimal training parameters for a fixed compute budget, so we replace _S_ =
_C/_ (6 _NB_ ( _L_ )), where _C_ is the number of FLOPs used in the training run:




- _<u>N</u>_
6 _B∗Sc_
_L_ <sup>1</sup> <sup>_/αB_</sup> _C_




- _αS_
_._ (B.3)



_L_ ( _N, C_ ) =




- _<u>Nc</u>_ - _αN_

+

_N_



Now, we set _∂N_ _L_ �� _C_ <sup>= 0 to find the condition for optimality:</sup>



0 = <sup>_<u>∂L</u>_</sup>
_∂N_



�� _C_




- _<u>N</u>_
6 _B∗Sc_
_L_ <sup>1</sup> <sup>_/αB_</sup> _C_




- _αN_
+ <sup>_<u>αS</u>_</sup>

_N_




- _αS_ �1 _−_ 5 <sup>_<u>N</u>_</sup> _<u>∂L</u>_ �� _C_
_L_ _∂N_







= _−_ <sup>_<u>αN</u>_</sup>

_N_




- _<u>Nc</u>_
_N_



�� _C_




- _<u>N</u>_
6 _B∗Sc_
_L_ <sup>1</sup> <sup>_/αB_</sup> _C_




- _αN_
=




- _αS_
(B.4)



= _⇒_ <sup>_<u>αN</u>_</sup>

_αS_




- _<u>Nc</u>_
_N_



Equation (B.3) and (B.4) together determine the compute-efficient frontier.


**B.2** **Efficient Training**


Now we assemble the implications of (B.3) and (B.4). First, note that inserting (B.4) into (B.3) yields




_L_ ( _N_ eff _, ∞_ ) _,_ (B.5)



_L_ ( _N_ eff ( _C_ ) _, C_ ) =



�1 + <sup>_<u>αN</u>_</sup>
_αS_



which implies that for compute-efficient training, we should train to a **fixed percentage** <sup>_<u>α</u>_</sup> _α_ <sup>_<u>N</u>_</sup> _S_ <sup>_≈_</sup> <sup>10%</sup> <sup>above</sup>

the converged loss. Next, let’s determine how the optimal loss depends on the compute budget. Eliminating
_N_ yields a power-law dependence of performance on compute:




- _αC_
(B.6)



_L_ ( _C_ ) =




- _<u>Cc</u>_
_C_



where we defined



_αC_ = 1 _/_ (1 _/αS_ + 1 _/αB_ + 1 _/αN_ ) _≈_ 0 _._ 052 (B.7)



�1 _/αS_
_._ (B.8)



�1 _/αS_ +1 _/αN_ - _<u>αS</u>_
_αN_



_Cc_ = 6 _NcB∗Sc_



�1 + <sup>_<u>αN</u>_</sup>
_αS_



Similarly, we can eliminate _L_ to find _N_ ( _C_ ):



�1 _/αN_
(B.9)



_<u>N</u>_ <u>(</u> _<u>C</u>_ <u>)</u>

_Nc_



=




- _<u>C</u>_ - _αC_ _/αN_ �1 + <sup>_<u>αN</u>_</sup>
_Cc_ _αS_



and




- _−_ 1 _/αN_ - _<u>C</u>_
_Cc_




- _αC_ _/αS_
(B.10)



_<u>Cc</u>_
_S_ ( _C_ ) =
6 _NcB∗_



�1 + <sup>_<u>αN</u>_</sup>
_αS_



9There is a slight ambiguity here: we can imagine training either at a constant batch size _B_ ( _L_ target), or we could
instead train at a variable batch size _B_ <sup>˜</sup> ( _L_ ), where _B_ <sup>˜</sup> is the instantaneous critical batch size (as opposed to _B_, which is
the averaged version). These two prescriptions result in the same number of steps, so we can ignore this subtlety (see

[MKAT18]).


21




<!-- page: ? -->

**B.3** **Comparison to Inefficient**


Typically, researchers train models until they appear to be close to convergence. In this section, we compare
the efficient training procedure described above to this more typical setup. We define a the convergence factor
_f_ as the percent deviation from the converged loss:


_L_ ( _N, C_ ) = (1 + _f_ ) _L_ ( _N, ∞_ ) _._ (B.11)


For compute-efficient training we have _f_ = _αN_ _/αS_ _≈_ 10% from the previous section, but researchers
typically use a much smaller value. Here, we choose _f_ <sup>_′_</sup> = 2% as an estimate. For a fixed value of the loss,
we predict:



�1 _/αN_
_≈_ 2 _._ 7 (B.12)


�1 _/αS_
_≈_ 0 _._ 13 (B.13)



_<u>Nf</u>_
_Nf ′_ <sup>=</sup>


_<u>Sf</u>_
_Sf ′_ <sup>=</sup>




- <u>1 +</u> _<u>f</u>_
1 + _f_ <sup>_′_</sup>


- 1 + _<u>f</u>_ <sup><u>1</u></sup>
1 + _f_ <sup><u>1</u></sup> <sup>_′_</sup>



_CCff ′_ <sup>=</sup> _N_ <sup>_<u>N</u>_</sup> _f_ <sup>_<u>f</u>_</sup> _′_



_<u>Sf</u>_
_Sf ′_ <sup>_≈_</sup> <sup>0</sup> <sup>_._</sup> <sup>35</sup> (B.14)



So that compute-efficient training uses 7.7x fewer parameter updates, 2.7x more parameters, and 65% less
compute to reach the same loss.


**B.4** **Suboptimal Model Sizes**


We can solve A.1 to find an expression for the amount of compute needed to reach a given value of the loss
_L_ with a model of size _N_ :



��
_L −_




- _<u>Nc</u>_
_N_




- _αN_ - _−_ 1 _/αS_
_._ (B.15)



_C_ ( _N, L_ ) =




- _<u>N</u>_
6 _B∗Sc_
_L_ <sup>1</sup> <sup>_/αB_</sup>



Using A.6 and A.9, we can eliminate _L_ in favor of _N_ eff ( _L_ ), the model size which reaches _L_ most efficiently.
From there, we find an expression for the excess compute needed as a consequence of using a suboptimal
model size:




- - _<u>N</u>_ <u>ef</u>
1 _−_
_N_




- _αN_ �� _−_ 1 _/αS_
_._ (B.16)



_<u>C</u>_ <u>(</u> _<u>N, N</u>_ <u>eff</u> <u>)</u> _<u>N</u>_
_C_ ( _N_ eff _, N_ eff ) <sup>=</sup> _N_ eff



�1 + <sup>_<u>αS</u>_</sup>
_αN_



The result is shown in Figure X. Models between 0.6x and 2.2x the optimal size can be used with only a
20% increase in compute budget. Using a smaller model is useful when accounting for the cost inference. A
larger model can be trained the the same level of performance in fewer steps, allowing for more parallelism
and faster training if sufficient harware is available (see Figure Y):




- - _<u>N</u>_ <u>ef</u>
1 _−_
_N_




- _αN_ �� _−_ 1 _/αS_
_._ (B.17)



_<u>S</u>_ <u>(</u> _<u>N, N</u>_ <u>eff</u> <u>)</u>
_S_ ( _N_ eff _, N_ eff ) <sup>=</sup>



�1 + <sup>_<u>αS</u>_</sup>
_αN_



A 2.2x larger model requires 45% fewer steps at a cost of 20% more training compute. Note that this equation
should not be trusted for very large models, as it is only valid in the power-law region of the learning curve
after initial transient effects.


**C** **Caveats**


In this section we list some potential caveats to our analysis.


_•_ At present we do not have a solid theoretical understanding for any of our proposed scaling laws.
The scaling relations with model size and compute are especially mysterious. It may be possible to
understand scaling at very large _D_ holding model size fixed [AS17], and also the shape of learning
curves late in training, by modeling the loss with a noisy quadratic. But the scaling with _D_ at very
large model size still remains mysterious. Without a theory or a systematic understanding of the
corrections to our scaling laws, it’s difficult to determine in what circumstances they can be trusted.


22




<!-- page: ? -->

|Col1|Col2|
|---|---|
|Test Loss<br>Train Loss|Test Loss<br>Train Loss|


Step



10 <sup>5</sup>


10 <sup>4</sup>


10 <sup>3</sup>



<u>Early Stopping Step</u>


10 <sup>3</sup> 10 <sup>4</sup> 10 <sup>5</sup>

Sc × [L(N, D) L(N, )] 1/ S



![](results/scaling_laws_llm/images/scaling_laws_llm.pdf-22-0.png)





6


5


4


3


2



10 <sup>10</sup>


10 <sup>9</sup>


10 <sup>8</sup>



**Figure** **16** **Left:** We characterize the step on which early stopping occurs, as a function of the extent of
overfitting. The red line indicates a lower bound for early stopping that is derived in Section 5.3. **Right:**
We display train and test loss for a series of 300M parameter models trained on different sized dataset subsamples. The test loss typically follows that of a run done with unrestricted data until diverging. Note that the
degree of overfitting (as compared to the infinite data limit) is significantly overestimated by _L_ test _−_ _L_ train
(denoted by a black bar for each run).


_•_ We are not especially confident in the prediction of _B_ crit( _L_ ) for values of the loss far outside the
range we have explored. Changes in _B_ crit could have a significant impact on trade-offs between
data parallelism and the number of serial training steps required, which would have a major impact
on training time.

_•_ We did not thoroughly investigate the small data regime, and our fits for _L_ ( _N, D_ ) were poor for
the smallest values of _D_ (where an epoch corresponded to only 40 steps). Furthermore, we did
not experiment with regularization and data augmentation. Improvements in these could alter our
results, quantitatively or qualitatively.

_•_ We used the estimated training compute _C_ _≈_ 6 _NBS_, which did not include contributions proportional to _n_ ctx (see Section 2.1). So our scalings with compute may be confounded in practice in the
regime of very large _n_ ctx, specifically where _n_ ctx ≳ 12 _d_ model.

_•_ We tuned learning rates, and we experimented with learning rate schedules. But we may have
neglected to tune some hyperparameter (e.g. intialization scale or momentum) that have an important
effect on scaling.

_•_ The optimal choice of learning rate is sensitive to the target loss. When training close to convergence,
it may be necessary to use a smaller learning rate to avoid divergences. But when conducting a short
training run (eg due to compute limitations), it may be possible to use a larger learning rate. We did
not experiment with higher learning rates for training runs that did not proceed to convergence.


**D** **Supplemental Figures**


**D.1** **Early Stopping and Test vs Train**


In section 5.3 we described the result shown in Figure 16, which provides a prediction for a lower bound on
the early stopping step. We also show the train and test loss for a given model size when training on different
sized datasets.


**D.2** **Universal Transformers**


We compare the performance of standard Transformers to recurrent Transformers [DGV <sup>+</sup> 18] in Figure 17.
These models re-use parameters, and so perform slightly better as a function of _N_, but slightly worse as a
function of compute _C_ . We include several different different possibilities for parameter re-use.


**D.3** **Batch Size**


We measure the critical batch size using the data displayed in figure 18. This made it possible to estimate
_B_ crit( _L_ ) in figure 10.


23




<!-- page: ? -->

4.5


4.0


3.5


3.0


2.5



![](results/scaling_laws_llm/images/scaling_laws_llm.pdf-23-0.png)



![](results/scaling_laws_llm/images/scaling_laws_llm.pdf-23-1.png)



4.5


4.0


3.5


3.0


2.5



10 <sup>5</sup> 10 <sup>6</sup> 10 <sup>7</sup> 10 <sup>8</sup> 10 <sup>9</sup>

Parameters, including reuse (non-embedding)



10 <sup>5</sup> 10 <sup>6</sup> 10 <sup>7</sup> 10 <sup>8</sup> 10 <sup>9</sup>

Parameters (non-embedding)



**Figure** **17** We compare recurrent Transformers [DGV <sup>+</sup> 18], which re-use parameters, to standard Transformers. Recurrent Transformers perform slightly better when comparing models with equal parameter count,
but slightly worse when accounting for reuse and comparing per FLOP.



<u>Batch Size Scan - 85M Params</u>


10 <sup>10</sup>


10 <sup>8</sup>


10 <sup>6</sup>


10 <sup>1</sup> 10 <sup>2</sup> 10 <sup>3</sup> 10 <sup>4</sup> 10 <sup>5</sup>

Step



10


8


6


4



10 <sup>11</sup>


10 <sup>10</sup>


10 <sup>9</sup>


10 <sup>8</sup>


10 <sup>7</sup>


10 <sup>6</sup>



<u>Batch Size Scan - 3M Params</u>


10 <sup>2</sup> 10 <sup>3</sup> 10 <sup>4</sup> 10 <sup>5</sup>

Step



10


8


6


4



![](results/scaling_laws_llm/images/scaling_laws_llm.pdf-23-2.png)

![](results/scaling_laws_llm/images/scaling_laws_llm.pdf-23-3.png)

**Figure 18** These figures demonstrate fits to Equation (5.1) for a large number of values of the loss _L_, and
for two different Transformer model sizes. These fits were used to measure _B_ crit( _L_ ) for Figure 10.


**D.4** **Sample Efficiency vs Model Size**


It is easy to see from figure 2 that larger models train faster, and are therefore more sample efficient. We
provide another way of looking at this phenomenon in figure 19, which shows when different models reach
various fixed values of the loss.



10 <sup>6</sup> 10 <sup>7</sup> 10 <sup>8</sup>

Parameters (non-embedding)



5.5


5.0


4.5


4.0


3.5


3.0


2.5



![](results/scaling_laws_llm/images/scaling_laws_llm.pdf-23-4.png)





5.5


5.0


4.5


4.0


3.5


3.0


2.5



10 <sup>6</sup> 10 <sup>7</sup> 10 <sup>8</sup>

Parameters (non-embedding)



![](results/scaling_laws_llm/images/scaling_laws_llm.pdf-23-5.png)

**Figure 19** The number of minimum serial steps needed to reach any fixed value of the test loss decreases
precipitously with model size. Sample efficiency (show here for training far below the critical batch size)
improves greatly as well, improving by a factor of almost 100 when comparing the smallest possible model
to a very large one.


24




<!-- page: ? -->

8


7


6


5


4


3



![](results/scaling_laws_llm/images/scaling_laws_llm.pdf-24-1.png)

10 <sup>0</sup> 10 <sup>1</sup> 10 <sup>2</sup> 10 <sup>3</sup>

Token Index



10 <sup>8</sup>


10 <sup>7</sup>


10 <sup>6</sup>



<u>Per-token Loss (774M Params)</u>


10 <sup>1</sup> 10 <sup>3</sup> 10 <sup>5</sup>

Step



10 <sup>3</sup>


10 <sup>2</sup>


10 <sup>1</sup>


10 <sup>0</sup>



10


8


6


4


2



![](results/scaling_laws_llm/images/scaling_laws_llm.pdf-24-0.png)

**Figure** **20** This figure provides information about the performance per token as a function of model size
and training time. **Left:** Loss per token as a function of its position _T_ in the 1024-token context. Loss scales
predictably as a power-law in _T_ . **Right:** Test loss per token as a function of training step.



7.5


6.0


4.5


3.0



![](results/scaling_laws_llm/images/scaling_laws_llm.pdf-24-2.png)



10 <sup>4</sup> 10 <sup>5</sup> 10 <sup>6</sup> 10 <sup>7</sup> 10 <sup>8</sup> 10 <sup>9</sup>

Parameters (excl. embedding)



**Figure** **21** In addition to the averaged loss, individual tokens within the 1024-token context also improve
smoothly as model size increases. Training runs with shorter context _n_ ctx = 8 (dashed lines) perform better
on early tokens, since they can allocate all of their capacity to them.


**D.5** **Context Dependence**


The trends for loss as a function of model size are displayed for different tokens in the context in Figure 21.
We see that models trained on _n_ ctx = 1024 show steady improvement with model size on all but the first
token.


Fixing model size, it appears that the loss scales as a power-law as a function of position _T_ in the context, see
Figure 20. This may be a consequence of underlying power-law correlations in language [EP94, ACDE12,
LT16], or a more general feature of the model architecture and optimization. It provides some suggestion for
the potential benefits (or lack thereof) from training on larger contexts. Not only do larger models converge
to better performance at _T_ = 1024, but they also improve more quickly at early tokens, suggesting that larger
models are more efficient at detecting patterns with less contextual information. In the right-hand plot we
show how per-token performance varies for a fixed model as a function of the training step. The model begins
by learning short-range information, and only learns longer-range correlations later in training.


We have also included models trained with a tiny context _n_ ctx = 8 in order to compare with our longer
context models. Even modestly sized models trained on _n_ ctx = 8 can dominate our largest _n_ ctx = 1024
models on very early tokens. This also suggests that further improvements should be possible with much
larger models trained on large contexts.


**D.6** **Learning Rate Schedules and Error Analysis**


We experimented with a variety of learning rates and schedules. A host of schedules and resulting test
performances for a small language model are plotted in Figure 22. We conclude that the choice of learning
rate schedule is mostly irrelevant, as long as the total summed learning rate is sufficiently large, and the
schedule includes a warmup period and a final decay to near-vanishing learning rate. Variations among


25




<!-- page: ? -->

50 100 150 200 250
LR Summed Over Steps



0.0010


0.0008


0.0006


0.0004


0.0002


0.0000



![](results/scaling_laws_llm/images/scaling_laws_llm.pdf-25-0.png)

0 50000 100000 150000 200000 250000
Step



3.90


3.85


3.80


3.75


3.70


3.65



**Figure 22** We test a variety of learning rate schedules including cosine decay, linear decay, as well as other
faster/slower decays schedules on a 3 million parameter model, shown on the left. For these experiments we
do not decay to zero, since we find that this tends to give a fixed improvement close to the end of training.
We find that, as long as the learning rate is not too small and does not decay too quickly, performance does
not depend strongly on learning rate. Run-to-run variation is at the level of 0.05 in the loss, so averaging
multiple runs is necessary to validate performance changes smaller than this level.



6


5


4


3


2



![](results/scaling_laws_llm/images/scaling_laws_llm.pdf-25-1.png)

10 <sup>4</sup> 10 <sup>5</sup> 10 <sup>6</sup> 10 <sup>7</sup> 10 <sup>8</sup> 10 <sup>9</sup>

Parameters (non-embedding)







**Figure 23** The trend for performance as a function of parameter count, _L_ ( _N_ ), is fit better by a power law
than by other functions such as a logarithm at a qualitative level.


schedules appear to be statistical noise, and provide a rough gauge for the scale of variation between different
training runs. Experiments on larger models suggest that the variation in the final test loss between different
random seeds is roughly constant in magnitude for different model sizes.


We found that larger models require a smaller learning rate to prevent divergence, while smaller models can
tolerate a larger learning rate. To implement this, the following rule of thumb was used for most runs:


LR( _N_ ) _≈_ 0 _._ 003239 + _−_ 0 _._ 0001395 log( _N_ ) (D.1)


We expect that this formula could be improved. There may be a dependence on network width, likely set by
the initialization scale. The formula also breaks down for _N_ _>_ 10 <sup>10</sup> parameters. Nevertheless, we found that
it works sufficiently well for the models we considered.


**D.7** **Fit Details and Power Law Quality**


We experimented with a number of functional forms for the fits to _L_ ( _N_ ) _, L_ ( _C_ ), and _L_ ( _D_ ); the power-law
fits were qualitatively much more accurate than other functions such as logarithms (see Figure 23).


For _L_ ( _C_ ), we do not include small models with only 1 layer in the fit, as the transition from 1 to 2 layers
causes a noticable lump in the data. For _L_ ( _N_ ) we also do not include very small models with only 1 layer in
the fit, and we exclude the largest models that have not trained fully to convergence. Fit parameters change
marginally if we do include them, and the trend extrapolates well in both directions regardless.


**D.8** **Generalization and Architecture**


In figure 24 we show that generalization to other data distributions does not depend on network depth when we
hold the total parameter count fixed. It seems to depend only on the performance on the training distribution.


26




<!-- page: ? -->

2.8


2.7


2.6


2.5


2.4


2.3





|Col1|Col2|
|---|---|
||Wikipedia<br>Books<br>Internet Books<br>Common Crawl<br>WebText2 (Train)<br>WebText2 (Test)|
|||


Depth



**Figure 24** We show evaluations on a series of datasets for models with approximately 1.5 Billion parameters. We observe no effect of depth on generalization; generalization performance depends primarily on
training distribution performance. The 12-layer model overfit the Internet Books dataset and we show the
early-stopped performance; we have not seen this surprising result in other experiments.


**List of Figures**


1 Summary of simple power laws. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3


2 Illustration of sample efficiency and compute efficiency. . . . . . . . . . . . . . . . . . . . . 4


3 How to scale up model size, batch size, and serial steps . . . . . . . . . . . . . . . . . . . . 4


4 Performance when varying model and data size, or model and training steps, simultaneously 5


5 Weak dependence of performance on hyperparameter tuning . . . . . . . . . . . . . . . . . 8


6 Comparison of performance trend when including or excluding embeddings . . . . . . . . . 8


7 LSTM and Transformer performance comparison . . . . . . . . . . . . . . . . . . . . . . . 9


8 Generalization to other test datasets . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10


9 Universality of overfitting . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11


10 Critical batch size . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12


11 Performance versus compute budget or number of parameter updates . . . . . . . . . . . . . 14


12 Training on suboptimal models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15


13 Comparison between empirical and adjusted compute trends . . . . . . . . . . . . . . . . . 15


14 Optimal model size and serial number of steps versus compute budget . . . . . . . . . . . . 16


15 Contradiction between compute and data trends . . . . . . . . . . . . . . . . . . . . . . . . 17


16 Early stopping lower bound and training curves for overfit models . . . . . . . . . . . . . . 23


17 Universal transformers . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24


18 Batch size scans . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24


19 Another look at sample efficiency . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24


20 Power-law dependence of performance on position in context . . . . . . . . . . . . . . . . . 25


21 Performance at different context positions versus model size . . . . . . . . . . . . . . . . . 25


22 Learning rate schedule scan . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26


23 Comparison of Power-Law and Logarithmic Fits . . . . . . . . . . . . . . . . . . . . . . . 26


24 Generalization versus depth . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27


27




<!-- page: ? -->

**List of Tables**


1 Parameter and compute counts for Transformer . . . . . . . . . . . . . . . . . . . . . . . . 7


2 Fits to _L_ ( _N, D_ ) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11


3 Fits to _L_ ( _N, S_ ) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14


4 Key trend equations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20


5 Key parameters to trend fits . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20


6 Trends for compute-efficient training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20


**References**


[ACDE12] Eduardo G Altmann, Giampaolo Cristadoro, and Mirko Degli Esposti. On the origin of longrange correlations in texts. _Proceedings of the National Academy of Sciences_, 109(29):11582–
11587, 2012. 25


[AS17] Madhu S. Advani and Andrew M. Saxe. High-dimensional dynamics of generalization error in
neural networks. _arXiv_ [, 2017, 1710.03667.](http://arxiv.org/abs/1710.03667) 11, 18, 22


[BB01] Michele Banko and Eric Brill. Scaling to very very large corpora for natural language disambiguation. In _Proceedings of the 39th annual meeting on association for computational linguis-_
_tics_, pages 26–33. Association for Computational Linguistics, 2001. 18


[BHMM18] Mikhail Belkin, Daniel Hsu, Siyuan Ma, and Soumik Mandal. Reconciling modern machine

learning and the bias-variance trade-off. _arXiv_ [, 2018, 1812.11118.](http://arxiv.org/abs/1812.11118) 18


[Bia12] GÃŠrard Biau. Analysis of a random forests model. _Journal_ _of_ _Machine_ _Learning_ _Research_,
13(Apr):1063–1095, 2012. 18


[CGRS19] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with
sparse transformers. _CoRR_, abs/1904.10509, 2019, [1904.10509.](http://arxiv.org/abs/1904.10509) URL `[http://arxiv.org/](http://arxiv.org/abs/1904.10509)`
`[abs/1904.10509](http://arxiv.org/abs/1904.10509)` . 19


[DCLT18] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep
[bidirectional transformers for language understanding, 2018, arXiv:1810.04805.](http://arxiv.org/abs/arXiv:1810.04805) 2

[DGV <sup>+</sup> 18] Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Lukasz Kaiser. Uni
versal transformers. _CoRR_, abs/1807.03819, 2018, [1807.03819.](http://arxiv.org/abs/1807.03819) URL `[http://arxiv.org/](http://arxiv.org/abs/1807.03819)`
`[abs/1807.03819](http://arxiv.org/abs/1807.03819)` . 6, 9, 23, 24


[EP94] Werner Ebeling and Thorsten Pöschel. Entropy and long-range correlations in literary english.
_EPL (Europhysics Letters)_, 26(4):241, 1994. 25


[Fou] The Common Crawl Foundation. Common crawl. URL `[http://commoncrawl.org](http://commoncrawl.org)` . 7


[GARD18] Guy Gur-Ari, Daniel A. Roberts, and Ethan Dyer. Gradient descent happens in a tiny subspace.

[2018, arXiv:1812.04754.](http://arxiv.org/abs/arXiv:1812.04754) 18

[GJS <sup>+</sup> 19] Mario Geiger, Arthur Jacot, Stefano Spigler, Franck Gabriel, Levent Sagun, Stéphane d’Ascoli,
Giulio Biroli, Clément Hongler, and Matthieu Wyart. Scaling description of generalization with
number of parameters in deep learning. _arXiv_ [, 2019, 1901.01608.](http://arxiv.org/abs/1901.01608) 18


[GKX19] Behrooz Ghorbani, Shankar Krishnan, and Ying Xiao. An investigation into neural net optimization via hessian eigenvalue density. _CoRR_, abs/1901.10159, 2019, [1901.10159.](http://arxiv.org/abs/1901.10159) URL
`[http://arxiv.org/abs/1901.10159](http://arxiv.org/abs/1901.10159)` . 18


[Goo01] Joshua Goodman. A bit of progress in language modeling. _CoRR_, cs.CL/0108005, 2001. URL
`[http://arxiv.org/abs/cs.CL/0108005](http://arxiv.org/abs/cs.CL/0108005)` . 18


[GRK17] Scott Gray, Alec Radford, and Diederik P Kingma. Gpu kernels for block-sparse weights. _ope-_
_nai.com_, 2017. 19


[HAD19] Joel Hestness, Newsha Ardalani, and Gregory Diamos. Beyond human-level accuracy: Computational challenges in deep learning. In _Proceedings of the 24th Symposium on Principles and_
_Practice of Parallel Programming_, PPoPP ’19, pages 1–14, New York, NY, USA, 2019. ACM.
[doi:10.1145/3293883.3295710.](http://dx.doi.org/10.1145/3293883.3295710) 18


28




<!-- page: ? -->

[HCC <sup>+</sup> 18] Yanping Huang, Yonglong Cheng, Dehao Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V. Le,
and Zhifeng Chen. Gpipe: Efficient training of giant neural networks using pipeline parallelism.
_CoRR_ [, abs/1811.06965, 2018, 1811.06965.](http://arxiv.org/abs/1811.06965) URL `[http://arxiv.org/abs/1811.06965](http://arxiv.org/abs/1811.06965)` . 19

[HNA <sup>+</sup> 17] Joel Hestness, Sharan Narang, Newsha Ardalani, Gregory Diamos, Heewoo Jun, Hassan Kia
ninejad, Md. Mostofa Ali Patwary, Yang Yang, and Yanqi Zhou. Deep learning scaling is pre[dictable, empirically, 2017, 1712.00409.](http://arxiv.org/abs/1712.00409) 18


[JGH18] Arthur Jacot, Franck Gabriel, and Clément Hongler. Neural tangent kernel: Convergence and
generalization in neural networks. In _Advances in neural information processing systems_, pages
8571–8580, 2018. 18


[KB14] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization, 2014,
[1412.6980.](http://arxiv.org/abs/1412.6980) 7


[Kom19] Aran Komatsuzaki. [One epoch is all you need, 2019, arXiv:1906.06669.](http://arxiv.org/abs/arXiv:1906.06669) 18


[KSH12] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. Imagenet classification with deep
convolutional neural networks. In _Proceedings of the 25th International Conference on Neural_
_Information_ _Processing_ _Systems_ _-_ _Volume_ _1_, NIPS’12, pages 1097–1105, USA, 2012. Curran
Associates Inc. URL `[http://dl.acm.org/citation.cfm?id=2999134.2999257](http://dl.acm.org/citation.cfm?id=2999134.2999257)` . 19

[LCG <sup>+</sup> 19] Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu
Soricut. Albert: A lite bert for self-supervised learning of language representations, 2019,
[1909.11942.](http://arxiv.org/abs/1909.11942) 9

[LOG <sup>+</sup> 19] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike
Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized BERT pretraining approach. _CoRR_, abs/1907.11692, 2019, [1907.11692.](http://arxiv.org/abs/1907.11692) URL `[http://arxiv.org/abs/](http://arxiv.org/abs/1907.11692)`
`[1907.11692](http://arxiv.org/abs/1907.11692)` . 2

[LSP <sup>+</sup> 18] Peter J. Liu, Mohammad Saleh, Etienne Pot, Ben Goodrich, Ryan Sepassi, Lukasz Kaiser, and
Noam Shazeer. Generating wikipedia by summarizing long sequences. _arXiv:1801.10198 [cs]_,
[2018, 1801.10198.](http://arxiv.org/abs/1801.10198) URL `[http://arxiv.org/abs/1801.10198](http://arxiv.org/abs/1801.10198)` . 2, 6


[LT16] Henry W Lin and Max Tegmark. Criticality in formal languages and statistical physics. _arXiv_
_preprint arXiv:1606.06737_, 2016. 25

[LXS <sup>+</sup> 19] Jaehoon Lee, Lechao Xiao, Samuel S. Schoenholz, Yasaman Bahri, Roman Novak, Jascha SohlDickstein, and Jeffrey Pennington. Wide neural networks of any depth evolve as linear models
[under gradient descent, 2019, arXiv:1902.06720.](http://arxiv.org/abs/arXiv:1902.06720) 18


[MKAT18] Sam McCandlish, Jared Kaplan, Dario Amodei, and OpenAI Dota Team. An empirical model

[of large-batch training, 2018, arXiv:1812.06162.](http://arxiv.org/abs/arXiv:1812.06162) 3, 5, 6, 12, 13, 21


[Pap18] Vardan Papyan. The full spectrum of deep net hessians at scale: Dynamics with sample size.
_CoRR_ [, abs/1811.07062, 2018, 1811.07062.](http://arxiv.org/abs/1811.07062) URL `[http://arxiv.org/abs/1811.07062](http://arxiv.org/abs/1811.07062)` . 18


[RNSS18] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language
understanding by generative pre-training. _URL_ _https://s3-us-west-2._ _amazonaws._ _com/openai-_
_assets/research-covers/languageunsupervised/language understanding paper. pdf_, 2018. 2, 6


[RRBS19a] Jonathan S. Rosenfeld, Amir Rosenfeld, Yonatan Belinkov, and Nir Shavit. A constructive
[prediction of the generalization error across scales, 2019, 1909.12673.](http://arxiv.org/abs/1909.12673) 18


[RRBS19b] Jonathan S. Rosenfeld, Amir Rosenfeld, Yonatan Belinkov, and Nir Shavit. A constructive
[prediction of the generalization error across scales, 2019, arXiv:1909.12673.](http://arxiv.org/abs/arXiv:1909.12673) 18

[RSR <sup>+</sup> 19] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena,
Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified
[text-to-text transformer, 2019, arXiv:1910.10683.](http://arxiv.org/abs/arXiv:1910.10683) 2

[RWC <sup>+</sup> 19] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language

models are unsupervised multitask learners. _openai.com_, 2019. 2, 5, 6, 7, 8

[SCP <sup>+</sup> 18] Noam Shazeer, Youlong Cheng, Niki Parmar, Dustin Tran, Ashish Vaswani, Penporn Koanantakool, Peter Hawkins, HyoukJoong Lee, Mingsheng Hong, Cliff Young, Ryan Sepassi, and
Blake Hechtman. Mesh-tensorflow: [Deep learning for supercomputers, 2018, 1811.02084.](http://arxiv.org/abs/1811.02084) 19


[SHB15] Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words
with subword units. _CoRR_ [, 2015, 1508.07909.](http://arxiv.org/abs/1508.07909) 6


29




<!-- page: ? -->

[SLA <sup>+</sup> 18] Christopher J. Shallue, Jaehoon Lee, Joe Antognini, Jascha Sohl-Dickstein, Roy Frostig, and
George E. Dahl. Measuring the effects of data parallelism on neural network training, 2018,
[arXiv:1811.03600.](http://arxiv.org/abs/arXiv:1811.03600) 12

[SS18] Noam Shazeer and Mitchell Stern. Adafactor: Adaptive learning rates with sublinear memory
cost. _CoRR_ [, abs/1804.04235, 2018, 1804.04235. URL](http://arxiv.org/abs/1804.04235) `[http://arxiv.org/abs/1804.04235](http://arxiv.org/abs/1804.04235)` .
7

[THK18] Stefan Thurner, Rudolf Hanel, and Peter Klimek. _Introduction to the theory of complex systems_ .
Oxford University Press, 2018. 18

[TL19] Mingxing Tan and Quoc V. Le. Efficientnet: Rethinking model scaling for convolutional neural
networks. _CoRR_ [, abs/1905.11946, 2019, 1905.11946.](http://arxiv.org/abs/1905.11946) URL `[http://arxiv.org/abs/1905.](http://arxiv.org/abs/1905.11946)`
`[11946](http://arxiv.org/abs/1905.11946)` . 18

[VSP <sup>+</sup> 17] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. V. Luxburg,
S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, _Advances in Neural_
_Information_ _Processing_ _Systems_ _30_, pages 5998–6008. Curran Associates, Inc., 2017. URL
`[http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)` . 2, 6

[VWB16] Andreas Veit, Michael Wilber, and Serge Belongie. Residual networks behave like ensembles
[of relatively shallow networks, 2016, arXiv:1605.06431.](http://arxiv.org/abs/arXiv:1605.06431) 8, 18

[Was06] Larry Wasserman. _All of nonparametric statistics_ . Springer Science & Business Media, 2006.
18

[WPN <sup>+</sup> 19] Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill,

Omer Levy, and Samuel R. Bowman. Superglue: A stickier benchmark for general-purpose
[language understanding systems, 2019, 1905.00537.](http://arxiv.org/abs/1905.00537) 2

[WRH17] Yu-Xiong Wang, Deva Ramanan, and Martial Hebert. Growing a brain: Fine-tuning by increasing model capacity. _2017 IEEE Conference on Computer Vision and Pattern Recognition_
_(CVPR)_, Jul 2017. [doi:10.1109/cvpr.2017.323.](http://dx.doi.org/10.1109/cvpr.2017.323) 19

[WYL19] Wei Wen, Feng Yan, and Hai Li. Autogrow: Automatic layer growing in deep convolutional
[networks, 2019, 1906.02909.](http://arxiv.org/abs/1906.02909) 19

[YDY <sup>+</sup> 19] Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, and Quoc V.

Le. Xlnet: Generalized autoregressive pretraining for language understanding, 2019,
[arXiv:1906.08237.](http://arxiv.org/abs/arXiv:1906.08237) 2

[ZK16] Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. _Procedings_ _of_ _the_ _British_
_Machine Vision Conference 2016_, 2016. [doi:10.5244/c.30.87.](http://dx.doi.org/10.5244/c.30.87) 18

[ZKZ <sup>+</sup> 15] Yukun Zhu, Ryan Kiros, Rich Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. Aligning books and movies: Towards story-like visual explanations by
watching movies and reading books. _2015 IEEE International Conference on Computer Vision_
_(ICCV)_, Dec 2015. [doi:10.1109/iccv.2015.11.](http://dx.doi.org/10.1109/iccv.2015.11) 7

[ZLN <sup>+</sup> 19] Guodong Zhang, Lala Li, Zachary Nado, James Martens, Sushant Sachdeva, George E. Dahl,
Christopher J. Shallue, and Roger B. Grosse. Which algorithmic choices matter at which batch
sizes? insights from a noisy quadratic model. _CoRR_ [, abs/1907.04164, 2019, 1907.04164.](http://arxiv.org/abs/1907.04164) URL
`[http://arxiv.org/abs/1907.04164](http://arxiv.org/abs/1907.04164)` . 12, 18


30


