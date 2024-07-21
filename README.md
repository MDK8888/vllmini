# vllmini
A kernel-up, minimal implementation of vllm.

## About this Project
Since [vllm](https://github.com/vllm-project/vllm) was introduced last year, it has taken the world by storm - it's the best open source AI inference engine out there. With over 20,000 stars, 3,000 forks, and hundreds of contributors, it's many things. Fast. Powerful. Scalable. 

Given that that's the case, then, why build this project? Why build a smaller, less powerful version of a project that people already use in production? The answer to that, my dear reader, lies not in the project as it currently is, but rather in its future. We created vllmini not to be used in production, but as 
a stepping stone for developers and scientists who are just dipping their toes into the deep ocean of AI infrastructure. The way that we see it, the more 
people that understand vllm, the more contributions it can receive, and the virtuous cycle of open source development continues! 

With that philosophical explanation out of the way, we can now dive into the technical details. We build vllm from the ground upwards, starting from the same kernels as vllm, then building GPT2 with the kernels integrated, followed by the KVCache manager, the request scheduler, and finally the FastAPI server on top. We will go through each step of this inference stack, diving deep into how each individual part works and how they all fit together. 

Are you ready? Let's get started with installation 😎

## Getting Started
To get started, you want to make sure that you're on a CUDA-enabled machine. Pop open your linux terminal, and simply run the following in your terminal:
```
./build.sh
``` 

Then, to spin up the server, you simply run:
```
python -m vllmini.main
```

To make requests to the server, just use curl from the terminal:
```
curl -X POST "http://localhost:8000/generate" -H "Content-Type: application/json" -d '{"prompt": "Once upon a time", "max_length": 20}'
```

This will return a seq_id to you. Don't lose it, as you will need it to query the result of your prompt:
```
curl "http://localhost:8000/result/{seq_id}"
```

If everything worked, great! Let's pop open the hood and dive into how vllm really works.

## The KV cache in vllm
If you're familiar with LLM inference, you may know that the intermediate keys and values that the model generates can be stored for future steps in 
token generation. These keys and values are stored in what is (surprisingly) called the key-value cache, or ** KV cache ** for short.

The key and value caches in our implementation are referenced [here](https://github.com/MDK8888/vllmini/blob/66a4056abd71a9081b526d2d3ee5dfd938723171/vllmini/kv_cache.py#L13C9-L15C1): 

```python
    self.key_cache = torch.zeros(num_blocks, num_heads, head_size // 8, block_size, 8, dtype=torch.float16, device='cuda')
    self.value_cache = torch.zeros(num_blocks, num_heads, head_size, block_size, dtype=torch.float16, device='cuda')
```

An important thing to note here is that a traditional KV cache stores the intermediate keys and values of a single generation request. However, the KV cache in vllm represents the KV cache ** for ALL generation requests across the lifetime of the server. ** This is an important distinction to make because it 
represents a transition away from local inference towards inference in production. 

Now, you may notice that the shape of the key cache is a little bit funky compared to the value cache, especially with the divide by 8 it has on the head_size. We will ignore this for now and focus on the value cache. 

The first dimension of the value cache is `num_blocks`. Blocks are the top level of vllm's memory hierarchy, and inside of a block we store the keys and values of a sequence of tokens. The next two dimensions are `num_heads` and `head_size`, which should be familiar to anyone who knows how multi-headed attention works. The last dimension is `block_size`, and this represents how many keys and values we can store in each block, where each key and value stored in the block are for ** one token. **

We will dive more deeply into how this KV cache structure interacts with vllm inference, but essentially each request is allocated some blocks in the 
KV cache to begin with, and then we will fill up each block with the key and value vectors of the tokens in generation, allocating more blocks as 
necessary. 

We are now ready to move onto the paged attention kernels, which form the backbone of vllmini and vllm. 

## Paged Attention Kernels 

There are many kernels from vllm, but we choose to focus on two critical ones: `cache_and_reshape` and `paged_attention_v1`. 

### cache_and_reshape

The cache_and_reshape kernel does exactly what it seems. Given the keys, values, key_cache, and value_cache, it will store the key and value into the 
corresponding cache and reshape them. For the sake of brevity, I will not be diving into the details of the kernel, and the first four arguments of the kernel are fairly self explanatory. However, there is one argument that is critical, and this is the `slot_mapping`. Essentially, the slot_mapping represents the ** physical index ** where your key and value will be stored if the KV cache ** were flattened. ** This is different from the block index but it is related. 

To see an example of this, suppose that your block_size is 16. Then, physical index 17 would correspond to block index 1 (17 // 16 = 1), and inside of the 
block your index is also 1 (17 % 16 = 1). More generally, if your block_size is n, then your block index is `physical_index // n`, and the index inside 
of the block is `physical index % n`.

### paged_attention_v1

Ah, yes. This is the kernel that makes vllm and vllmini work, and which represent a fundamental leap forward in AI infrastructure. There are too many arguments to go into and this .README file is long enough as is, so the only important argument I will be covering are the `block_tables`, `seq_len`, and
`max_seq_len`. 

The most important of these arguments is the `block_table`. This is a tensor of shape `(batch_size, max_num_blocks_per_seq)`. Each element in this array 
represents a ** block index ** where a piece of your KV cache resides. For example, if your batch size is 1 and your block_size is 16, your `block_table` could be `torch.tensor([[0, 4, -1, -1]])`, which means that the first 16 key and value vectors are in the first block, the second 16 are in the fifth block, etc. Note that in this case, `max_num_blocks_per_seq` is 4, so you could generate `4 * 16 = 64` tokens in total per sequence. 

With that out of the way, we can dive into GPT2!

## GPT2 with Paged Attention

This section is kind of short since many people are already familiar with GPT2, so I will only dive into the modifications that I made to GPT2Attention. 
The arguments to GPT2Attention now take in the `slot_mapping` and `block_tables` arguments, which are for the kernels above. For both prefill and decoding, we use the `cache_and_reshape` kernel discussed above to store the intermediate key and value vectors. For prefilling, we use vanilla attention, but for decoding, we use `paged_attention_v1`. 

*But what about the layers?* An astute reader may be wondering - you've talked about the KV cache from an abstract perspective, but in practice, transformers
have many different layers, and a single forward pass generates many keys and values, so how do we handle that?

We essentially allocate different slot_mappings and block_tables for different layers to take care of this. For example, the first layer of the transformer 
may get a `slot_mapping` of `torch.tensor([0, 1, 2])`, while the next layer may get `torch.tensor([16, 17, 18])`, if `block_size = 16`. We handle the block_table in a similar manner, where the first block would be `torch.tensor([0, -1, -1, -1])`, and the second block would be `torch.tensor([1, -1, -1, -1])`.

## Block Manager and KV cache Manager 

Climbing up the stack, we now find ourselves at the KVCache class and the BlockManager class. 

### KVCache

This class has many methods, but the most important ones are `allocate_for_prefill` and `append_block`. The former is responsible for returning the initial 
`slot_mappings` and `block_tables` that we will need for the decoding step. To retrieve `slot_mappings`, we take the free blocks and calculate physical indices based off of the free_blocks. For the block_tables, we simply return the free blocks as our initial block_table. Note that we do this for 
however many layers there are in our model, so for GPT2 we would need 12 sets of `slot_mappings` and `block_tables`.

As for the `append_block` method, this is called when we are generating tokens and have filled up our current block. In this case, we take the existing 
free blocks and allocate new blocks for the `block_tables`, modifying them so that they are ready to be used in the next decoding step. 

### BlockManager

This class wraps around `KVCache`, and the key method here is the `decode_step`. This method essentially finds us the `block_tables` and `slot_mappings`
that we need for the next decoding step. It uxses the `append_block` method to modify the `block_tables`, and also modifies the `slot_mappings` for the next iteration of decoding. We find out our current block, and based off of the current block we calculate the physical index where our next key or value vector should go and return it along with the modified `block_tables` from the `append_block` method. 

Ok, this has been a very long introduction, but we are almost done, you got this! 😊 (I am saying this to myself too as I grind this .README file out 
at 10:30 at night 😫)

## Scheduler

Ok, so this is the layer where all of the above comes together. The actual structure and function of this class is very simple. The most important methods in 
this class are `add_sequence` and `run`. In the `add_sequence` method, we will call the `allocate_for_prefill` function to get the initial `block_tables` and `slot_mappings`, and then we will run these through our model to get the logits and to fill the KV cache with the initial key and value vectors. This generation request is then queued. 

In the `run` method, we take a request from the top of the queue, sample from the corresponding logits, call `decode_step` from above to get the new `block_tables` and `slot_mappings` for this iteration of decoding, run these through the model. If we have reached our maxium length or if we have reached
the end-of-sequence token, we do not requeue, otherwise we do. We run this inside of a `while()` loop. 

## Server

This is the topmost layer of the inference stack - if you've made it this far, congratulations! The server essentially runs the scheduler in the background 
across its lifetime. The server has two endpoints: `/generate` and `/result/{seq_id}`. As we've touched upon waaayyyyy up top at the start of this .README
file, `/generate` is used to send requests to the server, and `/result/{seq_id}` can be used to see how much we have generated so far.

## Future Directions 

The most immediate thing that we need to do is to make this project production ready - this involves supporting more models, and overall making it more 
robust so that it can be used in production. In particular, we are interested in integrating the best parts of all the inference stacks that we know the 
technical details of, including [Character.ai](https://research.character.ai/optimizing-inference/?ref=blog.character.ai), and [Fireworks.ai](https://fireworks.ai/blog/fireattention-v2-long-context-inference). We eventually hope to become the firebase of AI inference - lightweight, but still very 
powerful. Essentially, a platform for AI infrastructure for people who don't want to deal with AI infrastructure 😁

## Conclusion 

Anyways, it's almost 11 where I am, so I'm going to head to bed soon. I hope that you had more fun reading this than I had writing it, and you liked my 
jokes 😂! (All jokes aside, working on this project overall and writing this was very enjoyable for me). Until the next update, so long, and happy building!


