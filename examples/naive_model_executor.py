import sys
import typing as t
import uuid

import torch
from lightllm.models.llama2.layer_infer.model import Llama2Model
from lightllm.server.router.model_infer.infer_batch import InferBatch
from lightllm.server.router.model_infer.post_process import sample
from lightllm.server.sampling_params import SamplingParams
from lightllm.server.tokenizer import get_tokenizer
from lightllm.utils.weights_utils import get_weight_dir

MAX_TOTAL_TOKEN_NUM = 8192

INPUTS = {
    0: [
        "Create an analogy comparing the internet to something from nature.",
        "Explain Superconductivity",
    ],
    15: [
        "Explain KV cache in LLM inference",
    ],
    35: [
        "How would you explain quantum computing to a 10-year-old",
        "If colors had personalities, what would purple be like and why?",
        'Write a haiku about climate change without using the words "earth," "climate," or "weather."',
    ],
}



def load_model(weight_dir, max_total_token_num):
    tokenizer = get_tokenizer(weight_dir)
    model = Llama2Model(weight_dir, max_total_token_num=max_total_token_num)
    return model, tokenizer


def gen_new_batch(reqs: t.List[str], model, tokenizer, batch_id=None):
    if batch_id is None:
        batch_id = uuid.uuid4().hex
    mini_batch_reqs = []
    for req in reqs:
        messages = [dict(role="user", content=req)]
        chat_token_ids = tokenizer.apply_chat_template(messages)
        req_id = uuid.uuid4().hex
        d = {
            "request_id": req_id,
            "orig_input": req,
            "input_id": chat_token_ids,
            "output_len": 2048,
            "sampling_param": SamplingParams().to_dict(),
        }
        mini_batch_reqs.append(d)
    mini_batch = InferBatch.init_batch(
        batch_id, mini_batch_reqs,
        torch.float16, torch.cuda.current_device(),
        model.mem_manager, model.vocab_size,
    )
    return mini_batch


# similar to ModelRpcServer's forward
def model_forward(model, batch, is_prefill):
    kwargs = {
        "batch_size": len(batch),
        "total_token_num": batch.nopad_total_token_num,
        "max_len_in_batch": batch.nopad_max_len_in_batch,
        "input_ids": batch.input_ids,
        "b_loc": batch.nopad_b_loc,
        "b_start_loc": batch.nopad_b_start_loc,
        "b_seq_len": batch.nopad_b_seq_len,
        "is_prefill": is_prefill
    }

    logits = model.forward(**kwargs)
    next_token_ids = sample(logits, batch)
    output_dict = {}
    new_input_ids = []        
    next_token_ids = next_token_ids.detach().cpu().numpy()
    for i, (r, all_input_ids, next_token_id) in enumerate(zip(batch.requests, batch.all_input_ids, next_token_ids)):
        all_input_ids.append(int(next_token_id))
        new_input_ids.append(next_token_id)
        batch.all_input_ids[i] = all_input_ids
        batch.input_lengths[i] += 1
        batch.out_token_id_counts[i][next_token_id] += 1
        output_dict[r["request_id"]] = int(next_token_id)

    batch.input_ids = torch.tensor(new_input_ids, dtype=torch.long).cuda()
    batch.nopad_b_start_loc = batch.nopad_b_start_loc + torch.arange(0, len(batch), dtype=torch.int32, device="cuda")
    batch.nopad_total_token_num += len(batch)
    batch.nopad_max_len_in_batch += 1
    batch.nopad_b_seq_len += 1
    return output_dict


def main(weight_dir, max_total_token_num):
    weight_dir = get_weight_dir(weight_dir, ignore_patterns=["*.pth", "*.bin"])
    model, tokenizer = load_model(weight_dir, max_total_token_num)
    eos_id = tokenizer.eos_token_id

    running = True
    step = 0
    decoding_batch = None

    while True:
        new_requests = INPUTS.get(step)

        # we will use a naive scheduling: do prefilling as soon as new
        # requests come in, otherwise do decoding. All requests are
        # grouped in a single batch because we don't need to worry
        # about the situation that total_token_num > max_total_token_num
        if new_requests:
            print(f"new requests added in step {step}")
            mini_batch = gen_new_batch(new_requests, model, tokenizer)
            output_dict = model_forward(model, mini_batch, is_prefill=True)
            if decoding_batch:
                decoding_batch = InferBatch.merge(decoding_batch, mini_batch)
            else:
                decoding_batch = mini_batch
            
        else:
            output_dict = model_forward(model, decoding_batch, is_prefill=False)

        finished_req_ids = set()
        unfinished_req_ids = set()
        for req_id, next_token_id in output_dict.items():
            if next_token_id == eos_id:
                finished_req_ids.add(req_id)
            else:
                unfinished_req_ids.add(req_id)

        if finished_req_ids:
            for req in decoding_batch.requests:
                req_id = req["request_id"]
                if req_id in finished_req_ids:
                    req_txt = tokenizer.decode(req["input_id"])
                    print("\n\n----------------\n")
                    print(f"request {req_id} finished, content: {req_txt}\n\n")

            if unfinished_req_ids:
                decoding_batch = decoding_batch.filter(unfinished_req_ids)
            else:
                running = False

        if not running:
            print("all requests processed")
            break

        step += 1
        if step % 20 == 0:
            print(f"step={step}, batch_size={len(decoding_batch)}")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        model_tag_or_dir = sys.argv[1]
    else:
        model_tag_or_dir = "meta-llama/Llama-3.2-3B-Instruct"
    main(model_tag_or_dir, MAX_TOTAL_TOKEN_NUM)
