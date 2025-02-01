########################################################################################################
#
# The RWKV-7 "Goose" Language Model - https://github.com/BlinkDL/RWKV-LM
#
########################################################################################################
print(f'\n### RWKV-7 "Goose" enabled ###\n')

import torch, os
from torch.nn import functional as F

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch._C._jit_set_autocast_mode(False)

if os.environ.get('RWKV_JIT_ON') == '1':
    uesModule = torch.jit.ScriptModule
    useFunction = torch.jit.script_method
    useStatic = torch.jit.script
else:
    uesModule = torch.nn.Module
    useFunction = lambda x: x
    useStatic = lambda x: x

HEAD_SIZE = 64
current_path = os.path.dirname(os.path.abspath(__file__))
time_shift = torch.nn.ZeroPad2d((0, 0, 1, -1))

from typing import List

########################################################################################################

class RWKV_x070_seq(uesModule):

    def __init__(self, model_path, rescale_layer: int = 0, device='cpu'):
        super().__init__()
        self.eval()
        with torch.no_grad():
            global HEAD_SIZE
            z = torch.load(
                model_path + ".pth" if ".pth" not in model_path else model_path,
                map_location="cpu",
                weights_only=True,
            )
            self.n_head, self.head_size = z["blocks.0.att.r_k"].shape
            self.vocab_size, self.n_embd = z["emb.weight"].shape
            self.rescale_layer = rescale_layer
            self.device = device

            keys = list(z.keys())
            self.n_layer = 0
            for k in keys:
                layer_id = int(k.split(".")[1]) if ("blocks." in k) else 0
                self.n_layer = max(self.n_layer, layer_id + 1)
                if (
                    "key.weight" in k
                    or "value.weight" in k
                    or "receptance.weight" in k
                    or "output.weight" in k
                    or "head.weight" in k
                ):
                    z[k] = z[k].t()
                z[k] = z[k].squeeze().to(dtype=torch.float32, device=self.device)
                if k.endswith("att.r_k"):
                    z[k] = z[k].flatten()

            assert self.head_size == self.head_size

            z["emb.weight"] = F.layer_norm(
                z["emb.weight"],
                (self.n_embd,),
                weight=z["blocks.0.ln0.weight"],
                bias=z["blocks.0.ln0.bias"],
            )
            z["blocks.0.att.v0"] = z["blocks.0.att.a0"]  # actually ignored
            z["blocks.0.att.v1"] = z["blocks.0.att.a1"]  # actually ignored
            z["blocks.0.att.v2"] = z["blocks.0.att.a2"]  # actually ignored

            self.z = z
            HEAD_SIZE = self.head_size
            print(f"Model Structure： \n    L{self.n_layer}D{self.n_embd}")

    @useFunction
    def forward(self, tokens:List[List[int]], full_output:bool=True):
        with torch.no_grad(): 
            z = self.z
            if len(tokens) == 1:
                x = z['emb.weight'][tokens]
                x.unsqueeze_(0)
            else:
                x = torch.cat([z['emb.weight'][ts].unsqueeze_(0) for ts in tokens], dim=0)

            x = x.to(device=self.device)

            v_first = torch.empty_like(x, device=self.device)
            for i in range(self.n_layer):
                block_id = f'blocks.{i}.'
                tmix_layer_id = f'blocks.{i}.att.'
                cmix_layer_id = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[f"{block_id}ln1.weight"], bias=z[f"{block_id}ln1.bias"])

                xx, v_first = RWKV_x070_TMix_seq(i, self.n_head, self.head_size, xx, v_first,
                    z[f"{tmix_layer_id}ln_x.weight"], z[f"{tmix_layer_id}ln_x.bias"],
                    z[f"{tmix_layer_id}x_r"], z[f"{tmix_layer_id}x_w"], z[f"{tmix_layer_id}x_k"], 
                    z[f"{tmix_layer_id}x_v"], z[f"{tmix_layer_id}x_a"], z[f"{tmix_layer_id}x_g"],
                    z[f"{tmix_layer_id}a0"], z[f"{tmix_layer_id}a1"], z[f"{tmix_layer_id}a2"], 
                    z[f"{tmix_layer_id}v0"], z[f"{tmix_layer_id}v1"], z[f"{tmix_layer_id}v2"],
                    z[f"{tmix_layer_id}g1"], z[f"{tmix_layer_id}g2"], 
                    z[f"{tmix_layer_id}w0"], z[f"{tmix_layer_id}w1"], z[f"{tmix_layer_id}w2"], 
                    z[f"{tmix_layer_id}k_k"], z[f"{tmix_layer_id}k_a"], z[f"{tmix_layer_id}r_k"],
                    z[f"{tmix_layer_id}receptance.weight"], z[f"{tmix_layer_id}key.weight"],
                    z[f"{tmix_layer_id}value.weight"], z[f"{tmix_layer_id}output.weight"]
                    )
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[f"{block_id}ln2.weight"], bias=z[f"{block_id}ln2.bias"])

                xx = RWKV_x070_CMix_seq(xx, z[f"{cmix_layer_id}x_k"], 
                    z[f'{cmix_layer_id}key.weight'], z[f'{cmix_layer_id}value.weight']
                    )
                x = x + xx

                if self.rescale_layer > 0:
                    if (i+1) % self.rescale_layer == 0:
                        x = x / 2

            if not full_output: x[-1,:] 

            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = x @ z['head.weight']

            return x.cpu()


########################################################################################################

if os.environ.get('RWKV_CUDA_ON') == '1':
    from torch.utils.cpp_extension import load

    load(name="wkv7", sources=["cuda/wkv7_op.cpp", f"cuda/wkv7.cu"], is_python_module=False,
                        verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"])
    
    class WKV_7(torch.autograd.Function):
        with torch.no_grad():
            @staticmethod
            def forward(ctx, r, w, k, v, a, b):
                with torch.no_grad():
                    B, T, C = r.size()
                    H = C // HEAD_SIZE

                    assert HEAD_SIZE == C // H
                    assert w.dtype == torch.float32
                    assert all(x.is_contiguous() for x in [r,w,k,v,a,b])

                    y = torch.empty((B, T, C), device=k.device, dtype=w.dtype, memory_format=torch.contiguous_format)
                    torch.ops.wkv7.forward(B, T, C, H, r, w, k, v, a, b, y)
                    return y

    def RWKV7_OP(r, w, k, v, a, b):
        return WKV_7.apply(r, w, k, v, a, b)

else:
    def RWKV7_OP(r, w, k, v, a, b):
        with torch.no_grad():
            B, T, C = r.size()
            H = C // HEAD_SIZE
            N = HEAD_SIZE
            r = r.view(B, T, H, N).float()
            k = k.view(B, T, H, N).float()
            v = v.view(B, T, H, N).float()
            a = a.view(B, T, H, N).float()
            b = b.view(B, T, H, N).float()
            w = torch.exp(-torch.exp(w.view(B, T, H, N).float()))
            out = torch.zeros((B, T, H, N), device=r.device, dtype=torch.float)
            state = torch.zeros((B, H, N, N), device=r.device, dtype=torch.float)

            for t in range(T):
                kk = k[:, t, :].view(B, H, 1, N)
                rr = r[:, t, :].view(B, H, N, 1)
                vv = v[:, t, :].view(B, H, N, 1)
                aa = a[:, t, :].view(B, H, N, 1)
                bb = b[:, t, :].view(B, H, 1, N)
                state = state * w[: , t, :, None, :] + state @ aa @ bb + vv @ kk
                out[:, t, :] = (state @ rr).view(B, H, N)

            return out.view(B, T, C)

 
@useStatic
def RWKV_x070_TMix_seq(layer_id: int, H:int, N:int, x_ln, v_first, ln_w, ln_b, x_r, x_w, x_k, x_v, x_a, x_g, a0, a1, a2, v0, v1, v2, g1, g2, w0, w1, w2, k_k, k_a, r_k, rw, kw, vw, ow):
    with torch.no_grad():
        B, T, C = x_ln.size()
        x = time_shift(x_ln) - x_ln
        xr, xw, xk, xv, xa, xg = x_ln+x*x_r, x_ln+x*x_w, x_ln+x*x_k, x_ln+x*x_v, x_ln+x*x_a, x_ln+x*x_g

        r = xr @ rw
        k = xk @ kw
        v = xv @ vw

        w = -F.softplus(-(w0 + torch.tanh(xw @ w1) @ w2)) - 0.5
        a = torch.sigmoid(a0 + (xa @ a1) @ a2)
        g = torch.sigmoid(xg @ g1) @ g2

        kk = F.normalize((k * k_k).view(B, T, H, N), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a-1) * k_a)
        if layer_id == 0: v_first = v
        else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

        x = RWKV7_OP(r, w, k, v, -kk, kk*a)
        x = F.group_norm(x.view(B*T, C), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(B, T, C)

        x = x + ((r * k * r_k).view(B, T, H, N).sum(dim=-1, keepdim=True) * v.view(B, T, H, N)).view(B, T, C)
        out = (x * g) @ ow

        return out , v_first


@useStatic
def RWKV_x070_CMix_seq(x_ln, x_k, kw, vw):
    with torch.no_grad():
        xx = time_shift(x_ln) - x_ln
        kx = x_ln + xx * x_k
        vx = torch.relu(kx @ kw) ** 2

        return vx @ vw


########################################################################################################
# Utils
########################################################################################################


def sample_logits(logits: torch.tensor, temperature: float = 1.4, top_p: float = 0.3):
    with torch.no_grad():
        # 确保logits为浮点类型并应用温度调节
        logits = logits.float()
        if temperature != 1.0:
            logits.div_(temperature)
        
        # 计算概率分布
        probs = F.softmax(logits, dim=-1)
        
        # # Top-k过滤
        # if top_k > 0:
        #     # 高效获取topk并创建掩码
        #     topk_probs, topk_indices = torch.topk(probs, k=top_k, dim=-1)
        #     mask = torch.zeros_like(probs, dtype=torch.bool)
        #     mask.scatter_(-1, topk_indices, True)
        #     probs = torch.where(mask, probs, 0.0)
        
        # Top-p过滤（核采样）
        if top_p < 1.0:
            # 排序并计算累积概率
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # 找到满足累积概率≥top_p的最小截断点
            cutoff_index = torch.searchsorted(cumulative_probs, top_p, side='left')
            cutoff_index = cutoff_index.clamp(max=cumulative_probs.size(-1)-1) + 1  # 包含当前索引
            
            # 生成保留掩码
            keep_mask = torch.zeros_like(probs, dtype=torch.bool)
            keep_indices = sorted_indices[..., :cutoff_index]
            keep_mask.scatter_(-1, keep_indices, True)
            probs = torch.where(keep_mask, probs, 0.0)
        
        # 数值稳定性处理（防止全零）
        probs += 1e-10  # 避免除零错误
        return torch.multinomial(probs, num_samples=1).item()


if __name__ == '__main__':
    # content = '难道你又不更远一点想到这样枝枝叶叶靠紧团结，力求上进的白杨树，\
    #     宛然象征了今天在华北平原纵横决荡用血写出新中国历史的那种精神和意志。'
    content = "那是力争上游的一种树，笔直的干，笔直的枝。它的干呢，通常是丈把高，像是加以人工似的，一丈以内，绝无旁枝;\
它所有的丫枝呢，一律向上，而且紧紧靠拢，也像是加以人工似的，成为一束，绝无横斜逸出;它的宽大的叶子也是片片向上，\
几乎没有斜生的，更不用说倒垂了;它的皮，光滑而有银色的晕圈，微微泛出淡青色。这是虽在北方的风雪的压迫下却保持着倔强挺立的一种树!\
哪怕只有碗来粗细罢，它却努力向上发展，高到丈许，两丈，参天耸立，不折不挠，对抗着西北风。这就是白杨树，西北极普通的一种树，\
然而决不是平凡的树!它没有婆娑的姿态，没有屈曲盘旋的虬枝，也许你要说它不美丽，──如果美是专指“婆娑”或“横斜逸出”之类而言，\
那么白杨树算不得树中的好女子;但是它却是伟岸，正直，朴质，严肃，也不缺乏温和，更不用提它的坚强不屈与挺拔，它是树中的伟丈夫!\
当你在积雪初融的高原上走过，看见平坦的大地上傲然挺立这么一株或一排白杨树，难道你就只觉得树只是树，难道你就不想到它的朴质，严肃，\
坚强不屈，至少也象征了北方的农民;难道你竟一点也不联想到，在敌后的广大土地上，到处有坚强不屈，就像这白杨树一样傲然挺立的守卫他们家乡的哨兵!\
难道你又不更远一点想到这样枝枝叶叶靠紧团结，力求上进的白杨树，宛然象征了今天在华北平原纵横决荡用血写出新中国历史的那种精神和意志。"

    from rwkv_tokenizer import TRIE_TOKENIZER
    tokenizer = TRIE_TOKENIZER('rwkv_vocab_v20230424.txt')

    model = RWKV_x070_seq('D:\RWKV-Runner\models\RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth')
    i_tokens = tokenizer.encode(content)
    split = len(i_tokens) // 4
    tokens_list = [i_tokens[i*split: (i+1)*split] for i in range(4)]

    out = model.forward(i_tokens)
    print(out.shape)
    o_tokens = torch.argmax(F.softmax(out, dim=-1), dim=-1).tolist()
    # tokens = [sample_logits(o) for o in out[0]]

    print("="*64)
    print(f'Input tokens number: {len(i_tokens)}\n{content}')
    print("="*64)
    print(f'Output tokens number: {len(o_tokens[-1])}\n{tokenizer.decode(o_tokens[-1])}')