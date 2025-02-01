import random as rd, torch

def generate_tokens_all_insertions(tokens: str, window_len: int = 90, stride_len: int = 42, delta_ratio=0.1, cot_token=[[61, 7277, 63], [754, 7277, 63]]):
    """
    :param tokens_list: 输入的token序列列表（实际上传入的是一个字符串，本函数内部将其转换为token子序列列表）
    :param window_len: 每个子序列的长度，默认为64
    :param stride_len: 窗口移动的步长，默认为42
    :param delta_ratio: 中间位置随机范围的比例（默认中间±10%长度）
    :return: 生成器，每次产出一个元组：(原序列, 对应的插入序列列表, 插入位置)

    The prefix "cot-" and the suffix.
    """
    # 根据窗口长度和步长将整个token序列划分为多个子序列
    tokens_list = [tokens[i:i + window_len] for i in range(0, len(tokens) - window_len + 1, stride_len)]
    print(f'Tokens number: {len(tokens)}, window len: {window_len}, stride len: {stride_len}. split tokens_list size: {len(tokens_list)}\n\n')

    # 遍历所有的子序列
    for tokens in tokens_list:
        if len(tokens) < 1:  # 空序列处理，若当前子序列为空则跳过
            continue

        n = len(tokens)
        # 计算当前子序列的中间位置
        mid = n // 2
        # 根据delta_ratio计算随机范围，确保至少有±1的位置范围
        delta = max(1, int(n * delta_ratio))
        # 在中间位置的左右delta范围内随机选择一个插入位置
        ki_pos = rd.randint(max(0, mid - delta), min(n, mid + delta))
        # ki_pos = 73  # 此行为调试或示例时使用的固定插入位置，目前被注释掉

        # 用于存放为当前子序列生成的所有插入序列
        kot_group = []

        # 遍历子序列中的每个token，为每个token生成一个新的序列，其中token被插入到ki_pos位置
        for token in tokens:
            # 构造新的序列：在ki_pos位置之前的部分 + 当前token + ki_pos位置之后的部分
            new_tokens = tokens[:ki_pos] + cot_token[0] + [token] + cot_token[1] + tokens[ki_pos:]
            kot_group.append(new_tokens)

        # 通过生成器返回当前子序列、对应的所有插入序列以及所选的插入位置
        yield tokens, kot_group, ki_pos


def calculate_subtoken_perplexity(
    model, 
    source_tokens: list, 
    candidate_token_groups: list, 
    key_position: int,  # 关键标记的起始位置（从0开始）
    batch_size: int = 16  # 批处理大小
):
    """
    计算并比较源标记与候选标记组的子标记级困惑度
    参数：
        model: 语言模型
        source_tokens: 源标记序列（token IDs列表）
        candidate_token_groups: 候选标记组的列表（每个元素是token IDs列表）
        key_position: 需要评估的关键子标记起始位置
        batch_size: 批处理大小（影响内存使用）
    """
    # -------------------- 输入校验 --------------------
    if key_position <= 0:
        raise ValueError("关键位置必须大于0（从1开始计数）")

    # -------------------- 准备目标子标记 --------------------
    # 目标子标记：源序列从key_position开始的所有标记
    # 形状：(1, target_length) -> 扩展为 (1, 1, target_length) 用于后续gather操作
    target_subtokens = torch.tensor(source_tokens[key_position:]).unsqueeze(0).unsqueeze(0)

    # -------------------- 计算源序列的log概率 --------------------
    # 获取源序列的logits，并截取关键位置之后的预测结果
    # logits形状: (1, seq_len, vocab_size) -> 截取到 (key_position-1到-2的位置)
    # 即预测位置为key_position开始的所有标记
    source_logits = model.forward([source_tokens])[:, key_position-1:-1, :]  # (1, target_length, vocab_size)
    
    # 计算log概率并提取目标标记对应的值
    source_log_probs = torch.log_softmax(source_logits, dim=2)  # 在词汇维度做归一化
    source_subtoken_logits = torch.gather(
        source_log_probs, 
        dim=2, 
        index=target_subtokens
    ).squeeze(dim=1)

    # -------------------- 分批处理候选序列 --------------------
    candidate_blocks = []
    for batch_start in range(0, len(candidate_token_groups), batch_size):
        batch_candidates = candidate_token_groups[batch_start:batch_start+batch_size]
        
        # 将候选组转换为张量并输入模型
        # 输入形状: (batch_size, seq_len)
        candidate_logits = model.forward(batch_candidates)  # (batch_size, seq_len, vocab_size)
        
        # 截取关键位置之后的预测结果
        candidate_logits = candidate_logits[:, key_position+6:-1, :]  # (batch_size, target_length, vocab_size)
        
        # 计算log概率并提取目标标记对应的值
        candidate_log_probs = torch.log_softmax(candidate_logits, dim=2)
        gathered_logits = torch.gather(
            candidate_log_probs,
            dim=2,
            index=target_subtokens.expand(candidate_log_probs.shape[0], -1, -1)
        ).squeeze()  # (batch_size, target_length)
        
        candidate_blocks.append(gathered_logits)

    # 合并所有候选的log概率
    candidate_subtoken_logits = torch.cat(candidate_blocks)  # (num_candidates, target_length)

    # -------------------- 计算困惑度 --------------------
    # 源序列困惑度计算
    avg_source_log_prob = source_subtoken_logits.mean(dim=1)  # 平均log概率
    source_perplexity = torch.exp(-avg_source_log_prob)  # 困惑度公式：exp(-avg_log_prob)

    # 候选序列困惑度计算（批量处理）
    avg_candidate_log_probs = candidate_subtoken_logits.mean(dim=1)  # (num_candidates,)
    candidate_perplexities = torch.exp(-avg_candidate_log_probs)

    # -------------------- 结果分析 --------------------
    print(f"[Debug] 源序列logits长度: {source_subtoken_logits.shape}，候选logits长度: {candidate_subtoken_logits.shape}")

    # 输出源序列的困惑度
    print(f"源序列困惑度: {source_perplexity.item():.2f}")

    # 统计候选困惑度优于源序列的数量
    num_better_candidates = (candidate_perplexities < source_perplexity).sum().item()
    print(f"优于源序列的候选数量: {num_better_candidates}/{len(candidate_token_groups)}")

    # 找到最佳候选（最低困惑度）
    best_candidate_idx = torch.argmin(candidate_perplexities).item()
    best_ppl = candidate_perplexities[best_candidate_idx].item()
    print(f"最佳候选困惑度: {best_ppl:.2f} (索引:{best_candidate_idx})")

    # -------------------- 可视化标记 --------------------
    # 解码并高亮显示关键位置的变化
    original_token = source_tokens[best_candidate_idx]  # 假设存在一一对应关系
    decoded_char = tokenizer.decode([original_token])
    
    full_sequence = tokenizer.decode(source_tokens)
    # 插入分隔符显示关键位置：...A|BC...
    highlighted_sequence = (
        f"{full_sequence[:key_position]}|{full_sequence[key_position:]}"
    ).replace(decoded_char, f"[{decoded_char}]", 1)  # 高亮第一个差异字符
    
    print("关键位置可视化:\n" + highlighted_sequence)




if __name__ == '__main__':
    from rwkv_tokenizer import TRIE_TOKENIZER
    from v7_seq_model import RWKV_x070_seq
    import os

    os.environ['RWKV_JIT_ON'] = '1'
    os.environ['RWKV_CUDA_ON'] = '1'

    with open('test_data.txt', 'r') as f:
        content_list = f.read().split('\n\n')
    
    model = RWKV_x070_seq('/home/beortust/nfs_share/RwkvModelLib/RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth', device='cuda')
    tokenizer = TRIE_TOKENIZER('rwkv_vocab_v20230424.txt')
    
    for content in content_list:
        tokens = tokenizer.encode(content)
        kot_dataset = generate_tokens_all_insertions(tokens)

        for i, (tl, tg, ip) in enumerate(kot_dataset):
            print('='*64)
            calculate_subtoken_perplexity(model, tl, tg, ip)
            print("\n")

    

