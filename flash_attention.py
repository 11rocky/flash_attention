import numpy as np
import unittest


class Attention:
    @staticmethod
    def _validity_qkv(q: np.ndarray, k: np.ndarray, v: np.ndarray):
        assert q.ndim == 3, "q should be a 3D tensor"      # [batch_size, seq_len, hidden_size]
        assert k.ndim == 3, "k should be a 3D tensor"
        assert v.ndim == 3, "v should be a 3D tensor"
        assert q.shape[0] == k.shape[0], "batch_size of q and k should be the same"
        assert q.shape[2] == k.shape[2] == v.shape[2], "hidden_size of q, k and v should be the same"

    def forward(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
        self._validity_qkv(q, k, v)
        return self.forward_impl(q, k, v)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    


class StandardAttention(Attention):
    @staticmethod
    def softmax(x: np.ndarray):
        max_x = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - max_x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward_impl(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
        denom = np.sqrt(q.shape[-1])
        attn = np.matmul(q, k.transpose(0, 2, 1))       # [batch_size, q_len, k_len]
        attn = attn / denom
        s = self.softmax(attn)
        out = np.matmul(s, v)                        # [batch_size, q_len, hidden_size]
        return out


class FlashAttention(Attention):
    def __init__(self, br: int, bc: int) -> None:
        self.Br = br
        self.Bc = bc
    
    @staticmethod
    def load(global_data: np.ndarray, start: int, end: int, step: int) -> np.ndarray:
        return global_data[:, start * step : end * step]
    
    @staticmethod
    def store(global_data: np.ndarray, local_data: np.ndarray, start: int, end: int, step: int):
        global_data[:, start * step : end * step] = local_data

    def _validity_check(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> None:
        assert q.shape[1] % self.Br == 0 and k.shape[1] % self.Bc == 0, "seq_len should be divisible by block_size"

    def fa_forward_impl(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
        return None

    def forward_impl(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
        self._validity_check(q, k, v)
        return self.fa_forward_impl(q, k, v)


class FlashAttentionV1(FlashAttention):
    def __init__(self, br: int, bc: int) -> None:
        super().__init__(br, bc)
    
    def fa_forward_impl(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
        batch_size, q_len, hidden_size = q.shape
        k_len = k.shape[1]
        # (Line 2): initialize O, l and m
        # O: output, will be updated in a row-block-wise manner
        out = np.zeros((batch_size, q_len, hidden_size))
        # l: exp-sum of each row block, will be the denominator in softmax. 
        # l will be updated in a exponential moving average way.
        l = np.zeros((batch_size, q_len, 1))
        # m: max of each row block, will be part of the numerator in softmax.
        # m will also be updated in a exponential moving average way.
        m = np.zeros((batch_size, q_len, 1))
        m.fill(-np.inf)

        # (Line 3): divide q into row blocks and k, v into column blocks
        Tr = q_len // self.Br       # Tr: number of row blocks
        Tc = k_len // self.Bc       # Tc: number of column blocks

        for j in range(Tc):
            # (Line 6), load the key and value block
            # kj: key block, [batch_size, Bc, hidden_size]
            # vj: value block, [batch_size, Bc, hidden_size]
            kj = self.load(k, j, j + 1, self.Bc)
            vj = self.load(v, j, j + 1, self.Bc)

            # (Line 7): iterate over row blocks
            for i in range(Tr):
                # (Line 8): load the query block. [batch_size, Br, hidden_size]
                qi = self.load(q, i, i + 1, self.Br)
                oi = self.load(out, i, i + 1, self.Br)
                mi = self.load(m, i, i + 1, self.Br)
                li = self.load(l, i, i + 1, self.Br)

                # (Line 9): compute the dot-product attention score
                sij = np.matmul(qi, kj.transpose(0, 2, 1)) / np.sqrt(hidden_size)  # [batch_size, Br, Bc]

                # (Line 10): compute max, softmax, and exp-sum
                mij = np.max(sij, axis=-1, keepdims=True)            # [batch_size, Br, 1]
                pij = np.exp(sij - mij)                              # [batch_size, Br, Bc]
                lij = np.sum(pij, axis=-1, keepdims=True)            # [batch_size, Br, 1]

                # (Line 11): update m and l
                # 11.a. update m, the max of each row block
                m_new = np.maximum(mi, mij)
                # 11.b. update l, the accumulated exp-sum of each row block
                l_new = np.exp(mi - m_new) * li + np.exp(mij - m_new) * lij

                # (Line 12): update output
                temp = li * np.exp(mi - m_new) * oi + np.exp(mij - m_new) * np.matmul(pij, vj)
                temp /= l_new
                self.store(out, temp, i, i + 1, self.Br)

                # (Line 13): store the m and l of current row block to the global m and l
                self.store(m, m_new, i, i + 1, self.Br)
                self.store(l, l_new, i, i + 1, self.Br)

        return out


class FlashAttentionV2(FlashAttention):
    def __init__(self, br: int, bc: int) -> None:
        super().__init__(br, bc)
    
    def fa_forward_impl(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
        batch_size, q_len, hidden_size = q.shape
        k_len = k.shape[1]

        out = np.zeros((batch_size, q_len, hidden_size))
        # (Line 3): divide q into row blocks and k, v into column blocks
        Tr = q_len // self.Br       # Tr: number of row blocks
        Tc = k_len // self.Bc       # Tc: number of column blocks

        for i in range(Tr):
            # (Line 4): load the query block. [batch_size, Br, hidden_size]
            qi = self.load(q, i, i + 1, self.Br)
            # (Line 5): initialize Oi, li and mi
            oi = np.zeros_like(qi)
            li = np.zeros((batch_size, self.Br, 1))
            mi = np.zeros((batch_size, self.Br, 1))
            mi.fill(-np.inf)

            for j in range(Tc):
                # kj: key block, [batch_size, Bc, hidden_size]
                # vj: value block, [batch_size, Bc, hidden_size]
                kj = self.load(k, j, j + 1, self.Bc)
                vj = self.load(v, j, j + 1, self.Bc)

                # (Line 8): compute sij
                sij = np.matmul(qi, kj.transpose(0, 2, 1)) / np.sqrt(hidden_size)  # [batch_size, Br, Bc]
                # (Line 9): compute max, softmax, and exp-sum
                mij = np.maximum(mi, np.max(sij, axis=-1, keepdims=True)) # [batch_size, Br, 1]
                pij = np.exp(sij - mij)                                   # [batch_size, Br, Bc]

                lij = np.exp(mi - mij) * li + np.sum(pij, axis=-1, keepdims=True)            # [batch_size, Br, 1]
                # (Line 10): compute oij
                oij = np.exp(mi - mij) * oi + np.matmul(pij, vj)

                mi = mij
                li = lij
                oi = oij
            oi /= li
            self.store(out, oi, i, i + 1, self.Br)

        return out



class FlashAttentionTest(unittest.TestCase):
    def run_test(self, batch_size, q_len, k_len, hidden_size, row_block_size, col_block_size):
        # generate random inputs
        q = np.random.randn(batch_size, q_len, hidden_size)
        k = np.random.randn(batch_size, k_len, hidden_size)
        v = np.random.randn(batch_size, k_len, hidden_size)

        # standard attention
        attn_std = StandardAttention()
        # flash attention v1
        attn_v1 = FlashAttentionV1(row_block_size, col_block_size)
        # flash attention v2
        attn_v2 = FlashAttentionV2(row_block_size, col_block_size)

        standard_out = attn_std(q, k, v)
        flash_v1_out = attn_v1(q, k, v)
        flash_v2_out = attn_v2(q, k, v)

        eps = 1e-8
        self.assertTrue(np.allclose(standard_out, flash_v1_out, atol=eps))
        self.assertTrue(np.allclose(standard_out, flash_v2_out, atol=eps))

    def test(self):
        batch_size = 2
        for row_block_size in (2, 4):
            for col_block_size in (4, 8):
                for factor in (10, 20):
                    q_len = row_block_size * factor
                    k_len = col_block_size * factor
                    for hidden_size in (8, 16, 32, 64):
                        self.run_test(batch_size, q_len, k_len, hidden_size, row_block_size, col_block_size)


if __name__  == "__main__":
    unittest.main()

