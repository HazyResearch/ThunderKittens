'''
Simulating the attention partial computation on a single SM with a few different tiling strategies.
'''

import numpy as np

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

class H100:
    sram_size = 226000
    reg_size = 256000
    tc_size = (64, 64)

    def __init__(self):
        self.sram_util = 0
        self.reg_util = 0
        self.tc_ops = []
        self.tc_util = []
        self.io_ops = []
        self.all_ops = []

    def mm(self, C, A, B, name=''):
        m, k = A.shape
        k, n = B.shape
        flops = m * n * k * 2
        m_util = 1 if m > self.tc_size[0] else m / self.tc_size[0]
        n_util = 1 if n > self.tc_size[1] else n / self.tc_size[1]

        self.tc_ops.append(flops)
        self.tc_util.append(m_util * n_util)

        ret = np.matmul(A, B)

        C[:m, :n] = ret

        self.all_ops.append({
            'type': 'mm',
            'm': m,
            'n': n,
            'flops': flops,
            'm_util': m_util,
            'n_util': n_util,
            'name': name,
        })

    def mma(self, C, A, B, name=''):
        m, k = A.shape
        k, n = B.shape
        flops = m * n * k * 2
        m_util = 1 if m > self.tc_size[0] else m / self.tc_size[0]
        n_util = 1 if n > self.tc_size[1] else n / self.tc_size[1]

        self.tc_ops.append(flops)
        self.tc_util.append(m_util * n_util)

        ret = np.matmul(A, B)

        C[:m, :n] += ret

        self.all_ops.append({
            'type': 'mma',
            'm': m,
            'n': n,
            'flops': flops,
            'm_util': m_util,
            'n_util': n_util,
            'name': name,
        })

    def alloc_sram(self, m, n):
        size = 2 * m * n
        if self.sram_util + size > self.sram_size:
            raise Exception("Out of SRAM")
        self.sram_util += size
        return np.zeros((m, n), dtype=np.float16)
    
    def alloc_reg(self, m, n, multiplier=1):
        # multiplier is for multiple warps/warpgroups
        size = m * n * multiplier
        if self.reg_util + size > self.reg_size:
            raise Exception("Out of registers")
        self.reg_util += size
        return np.zeros((m, n), dtype=np.float16)

    def load(self, dst, src, name=''):
        assert dst.shape == src.shape
        m, n = dst.shape
        self.io_ops.append(m * n * 2)
        dst = src
        self.all_ops.append({
            'type': 'load',
            'm': m,
            'n': n,
            'name': name,
        })

    def store(self, dst, src, name=''):
        assert dst.shape == src.shape
        m, n = dst.shape
        self.io_ops.append(m * n * 2)
        dst = src
        self.all_ops.append({
            'type': 'store',
            'm': m,
            'n': n,
            'name': name,
        })

    def get_io(self):
        return sum(self.io_ops)
    
    def get_tc_util(self):
        if len(self.tc_ops) == 0:
            return 0
        total_flops = sum(self.tc_ops)
        total_flop_util = sum([flops * util for flops, util in zip(self.tc_ops, self.tc_util)])
        return total_flop_util / total_flops
    
    def get_flops(self):
        return sum(self.tc_ops)
    
    def print_stats(self):
        print(f'IO: {self.get_io()}')
        print(f'TC Util: {self.get_tc_util()}')
        print(f'Flops: {self.get_flops()}')
        print(f'SRAM Util: {self.sram_util / self.sram_size}')
        print(f'Reg Util: {self.reg_util / self.reg_size}')

    def print_ops_timeline(self):
        # go through all the ops, and print the timeline visually
        # load and store ops are on the same line
        # mm and mma ops are on the line below
        # blocks are stacked and overlap with each other on the timeline
        if not self.all_ops:
            print("No operations to display")
            return
        
        # Set a fixed timeline length
        timeline_length = 300
        
        # Create two lines for the timeline
        io_line = [" "] * timeline_length
        compute_line = [" "] * timeline_length
        
        # Track current time position
        load_time = 0
        compute_time = -1
        
        for op in self.all_ops:
            op_type = op['type']
            name_str = op.get('name', '')
            
            # Use a fixed duration for visualization
            fixed_duration = 8
            
            # Format the operation visualization
            if op_type == 'load':
                prefix = "L:"
                op_str = f"{prefix}{name_str}"
                # Calculate padding needed to make the total length fixed_duration - 2 (for the | characters)
                padding = fixed_duration - 2 - len(op_str)
                op_visual = f"|{op_str}{'-' * max(0, padding)}|"
                line = io_line
            elif op_type == 'store':
                prefix = "S:"
                op_str = f"{prefix}{name_str}"
                padding = fixed_duration - 2 - len(op_str)
                op_visual = f"|{op_str}{'-' * max(0, padding)}|"
                line = io_line
            elif op_type == 'mm':
                prefix = "MM:"
                op_str = f"{prefix}{name_str}"
                padding = fixed_duration - 2 - len(op_str)
                op_visual = f"|{op_str}{'-' * max(0, padding)}|"
                line = compute_line
            elif op_type == 'mma':
                prefix = "MMA:"
                op_str = f"{prefix}{name_str}"
                padding = fixed_duration - 3 - len(op_str)
                op_visual = f"|{op_str}{'-' * max(0, padding)}|"
                line = compute_line
            
            # Place the operation in the timeline
            for i in range(len(op_visual)):
                if op_type in ['load', 'store']:
                    if load_time < compute_time:
                        load_time = compute_time
                    if load_time + i < timeline_length:
                        line[load_time + i] = op_visual[i]
                else:
                    if compute_time == -1:
                        compute_time = load_time
                    if compute_time < load_time:
                        compute_time = load_time - fixed_duration
                    if compute_time + i < timeline_length:
                        line[compute_time + i] = op_visual[i]
            
            # Advance time for all operations to avoid gaps
            # For compute operations, we'll advance less to create some overlap
            if op_type in ['load', 'store']:
                load_time += fixed_duration
            else:
                # For compute operations, we advance but with some overlap
                compute_time += fixed_duration
        
        # Print the timeline
        print("Operations Timeline:")
        print("I/O:     " + "".join(io_line))
        print("Compute: " + "".join(compute_line))
        print("Legend: L=Load, S=Store, MM=Matrix Multiply, MMA=Matrix Multiply Accumulate")

class B200(H100):
    sram_size = 256000
    reg_size = 256000
    tm_size = 256000
    tc_size = (128, 128)

    def __init__(self):
        super().__init__()
        self.tm_util = 0

    def alloc_tm(self, m, n):
        size = m * n * 2
        if self.tm_util + size > self.tm_size:
            raise Exception("Out of TM")
        self.tm_util += size
        return np.zeros((m, n), dtype=np.float16)
    
    def print_stats(self):
        super().print_stats()
        print(f'TM Util: {self.tm_util / self.tm_size}')

class MLASimple:
    def __init__(
        self,
        smclass = H100,
        qrows: int = 64,
        qdim: int = 576,
        krows: int = 32,
        kdim: int = 576,
        vrows: int = 32,
        vdim: int = 512,
        pipeline_depth: int = 3,
    ):
        self.sm = smclass()
        self.qrows = qrows
        self.qdim = qdim
        self.krows = krows
        self.kdim = kdim
        self.vrows = vrows
        self.vdim = vdim

        self.pipeline_depth = pipeline_depth

        self.kblock = [
            self.sm.alloc_sram(self.krows, self.kdim) for _ in range(pipeline_depth)
        ]
        self.qblock = self.sm.alloc_sram(self.qrows, self.qdim)
        self.oblock = self.sm.alloc_reg(self.qrows, self.vdim)
        self.att_reg = self.sm.alloc_reg(self.qrows * 2, self.krows)
        self.att_block = self.sm.alloc_sram(self.qrows, self.krows)

    def run(self, Q, K, V, out, krows_partial):
        '''
        Very simplified MLA implementation, only intended to show tiling strategy.

        Q shape: (64, 572)
        K shape: (krows_partial, 572)

        V is ignored for simplicity

        Out: (64, 572)
        '''
        assert krows_partial % self.krows == 0
        num_partials = krows_partial // self.krows

        # producer load, preload for simplicity
        for i in range(self.pipeline_depth):
            self.sm.load(
                self.kblock[i], K[i * self.krows:(i + 1) * self.krows],
                name='k'
            )

        # consumer setup
        self.sm.load(self.qblock, Q, name='q')
        for i in range(num_partials):
            # consumer compute
            pipline_idx = i % self.pipeline_depth

            # q @ k.T
            self.sm.mm(
                self.att_block, self.qblock, self.kblock[pipline_idx].T,
                name='qk'
            )

            # one part of softmax, skip the scaling for simplicity
            self.att_block = softmax(self.att_block)

            # o = att_block @ v; v == k in this case
            self.sm.mma(
                self.oblock, self.att_block, self.kblock[pipline_idx][:, :V.shape[1]],
                name='av'
            )

            # this is the producer stage, now we can load the next kblock
            next_load_idx = self.pipeline_depth + i
            if next_load_idx < num_partials:
                self.sm.load(
                    self.kblock[next_load_idx % self.pipeline_depth],
                    K[next_load_idx * self.krows:(next_load_idx + 1) * self.krows],
                    name='k'
                )

        # consumer finish
        self.sm.store(out, self.oblock, name='o')

    def stats(self):
        return {
            'io': self.sm.get_io(),
            'tc_util': self.sm.get_tc_util(),
            'flops': self.sm.get_flops(),
            'sram_util': self.sm.sram_util / self.sm.sram_size,
            'reg_util': self.sm.reg_util / self.sm.reg_size,
        }
    
    def print_stats(self):
        self.sm.print_stats()
    
    def repr(self):
        return f'{self.sm.__class__.__name__} {self.__class__.__name__} pipeline {self.pipeline_depth}'
    
class MLAB200(MLASimple):
    def __init__(
        self,
        smclass = B200,
        qrows: int = 128,
        qdim: int = 64,
        krows: int = 128,
        kdim: int = 64,
        vrows: int = 128,
        vdim: int = 128,
        pipeline_depth: int = 5,
    ):
        self.sm = smclass()
        self.qrows = qrows
        self.qdim = qdim
        self.krows = krows
        self.kdim = kdim
        self.vrows = vrows
        self.vdim = vdim
        self.pipeline_depth = pipeline_depth


        self.qblock = [
            self.sm.alloc_sram(self.qrows, self.qdim) for _ in range(pipeline_depth)
        ]
        self.kblock = [
            self.sm.alloc_sram(self.krows, self.kdim) for _ in range(pipeline_depth)
        ]
        # re-using the same memory for vblock
        self.vblock = [
            np.zeros((self.vrows, self.vdim), dtype=np.float16) for _ in range(pipeline_depth)
        ]
        self.oblock_tm = [
            self.sm.alloc_tm(self.qrows, self.vdim)
            for _ in range(2)
        ]
        self.att_block = self.sm.alloc_sram(self.qrows, self.krows)
        self.att_block_tm = self.sm.alloc_tm(self.qrows, self.krows)
        self.att_block_tm2 = self.sm.alloc_tm(self.qrows, self.krows) # just for float
        self.oblock_tm2 = [
            self.sm.alloc_tm(self.qrows, self.vdim)
            for _ in range(2)
        ]

    def run(self, Q, K, V, out, krows_partial):
        '''
        Very simplified MLA implementation, only intended to show tiling strategy.

        Q shape: (128, 576)
        K shape: (krows_partial, 576)
        V shape: (krows_partial, 512)
        Out: (128, 576)
        '''
        assert krows_partial % self.krows == 0
        num_partials = krows_partial // self.krows

        for i in range(num_partials):
            iters = Q.shape[1] // self.qdim + V.shape[1] // self.vdim

            for j in range(self.pipeline_depth - 2):
                # producer load
                self.sm.load(
                    self.kblock[j], K[i * self.krows:(i + 1) * self.krows, j * self.kdim:(j + 1) * self.kdim],
                    name=f'k{j}'
                )
                self.sm.load(
                    self.qblock[j], Q[:, j * self.qdim:(j + 1) * self.qdim],
                    name=f'q{j}'
                )

            # consumer compute
            for j in range(iters):
                # consumer compute
                if j < Q.shape[1] // self.qdim:
                    self.sm.mma(
                        self.att_block, self.qblock[j % self.pipeline_depth], self.kblock[j % self.pipeline_depth].T,
                        name=f'qk{j}'
                    )
                else:
                    jnew = j - Q.shape[1] // self.qdim
                    if jnew < 2:
                        self.sm.mma(
                            self.oblock_tm[jnew], self.att_block, self.vblock[jnew],
                            name=f'av{jnew}'
                        )
                    else:
                        self.sm.mma(
                            self.oblock_tm2[jnew % 2], self.att_block, self.vblock[jnew],
                            name=f'av{jnew}'
                        )

                next_load_idx = self.pipeline_depth + j - 2
                if next_load_idx < iters:
                    # producer load
                    if next_load_idx < Q.shape[1] // self.qdim:
                        self.sm.load(
                            self.kblock[next_load_idx % self.pipeline_depth], K[i * self.krows:(i + 1) * self.krows, next_load_idx * self.kdim:(next_load_idx + 1) * self.kdim],
                            name=f'k{next_load_idx}'
                        )
                        self.sm.load(
                            self.qblock[next_load_idx % self.pipeline_depth], Q[:, next_load_idx * self.qdim:(next_load_idx + 1) * self.qdim],
                            name=f'q{next_load_idx}'
                        )
                    elif next_load_idx - Q.shape[1] // self.qdim < V.shape[1] // self.vdim:
                        actual_load_idx = next_load_idx - Q.shape[1] // self.qdim
                        self.sm.load(
                            self.vblock[actual_load_idx % self.pipeline_depth], V[i * self.krows:(i + 1) * self.krows, actual_load_idx * self.vdim:(actual_load_idx + 1) * self.vdim],
                            name=f'v{actual_load_idx}'
                        )

        
        # consumer finish
        self.sm.store(out[i * self.qrows:(i + 1) * self.qrows, :self.vdim], self.oblock_tm[0], name='o')
        self.sm.store(out[i * self.qrows:(i + 1) * self.qrows, self.vdim:2*self.vdim], self.oblock_tm[1], name='o')
        self.sm.store(out[i * self.qrows:(i + 1) * self.qrows, 2*self.vdim:3*self.vdim], self.oblock_tm2[0], name='o')
        self.sm.store(out[i * self.qrows:(i + 1) * self.qrows, 3*self.vdim:], self.oblock_tm2[1], name='o')

        
def run_exp(smclass=H100, mla_class=MLASimple, qrows=64, krows_partial=256):
    mla = mla_class(smclass=smclass)

    Q = np.random.rand(qrows, 576).astype(np.float16)
    K = np.random.rand(krows_partial, 576).astype(np.float16)
    V = np.random.rand(krows_partial, 512).astype(np.float16)
    out = np.zeros((qrows, 512), dtype=np.float16)

    mla.run(Q, K, V, out, krows_partial)

    return mla, mla.stats(), Q.shape, K.shape, out.shape

def print_exp(smclass=H100, mla_class=MLASimple, qrows=64, krows_partial=256):
    mla, stats, Q_shape, K_shape, out_shape = run_exp(smclass, mla_class, qrows, krows_partial)
    print(f'------{mla.repr()}------')
    print(f'Q shape: {Q_shape}')
    print(f'K shape: {K_shape}')
    print(f'Out shape: {out_shape}')
    mla.print_stats()
    mla.sm.print_ops_timeline()
    print()

if __name__ == '__main__':
    print_exp(H100, MLASimple, qrows=64, krows_partial=128)
    print_exp(B200, MLASimple, qrows=64, krows_partial=128)
    print_exp(B200, MLAB200, qrows=128, krows_partial=128)
