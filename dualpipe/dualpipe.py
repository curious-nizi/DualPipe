from typing import Tuple, List, Union, Callable, Optional

import torch
import torch.nn as nn
import torch.distributed as dist

import dualpipe.comm as comm
from dualpipe.utils import WeightGradStore, run_backward, scatter, gather


class DualPipe(nn.Module):
    def __init__(
        self,
        modules: Tuple[nn.Module, nn.Module],
        batch_dim: int = 0,
        process_group: Optional[dist.ProcessGroup] = None,
        rank_mapping: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        assert next(modules[0].parameters()).device == torch.device(torch.cuda.current_device())
        self.module = nn.ModuleList(modules)
        self.overlapped_forward_backward = type(modules[0]) == type(modules[1]) and hasattr(type(modules[0]), "overlapped_forward_backward")
        self.batch_dim = batch_dim
        self.group = process_group or dist.distributed_c10d._get_default_group()
        self.num_ranks = self.group.size()

        # rank_mapping: Map rank in process_group to actual pp rank.
        # rank_inverse_mapping: Map actual pp rank to rank in process_group.
        if rank_mapping is None:
            rank_mapping = list(range(self.num_ranks)) # [0,1,2,3,4,5,6,7]
        rank_inverse_mapping = [None] * (self.num_ranks + 1) # [None, None, None, None, None, None, None, None, None]
        for i in range(self.num_ranks):
            rank_inverse_mapping[rank_mapping[i]] = i # [0,1,2,3,4,5,6,7,None], pp rank -> process rank

        self.rank = rank_mapping[self.group.rank()] # pp rank
        self.first_rank = rank_inverse_mapping[0] # process rank of the first pp rank (0)
        self.prev_rank = rank_inverse_mapping[self.rank - 1] # process rank of the previous pp rank
        self.next_rank = rank_inverse_mapping[self.rank + 1] # process rank of the next pp rank
        self.last_rank = rank_inverse_mapping[self.num_ranks - 1] # process rank of the last pp rank (7)

        self.is_first_rank = self.rank == 0
        self.is_last_rank = self.rank == self.num_ranks - 1
        self.is_in_second_half = self.rank >= self.num_ranks // 2 # rank = 4,5,6,7
        self.is_middle_rank = (self.rank == self.num_ranks // 2 - 1) or (self.rank == self.num_ranks // 2) # rank = 3,4

    def _reset_states(self) -> None:
        WeightGradStore.clear()

        # inputs are microbatches (numbered from 0 to num_chunks//2-1)
        self.input_chunks: Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]] = ([], [])
        # outputs are activations of each microbatch (numeberd from 0 to num_chunks//2-1)
        self.output_chunks: Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]] = ([], [])
        self.input_grad_chunks: Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]] = ([], [])
        self.output_grad_chunks: Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]] = ([], [])
        self.labels: Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]] = None
        self.loss_chunks: List[torch.Tensor] = []
        self.criterion: Callable = None

        self.current_f_chunk_id: List[int] = [0, 0]
        self.current_b_chunk_id: List[int] = [0, 0]
        self.current_send_f_chunk_id: List[int] = [0, 0]
        self.current_send_b_chunk_id: List[int] = [0, 0]
        self.current_recv_f_chunk_id: List[int] = [0, 0]
        self.current_recv_b_chunk_id: List[int] = [0, 0]
        self.comm_ops: List[dist.P2POp] = []
        self.to_free: List[torch.Tensor] = []

    def _forward_compute_chunk(self, phase: int) -> None:
        # phase specifies which one of the dual pipelines to process
        # the first half of ranks and the second half of ranks are like mirror images of each other
        # phase = 0: the first half of ranks processes the first pipeline, the second half of ranks processes the second pipeline
        # phase = 1: the first half of ranks processes the second pipeline, the second half of ranks processes the first pipeline
        phase ^= self.is_in_second_half
        chunk_id = self.current_f_chunk_id[phase]
        self.current_f_chunk_id[phase] += 1
        inputs = self.input_chunks[phase][chunk_id]
        if self.forward_only: # inference only
            self.input_chunks[phase][chunk_id] = None

        is_last_stage = (self.is_first_rank and phase == 1) or (self.is_last_rank and phase == 0)

        outputs = self.module[phase](*inputs)
        outputs = [outputs] if isinstance(outputs, torch.Tensor) else outputs
        if is_last_stage and self.criterion is not None:
            labels = self.labels[phase][chunk_id]
            loss = self.criterion(*outputs, *labels)
            self.loss_chunks.append(loss)

        if (not is_last_stage) or self.return_outputs:
            self.output_chunks[phase].append(outputs)

    def _backward_compute_chunk(self, phase: int, enable_zb: bool = False) -> None:
        if self.forward_only:
            return

        phase ^= self.is_in_second_half
        chunk_id = self.current_b_chunk_id[phase]
        self.current_b_chunk_id[phase] += 1

        is_last_stage = (self.is_first_rank and phase == 1) or (self.is_last_rank and phase == 0)

        WeightGradStore.enabled = enable_zb
        if is_last_stage:
            loss = self.loss_chunks[chunk_id]
            loss.backward()
            loss.detach_()
        else:
            outputs = self.output_chunks[phase][chunk_id]
            if not self.return_outputs:
                self.output_chunks[phase][chunk_id] = None
            output_grads = self.output_grad_chunks[phase][chunk_id]
            self.output_grad_chunks[phase][chunk_id] = None
            non_empty = [(t, g) for t, g in zip(outputs, output_grads) if g is not None]
            outputs, output_grads = list(zip(*non_empty))
            if len(outputs) > 0:
                run_backward(outputs, output_grads)
        WeightGradStore.enabled = False
        if enable_zb:
            WeightGradStore.flush()

        inputs = self.input_chunks[phase][chunk_id]
        self.input_chunks[phase][chunk_id] = None
        input_grads = [t.grad for t in inputs]
        self.input_grad_chunks[phase].append(input_grads)

    def _forward_backward_compute_chunk(self, phase0: int, phase1: int) -> None:
        if self.forward_only:
            self._forward_compute_chunk(phase0)
            return

        if not self.overlapped_forward_backward:
            self._forward_compute_chunk(phase0)
            self._backward_compute_chunk(phase1)
            return

        # pre-forward
        phase0 ^= self.is_in_second_half
        chunk_id0 = self.current_f_chunk_id[phase0]
        self.current_f_chunk_id[phase0] += 1
        module0 = self.module[phase0]
        inputs0 = self.input_chunks[phase0][chunk_id0]
        is_last_stage0 = (self.is_first_rank and phase0 == 1) or (self.is_last_rank and phase0 == 0)

        if is_last_stage0 and self.criterion is not None:
            labels0 = self.labels[phase0][chunk_id0]
            criterion0 = self.criterion
        else:
            labels0 = []
            criterion0 = None

        # pre-backward
        phase1 ^= self.is_in_second_half
        chunk_id1 = self.current_b_chunk_id[phase1]
        self.current_b_chunk_id[phase1] += 1
        module1 = self.module[phase1]
        is_last_stage1 = (self.is_first_rank and phase1 == 1) or (self.is_last_rank and phase1 == 0)

        if is_last_stage1:
            loss1 = self.loss_chunks[chunk_id1]
            outputs1 = []
            output_grads1 = []
        else:
            loss1 = None
            outputs1 = self.output_chunks[phase1][chunk_id1]
            if not self.return_outputs:
                self.output_chunks[phase1][chunk_id1] = None
            output_grads1 = self.output_grad_chunks[phase1][chunk_id1]
            self.output_grad_chunks[phase1][chunk_id1] = None
            non_empty = [(t, g) for t, g in zip(outputs1, output_grads1) if g is not None]
            outputs1, output_grads1 = list(zip(*non_empty))

        # forward & backward
        outputs0, loss0 = type(module0).overlapped_forward_backward(
            module0, inputs0, criterion0, labels0,
            module1, loss1, outputs1, output_grads1,
        )

        # post-forward
        if (not is_last_stage0) or self.return_outputs:
            self.output_chunks[phase0].append(outputs0)
        if is_last_stage0 and self.criterion is not None:
            self.loss_chunks.append(loss0)

        # post-backward
        inputs = self.input_chunks[phase1][chunk_id1]
        self.input_chunks[phase1][chunk_id1] = None
        input_grads1 = [t.grad for t in inputs]
        self.input_grad_chunks[phase1].append(input_grads1)

    def _forward_chunk(self, phase: int, recv: bool = True, send: bool = True) -> None:
        if recv:
            self._recv_forward(phase)
        self._commit_and_wait_comm()

        self._forward_compute_chunk(phase) # forward pass

        if send:
            self._send_forward(phase)

    def _backward_chunk(self, phase: int, enable_zb: bool = False, recv: bool = True, send: bool = True) -> None:
        if recv:
            self._recv_backward(phase)
        self._commit_and_wait_comm()

        self._backward_compute_chunk(phase, enable_zb)

        if send:
            self._send_backward(phase)

    def _forward_backward_chunk(self, phase0: int, phase1: int, recv0: bool = True) -> None:
        if recv0:
            self._recv_forward(phase0)
        self._recv_backward(phase1)
        self._commit_and_wait_comm()

        self._forward_backward_compute_chunk(phase0, phase1)

        self._send_forward(phase0)
        self._send_backward(phase1)

    def _weight_chunk(self) -> None:
        if self.forward_only:
            return

        self._commit_and_wait_comm()

        # Assume FIFO
        WeightGradStore.pop()

    def _free_tensors(self) -> None:
        for tensor in self.to_free:
            assert tensor._base is None, f"pipeline stage should not return view tensors {dist.get_rank(), tensor.shape}"
            tensor.data = torch.Tensor()
        self.to_free = []

    def _recv_forward(self, phase: int) -> None:
        phase ^= self.is_in_second_half
        is_first_stage = (self.is_first_rank and phase == 0) or (self.is_last_rank and phase == 1)
        if is_first_stage:
            return

        self.current_recv_f_chunk_id[phase] += 1
        # for the first pipeline (phase=0, top down), recv from the previous rank
        # for the second pipeline (phase=1, bottom up), recv from the next rank
        tensors = comm.append_irecv(self.comm_ops, self.prev_rank if phase == 0 else self.next_rank, self.group)
        self.input_chunks[phase].append(tensors)

    def _send_forward(self, phase: int) -> None:
        phase ^= self.is_in_second_half
        is_last_stage = (self.is_first_rank and phase == 1) or (self.is_last_rank and phase == 0)
        if is_last_stage:
            return

        chunk_id = self.current_send_f_chunk_id[phase]
        self.current_send_f_chunk_id[phase] += 1
        tensors = self.output_chunks[phase][chunk_id]

        comm.append_isend(self.comm_ops, tensors, self.next_rank if phase == 0 else self.prev_rank, self.group)

        if not self.return_outputs:
            self.to_free.extend(tensors)

    def _recv_backward(self, phase: int) -> None:
        if self.forward_only:
            return

        phase ^= self.is_in_second_half
        is_last_stage = (self.is_first_rank and phase == 1) or (self.is_last_rank and phase == 0)
        if is_last_stage:
            return

        self.current_recv_b_chunk_id[phase] += 1
        tensors = comm.append_irecv(self.comm_ops, self.next_rank if phase == 0 else self.prev_rank, self.group)
        self.output_grad_chunks[phase].append(tensors)

    def _send_backward(self, phase: int) -> None:
        if self.forward_only:
            return

        phase ^= self.is_in_second_half
        is_first_stage = (self.is_first_rank and phase == 0) or (self.is_last_rank and phase == 1)
        if is_first_stage:
            return

        chunk_id = self.current_send_b_chunk_id[phase]
        self.current_send_b_chunk_id[phase] += 1
        tensors = self.input_grad_chunks[phase][chunk_id]
        self.input_grad_chunks[phase][chunk_id] = None

        comm.append_isend(self.comm_ops, tensors, self.prev_rank if phase == 0 else self.next_rank, self.group)

    def _commit_and_wait_comm(self) -> None:
        if not self.comm_ops:
            return
        reqs = dist.batch_isend_irecv(self.comm_ops)
        for req in reqs:
            req.wait()
        self.comm_ops = []
        self._free_tensors()

    def step(
        self,
        # if i'm the first pp rank, inputs is the first half of the batch
        # if i'm the last pp rank, inputs is the second half of the batch
        *inputs: Optional[torch.Tensor],
        num_chunks: int = 0,
        criterion: Optional[Callable] = None,
        # if i'm the first pp rank, labels is the second half of the batch's labels
        # if i'm the last pp rank, labels is the first half of the batch's labels
        labels: List[Optional[torch.Tensor]] = [],
        return_outputs: bool = False,
    ) -> Tuple[Optional[torch.Tensor], Optional[Union[torch.Tensor, Tuple[torch.Tensor]]]]:
        """
        Execute a training or inference step.

        Arguments:
            *inputs: Module inputs. Required only on the first/last ranks.
            num_chunks: The number of micro-batches.
            criterion: Loss function, invoked as ``criterion(*outputs, *labels)``. Required only on the first/last ranks.
            labels: Labels of the loss function. Required only on the first/last ranks.
                labels on the first rank corresponds to inputs on the last rank.
                labels on the last rank corresponds to inputs on the first rank.
            return_outputs: Whether to return outputs on the first/last ranks. Default: ``False``.

        Returns: (loss, outputs)
            loss: Loss for the batch.
                loss on the first rank corresponds to inputs on the last rank.
                loss on the last rank corresponds to inputs on the first rank.
                Otherwise: ``None``.
            outputs: Returned only if ``return_outputs=True``.
                outputs on the first rank corresponds to inputs on the last rank.
                outputs on the last rank corresponds to inputs on the first rank.
                Otherwise: ``None``.

        """
        # need to know the shape of a microbatch (microbatch_size, seq_len, hidden_size)
        assert comm.TENSOR_SHAPES is not None and comm.TENSOR_DTYPE is not None, \
            "You need to call set_p2p_tensor_shapes and set_p2p_tensor_dtype before doing a step."
        self.forward_only = not torch.is_grad_enabled()
        self.return_outputs = return_outputs

        # suppose self.num_ranks = 8, self.rank = 1
        rank = self.rank
        num_ranks = self.num_ranks
        assert num_ranks % 2 == 0
        assert num_chunks > 0 and num_chunks % 2 == 0 and num_chunks >= num_ranks * 2, f"{num_chunks=}, {num_ranks=}"
        num_half_ranks = num_ranks // 2 # num_half_ranks = 4
        half_rank = min(rank, num_ranks - 1 - rank) # half_rank = min(1, 8-1-1) = min(1, 6) = 1
        # rank:      0 1 2 3 4 5 6 7
        # 7-rank:    7 6 5 4 3 2 1 0
        # half_rank: 0 1 2 3 3 2 1 0
        
        # suppose num_chunks (microbatches) = 20
        half_num_chunks = num_chunks // 2 # 10
        self.num_half_ranks = num_half_ranks # 4
        self.half_rank = half_rank

        if not self.forward_only and (self.is_first_rank or self.is_last_rank):
            assert criterion is not None # loss function is required on the first/last ranks

        self._reset_states()

        # self.batch_dim: the dimension along which to split the batch (default: 0)
        # note: inputs is either the *first half* of the batch or the *second half* of the batch
        # therefore, we only need to split inputs further by half_num_chunks.
        # before scatter: inputs = (tensor1, tensor2)
        # after scatter: inputs = [(tensor1_chunk1, tensor2_chunk1), , (tensor1_chunk10, tensor2_chunk10)]
        inputs = scatter(inputs, half_num_chunks, self.batch_dim)
        labels = scatter(labels, half_num_chunks, self.batch_dim)
        if self.is_first_rank:
            self.input_chunks = (inputs, [])
            self.labels = ([], labels)
        elif self.is_last_rank:
            self.input_chunks = ([], inputs)
            self.labels = (labels, [])
        self.criterion = criterion

        # For the first half of the ranks: phase 0 means forward direction, phase 1 means reverse direction.
        # For the second half of the ranks: phase 0 means reverse direction, phase 1 means forward direction.

        # Step 1: nF0
        # loop n times, each iteration executes fwd(0)
        # from the persective of the first half of the ranks, pipeline 0 is the first pipeline (top down)
        # from the perspective of the second half of the ranks, pipeline 0 is the second pipeline (bottom up)
        # rank:      0,1,2,3,4,5,6,7
        # half_rank: 0,1,2,3,3,2,1,0
        # step_1:    6,4,2,0,0,2,4,6
        step_1 = (num_half_ranks - half_rank - 1) * 2
        # device 0 (rank=0) processes fwd_0 .. fwd_5
        # device 1 (rank=1) processes fwd_0 .. fwd_3
        # device 2 (rank=2) processes fwd_0 .. fwd_1
        # device 3 (rank=3) skip (step_1=0)
        # device 4 (rank=4) skip (step_1=0)
        # device 5 (rank=5) processes fwd_10 .. fwd_11
        # device 6 (rank=6) processes fwd_10 .. fwd_13
        # device 7 (rank=7) processes fwd_10 .. fwd_15
        for i in range(step_1):
            self._forward_chunk(0)

        # Step 2: nF0F1
        # loop n times, each iteration executes fwd(0), fwd(1)
        step_2 = half_rank + 1
        
        # *prepare* receiving outputs from previous ranks (allocate tensors and append receiving ops to self.comm_ops)
        # rank 0: receive 6 (first rank, no op)
        # rank 1: receive outputs for microbatch 4 from rank 0
        # rank 2: receive outputs for microbatch 2 from rank 1
        # rank 3: receive outputs for microbatch 0 from rank 2
        # rank 4: receive outputs for microbatch 10 from rank 5
        # rank 5: receive outputs for microbatch 12 from rank 6
        # rank 6: receive outputs for microbatch 14 from rank 7
        # rank 7: receive 16 (first rank, no op)
        self._recv_forward(0)
        
        # rank:      0,1,2,3,4,5,6,7
        # half_rank: 0,1,2,3,3,2,1,0
        # step_2:    1,2,3,4,4,3,2,1 # step_2 = half_rank + 1
        for i in range(step_2):
            # alternate processing the first and second pipeline
            # rank 0: processes fwd_6, fwd_10
            # rank 1: processes fwd_4, fwd_10, fwd_5, fwd_11
            # rank 2: processes fwd_2, fwd_10, fwd_3, fwd_11, fwd_4, fwd_12
            # rank 3: processes fwd_0, fwd_10, fwd_1, fwd_11, fwd_2, fwd_12, fwd_3, fwd_13
            # rank 4: processes fwd_10, fwd_0, fwd_11, fwd_1, fwd_12, fwd_2, fwd_13, fwd_3
            # rank 5: processes fwd_12, fwd_0, fwd_13, fwd_1, fwd_14, fwd_2
            # rank 6: processes fwd_14, fwd_0, fwd_15, fwd_1
            # rank 7: processes fwd_16, fwd_0
            
            # only rank 3 and 4 will send outputs to the next rank, others will wait to send at the end of this iteration
            self._forward_chunk(0, recv=False, send=self.is_middle_rank)
            # this recv is seperated out from _forward_chunk
            self._recv_forward(0)
           
            # send outputs to next ranks except in the last iteration of rank 3 and 4 (middle rank)
            self._forward_chunk(1, send=(not self.is_middle_rank) or (i < step_2 - 1))
            
            # non-middle ranks send outputs from the prev _forward_chunk(0) to next ranks
            if not self.is_middle_rank:
                self._send_forward(0)
        
        # Step 3: nB1W1F1 (Use zero bubble)
        # loop n times, each iteration execute bwdi(1),bwdw(1),fwd(1)
        step_3 = num_half_ranks - half_rank - 1
        # rank:      0,1,2,3,4,5,6,7
        # half_rank: 0,1,2,3,3,2,1,0
        # step_3:    3,2,1,0,0,1,2,3 # step_3 = 4 - half_rank - 1 = 3 - half_rank
        for i in range(step_3):
            # rank0: processes bwdi_10,bwdw_10,fwd_11,bwdi_11,bwdw_11,fwd_12,bwdi_12,bwdw_12,fwd_13
            # rank1: processes bwdi_10,bwdw_10,fwd_12,bwdi_11,bwdw_11,fwd_13
            # rank2: processes bwdi_10,bwdw_10,fwd_13
            # rank3: skip (step_3 = 0)
            # rank4: skip (step_3 = 0)
            # rank5: process bwdi_0,bwdw_0,fwd_4
            # rank6: processes bwdi_0,bwdw_0,fwd_2,bwdi_1,bwdw_1,fwd_3
            # rank7: processes bwdi_0,bwdw_0,fwd_1,bwdi_1,bwdw_1,fwd_2,bwdi_2,bwdw_2,fwd_3
            self._backward_chunk(1, enable_zb=True) # enable zero bubble - decouple backward for inputs and backward for weights
            self._recv_forward(1)
            self._weight_chunk()
            self._forward_chunk(1, recv=False)

        # Step 4 (Main step): nF0B1F1B0
        # loop n times, each iteration executes fwd(0),bwd(1),fwd(1),bwd(0)
        step_4 = half_num_chunks - num_ranks + half_rank + 1
        # half_num_chunks - num_ranks + 1 = 10 - 8 + 1 = 3
        # rank:      0,1,2,3,4,5,6,7
        # half_rank: 0,1,2,3,3,2,1,0
        # step_4:    3,4,5,6,6,5,4,3  # step_4 = 3 + half_rank
        for i in range(step_4):
            # rank0: processes fwd_7,bwd_13,fwd_14,bwd_0,fwd_8,bwd_14,fwd_15,bwd_1,fwd_9,bwd_15,fwd_16,bwd_2
            # rank1: processes fwd_6,bwd_12,fwd_14,bwd_0,fwd_7,bwd_13,fwd_15,bwd_1,fwd_8,bwd_14,fwd_16,bwd_2,fwd_9,bwd_15,fwd_17,bwd_3
            # rank2: processes fwd_5,bwd_11,fwd_14,bwd_0,fwd_6,bwd_12,fwd_15,bwd_1,fwd_7,bwd_13,fwd_16,bwd_2,fwd_8,bwd_14,fwd_17,bwd_3,fwd_9,bwd_15,fwd_18,bwd_4
            # rank3: processes fwd_4,bwd_10,fwd_14,bwd_0,fwd_5,bwd_11,fwd_15,bwd_1,fwd_6,bwd_12,fwd_16,bwd_2,fwd_7,bwd_13,fwd_17,bwd_3,fwd_8,bwd_14,fwd_18,bwd_4,fwd_9,bwd_15,fwd_19,bwd_5
            # rank4: processes fwd_14,bwd_0,fwd_4,bwd_10,fwd_15,bwd_1,fwd_5,bwd_11,fwd_16,bwd_2,fwd_6,bwd_12,fwd_17,bwd_3,fwd_7,bwd_13,fwd_18,bwd_4,fwd_8,bwd_14,fwd_19,bwd_5,fwd_9,bwd_15
            # rank5: processes fwd_15,bwd_1,fwd_4,bwd_10,fwd_16,bwd_2,fwd_5,bwd_11,fwd_17,bwd_3,fwd_6,bwd_12,fwd_18,bwd_4,fwd_7,bwd_13,fwd_19,bwd_5,fwd_8,bwd_14
            # rank6: processes fwd_16,bwd_2,fwd_4,bwd_10,fwd_17,bwd_3,fwd_5,bwd_11,fwd_18,bwd_4,fwd_6,bwd_12,fwd_19,bwd_5,fwd_7,bwd_13
            # rank7: processes fwd_17,bwd_3,fwd_4,bwd_10,fwd_18,bwd_4,fwd_5,bwd_11,fwd_19,bwd_5,fwd_6,bwd_12
            if i == 0:
                if self.is_middle_rank:
                    # NOTE: We don't overlap these two chunks to further reduce bubble size.
                    self._forward_chunk(0, recv=False, send=False)
                    self._send_forward(1)
                    self._backward_chunk(1, send=False)
                    self._send_forward(0)
                    self._send_backward(1)
                else:
                    self._forward_backward_chunk(0, 1, recv0=False)
            else:
                self._forward_backward_chunk(0, 1)
            self._forward_backward_chunk(1, 0)

        # Step 5: nB1F1B0
        step_5 = num_half_ranks - half_rank - 1
        for i in range(step_5):
            self._backward_chunk(1)
            self._forward_backward_chunk(1, 0)

        # Step 6: nB1B0 (The second half of the chunks use zero bubble)
        step_6 = half_rank + 1
        enable_zb = False
        for i in range(step_6):
            if i == step_6 // 2 and half_rank % 2 == 1:
                enable_zb = True
            self._backward_chunk(1, enable_zb=enable_zb)
            if i == step_6 // 2 and half_rank % 2 == 0:
                enable_zb = True
            self._backward_chunk(0, enable_zb=enable_zb)

        # Step 7: nWB0 (Use zero bubble)
        step_7 = num_half_ranks - half_rank - 1
        for i in range(step_7):
            self._weight_chunk()
            self._backward_chunk(0, enable_zb=True)

        # Step 8: nW
        step_8 = half_rank + 1
        for i in range(step_8):
            self._weight_chunk()
        assert WeightGradStore.funcs_queue.empty()

        self._commit_and_wait_comm()

        loss, outputs = None, None
        if self.is_first_rank or self.is_last_rank:
            if criterion is not None:
                loss = torch.stack(self.loss_chunks)
            if return_outputs:
                outputs = gather(self.output_chunks[self.is_first_rank], self.batch_dim)
                if len(outputs) == 1:
                    outputs = outputs[0]

        self._reset_states()

        return loss, outputs
